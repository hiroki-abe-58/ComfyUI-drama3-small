"""Text -> speech with drama3-small: prompt, prefill, frame-by-frame decode, codec.

Sampling follows audio.cpp's fish_drama3 generator, which the drama3-small listening tests
used: inclusive top-p then top-k on the main token, a repetition-aware redraw (RAS: when
the draw repeats one of the last 16 generated main tokens, a temperature 1.0 / top-k 30 /
top-p 0.95 draw replaces it) and the same sampler for codes 1..9.

drama3-small's pause fix: after `max_pause_frames` consecutive silence frames, silence codes are
masked from the main draw, so the model must speak or end; a pause past `pause_end_ratio` x the
text's expected length ends the take and `pause_cap_ratio` x ends it regardless. Silence frames
are those whose semantic code is on silence_codes.json (v4: 286 codes that are >= 90 % quiet in
drama3-small's own renders) or on the take's own extension of it: a run of up to 3 unlisted
frames between listed silence frames joins the take's list. That stops the model from escaping
the mask with rarer quiet codes ("10 silent frames, 1 unlisted quiet frame, ..." loops).

Long text is split into chunks of about 200 characters; each chunk sees the previous one as
history. On CUDA (and ROCm) one decode step is captured as a CUDA graph per attended-cache bucket.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from . import kernels
from .codec import SAMPLE_RATE, FishDAC
from .model import Drama3Small
from .text import Take, build_prompt, chunk_text, expected_frames

FPS = SAMPLE_RATE / 2048  # 21.53 frames per second
RAS_WINDOW = 16
ESCAPE_FRAMES = 3  # unlisted frames between listed silence that still count as the same pause
TOP_K_CAP = 128
KV_BUCKETS = (1024, 2048, 4096, 8192)
DEFAULT_SILENCE_CODES = Path(__file__).resolve().parent.parent / "silence_codes.json"


@dataclass
class Sampling:
    temperature: float = 0.7
    top_p: float = 0.7
    top_k: int = 30
    max_frames: int = 645  # per chunk (30 s)
    seed: int = 0
    max_pause_frames: int = 10  # 0 disables the pause fix
    pause_end_ratio: float = 2.0
    pause_cap_ratio: float = 3.0
    chunk_chars: int = 200


@dataclass
class Result:
    audio: Tensor  # float32 [samples] at 44.1 kHz, CPU
    codes: Tensor  # long [10, frames], CPU
    chunks: list[dict] = field(default_factory=list)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE


def sample(logits: Tensor, temperature: Tensor | float, top_k: Tensor | int, top_p: Tensor | float,
           generator: torch.Generator | None = None) -> Tensor:
    """One id per row: softmax(logits / T), keep the top-ranked token, drop ranks whose
    cumulative probability exceeds top_p or whose rank is >= top_k, then draw."""
    if isinstance(temperature, Tensor):
        temperature = temperature.clamp(min=1e-5)
    else:
        temperature = max(temperature, 1e-5)
    scaled = logits.float() / temperature
    lse = torch.logsumexp(scaled, dim=-1, keepdim=True)
    values, ids = scaled.topk(min(TOP_K_CAP, scaled.shape[-1]), dim=-1)
    probs = (values - lse).exp()
    ranks = torch.arange(values.shape[-1], device=logits.device)
    drop = (probs.cumsum(-1) > top_p) | (ranks >= top_k)
    drop[..., 0] = False
    probs = probs.masked_fill(drop, 0.0)
    noise = torch.empty_like(probs).exponential_(generator=generator)
    return ids.gather(-1, (probs / noise).argmax(-1, keepdim=True)).squeeze(-1)


class Decoder:
    """Static buffers and one decode step (optionally CUDA-graphed)."""

    def __init__(self, model: Drama3Small, silence_codes: list[int]) -> None:
        self.model = model
        dev = model.device
        width = 1 + model.tower_cfg.codebooks
        self.hidden = torch.zeros(1, model.text_cfg.hidden, device=dev)
        self.next_hidden = torch.zeros_like(self.hidden)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.frame = torch.zeros(1, width, dtype=torch.long, device=dev)
        self.sampled = torch.zeros(1, width, dtype=torch.long, device=dev)
        self.selected = torch.zeros(1, dtype=torch.long, device=dev)
        self.recent = torch.full((RAS_WINDOW,), -1, dtype=torch.long, device=dev)
        self.mask_silence = torch.zeros((), dtype=torch.bool, device=dev)
        self.temperature = torch.zeros((), device=dev)
        self.top_k = torch.zeros((), dtype=torch.long, device=dev)
        self.top_p = torch.zeros((), device=dev)
        vocab = model.tower_cfg.vocab
        self.silence = torch.zeros(vocab + 1, dtype=torch.bool, device=dev)
        codes = torch.tensor([c for c in silence_codes if 0 <= c < vocab], dtype=torch.long, device=dev)
        self.silence[codes + 1] = True
        self.silence_set = set(int(c) for c in silence_codes)
        self.static_silence = self.silence.clone()
        self.adaptive = os.environ.get("DRAMA3_ADAPTIVE_SILENCE", "1") != "0"
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = None
        self.arena = kernels.StaticArena()
        # A private generator: sampling never touches the process-wide RNG, and a failed graph
        # capture (which leaves its generator in capture mode) only costs this one.
        self.use_graphs = dev.type == "cuda" and os.environ.get("DRAMA3_CUDA_GRAPH", "1") != "0"
        self.gen = _generator(dev)
        self.graph_rng = hasattr(torch.cuda.CUDAGraph, "register_generator_state")
        if self.use_graphs and not self.graph_rng:  # torch < 2.4: graphs can only use the default generator
            self.gen = None

    def body(self, kv_len: int) -> None:
        m = self.model
        tower, vocab = m.tower, m.tower_cfg.vocab
        logits = m.logits(self.hidden)  # [1, 1 + vocab]
        logits = torch.where(self.mask_silence & self.silence, float("-inf"), logits)
        first = sample(logits, self.temperature, self.top_k, self.top_p, self.gen)
        retry = sample(logits, 1.0, 30, 0.95, self.gen)
        repeats = (first != 0) & (self.recent == first).any()
        selected = torch.where(repeats, retry, first)
        self.selected.copy_(selected)
        semantic = selected != 0
        main = torch.where(semantic, selected - 1 + m.semantic_begin, torch.full_like(selected, m.eos))
        c0 = (selected - 1).clamp(0, vocab - 1)
        h = tower.run_shared(self.hidden, c0, semantic, self.pos, kv_len, self.frame)
        codes = [c0]
        for layer, head in zip(tower.residual, tower.heads):
            h = layer(h, codes[-1], semantic, self.pos, kv_len, None, m.dtype)
            codes.append(sample(head(h.to(m.dtype), torch.float32), self.temperature, self.top_k, self.top_p, self.gen))
        frame = torch.stack((main, *codes), dim=-1)
        self.sampled.copy_(frame)
        nxt = self.pos + 1
        self.next_hidden.copy_(m.backbone(m.embed(frame), nxt, kv_len, continue_state=True))

    def bucket(self, position: int) -> int:
        needed = position + 2  # the backbone step writes row position + 1
        for b in KV_BUCKETS:
            if b >= needed and b <= self.model.capacity:
                return b
        return self.model.capacity

    def ensure(self, rows: int) -> None:
        """Caches for `rows` positions plus every graph bucket, before the prefill.

        Capturing runs the decode step, which advances the delta-net states, so it must happen
        before a prefill resets them, never mid-take. Growing the caches re-captures all graphs.
        """
        capacity = next((b for b in KV_BUCKETS if b >= rows), rows)
        if capacity > self.model.capacity:
            self.graphs.clear()
            self.pool = None
            self.arena.clear()
            self.model.allocate(capacity)
        if self.use_graphs:
            for kv_len in sorted({self.bucket(p) for p in (0, *(b - 2 for b in KV_BUCKETS if b <= self.model.capacity))}):
                if kv_len not in self.graphs:
                    self.capture(kv_len)

    def step(self, position: int) -> None:
        kv_len = self.bucket(position)
        if not self.use_graphs:
            self.body(kv_len)
            return
        self.graphs[kv_len].replay()

    def capture(self, kv_len: int) -> None:
        """Warm up (Triton compilation, lazily created tables, static arena) then capture.

        A failed capture switches this decoder to eager steps instead of failing the request."""
        try:
            stream = torch.cuda.Stream(self.model.device)
            stream.wait_stream(torch.cuda.current_stream(self.model.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    kernels.ARENA = self.arena
                    self.arena.begin()
                    try:
                        self.body(kv_len)
                    finally:
                        self.arena.end()
            torch.cuda.current_stream(self.model.device).wait_stream(stream)
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            if self.gen is not None:
                graph.register_generator_state(self.gen)
            self.arena.begin()
            try:
                with torch.cuda.graph(graph, pool=self.pool):
                    self.body(kv_len)
            finally:
                self.arena.end()
            self.graphs[kv_len] = graph
        except Exception as error:  # noqa: BLE001 - any capture problem: decode eagerly
            torch.cuda.synchronize(self.model.device)
            logging.getLogger("drama3-small").warning("CUDA graph capture failed (%s); decoding without graphs", error)
            self.graphs.clear()
            self.use_graphs = False
            self.gen = _generator(self.model.device)


class Drama3TTS:
    """drama3-small loaded from its GGUF: reference encoding, generation and decoding."""

    def __init__(self, gguf: str | Path, device: str | torch.device = "cuda", dtype: torch.dtype | None = None,
                 silence_codes: str | Path | list[int] | None = None, expert_offload: float = 0.0) -> None:
        from tokenizers import Tokenizer

        started = time.time()
        self.model = Drama3Small(str(gguf), device, dtype, expert_offload)
        self.device = self.model.device
        self.codec = FishDAC(self.model.reader, self.device)
        self.tokenizer = Tokenizer.from_str(self.model.files["tokenizer.json"].decode("utf-8"))
        if silence_codes is None or isinstance(silence_codes, (str, Path)):
            path = Path(silence_codes or DEFAULT_SILENCE_CODES)
            data = json.loads(path.read_text())
            silence_codes = data["codes"] if isinstance(data, dict) else data
        self.decoder = Decoder(self.model, list(silence_codes))
        self.load_seconds = time.time() - started

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def to(self, device: str | torch.device) -> "Drama3TTS":
        """Move every weight, cache and buffer (e.g. to the CPU to free VRAM, and back)."""
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device == self.device:
            return self
        self.decoder.graphs.clear()
        self.decoder.pool = None
        self.decoder.arena.clear()
        seen: set[int] = set()
        for obj in (self.model, self.codec, self.decoder):
            _move_tensors(obj, device, seen)
        self.device = self.model.device = self.codec.device = device
        self.decoder.use_graphs = device.type == "cuda" and os.environ.get("DRAMA3_CUDA_GRAPH", "1") != "0"
        self.decoder.gen = _generator(device) if (self.decoder.graph_rng or not self.decoder.use_graphs) else None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def encode_reference(self, audio: Tensor, sample_rate: int) -> Tensor:
        """Waveform [samples] or [channels, samples] -> codes [10, frames] (CPU long)."""
        if audio.ndim == 2:
            audio = audio.mean(0)
        return self.codec.encode(audio, sample_rate).cpu()

    @torch.inference_mode()
    def generate(self, text: str, references: list[Take] | None = None, history: list[Take] | None = None,
                 sampling: Sampling | None = None, progress=None, interrupt=None, streaming: bool = False,
                 on_audio=None, block_frames: int | None = None) -> Result:
        """Speak `text`. references clone a voice (none = the model invents one); history holds
        earlier takes this one continues. progress(done_frames, chunk, chunks) and interrupt()
        (raises to stop) are optional callbacks.

        streaming=True decodes audio while frames are generated, in blocks of `block_frames`
        (default 43, about 2 s), hands each finished block to on_audio(samples) and keeps it in
        host memory: VRAM stays flat however long the text is. The result is the same audio
        either way; without streaming, decoding runs after generation in 4 s blocks."""
        s = sampling or Sampling()
        references, history = list(references or []), list(history or [])
        chunks = chunk_text(text, s.chunk_chars)
        if not chunks:
            raise ValueError("text is empty")
        block = block_frames or (43 if streaming else 86)
        stream = self.codec.stream()
        pieces: list[Tensor] = []
        pending: list[Tensor] = []

        def flush(final: bool = False) -> None:
            while pending and (final or len(pending) >= block):
                take = torch.stack(pending[:block], dim=1)
                del pending[:block]
                audio = stream.push(take).float().cpu()
                pieces.append(audio)
                if on_audio is not None:
                    on_audio(audio)

        def on_frame(codes: Tensor) -> None:
            pending.append(codes)
            flush()

        generated: list[Take] = []
        info = []
        with _seeded(self.device, self.decoder.gen is None):
            for index, chunk in enumerate(chunks):
                context = history + generated[-1:]
                codes, meta = self._generate_chunk(chunk, references, context, s, s.seed + index, progress, index,
                                                   len(chunks), interrupt, on_frame if streaming else None)
                generated.append(Take(chunk, codes))
                info.append({"text": chunk, **meta})
                if not streaming:
                    pending.extend(codes.to(self.device).unbind(1))
            flush(final=True)
        codes = torch.cat([t.codes for t in generated], dim=1)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()  # give prefill / codec scratch back to other models
        return Result(audio=torch.cat(pieces), codes=codes, chunks=info)

    def _generate_chunk(self, text, references, history, s: Sampling, seed, progress, index, total, interrupt,
                        on_frame=None):
        m, dec = self.model, self.decoder
        prompt = build_prompt(self.tokenizer, text, references, history, m.semantic_begin, m.tower_cfg.codebooks)
        rows = prompt.shape[0]
        dec.ensure(rows + s.max_frames + 2)
        if rows + s.max_frames + 2 > m.capacity:
            raise ValueError("prompt plus max length exceeds the context; shorten the reference or text")
        seed = int(seed) & 0xFFFFFFFFFFFF
        if dec.gen is not None:
            dec.gen.manual_seed(seed)
        else:
            _manual_seed(self.device, seed)
        started = time.time()
        dec.hidden.copy_(m.prefill(prompt))
        prefill_s = time.time() - started
        dec.pos.fill_(rows - 1)
        dec.frame.copy_(prompt[-1:].to(self.device))
        dec.recent.fill_(-1)
        dec.temperature.fill_(s.temperature)
        dec.top_k.fill_(max(1, min(int(s.top_k), TOP_K_CAP)))
        dec.top_p.fill_(s.top_p)
        clamp = s.max_pause_frames > 0
        expected = expected_frames(text)
        end_frames, cap_frames = int(s.pause_end_ratio * expected), int(s.pause_cap_ratio * expected)
        silent_run, frames, stop = 0, [], "length"
        # The take's own silence list: the static list plus any short run (<= ESCAPE_FRAMES) of
        # unlisted codes that sits between listed silence frames. Without it the model loops
        # "10 silent frames, 1-2 unlisted quiet frames" under the mask and the pause never ends;
        # with it, each escape route is masked after one use.
        take_silence, pending = set(dec.silence_set), []
        dec.silence.copy_(dec.static_silence)
        started = time.time()
        for step in range(s.max_frames):
            if interrupt is not None:
                interrupt()
            if clamp and step >= cap_frames:
                stop = "length_guard"
                break
            if clamp and silent_run >= s.max_pause_frames and step >= end_frames:
                stop = "pause_after_text"
                break
            forcing = clamp and silent_run >= s.max_pause_frames
            dec.mask_silence.fill_(forcing)
            dec.step(rows - 1 + step)
            frame = dec.sampled[0].clone()
            selected = int(dec.selected.item())
            if selected == 0:
                stop = "eos"
                break
            frames.append(frame)
            if on_frame is not None:
                on_frame(frame[1:])
            code = selected - 1
            if code in take_silence:
                if pending:  # the unlisted frames in between were part of the pause
                    for c in pending:
                        if c not in take_silence:
                            take_silence.add(c)
                            dec.silence[c + 1] = True
                    silent_run += len(pending)
                    pending = []
                silent_run += 1
            elif silent_run > 0 and dec.adaptive and len(pending) < ESCAPE_FRAMES:
                pending.append(code)  # decided by what follows: more silence, or speech
            else:
                silent_run, pending = 0, []
            dec.recent.copy_(torch.cat((dec.recent[1:], dec.selected)))
            dec.hidden.copy_(dec.next_hidden)
            dec.frame.copy_(dec.sampled)
            dec.pos.add_(1)
            if progress is not None:
                progress(len(frames), index, total)
        if not frames:
            raise RuntimeError("drama3-small ended before producing audio; try another seed")
        decode_s = time.time() - started
        codes = torch.stack(frames)[:, 1:].T.cpu()
        return codes, {"frames": len(frames), "stop": stop, "prompt_rows": rows, "prefill_s": round(prefill_s, 3),
                       "decode_s": round(decode_s, 3), "fps": round(len(frames) / max(decode_s, 1e-6), 1)}


def _move_tensors(obj, device: torch.device, seen: set[int]):
    """Replace every tensor reachable through attributes / lists / tuples / dicts by its copy on device."""
    if isinstance(obj, Tensor):
        return obj.to(device)
    if id(obj) in seen or isinstance(obj, (str, bytes, int, float, bool, type(None), torch.dtype, torch.device)):
        return obj
    if getattr(obj, "stay_on_host", False):
        return obj
    seen.add(id(obj))
    if isinstance(obj, list):
        obj[:] = [_move_tensors(v, device, seen) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_move_tensors(v, device, seen) for v in obj)
    elif isinstance(obj, dict):
        for k in list(obj):
            obj[k] = _move_tensors(obj[k], device, seen)
    elif hasattr(obj, "__dict__") and type(obj).__module__.startswith(__package__ or "drama3_torch"):
        for k, v in vars(obj).items():
            setattr(obj, k, _move_tensors(v, device, seen))
    return obj


def _generator(device: torch.device) -> torch.Generator | None:
    try:
        return torch.Generator(device=device)
    except (RuntimeError, TypeError):  # devices without their own generators use the global one
        return None


def _manual_seed(device: torch.device, seed: int) -> None:
    seed = int(seed) & 0xFFFFFFFFFFFF
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)
    else:
        torch.manual_seed(seed)


@contextlib.contextmanager
def _seeded(device: torch.device, uses_global: bool = True):
    """Restore the device's global RNG afterwards when sampling had to use it (torch < 2.4)."""
    if not uses_global:
        yield
        return
    if device.type == "cuda":
        state = torch.cuda.get_rng_state(device)
        try:
            yield
        finally:
            torch.cuda.set_rng_state(state, device)
    else:
        state = torch.get_rng_state()
        try:
            yield
        finally:
            torch.set_rng_state(state)

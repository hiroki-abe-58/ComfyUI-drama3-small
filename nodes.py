"""ComfyUI nodes for drama3-small (pure PyTorch; no audio.cpp needed).

Built with Fish Audio. drama3-small is a compressed derivative of Fish Audio Drama 3 under the
Fish Audio Community License v1.0 (see LICENSE and NOTICE).
"""

from __future__ import annotations

import logging
import os
import re

import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths

from .drama3_torch.generate import FPS, Drama3TTS, Sampling
from .drama3_torch.text import Take

log = logging.getLogger("drama3-small")

MODEL_FOLDER = "drama3"
HF_REPO = "pentacoxian-dev/drama3-small-GGUF"
HF_FILE = "drama3-small-IQ3_XXS.gguf"
DOWNLOAD_SUFFIX = " (download)"
# VRAM to make room for, by expert offload setting (measured peaks plus margin).
VRAM_NEEDED = {"none": 11 * 1024**3, "half": 8 * 1024**3, "all": 5 * 1024**3}
OFFLOAD_SHARE = {"none": 0.0, "half": 0.5, "all": 1.0}
SCRATCH_NEEDED = 2 * 1024**3
CODES_KEY = "drama3_codes"

folder_paths.add_model_folder_path(MODEL_FOLDER, os.path.join(folder_paths.models_dir, MODEL_FOLDER))
if MODEL_FOLDER in folder_paths.folder_names_and_paths:
    paths, _ = folder_paths.folder_names_and_paths[MODEL_FOLDER]
    folder_paths.folder_names_and_paths[MODEL_FOLDER] = (paths, {".gguf"})


def model_choices() -> list[str]:
    files = folder_paths.get_filename_list(MODEL_FOLDER)
    return files if HF_FILE in files else [*files, HF_FILE + DOWNLOAD_SUFFIX]


def resolve_model(name: str) -> str:
    if name.endswith(DOWNLOAD_SUFFIX):
        name = name[: -len(DOWNLOAD_SUFFIX)]
    path = folder_paths.get_full_path(MODEL_FOLDER, name)
    if path:
        return path
    if name != HF_FILE:
        raise FileNotFoundError(f"{name} not found in models/{MODEL_FOLDER}")
    from huggingface_hub import hf_hub_download

    target = folder_paths.get_folder_paths(MODEL_FOLDER)[0]
    os.makedirs(target, exist_ok=True)
    log.info("drama3-small: downloading %s from %s (9.7 GiB) to %s", HF_FILE, HF_REPO, target)
    # local_dir keeps a real file named *.gguf next to the user's other models.
    return hf_hub_download(HF_REPO, HF_FILE, local_dir=target)


def device_choices() -> list[str]:
    choices = ["auto"]
    if torch.cuda.is_available():
        choices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        choices.append("mps")
    return choices + ["cpu"]


def spoken_text(text: str) -> str:
    """The words actually spoken: drop a director's note and leading [style] directions."""
    if text.startswith("Direction:") and "\nLine:" in text:
        text = text.split("\nLine:", 1)[1]
    return re.sub(r"^\s*(\[[^\]]*\]\s*)+", "", text).strip() or text.strip()


class Drama3SmallModel:
    """Holds the engine; offloads to the CPU between runs when asked."""

    def __init__(self, path: str, device: torch.device, dtype: torch.dtype | None, expert_offload: str = "none") -> None:
        self.path, self.device, self.dtype, self.expert_offload = path, device, dtype, expert_offload
        self.tts: Drama3TTS | None = None

    def load(self) -> Drama3TTS:
        if self.device.type in ("cuda", "mps"):
            # Let ComfyUI evict its own models: all of ours when loading, scratch space otherwise.
            resident = self.tts is not None and self.tts.device.type == self.device.type
            mm.free_memory(SCRATCH_NEEDED if resident else VRAM_NEEDED[self.expert_offload], self.device)
        if self.tts is None:
            self.tts = Drama3TTS(self.path, self.device, self.dtype, expert_offload=OFFLOAD_SHARE[self.expert_offload])
            log.info("drama3-small: loaded %s on %s in %.1fs", os.path.basename(self.path), self.device, self.tts.load_seconds)
        else:
            self.tts.to(self.device)
        return self.tts

    def offload(self) -> None:
        if self.tts is not None and self.device.type != "cpu":
            self.tts.to(torch.device("cpu"))
            mm.soft_empty_cache()


class Drama3SmallLoader:
    DESCRIPTION = ("Loads drama3-small, a 9.7 GiB GGUF compression of Fish Audio Drama 3 (Japanese + English, "
                   "expressive). Runs in pure PyTorch; fast on NVIDIA/AMD GPUs with Triton, slow without. "
                   "Built with Fish Audio.")
    CATEGORY = "audio/drama3-small"
    RETURN_TYPES = ("DRAMA3_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "gguf": (model_choices(), {"tooltip": f"GGUF in models/{MODEL_FOLDER}. '(download)' fetches it from "
                                                  f"huggingface.co/{HF_REPO} on first use."}),
            "device": (device_choices(), {"default": "auto"}),
            "precision": (["auto", "bf16", "fp16", "fp32"], {"default": "auto",
                                                               "tooltip": "auto: bf16 where supported, else fp16 (fp32 on CPU)."}),
            "expert_offload": (list(OFFLOAD_SHARE), {"default": "none", "tooltip":
                "Keep routed-expert weights in system RAM; the GPU reads the few it needs per frame over PCIe. "
                "Measured on an RTX 4090: none = 9.6 GB VRAM, 2.3x real time; half = 7.0 GB, 1.7x; all = 4.4 GB, 1.3x. "
                "Same audio. Needs CUDA/ROCm with Triton and about 5 GB of free RAM for 'all'."}),
        }}

    def load(self, gguf, device, precision, expert_offload="none"):
        path = resolve_model(gguf)
        dev = mm.get_torch_device() if device == "auto" else torch.device(device)
        dtype = {"auto": None, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        return (Drama3SmallModel(path, dev, dtype, expert_offload),)


class Drama3SmallTTS:
    DESCRIPTION = ("Speaks text with drama3-small. Without reference audio the model makes up a voice "
                   "(the seed picks it; the optional description steers it). With reference audio and its "
                   "transcript it clones that voice. Supports [tags] like [laughs] [sighs] [pause], "
                   "<whisper>...</whisper> spans and English [style directions]. Long text is split "
                   "into chunks automatically. Only clone voices you have consent to use.")
    CATEGORY = "audio/drama3-small"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "spoken_text")
    OUTPUT_TOOLTIPS = ("44.1 kHz mono speech.",
                       "The text without leading [style] directions: wire it with the audio into another node's "
                       "reference inputs to keep this voice for more lines.")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DRAMA3_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "[with bright eager excitement] 新しいGPUが届いたから、今夜はfine-tuningを回しまくろう！"}),
                "voice_description": ("STRING", {"default": "", "tooltip":
                    "Optional English description, e.g. \"a deep, calm older man's voice\". Sent as a leading "
                    "[direction]: it shapes the voice when there is no reference, and the delivery when cloning. "
                    "Describe a kind of voice, not a real person."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFF, "control_after_generate": "fixed",
                                 "tooltip": "Without a reference, each seed is a different voice."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.05, "max": 1.5, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.7, "min": 0.05, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 30, "min": 1, "max": 128}),
                "max_seconds": ("FLOAT", {"default": 45.0, "min": 3.0, "max": 120.0, "step": 1.0,
                                          "tooltip": "Length limit per text chunk (long text is split at about 200 characters)."}),
                "max_pause_seconds": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 2.0, "step": 0.05,
                                                "tooltip": "Pause fix: silences longer than this are cut short and runaway "
                                                           "takes are stopped. 0 disables it (not recommended)."}),
            },
            "optional": {
                "reference_audio": ("AUDIO", {"tooltip": "Voice to clone: 3-30 s of one speaker. Needs reference_text."}),
                "reference_text": ("STRING", {"multiline": True, "default": "", "forceInput": False,
                                              "tooltip": "Exact transcript of the reference audio."}),
                "streaming": ("BOOLEAN", {"default": False,
                                          "tooltip": "Decode audio in 2-second blocks while it is generated and move each "
                                                     "block to system RAM, so VRAM stays flat for very long text. Same audio; "
                                                     "slightly slower."}),
                "keep_model_loaded": ("BOOLEAN", {"default": True,
                                                  "tooltip": "Off: move the model to the CPU after generating, freeing VRAM "
                                                             "for other models (reloading takes a few seconds)."}),
            },
        }

    def generate(self, model: Drama3SmallModel, text, voice_description, seed, temperature, top_p, top_k, max_seconds,
                 max_pause_seconds, reference_audio=None, reference_text="", streaming=False, keep_model_loaded=True):
        text = text.strip()
        if not text:
            raise ValueError("drama3-small: text is empty")
        description = " ".join(voice_description.split())
        if any(c in description for c in "[]<>"):
            raise ValueError("drama3-small: write the voice description without brackets")
        prompt_text = f"[{description}] {text}" if description else text
        tts = model.load()
        try:
            references = []
            if reference_audio is not None:
                transcript = " ".join((reference_text or "").split())
                if not transcript:
                    raise ValueError("drama3-small: reference_audio needs reference_text (what is said in the clip)")
                codes = reference_audio.get(CODES_KEY)
                if codes is None:  # any other audio: encode it with the codec
                    waveform = reference_audio["waveform"][0]  # [channels, samples]
                    sample_rate = int(reference_audio["sample_rate"])
                    limit = 30 * sample_rate
                    if waveform.shape[-1] > limit:
                        log.warning("drama3-small: reference longer than 30 s; using the first 30 s")
                        waveform = waveform[..., :limit]
                    codes = tts.encode_reference(waveform.float(), sample_rate)
                elif codes.shape[1] > 30 * FPS:
                    log.warning("drama3-small: reference take is %.0f s long; 3-30 s works best", codes.shape[1] / FPS)
                references = [Take(transcript, codes)]
            sampling = Sampling(temperature=temperature, top_p=top_p, top_k=top_k, max_frames=max(1, int(max_seconds * FPS)),
                                seed=seed, max_pause_frames=round(max_pause_seconds * FPS) if max_pause_seconds > 0 else 0)
            bar = comfy.utils.ProgressBar(100)

            def progress(frames, chunk, chunks):
                if frames % 8 == 0:
                    bar.update_absolute(min(99, int(100 * (chunk + min(frames / sampling.max_frames, 0.95)) / chunks)), 100)

            result = tts.generate(prompt_text, references, sampling=sampling, progress=progress,
                                  interrupt=mm.throw_exception_if_processing_interrupted, streaming=streaming)
            bar.update_absolute(100, 100)
            log.info("drama3-small: %.1f s of audio; chunks %s", len(result.audio) / result.sample_rate,
                     [(c["frames"], c["stop"], c["fps"]) for c in result.chunks])
        finally:
            if not keep_model_loaded:
                model.offload()
        # The codes ride along so a "keep this voice" chain reuses them exactly instead of re-encoding;
        # nodes that change the audio build a new dict without them.
        audio = {"waveform": result.audio.reshape(1, 1, -1).contiguous(), "sample_rate": result.sample_rate,
                 CODES_KEY: result.codes}
        return (audio, spoken_text(text))


NODE_CLASS_MAPPINGS = {
    "Drama3SmallLoader": Drama3SmallLoader,
    "Drama3SmallTTS": Drama3SmallTTS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Drama3SmallLoader": "drama3-small Loader",
    "Drama3SmallTTS": "drama3-small Speak",
}

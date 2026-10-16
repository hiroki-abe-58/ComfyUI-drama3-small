"""Fish Audio DAC codec (Drama 3's modded_dac + DownsampleResidualVectorQuantize), inference only.

Ported from fish-speech `models/dac/{modded_dac,rvq}.py` and `configs/modded_dac_vq.yaml`
without the descript-audio-codec / audiotools dependencies. audio.cpp packages store
the weight-normalized convolutions already folded, so every conv here is a plain conv.
Weights stay fp16 on the device and are upcast per layer: activations run in fp32.

The convolution stacks are causal, so long audio is processed in chunks with left context
(identical output, bounded memory); the attention modules run on the whole sequence.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .gguf import GGUFReader

PREFIX = "codec_weights/"
SAMPLE_RATE = 44100
HOP = 512  # encoder samples per latent frame (2 * 4 * 8 * 8)
FRAME = HOP * 4  # samples per code frame (the quantizer downsamples latents by 4)
CONTEXT_LATENTS = 32  # left context for chunked conv stacks; the receptive field is about 12 latents


class Weights:
    def __init__(self, reader: GGUFReader, device: torch.device) -> None:
        self.r, self.device = reader, device

    def __call__(self, name: str) -> Tensor:
        t = self.r.tensors[PREFIX + name]
        np_dtype = {"F16": np.float16, "F32": np.float32}[t.qtype]
        array = np.array(self.r.raw(PREFIX + name)).view(np_dtype).reshape(t.shape)
        return torch.from_numpy(array).to(self.device, torch.float16)


class CausalConv:
    def __init__(self, w: Weights, name: str, stride: int = 1, dilation: int = 1, groups: int = 1) -> None:
        self.weight, self.bias = w(name + ".weight"), w(name + ".bias")
        self.stride, self.dilation, self.groups = stride, dilation, groups
        self.kernel = (self.weight.shape[-1] - 1) * dilation + 1
        self.padding = self.kernel - stride

    def __call__(self, x: Tensor) -> Tensor:
        length = x.shape[-1]
        frames = (length - self.kernel + self.padding) / self.stride + 1
        extra = (math.ceil(frames) - 1) * self.stride + (self.kernel - self.padding) - length
        x = F.pad(x, (self.padding, extra))
        return F.conv1d(x, self.weight.float(), self.bias.float(), self.stride, dilation=self.dilation, groups=self.groups)


class CausalTransConv:
    def __init__(self, w: Weights, name: str, stride: int) -> None:
        self.weight, self.bias, self.stride = w(name + ".weight"), w(name + ".bias"), stride
        self.trim = self.weight.shape[-1] - stride

    def __call__(self, x: Tensor) -> Tensor:
        y = F.conv_transpose1d(x, self.weight.float(), self.bias.float(), self.stride)
        return y[..., : y.shape[-1] - self.trim] if self.trim else y


class Snake:
    def __init__(self, w: Weights, name: str) -> None:
        self.alpha = w(name + ".alpha").float()

    def __call__(self, x: Tensor) -> Tensor:
        y = torch.sin(self.alpha * x)  # one temporary instead of four (these tensors are large)
        return y.pow_(2).mul_((self.alpha + 1e-9).reciprocal()).add_(x)


class ResidualUnit:
    def __init__(self, w: Weights, p: str, dilation: int) -> None:
        self.s1, self.c1 = Snake(w, p + "block.0"), CausalConv(w, p + "block.1.conv", dilation=dilation)
        self.s2, self.c2 = Snake(w, p + "block.2"), CausalConv(w, p + "block.3.conv")

    def __call__(self, x: Tensor) -> Tensor:
        y = self.c2(self.s2(self.c1(self.s1(x))))
        pad = x.shape[-1] - y.shape[-1]
        return (x[..., :-pad] if pad > 0 else x) + y


class Transformer:
    """WindowLimitedTransformer (causal, RoPE with bf16-rounded tables, layer scale)."""

    def __init__(self, w: Weights, p: str, layers: int, heads: int, window: int | None) -> None:
        self.layers = []
        for i in range(layers):
            q = f"{p}layers.{i}."
            self.layers.append(dict(
                attention_norm=w(q + "attention_norm.weight").float(), wqkv=w(q + "attention.wqkv.weight"),
                wo=w(q + "attention.wo.weight"), attention_scale=w(q + "attention_layer_scale.gamma").float(),
                ffn_norm=w(q + "ffn_norm.weight").float(), w1=w(q + "feed_forward.w1.weight"),
                w2=w(q + "feed_forward.w2.weight"), w3=w(q + "feed_forward.w3.weight"),
                ffn_scale=w(q + "ffn_layer_scale.gamma").float()))
        self.norm = w(p + "norm.weight").float()
        self.heads, self.window = heads, window
        self.dim = self.norm.shape[0]
        self.head_dim = self.dim // heads

    @staticmethod
    def rms(x: Tensor, weight: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight

    def rope(self, positions: Tensor) -> Tensor:
        device = positions.device
        freqs = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, device=device)[: self.head_dim // 2].float() / self.head_dim))
        angles = torch.outer(positions.float(), freqs)
        # fish-speech builds this table in bfloat16.
        return torch.stack((angles.cos(), angles.sin()), -1).to(torch.bfloat16).float()

    @staticmethod
    def apply_rope(x: Tensor, table: Tensor) -> Tensor:
        xs = x.reshape(*x.shape[:-1], -1, 2)
        t = table[None, :, None]
        return torch.stack((xs[..., 0] * t[..., 0] - xs[..., 1] * t[..., 1],
                            xs[..., 1] * t[..., 0] + xs[..., 0] * t[..., 1]), -1).flatten(3)

    def __call__(self, x: Tensor, start: int = 0, cache: list | None = None) -> Tensor:
        """x [1, C, T] channels first -> same.

        With `cache` (one (keys, values) pair per layer, initially None), the rows continue a
        stream at absolute position `start`: each layer attends to the window's cached keys and
        keeps the last window - 1 of them, so the output equals one pass over the whole sequence."""
        x = x.transpose(1, 2)
        length = x.shape[1]
        pos = torch.arange(start, start + length, device=x.device)
        table = self.rope(pos)
        for index, layer in enumerate(self.layers):
            h = self.rms(x, layer["attention_norm"])
            q, k, v = F.linear(h, layer["wqkv"].float()).split(self.dim, -1)
            q, k, v = (t.view(1, length, self.heads, self.head_dim) for t in (q, k, v))
            q, k = self.apply_rope(q, table), self.apply_rope(k, table)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            key_pos = pos
            if cache is not None:
                if cache[index] is not None:
                    k, v = torch.cat((cache[index][0], k), 2), torch.cat((cache[index][1], v), 2)
                    key_pos = torch.arange(start - (k.shape[2] - length), start + length, device=x.device)
                cache[index] = (k[:, :, -(self.window - 1):], v[:, :, -(self.window - 1):])
            mask = key_pos[None, :] <= pos[:, None]
            if self.window is not None:
                mask &= key_pos[None, :] >= (pos[:, None] - self.window + 1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            y = F.linear(y.transpose(1, 2).reshape(1, length, self.dim), layer["wo"].float())
            x = x + y * layer["attention_scale"]
            h = self.rms(x, layer["ffn_norm"])
            y = F.linear(F.silu(F.linear(h, layer["w1"].float())) * F.linear(h, layer["w3"].float()), layer["w2"].float())
            x = x + y * layer["ffn_scale"]
        return self.rms(x, self.norm).transpose(1, 2)


class ConvNeXt:
    def __init__(self, w: Weights, p: str) -> None:
        self.dwconv = CausalConv(w, p + "dwconv.conv", groups=w(p + "dwconv.conv.weight").shape[0])
        self.norm_w, self.norm_b = w(p + "norm.weight").float(), w(p + "norm.bias").float()
        self.pw1_w, self.pw1_b = w(p + "pwconv1.weight"), w(p + "pwconv1.bias")
        self.pw2_w, self.pw2_b = w(p + "pwconv2.weight"), w(p + "pwconv2.bias")
        self.gamma = w(p + "gamma").float()

    def __call__(self, x: Tensor) -> Tensor:
        y = self.dwconv(x).transpose(1, 2)
        y = F.layer_norm(y, (y.shape[-1],), self.norm_w, self.norm_b, 1e-6)
        y = F.gelu(F.linear(y, self.pw1_w.float(), self.pw1_b.float()))
        y = F.linear(y, self.pw2_w.float(), self.pw2_b.float()) * self.gamma
        return x + y.transpose(1, 2)


class VectorQuantizer:
    def __init__(self, w: Weights, p: str) -> None:
        self.in_w, self.in_b = w(p + "in_proj.weight"), w(p + "in_proj.bias")
        self.out_w, self.out_b = w(p + "out_proj.weight"), w(p + "out_proj.bias")
        self.codebook = w(p + "codebook.weight").float()
        self.normalized = F.normalize(self.codebook)

    @property
    def size(self) -> int:
        return self.codebook.shape[0]

    def encode(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """z [1, D, T] -> (codes [T], decoded contribution [1, D, T])."""
        z_e = F.conv1d(z, self.in_w.float(), self.in_b.float())[0].T  # [T, d]
        codes = (F.normalize(z_e) @ self.normalized.T).argmax(-1)
        return codes, self.decode(codes)

    def decode(self, codes: Tensor) -> Tensor:
        return F.conv1d(self.codebook[codes].T[None], self.out_w.float(), self.out_b.float())


class FishDAC:
    def __init__(self, reader: GGUFReader, device: torch.device) -> None:
        w = Weights(reader, device)
        self.device = device
        # Encoder: conv, 4 blocks (64 -> 128 -> 256 -> 512 -> 1024 channels), snake, conv.
        self.enc_in = CausalConv(w, "encoder.block.0.conv")
        self.enc_blocks = []
        for b, stride in enumerate((2, 4, 8, 8), start=1):
            p = f"encoder.block.{b}.block."
            self.enc_blocks.append((
                [ResidualUnit(w, f"{p}{i}.", d) for i, d in enumerate((1, 3, 9))],
                Snake(w, f"{p}3"), CausalConv(w, f"{p}4.conv", stride=stride)))
        self.enc_transformer = Transformer(w, "encoder.block.4.block.5.", layers=4, heads=16, window=512)
        self.enc_snake, self.enc_out = Snake(w, "encoder.block.5"), CausalConv(w, "encoder.block.6.conv")
        # Quantizer.
        q = "quantizer."
        self.down = [(CausalConv(w, f"{q}downsample.{i}.0.conv", stride=2), ConvNeXt(w, f"{q}downsample.{i}.1."))
                     for i in range(2)]
        self.up = [(CausalTransConv(w, f"{q}upsample.{i}.0.conv", stride=2), ConvNeXt(w, f"{q}upsample.{i}.1."))
                   for i in range(2)]
        self.pre = Transformer(w, q + "pre_module.", layers=8, heads=16, window=128)
        self.post = Transformer(w, q + "post_module.", layers=8, heads=16, window=128)
        self.semantic = VectorQuantizer(w, q + "semantic_quantizer.quantizers.0.")
        self.residual = [VectorQuantizer(w, f"{q}quantizer.quantizers.{i}.") for i in range(9)]
        # Decoder: conv, 4 blocks (1536 -> 768 -> 384 -> 192 -> 96 channels), snake, conv, tanh.
        self.dec_in = CausalConv(w, "decoder.model.0.conv")
        self.dec_blocks = []
        for b, stride in enumerate((8, 8, 4, 2), start=1):
            p = f"decoder.model.{b}.block."
            self.dec_blocks.append((
                Snake(w, f"{p}0"), CausalTransConv(w, f"{p}1.conv", stride),
                [ResidualUnit(w, f"{p}{i}.", d) for i, d in zip((2, 3, 4), (1, 3, 9))]))
        self.dec_snake, self.dec_out = Snake(w, "decoder.model.5"), CausalConv(w, "decoder.model.6.conv")

    # -- chunked causal stacks ------------------------------------------------

    @staticmethod
    def chunked(fn, x: Tensor, ratio: int, chunk_out: int, context_out: int) -> Tensor:
        """Apply a causal stack mapping `ratio` input steps to one output step, chunk by chunk."""
        total_out = math.ceil(x.shape[-1] / ratio)
        if total_out <= chunk_out + context_out:
            return fn(x)
        pieces = []
        for start in range(0, total_out, chunk_out):
            begin = max(0, start - context_out)
            y = fn(x[..., begin * ratio : (start + chunk_out) * ratio])
            pieces.append(y[..., start - begin : start - begin + chunk_out])
        return torch.cat(pieces, dim=-1)[..., :total_out]

    def encoder_convs(self, x: Tensor) -> Tensor:
        x = self.enc_in(x)
        for units, snake, conv in self.enc_blocks:
            for unit in units:
                x = unit(x)
            x = conv(snake(x))
        return x

    def decoder_convs(self, z: Tensor) -> Tensor:
        x = self.dec_in(z)
        for snake, conv, units in self.dec_blocks:
            x = conv(snake(x))
            for unit in units:
                x = unit(x)
        return torch.tanh(self.dec_out(self.dec_snake(x)))

    # -- public ---------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, audio: Tensor, sample_rate: int, chunk_seconds: float = 4.0) -> Tensor:
        """Mono waveform [T] -> codes [10, frames] (long, on the device)."""
        audio = audio.to(self.device, torch.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE:
            import torchaudio

            audio = torchaudio.functional.resample(audio, sample_rate, SAMPLE_RATE)
        length = audio.shape[-1]
        audio = F.pad(audio, (0, math.ceil(length / FRAME) * FRAME - length))[None, None]
        chunk = max(1, int(chunk_seconds * SAMPLE_RATE / HOP))
        z = self.chunked(self.encoder_convs, audio, HOP, chunk, CONTEXT_LATENTS)
        z = self.enc_out(self.enc_snake(self.enc_transformer(z)))
        for conv, block in self.down:
            z = block(conv(z))
        z = self.pre(z)
        semantic_codes, semantic_z = self.semantic.encode(z)
        residual = z - semantic_z
        codes = [semantic_codes]
        for quantizer in self.residual:
            c, zq = quantizer.encode(residual)
            residual = residual - zq
            codes.append(c)
        frames = math.ceil(length / FRAME)
        return torch.stack(codes)[:, :frames]

    @torch.inference_mode()
    def decode(self, codes: Tensor, block_frames: int = 86) -> Tensor:
        """Codes [10, frames] -> waveform fp32 [samples] at 44.1 kHz (on the device).

        Runs the streaming decoder over blocks of about 4 s: memory does not grow with length."""
        stream = self.stream()
        return torch.cat([stream.push(codes[:, i : i + block_frames]) for i in range(0, codes.shape[1], block_frames)])

    def stream(self) -> "StreamDecoder":
        return StreamDecoder(self)


class StreamDecoder:
    """Exact incremental decoding: push code frames, get their finished audio back.

    Everything after the codebooks is causal: the post-quantizer transformer (128-frame window)
    keeps per-layer key/value caches, and the upsampling + decoder convolutions are re-run with a
    16-frame tail of context (their receptive field is about 10 frames). The audio of frame t
    never changes once frame t has been pushed, so each push returns final samples."""

    TAIL = 16

    def __init__(self, dac: FishDAC) -> None:
        self.dac = dac
        self.start = 0
        self.cache: list = [None] * len(dac.post.layers)
        self.tail: Tensor | None = None

    @torch.inference_mode()
    def push(self, codes: Tensor) -> Tensor:
        """codes [10, n] -> fp32 samples [n * 2048] on the device."""
        dac = self.dac
        codes = codes.to(dac.device, torch.long)
        n = codes.shape[1]
        if n == 0:
            return torch.zeros(0, device=dac.device)
        z = dac.semantic.decode(codes[0].clamp(max=dac.semantic.size - 1))
        for i, quantizer in enumerate(dac.residual):
            z = z + quantizer.decode(codes[i + 1].clamp(max=quantizer.size - 1))
        z = dac.post(z, start=self.start, cache=self.cache)
        self.start += n
        context = 0 if self.tail is None else self.tail.shape[-1]
        if self.tail is not None:
            z = torch.cat((self.tail, z), dim=-1)
        self.tail = z[..., -self.TAIL :]
        for conv, block in dac.up:
            z = block(conv(z))
        return dac.decoder_convs(z)[0, 0][context * FRAME :]

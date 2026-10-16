"""drama3-small backbone + audio tower in PyTorch, weights kept quantized as in the GGUF.

Math follows fish-speech's fish_omni (Drama 3) reference: a 40-layer hybrid backbone
(gated delta-net and gated full attention, top-8 MoE with a gated shared expert), then a
codebook tower (4 shared + 9 residual layers) that predicts codes 1..9 of each frame.
drama3-small differences: per-layer routed expert counts, a speech-only output head
([EOS, semantic 0..4095]) and Q8_0 / IQ3_XXS weights.

The residual streams are fp32; matmuls run in `dtype` (bf16, or fp16 on GPUs without bf16)
with fp32 accumulation. Prefill handles a whole prompt at once; decode handles one frame
with static shapes so a CUDA graph can capture it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from . import kernels
from .gguf import GGUFReader
from .quants import q8_split

BACKBONE = "model_weights/model.language_model."
TOWER_SHARED = "model_weights/audio_decoder_mtp_shared.layers."
TOWER_RESIDUAL = "model_weights/audio_decoder_mtp.layers."
GDN_CHUNK = 64
EXPERT_CHUNK = 16  # prefill dequantizes this many experts at a time


# ---------------------------------------------------------------------------
# Weights


class Q8:
    """Q8_0 matrix [N, K] as int8 + per-32 fp16 scales."""

    def __init__(self, qs: Tensor, d: Tensor) -> None:
        self.qs, self.d = qs, d

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.qs.shape)

    def __call__(self, x: Tensor, out_dtype: torch.dtype | None = None) -> Tensor:
        return kernels.q8_linear(x, self.qs, self.d, out_dtype)

    def rows(self, ids: Tensor, dtype: torch.dtype) -> Tensor:
        """Embedding lookup: dequantized rows ids -> [..., K]."""
        flat = ids.reshape(-1)
        q = self.qs.index_select(0, flat).float().view(len(flat), -1, 32)
        out = (q * self.d.index_select(0, flat).float().unsqueeze(-1)).view(len(flat), -1)
        return out.to(dtype).view(*ids.shape, -1)

    def dense(self, dtype: torch.dtype) -> Tensor:
        return kernels.q8_dequant(self.qs, self.d, dtype)


class HostQ8(Q8):
    """A Q8 table that stays in host memory when the model moves between devices."""

    stay_on_host = True


class Banks:
    """IQ3_XXS expert banks [E, rows, bytes] of one layer, in VRAM."""

    def __init__(self, gate: Tensor, up: Tensor, down: Tensor) -> None:
        self.gate, self.up, self.down = gate, up, down

    @property
    def on_host(self) -> bool:
        return not self.gate.is_cuda


class HostBanks(Banks):
    """Expert banks in pinned host memory (expert offload). The Triton kernels read the selected
    experts' rows directly over PCIe, so they stay put when the model moves between devices."""

    stay_on_host = True


class Loader:
    """Reads named GGUF tensors onto one device."""

    def __init__(self, reader: GGUFReader, device: torch.device, dtype: torch.dtype) -> None:
        self.r, self.device, self.dtype = reader, device, dtype

    def raw(self, name: str) -> Tensor:
        return torch.from_numpy(np.array(self.r.raw(name))).to(self.device)

    def q8(self, name: str, device: torch.device | None = None) -> Q8:
        t = self.r.tensors[name]
        if t.qtype != "Q8_0":
            raise ValueError(f"{name}: expected Q8_0, got {t.qtype}")
        rows = int(np.prod(t.shape[:-1]))
        raw = torch.from_numpy(np.array(self.r.raw(name))).to(device or self.device)
        qs, d = q8_split(raw, rows, t.shape[-1])
        return Q8(qs, d)

    def dense(self, name: str, dtype: torch.dtype | None = None) -> Tensor:
        t = self.r.tensors[name]
        np_dtype = {"F32": np.float32, "F16": np.float16}.get(t.qtype)
        if np_dtype is None:
            if t.qtype == "Q8_0":
                return self.q8(name).dense(dtype or self.dtype).view(t.shape)
            raise ValueError(f"{name}: unsupported dense type {t.qtype}")
        array = np.array(self.r.raw(name)).view(np_dtype).reshape(t.shape)
        return torch.from_numpy(array).to(self.device, dtype or self.dtype)

    def iq3_bank(self, name: str, host: bool = False) -> Tensor:
        t = self.r.tensors[name]
        if t.qtype != "IQ3_XXS":
            raise ValueError(f"{name}: expected IQ3_XXS, got {t.qtype}")
        experts, rows, cols = t.shape
        raw = torch.from_numpy(np.array(self.r.raw(name)))
        raw = raw.pin_memory() if host else raw.to(self.device)
        return raw.view(experts, rows, cols // 256 * 98)


# ---------------------------------------------------------------------------
# Shared math


def rms_norm(x: Tensor, weight: Tensor, eps: float, zero_centered: bool = True) -> Tensor:
    """fp32 RMSNorm; scales by 1 + weight when zero_centered."""
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    return out * ((1.0 + weight) if zero_centered else weight)


def neox_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Half-split partial RoPE on x [T, heads, D]; cos / sin [T, rotary]."""
    rotary = cos.shape[-1]
    r, rest = x[..., :rotary].float(), x[..., rotary:]
    first, second = r.chunk(2, dim=-1)
    out = r * cos[:, None] + torch.cat((-second, first), dim=-1) * sin[:, None]
    return torch.cat((out.to(x.dtype), rest), dim=-1)


def l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def chunk_gated_delta_rule(q, k, v, decay, beta, state, chunk=GDN_CHUNK):
    """Gated delta rule over a sequence (fp32). q / k [T, H, dk], v [T, H, dv], decay / beta [T, H],
    state [H, dk, dv] or None -> (out [T, H, dv], final state)."""
    length, heads, dk = k.shape
    dv = v.shape[-1]
    q, k, v = (t.transpose(0, 1).float() for t in (q, k, v))  # [H, T, d]
    beta, decay = beta.transpose(0, 1).float(), decay.transpose(0, 1).float()
    q = l2norm(q) * dk**-0.5
    k = l2norm(k)
    pad = (chunk - length % chunk) % chunk
    q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    beta, decay = F.pad(beta, (0, pad)), F.pad(decay, (0, pad))
    n = (length + pad) // chunk
    vb, kb = v * beta[..., None], k * beta[..., None]
    q, k, kb, vb = (t.reshape(heads, n, chunk, -1) for t in (q, k, kb, vb))
    decay = decay.reshape(heads, n, chunk)
    upper = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).triu(1)
    cum = decay.cumsum(-1)
    pairwise = (cum[..., :, None] - cum[..., None, :]).masked_fill(upper, float("-inf")).exp()
    system = (kb @ k.transpose(-1, -2)) * pairwise
    intra = (q @ k.transpose(-1, -2)) * pairwise
    new_values = torch.linalg.solve_triangular(system, vb, upper=False, unitriangular=True)
    key_decay = torch.linalg.solve_triangular(system, kb * cum.exp()[..., None], upper=False, unitriangular=True)
    if state is None:
        state = q.new_zeros(heads, dk, dv)
    out = torch.empty_like(new_values)
    q = q * cum.exp()[..., None]
    k = k * (cum[..., -1:] - cum).exp()[..., None]
    chunk_decay = cum[..., -1].exp()[..., None, None]
    for i in range(n):
        corrected = new_values[:, i] - key_decay[:, i] @ state
        out[:, i] = q[:, i] @ state + intra[:, i] @ corrected
        state = state * chunk_decay[:, i] + k[:, i].transpose(-1, -2) @ corrected
    out = out.reshape(heads, -1, dv)[:, :length]
    return out.transpose(0, 1), state


# ---------------------------------------------------------------------------
# Backbone


@dataclass
class TextConfig:
    hidden: int
    layers: list[str]
    heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    eps: float
    k_heads: int
    v_heads: int
    dk: int
    dv: int
    conv_kernel: int
    top_k: int


class FullAttention:
    def __init__(self, ld: Loader, p: str, cfg: TextConfig) -> None:
        self.cfg = cfg
        self.q, self.k, self.v, self.o = (ld.q8(p + n) for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"))
        self.q_norm = ld.dense(p + "q_norm.weight", torch.float32)
        self.k_norm = ld.dense(p + "k_norm.weight", torch.float32)
        self.k_cache = self.v_cache = None

    def allocate(self, capacity: int, dtype: torch.dtype) -> None:
        shape = (self.cfg.kv_heads, capacity, self.cfg.head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=self.q.qs.device)
        self.v_cache = torch.zeros_like(self.k_cache)

    def __call__(self, x: Tensor, pos: Tensor, cos: Tensor, sin: Tensor, kv_len: int) -> Tensor:
        """x [T, H] at positions pos (contiguous) -> fp32 [T, H]; attends cache rows < kv_len."""
        c = self.cfg
        t = x.shape[0]
        qg = self.q(x).view(t, c.heads, 2 * c.head_dim)
        query, gate = qg[..., : c.head_dim], qg[..., c.head_dim :].reshape(t, -1)
        query = rms_norm(query, self.q_norm, c.eps).to(x.dtype)
        key = rms_norm(self.k(x).view(t, c.kv_heads, c.head_dim), self.k_norm, c.eps).to(x.dtype)
        value = self.v(x).view(t, c.kv_heads, c.head_dim)
        query = neox_rope(query, cos, sin).transpose(0, 1)  # [heads, T, D]
        key = neox_rope(key, cos, sin).transpose(0, 1)
        self.k_cache.index_copy_(1, pos, key.to(self.k_cache.dtype))
        self.v_cache.index_copy_(1, pos, value.transpose(0, 1).to(self.v_cache.dtype))
        groups = c.heads // c.kv_heads
        if t == 1:  # decode: fold each KV head's query heads into rows, so the cache is read once
            keys, values = self.k_cache[:, :kv_len], self.v_cache[:, :kv_len]
            mask = torch.arange(kv_len, device=x.device)[None, :] <= pos[:, None]
            out = F.scaled_dot_product_attention(query.reshape(c.kv_heads, groups, c.head_dim)[None], keys[None], values[None],
                                                 attn_mask=mask, scale=c.head_dim**-0.5)
        else:  # prefill from position 0: causal attention over the rows themselves
            assert kv_len == t, "prefill must start at position 0"
            keys, values = key.repeat_interleave(groups, 0), value.transpose(0, 1).repeat_interleave(groups, 0)
            out = F.scaled_dot_product_attention(query[None], keys[None], values[None], is_causal=True,
                                                 scale=c.head_dim**-0.5)
        out = out.reshape(c.heads, t, c.head_dim).transpose(0, 1).reshape(t, -1)
        return self.o(out * torch.sigmoid(gate.float()).to(out.dtype), torch.float32)


class GatedDeltaNet:
    def __init__(self, ld: Loader, p: str, cfg: TextConfig) -> None:
        self.cfg = cfg
        self.qkv = ld.q8(p + "in_proj_qkv.weight")
        self.z = ld.q8(p + "in_proj_z.weight")
        self.b = ld.q8(p + "in_proj_b.weight")
        self.a = ld.q8(p + "in_proj_a.weight")
        self.out = ld.q8(p + "out_proj.weight")
        self.conv = ld.dense(p + "conv1d.weight").reshape(-1, cfg.conv_kernel)  # [C, K]
        self.dt_bias = ld.dense(p + "dt_bias", torch.float32)
        self.a_neg = -ld.dense(p + "A_log", torch.float32).exp()
        self.norm = ld.dense(p + "norm.weight", torch.float32)
        self.key_dim = cfg.k_heads * cfg.dk
        self.conv_state = self.state = None

    def allocate(self, dtype: torch.dtype) -> None:
        c = self.cfg
        device = self.conv.device
        self.conv_state = torch.zeros(self.conv.shape[0], c.conv_kernel - 1, dtype=dtype, device=device)
        self.state = torch.zeros(c.v_heads, c.dk, c.dv, dtype=torch.float32, device=device)

    def __call__(self, x: Tensor, continue_state: bool) -> Tensor:
        c = self.cfg
        t = x.shape[0]
        mixed = self.qkv(x).T  # [C, T]
        history = c.conv_kernel - 1
        previous = self.conv_state.to(mixed.dtype) if continue_state else mixed.new_zeros(mixed.shape[0], history)
        padded = torch.cat((previous, mixed), dim=-1)
        self.conv_state.copy_(padded[:, -history:])
        conv = F.conv1d(padded[None], self.conv.to(mixed.dtype)[:, None], groups=mixed.shape[0])[0]
        mixed = F.silu(conv).T  # [T, C]
        q = mixed[:, : self.key_dim].reshape(t, c.k_heads, c.dk)
        k = mixed[:, self.key_dim : 2 * self.key_dim].reshape(t, c.k_heads, c.dk)
        v = mixed[:, 2 * self.key_dim :].reshape(t, c.v_heads, c.dv)
        z = self.z(x).reshape(t, c.v_heads, c.dv)
        beta = torch.sigmoid(self.b(x, torch.float32))
        decay = self.a_neg * F.softplus(self.a(x, torch.float32) + self.dt_bias)
        groups = c.v_heads // c.k_heads
        q = q.repeat_interleave(groups, dim=1)
        k = k.repeat_interleave(groups, dim=1)
        if continue_state and t == 1:
            qf = l2norm(q[0].float()) * c.dk**-0.5
            kf = l2norm(k[0].float())
            state = self.state * decay[0].exp()[:, None, None]
            remembered = (state * kf[..., None]).sum(-2)
            delta = (v[0].float() - remembered) * beta[0][:, None]
            state = state + kf[..., None] * delta[:, None, :]
            core = (state * qf[..., None]).sum(-2)[None]
            self.state.copy_(state)
        else:
            core, state = chunk_gated_delta_rule(q, k, v, decay, beta, self.state if continue_state else None)
            self.state.copy_(state)
        core = rms_norm(core, self.norm, c.eps, zero_centered=False) * F.silu(z.float())
        return self.out(core.to(x.dtype).reshape(t, -1), torch.float32)


class MoE:
    def __init__(self, ld: Loader, p: str, cfg: TextConfig, offload: bool = False) -> None:
        self.cfg = cfg
        self.router = ld.dense(p + "gate.weight", torch.float32)  # [E, H]; routing in fp32 keeps top-k stable
        self.banks = (HostBanks if offload else Banks)(
            *(ld.iq3_bank(f"{p}experts.{proj}_proj.weight", host=offload) for proj in ("gate", "up", "down")))
        self.inter = ld.r.tensors[p + "experts.gate_proj.weight"].shape[1]
        self.s_gate = ld.q8(p + "shared_expert.gate_proj.weight")
        self.s_up = ld.q8(p + "shared_expert.up_proj.weight")
        self.s_down = ld.q8(p + "shared_expert.down_proj.weight")
        self.shared_gate = ld.dense(p + "shared_expert_gate.weight", torch.float32).reshape(1, -1)

    def __call__(self, x: Tensor) -> Tensor:
        """x [T, H] (compute dtype) -> fp32 [T, H]."""
        xf = x.float()
        probs = torch.softmax(F.linear(xf, self.router), dim=-1)
        weights, ids = probs.topk(self.cfg.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        b = self.banks
        if x.shape[0] <= kernels.GEMV_MAX_ROWS:
            routed = kernels.iq3_moe(x, ids, weights, b.gate, b.up, b.down)
        else:
            routed = self.prefill_experts(x, ids, weights)
        shared = self.s_down(F.silu(self.s_gate(x, torch.float32)).mul_(self.s_up(x, torch.float32)).to(x.dtype), torch.float32)
        scale = torch.sigmoid(F.linear(xf, self.shared_gate))
        return routed + scale * shared

    def prefill_experts(self, x: Tensor, ids: Tensor, weights: Tensor) -> Tensor:
        tokens, hidden = x.shape
        top_k = ids.shape[1]
        flat = ids.reshape(-1)
        order = torch.argsort(flat, stable=True)
        rows = order // top_k
        pair_w = weights.reshape(-1)[order]
        counts = torch.bincount(flat, minlength=self.router.shape[0]).tolist()
        starts = np.concatenate(([0], np.cumsum(counts)))
        out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x.device)
        hit = [e for e, n in enumerate(counts) if n]
        b = self.banks
        if b.on_host:  # contiguous ranges of pinned memory, which the dequant kernel reads in place
            groups = [(e0, [e for e in hit if e0 <= e < e0 + EXPERT_CHUNK]) for e0 in range(0, len(counts), EXPERT_CHUNK)]
        else:
            groups = [(None, hit[c : c + EXPERT_CHUNK]) for c in range(0, len(hit), EXPERT_CHUNK)]
        for e0, chunk in groups:
            if not chunk:
                continue
            if e0 is None:
                sel = torch.tensor(chunk, device=x.device)
                gate, up, down = b.gate[sel], b.up[sel], b.down[sel]
                index = list(range(len(chunk)))
            else:
                end = min(e0 + EXPERT_CHUNK, len(counts))
                gate, up, down = b.gate[e0:end], b.up[e0:end], b.down[e0:end]
                index = [e - e0 for e in chunk]
            wg = kernels.iq3_dequant(gate, hidden, x.dtype, x.device)
            wu = kernels.iq3_dequant(up, hidden, x.dtype, x.device)
            wd = kernels.iq3_dequant(down, self.inter, x.dtype, x.device)
            for i, e in zip(index, chunk):
                r = rows[starts[e] : starts[e + 1]]
                xe = x.index_select(0, r)
                h = F.silu(F.linear(xe, wg[i]).float()) * F.linear(xe, wu[i]).float()
                o = F.linear(h.to(x.dtype), wd[i]).float() * pair_w[starts[e] : starts[e + 1], None]
                out.index_add_(0, r, o)
        return out


class Backbone:
    def __init__(self, ld: Loader, cfg: TextConfig, dtype: torch.dtype, offload_layers: set[int] = frozenset()) -> None:
        self.cfg, self.dtype = cfg, dtype
        # The 252k-row token table stays in host memory (prompt rows are embedded there); decoding
        # only ever feeds back EOS or a semantic token, whose rows live on the device.
        table = ld.q8(BACKBONE + "embed_tokens.weight", torch.device("cpu"))
        self.embed = HostQ8(table.qs, table.d)
        self.speech_embed: Q8 | None = None
        self.layers = []
        for i, kind in enumerate(cfg.layers):
            p = f"{BACKBONE}layers.{i}."
            mixer = FullAttention(ld, p + "self_attn.", cfg) if kind == "full_attention" else GatedDeltaNet(ld, p + "linear_attn.", cfg)
            self.layers.append((kind, ld.dense(p + "input_layernorm.weight", torch.float32), mixer,
                                ld.dense(p + "post_attention_layernorm.weight", torch.float32),
                                MoE(ld, p + "mlp.", cfg, offload=i in offload_layers)))
        self.norm = ld.dense(BACKBONE + "norm.weight", torch.float32)
        self.head = ld.q8("model_weights/lm_head.weight")  # [EOS, semantic 0..N-1]
        exponents = torch.arange(0, cfg.rotary_dim, 2, dtype=torch.float32, device=ld.device)
        self.inv_freq = 1.0 / (cfg.rope_theta ** (exponents / cfg.rotary_dim))

    def allocate(self, capacity: int) -> None:
        for kind, _, mixer, _, _ in self.layers:
            if kind == "full_attention":
                mixer.allocate(capacity, self.dtype)
            else:
                mixer.allocate(self.dtype)

    def __call__(self, x: Tensor, pos: Tensor, kv_len: int, continue_state: bool) -> Tensor:
        """x [T, H] fp32 input embeddings at positions pos -> final-norm hidden fp32 [T, H]."""
        angles = pos.float()[:, None] * self.inv_freq[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        cos, sin = angles.cos(), angles.sin()
        eps = self.cfg.eps
        for kind, in_norm, mixer, post_norm, moe in self.layers:
            h = rms_norm(x, in_norm, eps).to(self.dtype)
            if kind == "full_attention":
                x = x + mixer(h, pos, cos, sin, kv_len)
            else:
                x = x + mixer(h, continue_state)
            x = x + moe(rms_norm(x, post_norm, eps).to(self.dtype))
        return rms_norm(x, self.norm, eps)


# ---------------------------------------------------------------------------
# Audio tower


@dataclass
class TowerConfig:
    dim: int
    heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_base: float
    eps: float
    vocab: int
    codebooks: int
    shared_layers: int
    backbone_codebooks: int


class TowerLayer:
    def __init__(self, ld: Loader, p: str, cfg: TowerConfig, shared: bool) -> None:
        self.cfg = cfg
        self.embedding = ld.q8(p + ("semantic_embedding.weight" if shared else "codebook_embedding.weight"))
        self.enorm = ld.dense(p + "enorm.weight", torch.float32)
        self.hnorm = ld.dense(p + "hnorm.weight", torch.float32)
        self.eh_proj = ld.q8(p + "eh_proj.weight")
        t = p + "transformer_layer."
        self.attention_norm = ld.dense(t + "attention_norm.weight", torch.float32)
        self.ffn_norm = ld.dense(t + "ffn_norm.weight", torch.float32)
        self.wqkv = ld.q8(t + "attention.wqkv.weight")
        self.wo = ld.q8(t + "attention.wo.weight")
        self.q_norm = ld.dense(t + "attention.q_norm.weight", torch.float32)
        self.k_norm = ld.dense(t + "attention.k_norm.weight", torch.float32)
        self.w1 = ld.q8(t + "feed_forward.w1.weight")
        self.w2 = ld.q8(t + "feed_forward.w2.weight")
        self.w3 = ld.q8(t + "feed_forward.w3.weight")
        self.final_norm = ld.dense(p + "final_layernorm.weight", torch.float32)
        exponents = torch.arange(0, cfg.rotary_dim, 2, dtype=torch.float32, device=ld.device)
        self.inv_freq = cfg.rope_base ** (-exponents / cfg.rotary_dim)
        self.k_cache = self.v_cache = None

    def allocate(self, capacity: int, dtype: torch.dtype) -> None:
        self.k_cache = torch.zeros(self.cfg.kv_heads, capacity, self.cfg.head_dim, dtype=dtype, device=self.wo.qs.device)
        self.v_cache = torch.zeros_like(self.k_cache)

    def rotate(self, x: Tensor, pos: Tensor) -> Tensor:
        """Interleaved partial RoPE on x [T, heads, D] (fp32 math)."""
        width = self.cfg.rotary_dim
        angles = pos.float()[:, None] * self.inv_freq[None, :]
        cos, sin = angles.cos()[:, None], angles.sin()[:, None]
        even, odd = x[..., :width:2].float(), x[..., 1:width:2].float()
        rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1)
        return torch.cat((rotated.flatten(-2).to(x.dtype), x[..., width:]), dim=-1)

    def attention(self, x: Tensor, pos: Tensor, kv_len: int) -> Tensor:
        c = self.cfg
        t = x.shape[0]
        qw, kvw = c.heads * c.head_dim, c.kv_heads * c.head_dim
        q, gate, k, v = self.wqkv(x).split((qw, qw, kvw, kvw), -1)
        q = rms_norm(q.reshape(t, c.heads, c.head_dim), self.q_norm, c.eps).to(x.dtype)
        k = rms_norm(k.reshape(t, c.kv_heads, c.head_dim), self.k_norm, c.eps).to(x.dtype)
        q = self.rotate(q, pos).transpose(0, 1)
        k = self.rotate(k, pos).transpose(0, 1)
        self.k_cache.index_copy_(1, pos, k.to(self.k_cache.dtype))
        self.v_cache.index_copy_(1, pos, v.reshape(t, c.kv_heads, c.head_dim).transpose(0, 1).to(self.v_cache.dtype))
        groups = c.heads // c.kv_heads
        if t == 1:
            keys, values = self.k_cache[:, :kv_len], self.v_cache[:, :kv_len]
            mask = torch.arange(kv_len, device=x.device)[None, :] <= pos[:, None]
            out = F.scaled_dot_product_attention(q.reshape(c.kv_heads, groups, c.head_dim)[None], keys[None], values[None], attn_mask=mask)
        else:  # teacher-forced prefill from position 0
            assert kv_len == t, "prefill must start at position 0"
            keys = k.repeat_interleave(groups, 0)
            values = v.reshape(t, c.kv_heads, c.head_dim).transpose(0, 1).repeat_interleave(groups, 0)
            out = F.scaled_dot_product_attention(q[None], keys[None], values[None], is_causal=True)
        out = out.reshape(c.heads, t, c.head_dim).transpose(0, 1).reshape(t, qw)
        return self.wo(out * torch.sigmoid(gate.float()).to(out.dtype), torch.float32)

    def __call__(self, hidden: Tensor, codes: Tensor, semantic: Tensor, pos: Tensor, kv_len: int,
                 shared_input: Tensor | None, dtype: torch.dtype) -> Tensor:
        """hidden fp32 [T, D], codes / semantic [T] -> fp32 [T, D]."""
        c = self.cfg
        e = self.embedding.rows(codes.clamp(0, c.vocab - 1), torch.float32)
        e = torch.where(semantic[:, None], e, 0.0)
        if shared_input is not None:
            e = e + shared_input
        mixed = self.eh_proj(torch.cat((rms_norm(e, self.enorm, c.eps), rms_norm(hidden, self.hnorm, c.eps)), -1).to(dtype),
                             torch.float32)
        mixed = mixed + self.attention(rms_norm(mixed, self.attention_norm, c.eps).to(dtype), pos, kv_len)
        h = rms_norm(mixed, self.ffn_norm, c.eps).to(dtype)
        mixed = mixed + self.w2(F.silu(self.w1(h, torch.float32)).mul_(self.w3(h, torch.float32)).to(dtype), torch.float32)
        return rms_norm(mixed, self.final_norm, c.eps)


class Tower:
    def __init__(self, ld: Loader, cfg: TowerConfig, semantic_begin: int, dtype: torch.dtype) -> None:
        self.cfg, self.dtype = cfg, dtype
        self.semantic_begin = semantic_begin
        self.codebook_embeddings = ld.dense("model_weights/audio_decoder.codebook_embeddings.weight", dtype)
        self.shared = [TowerLayer(ld, f"{TOWER_SHARED}{i}.", cfg, True) for i in range(cfg.shared_layers)]
        self.residual = [TowerLayer(ld, f"{TOWER_RESIDUAL}{i}.", cfg, False) for i in range(cfg.codebooks - 1)]
        self.heads = [ld.q8(f"model_weights/audio_decoder_mtp.heads.{i}.weight") for i in range(cfg.codebooks - 1)]
        self.offsets = torch.arange(cfg.codebooks, device=ld.device) * cfg.vocab

    def allocate(self, capacity: int) -> None:
        for layer in (*self.shared, *self.residual):
            layer.allocate(capacity, self.dtype)

    def is_semantic(self, tokens: Tensor) -> Tensor:
        return (tokens >= self.semantic_begin) & (tokens < self.semantic_begin + self.cfg.vocab)

    def embedding_sum(self, frames: Tensor, first: int, end: int) -> Tensor:
        """Sum of embeddings of codes first..end-1 per frame row [T, 1 + N] -> fp32 [T, D]."""
        rows = frames[:, 1 + first : 1 + end] + self.offsets[first:end]
        return F.embedding_bag(rows, self.codebook_embeddings, mode="sum").float()

    def backbone_input(self, frames: Tensor, text_embeddings: Tensor) -> Tensor:
        """Semantic rows add codes 1..K to the token embedding and rescale; fp32 [T, D]."""
        k = self.cfg.backbone_codebooks
        mixed = (text_embeddings + self.embedding_sum(frames, 1, 1 + k)) / math.sqrt(1 + k)
        return torch.where(self.is_semantic(frames[:, 0])[:, None], mixed, text_embeddings)

    def shared_input(self, frames: Tensor) -> Tensor:
        """Codes the backbone did not see, for the first shared layer."""
        unseen = self.embedding_sum(frames, 1 + self.cfg.backbone_codebooks, self.cfg.codebooks)
        return torch.where(self.is_semantic(frames[:, 0])[:, None], unseen, 0.0)

    def run_shared(self, hidden, c0, semantic, pos, kv_len, frames):
        extra = self.shared_input(frames)
        for i, layer in enumerate(self.shared):
            hidden = layer(hidden, c0, semantic, pos, kv_len, extra if i == 0 else None, self.dtype)
        return hidden

    def prefill(self, hidden: Tensor, pos: Tensor, frames: Tensor, next_frames: Tensor) -> None:
        """Teacher-force the tower caches: row p is labelled with frame p + 1."""
        kv_len = int(pos[-1]) + 1
        semantic = self.is_semantic(next_frames[:, 0])
        hidden = self.run_shared(hidden, next_frames[:, 1], semantic, pos, kv_len, frames)
        for i, layer in enumerate(self.residual):
            hidden = layer(hidden, next_frames[:, i + 1], semantic, pos, kv_len, None, self.dtype)


# ---------------------------------------------------------------------------
# Whole model


class Drama3Small:
    """drama3-small loaded from an audio.cpp GGUF package."""

    def __init__(self, path: str, device: str | torch.device = "cuda", dtype: torch.dtype | None = None,
                 expert_offload: float = 0.0) -> None:
        """expert_offload: share of the routed-expert bytes (0..1) kept in pinned host memory instead of
        VRAM, largest layers first. Needs CUDA + Triton; each offloaded layer reads its 8 selected
        experts over PCIe every frame (about 10 MB), so decoding slows down as the share grows."""
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = dtype or default_dtype(self.device)
        reader = GGUFReader(path)
        config = reader.config()
        tc, ad = config["text_config"], config["audio_decoder_config"]
        self.config = config
        self.files = reader.embedded_files()
        self.eos = config["eos_token_id"]
        self.semantic_begin = config["semantic_start_token_id"]
        self.semantic_end = config["semantic_end_token_id"]
        head_dim = tc["head_dim"]
        rope = tc.get("rope_parameters") or tc
        rotary = int(head_dim * rope.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25)))
        layer_types = tc.get("layer_types") or [
            "full_attention" if (i + 1) % tc["full_attention_interval"] == 0 else "linear_attention"
            for i in range(tc["num_hidden_layers"])]
        self.text_cfg = TextConfig(
            hidden=tc["hidden_size"], layers=layer_types, heads=tc["num_attention_heads"],
            kv_heads=tc["num_key_value_heads"], head_dim=head_dim, rotary_dim=rotary, rope_theta=rope["rope_theta"],
            eps=tc["rms_norm_eps"], k_heads=tc["linear_num_key_heads"], v_heads=tc["linear_num_value_heads"],
            dk=tc["linear_key_head_dim"], dv=tc["linear_value_head_dim"], conv_kernel=tc["linear_conv_kernel_dim"],
            top_k=tc["num_experts_per_tok"])
        self.tower_cfg = TowerConfig(
            dim=ad["dim"], heads=ad["n_head"], kv_heads=ad["n_local_heads"], head_dim=ad["head_dim"],
            rotary_dim=int(ad["head_dim"] * ad["rotary_percent"]), rope_base=ad["rope_base"], eps=ad["norm_eps"],
            vocab=ad["vocab_size"], codebooks=ad["num_codebooks"], shared_layers=ad["shared_enrichment_layers"],
            backbone_codebooks=ad["vq_backbone_codebooks"])
        ld = Loader(reader, self.device, self.dtype)
        offload_layers = self.pick_offload_layers(reader, expert_offload)
        self.offloaded_layers = sorted(offload_layers)
        self.backbone = Backbone(ld, self.text_cfg, self.dtype, offload_layers)
        self.tower = Tower(ld, self.tower_cfg, self.semantic_begin, self.dtype)
        if self.backbone.head.shape[0] != self.tower_cfg.vocab + 1:
            raise ValueError("expected a speech-only head with rows [EOS, semantic 0..N-1]")
        speech = torch.cat((torch.tensor([self.eos]), torch.arange(self.semantic_begin, self.semantic_begin + self.tower_cfg.vocab)))
        table = self.backbone.embed
        self.backbone.speech_embed = Q8(table.qs[speech].to(self.device), table.d[speech].to(self.device))
        self.reader = reader
        self.capacity = 0

    def pick_offload_layers(self, reader: GGUFReader, share: float) -> set[int]:
        if share <= 0:
            return set()
        if not (self.device.type == "cuda" and kernels.HAS_TRITON):
            raise ValueError("expert offload needs a CUDA/ROCm GPU with Triton")
        sizes = {i: sum(reader.tensors[f"{BACKBONE}layers.{i}.mlp.experts.{p}_proj.weight"].n_bytes for p in ("gate", "up", "down"))
                 for i in range(len(self.text_cfg.layers))}
        target, chosen, total = share * sum(sizes.values()), set(), 0
        for i in sorted(sizes, key=lambda i: (-sizes[i], i)):
            if total >= target - 1:
                break
            chosen.add(i)
            total += sizes[i]
        return chosen

    def allocate(self, capacity: int) -> None:
        if capacity > self.capacity:
            self.backbone.allocate(capacity)
            self.tower.allocate(capacity)
            self.capacity = capacity

    def embed(self, frames: Tensor) -> Tensor:
        """Backbone input embeddings (fp32) for frame rows [T, 1 + N] on the device.

        One row (decoding) must be EOS or a semantic token and uses the device table; longer
        prompts are embedded from the host table."""
        tokens = frames[:, 0]
        if frames.shape[0] == 1:
            rows = torch.where(tokens == self.eos, 0, tokens - self.semantic_begin + 1).clamp(0, self.tower_cfg.vocab)
            text = self.backbone.speech_embed.rows(rows, torch.float32)
        else:
            text = self.backbone.embed.rows(tokens.cpu(), torch.float32).to(frames.device)
        return self.tower.backbone_input(frames, text)

    @torch.inference_mode()
    def prefill(self, prompt: Tensor) -> Tensor:
        """Run the prompt rows [T, 1 + N]; fills every cache; returns the last hidden row fp32 [1, H]."""
        prompt = prompt.to(self.device)
        t = prompt.shape[0]
        self.allocate(t + 1)
        pos = torch.arange(t, device=self.device)
        hidden = self.backbone(self.embed(prompt), pos, t, continue_state=False)
        if t > 1:
            self.tower.prefill(hidden[:-1], pos[:-1], prompt[:-1], prompt[1:])
        return hidden[-1:]

    def logits(self, hidden: Tensor) -> Tensor:
        """Main-token logits fp32 [T, 1 + vocab]: column 0 is EOS, column 1 + c is semantic code c."""
        return self.backbone.head(hidden.to(self.dtype), torch.float32)


def default_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32

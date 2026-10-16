"""GGML block formats used by drama3-small, as torch tensors on the compute device.

Q8_0 (32 weights: fp16 d, 32 x int8) is re-laid out at load time into an int8 matrix plus
an fp16 scale per 32 columns, which kernels read with plain strided loads. IQ3_XXS (256
weights in 98 bytes: fp16 d, 64 grid indices, 8 x uint32 scale+sign words) stays packed.
The pure-torch dequantizers here are the reference for the Triton kernels and the
fallback where Triton is unavailable.
"""

from __future__ import annotations

import numpy as np
import torch

IQ3_BLOCK_BYTES = 98
QK_K = 256

# ggml iq3xxs_grid (ggml-common.h, MIT): 256 entries of 4 packed uint8 magnitudes.
IQ3XXS_GRID = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
]

_TABLES: dict[tuple[str, torch.device], torch.Tensor] = {}


def iq3_grid(device: torch.device) -> torch.Tensor:
    """int32 [256] packed grid (values fit in 31 bits), cached per device."""
    key = ("grid", torch.device(device))
    if key not in _TABLES:
        _TABLES[key] = torch.tensor(np.array(IQ3XXS_GRID, dtype=np.uint32).view(np.int32), device=device)
    return _TABLES[key]


def iq3_grid_bytes(device: torch.device) -> torch.Tensor:
    """float32 [256, 4] grid magnitudes, cached per device (torch reference path)."""
    key = ("grid4", torch.device(device))
    if key not in _TABLES:
        g = np.array(IQ3XXS_GRID, dtype=np.uint32).view(np.uint8).reshape(256, 4)
        _TABLES[key] = torch.tensor(g.astype(np.float32), device=device)
    return _TABLES[key]


def _shifts(device: torch.device) -> torch.Tensor:
    """int64 [4] sign-index shifts, cached (creating it inside a CUDA-graph capture is illegal)."""
    key = ("shifts", torch.device(device))
    if key not in _TABLES:
        _TABLES[key] = torch.tensor([0, 7, 14, 21], dtype=torch.int64, device=device)
    return _TABLES[key]


def sign_bits(device: torch.device) -> torch.Tensor:
    """float32 [128, 8] of +-1: ggml ksigns_iq2xs (7 sign bits + even parity bit), cached."""
    key = ("signs", torch.device(device))
    if key not in _TABLES:
        idx = np.arange(128)
        parity = np.array([bin(i).count("1") & 1 for i in idx])
        ks = (idx | (parity << 7)).astype(np.uint8)
        bits = (ks[:, None] >> np.arange(8)) & 1
        _TABLES[key] = torch.tensor(1.0 - 2.0 * bits, dtype=torch.float32, device=device)
    return _TABLES[key]


# ---------------------------------------------------------------------------
# Q8_0


def q8_split(raw: torch.Tensor, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw Q8_0 bytes [..., rows * cols/32 * 34] -> (int8 [..., rows, cols], fp16 [..., rows, cols/32])."""
    lead = raw.shape[:-1]
    blocks = raw.reshape(*lead, rows, cols // 32, 34)
    d = blocks[..., :2].contiguous().view(torch.float16).reshape(*lead, rows, cols // 32)
    qs = blocks[..., 2:].contiguous().view(torch.int8).reshape(*lead, rows, cols)
    return qs, d


def q8_dequant_torch(qs: torch.Tensor, d: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    *lead, rows, cols = qs.shape
    w = qs.reshape(*lead, rows, cols // 32, 32).float() * d.float().unsqueeze(-1)
    return w.reshape(*lead, rows, cols).to(dtype)


# ---------------------------------------------------------------------------
# IQ3_XXS


def iq3_dequant_torch(raw: torch.Tensor, cols: int, dtype: torch.dtype) -> torch.Tensor:
    """Packed IQ3_XXS rows [..., cols/256 * 98] (uint8) -> dtype [..., cols]; mirrors ggml."""
    lead = raw.shape[:-1]
    b = raw.reshape(-1, IQ3_BLOCK_BYTES)
    n = b.shape[0]
    d = b[:, :2].contiguous().view(torch.float16).float()  # [n, 1]
    qs = b[:, 2:66].long()  # [n, 64]
    aux = b[:, 66:98].contiguous().view(torch.int32).long() & 0xFFFFFFFF  # [n, 8]
    db = (d * (0.5 + (aux >> 28).float()) * 0.5).reshape(n, 8, 1, 1)
    shifts = _shifts(b.device)
    sign_idx = (aux.reshape(n, 8, 1) >> shifts) & 0x7F  # [n, 8, 4]
    signs = sign_bits(b.device)[sign_idx]  # [n, 8, 4, 8]
    grid = iq3_grid_bytes(b.device)[qs].reshape(n, 8, 4, 8)
    return (db * grid * signs).reshape(*lead, cols).to(dtype)

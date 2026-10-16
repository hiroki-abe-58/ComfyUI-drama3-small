"""Minimal GGUF (v3) reader for audio.cpp Drama 3 packages; no `gguf` package needed.

audio.cpp packages name tensors `_audiocpp.N` (or keep the real name), record the real
names and logical PyTorch-order shapes in metadata arrays, and embed config.json and
tokenizer.json in one byte blob. Tensor data is read through a memory map.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ggml type id -> (name, block size, bytes per block). Only types this engine reads.
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    8: ("Q8_0", 32, 34),
    18: ("IQ3_XXS", 256, 98),
    30: ("BF16", 1, 2),
}
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


@dataclass
class Tensor:
    name: str  # real (audio.cpp) name
    qtype: str  # F32 / F16 / BF16 / Q8_0 / IQ3_XXS
    shape: tuple[int, ...]  # logical PyTorch order; shape[-1] is the contiguous row
    offset: int  # absolute file offset
    n_bytes: int


class GGUFReader:
    def __init__(self, path: str | Path, header_bytes: int = 256 << 20) -> None:
        self.path = Path(path)
        with open(self.path, "rb") as f:
            head = f.read(header_bytes)
        buf = memoryview(head)
        if bytes(buf[:4]) != b"GGUF":
            raise ValueError(f"{self.path} is not a GGUF file")
        version, n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
        if version != 3:
            raise ValueError(f"GGUF version {version} is not supported")
        pos = 24
        self.kv: dict[str, object] = {}
        for _ in range(n_kv):
            key, pos = _string(buf, pos)
            (vtype,) = struct.unpack_from("<I", buf, pos)
            self.kv[key], pos = _value(buf, pos + 4, vtype)
        infos = []
        for _ in range(n_tensors):
            name, pos = _string(buf, pos)
            (n_dims,) = struct.unpack_from("<I", buf, pos)
            dims = struct.unpack_from(f"<{n_dims}Q", buf, pos + 4)
            pos += 4 + 8 * n_dims
            ggml_type, offset = struct.unpack_from("<IQ", buf, pos)
            pos += 12
            infos.append((name, dims, ggml_type, offset))
        alignment = int(self.kv.get("general.alignment", 32))
        data_offset = (pos + alignment - 1) // alignment * alignment

        real = self.array("audiocpp.tensor_names") if "audiocpp.tensor_names" in self.kv else None
        ranks = self.array("audiocpp.tensor_ranks") if real else None
        shapes = self.array("audiocpp.tensor_shapes") if real else None
        self.tensors: dict[str, Tensor] = {}
        cursor = 0
        for index, (name, dims, ggml_type, offset) in enumerate(infos):
            if ggml_type not in GGML_TYPES:
                raise ValueError(f"{name}: ggml type {ggml_type} is not supported by this engine")
            qtype, block, type_size = GGML_TYPES[ggml_type]
            count = int(np.prod(dims))
            if real is not None:
                logical = tuple(int(d) for d in shapes[cursor : cursor + ranks[index]])
                cursor += ranks[index]
                name = real[index]
            else:
                logical = tuple(int(d) for d in reversed(dims))
            self.tensors[name] = Tensor(name, qtype, logical, data_offset + offset, count // block * type_size)

    def array(self, key: str) -> list:
        elem_type, count, offset = self.kv[key]
        with open(self.path, "rb") as f:
            f.seek(offset)
            if elem_type == _STRING:
                out = []
                for _ in range(count):
                    (n,) = struct.unpack("<Q", f.read(8))
                    out.append(f.read(n).decode("utf-8"))
                return out
            fmt = _SCALAR[elem_type]
            return list(struct.unpack(f"<{count}{fmt[1]}", f.read(struct.calcsize(fmt) * count)))

    def array_bytes(self, key: str) -> bytes:
        _, count, offset = self.kv[key]
        with open(self.path, "rb") as f:
            f.seek(offset)
            return f.read(count)

    def embedded_files(self) -> dict[str, bytes]:
        names = self.array("audiocpp.embedded_files.names")
        offsets = self.array("audiocpp.embedded_files.offsets")
        blob = self.array_bytes("audiocpp.embedded_files.data")
        return {n: blob[offsets[i] : offsets[i + 1]] for i, n in enumerate(names)}

    def config(self) -> dict:
        return json.loads(self.embedded_files()["config.json"])

    def raw(self, name: str) -> np.ndarray:
        """The tensor's bytes as a read-only uint8 memmap."""
        t = self.tensors[name]
        return np.memmap(self.path, dtype=np.uint8, mode="r", offset=t.offset, shape=(t.n_bytes,))


def _string(buf: memoryview, pos: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<Q", buf, pos)
    return bytes(buf[pos + 8 : pos + 8 + n]).decode("utf-8"), pos + 8 + n


def _value(buf: memoryview, pos: int, vtype: int) -> tuple[object, int]:
    if vtype in _SCALAR:
        fmt = _SCALAR[vtype]
        return struct.unpack_from(fmt, buf, pos)[0], pos + struct.calcsize(fmt)
    if vtype == _STRING:
        return _string(buf, pos)
    if vtype == _ARRAY:
        elem_type, count = struct.unpack_from("<IQ", buf, pos)
        pos += 12
        start = pos
        if elem_type in _SCALAR:
            pos += struct.calcsize(_SCALAR[elem_type]) * count
        elif elem_type == _STRING:
            for _ in range(count):
                (n,) = struct.unpack_from("<Q", buf, pos)
                pos += 8 + n
        else:
            raise ValueError(f"nested GGUF arrays are not supported (type {elem_type})")
        return (elem_type, count, start), pos
    raise ValueError(f"unknown GGUF value type {vtype}")

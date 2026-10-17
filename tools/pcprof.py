"""Writer/reader for the .pcprof profile format (mirror of src/profiles/pcprof.ts)."""
from __future__ import annotations

import json
import struct

import numpy as np

MAGIC = b"PCPF"
VERSION = 1


def write(path, header: dict, bank: np.ndarray, sqnorm: np.ndarray, thumbnail: bytes, bank_dtype="float16"):
    header = dict(header)
    header["bankDtype"] = bank_dtype
    header["M"], header["D"] = int(bank.shape[0]), int(bank.shape[1])
    header["thumbnailBytes"] = len(thumbnail)
    hj = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(hj)) % 4
    bank_bytes = bank.astype("<f2" if bank_dtype == "float16" else "<f4").tobytes()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(hj)))
        f.write(hj + b" " * pad)
        f.write(bank_bytes)
        f.write(sqnorm.astype("<f4").tobytes())
        f.write(thumbnail)


def read(path):
    data = open(path, "rb").read()
    assert data[:4] == MAGIC
    version, hl = struct.unpack_from("<II", data, 4)
    assert version == VERSION
    header = json.loads(data[12:12 + hl].decode("utf-8"))
    off = 12 + hl + ((-hl) % 4)
    m, d = header["M"], header["D"]
    dt = "<f2" if header["bankDtype"] == "float16" else "<f4"
    nb = m * d * np.dtype(dt).itemsize
    bank = np.frombuffer(data, dt, m * d, off).reshape(m, d).astype(np.float32)
    off += nb
    sq = np.frombuffer(data, "<f4", m, off)
    off += m * 4
    thumb = data[off:off + header["thumbnailBytes"]]
    return header, bank, sq, thumb

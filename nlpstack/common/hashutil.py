import hashlib
import io
from typing import Any

import dill


def hash_object(o: Any) -> int:
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
    return int.from_bytes(m.digest(), "big")


def murmurhash3(key: str, seed: int = 0) -> int:
    length = len(key)

    remainder = length & 3
    n = length - remainder
    h1 = seed
    c1 = 0xCC9E2D51
    c2 = 0x1B873593
    i = 0

    while i < n:
        k1 = (
            (ord(key[i]) & 0xFF)
            | ((ord(key[i + 1]) & 0xFF) << 8)
            | (ord(key[i + 2]) & 0xFF) << 16
            | (ord(key[i + 3]) & 0xFF) << 24
        )
        i += 4

        k1 = ((((k1 & 0xFFFF) * c1) + ((((k1 >> 16) * c1) & 0xFFFF) << 16))) & 0xFFFFFFFF
        k1 = (k1 << 15) | (k1 >> 17)
        k1 = ((((k1 & 0xFFFF) * c2) + ((((k1 >> 16) * c2) & 0xFFFF) << 16))) & 0xFFFFFFFF

        h1 ^= k1
        h1 = (h1 << 13) | (h1 >> 19)
        h1b = ((((h1 & 0xFFFF) * 5) + ((((h1 >> 16) * 5) & 0xFFFF) << 16))) & 0xFFFFFFFF
        h1 = ((h1b & 0xFFFF) + 0x6B64) + ((((h1b >> 16) + 0xE654) & 0xFFFF) << 16)

    k1 = 0
    if remainder == 3:
        k1 ^= (ord(key[i + 2]) & 0xFF) << 16
        k1 ^= (ord(key[i + 1]) & 0xFF) << 8
        k1 ^= ord(key[i]) & 0xFF
    elif remainder == 2:
        k1 ^= (ord(key[i + 1]) & 0xFF) << 8
        k1 ^= ord(key[i]) & 0xFF
    elif remainder == 1:
        k1 ^= ord(key[i]) & 0xFF
    k1 = ((((k1 & 0xFFFF) * c1) + ((((k1 >> 16) * c1) & 0xFFFF) << 16))) & 0xFFFFFFFF
    k1 = (k1 << 15) | (k1 >> 17)
    k1 = ((((k1 & 0xFFFF) * c2) + ((((k1 >> 16) * c2) & 0xFFFF) << 16))) & 0xFFFFFFFF
    h1 ^= k1

    h1 ^= length
    h1 ^= h1 >> 16
    h1 = ((((h1 & 0xFFFF) * 0x85EBCA6B) + ((((h1 >> 16) * 0x85EBCA6B) & 0xFFFF) << 16))) & 0xFFFFFFFF
    h1 ^= h1 >> 13
    h1 = ((((h1 & 0xFFFF) * 0xC2B2AE35) + ((((h1 >> 16) * 0xC2B2AE35) & 0xFFFF) << 16))) & 0xFFFFFFFF
    h1 ^= h1 >> 16

    return h1 >> 0

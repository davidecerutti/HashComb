"""Canonical serialization utilities for hash paths/prefixes."""
from __future__ import annotations

from typing import Sequence, List


_PREFIX = "v1"


def serialize_path(tokens: Sequence[str]) -> str:
    """
    Serialize a path into a stable, unambiguous string.

    Format: v1|L|len1:tok1|len2:tok2|...
    where L is the number of tokens and each token is length-prefixed.
    """
    parts: List[str] = [f"{_PREFIX}|{len(tokens)}"]
    for token in tokens:
        t = str(token)
        parts.append(f"{len(t)}:{t}")
    return "|".join(parts)


def deserialize_path(serialized: str) -> List[str]:
    """
    Parse a serialized path produced by serialize_path.
    """
    if not serialized.startswith(f"{_PREFIX}|"):
        raise ValueError("Unsupported or missing serialization prefix")

    rest = serialized[len(_PREFIX) + 1 :]
    try:
        length_str, tail = rest.split("|", 1)
        expected = int(length_str)
    except ValueError as exc:
        raise ValueError("Invalid serialized path header") from exc

    tokens: List[str] = []
    cursor = 0
    for _ in range(expected):
        colon = tail.find(":", cursor)
        if colon == -1:
            raise ValueError("Invalid serialized token length")
        try:
            token_len = int(tail[cursor:colon])
        except ValueError as exc:
            raise ValueError("Invalid token length value") from exc
        start = colon + 1
        end = start + token_len
        if end > len(tail):
            raise ValueError("Serialized token exceeds buffer length")
        tokens.append(tail[start:end])
        cursor = end
        if cursor < len(tail) and tail[cursor] == "|":
            cursor += 1

    if len(tokens) != expected:
        raise ValueError("Token count mismatch during deserialization")
    return tokens
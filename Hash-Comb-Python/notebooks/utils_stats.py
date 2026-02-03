"""Statistics helpers for HashComb notebooks."""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np


def histogram_from_tokens(tokens: Sequence[str]) -> dict[str, int]:
    """Count token occurrences (histogram)."""
    return dict(Counter(str(t) for t in tokens))


def histogram_from_values(values: Sequence[float], encoder) -> dict[str, int]:
    """Encode values and return token histogram."""
    tokens = encoder.encodeArray(values)
    return histogram_from_tokens(tokens)


def mean_from_counts(counts: Mapping[str, int], decoder) -> float:
    """Compute mean from token counts using decoded bin centers."""
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    s = 0.0
    for h, c in counts.items():
        center = decoder.decode(h)
        s += center * c
    return float(s / total)


def mean_from_tokens(tokens: Sequence[str], decoder) -> float:
    """Compute mean from a token sequence."""
    return mean_from_counts(histogram_from_tokens(tokens), decoder)


def quantization_error(values: Sequence[float], decoder, tokens: Sequence[str]) -> dict[str, float]:
    """Return mean_plain, mean_hash, abs_error, rel_error for a tokenized dataset."""
    mean_plain = float(np.mean(values))
    mean_hash = mean_from_tokens(tokens, decoder)
    abs_error = abs(mean_plain - mean_hash)
    rel_error = abs_error / max(abs(mean_plain), 1e-12)
    return {
        "mean_plain": mean_plain,
        "mean_hash": mean_hash,
        "abs_error": abs_error,
        "rel_error": rel_error,
    }

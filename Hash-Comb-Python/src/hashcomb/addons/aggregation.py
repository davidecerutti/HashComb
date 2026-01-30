"""Add-on utilities for ciphertext aggregation (paper Eq. 16)."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple, TypeVar, Callable
import operator


T = TypeVar("T")


def aggregate_ciphertexts(
    items: Iterable[Tuple[str, T]],
    *,
    add: Callable[[T, T], T] | None = None,
) -> Dict[str, T]:
    """
    Aggregate ciphertexts by token key using a user-supplied add operation.

    Parameters
    ----------
    items:
        Iterable of (token, ciphertext) tuples.
    add:
        Optional binary function to combine ciphertexts. Defaults to operator.add.

    Returns
    -------
    Dict[str, T]
        Map from token to aggregated ciphertext.
    """
    add_fn = add or operator.add
    agg: Dict[str, T] = {}
    for token, value in items:
        if token in agg:
            agg[token] = add_fn(agg[token], value)
        else:
            agg[token] = value
    return agg
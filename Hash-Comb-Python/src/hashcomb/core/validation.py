"""Validation helpers for HashComb inputs."""
from __future__ import annotations

import math
from typing import Any

from .exceptions import InvalidParameterError, OutOfRangeError


def validate_finite(name: str, value: Any) -> None:
    """Ensure a value is finite (not NaN/inf)."""
    try:
        v = float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise InvalidParameterError.param(name, value, f"{name} must be a number") from exc
    if not math.isfinite(v):
        raise InvalidParameterError.param(name, value, f"{name} must be finite")


def validate_channels(channels: Any, *, min_: int = 1, max_: int = 30) -> int:
    """Validate channels and return coerced int."""
    try:
        ch = int(channels)
    except Exception as exc:  # pragma: no cover - defensive
        raise InvalidParameterError.param("channels", channels, "channels must be an integer") from exc
    if not (min_ <= ch <= max_):
        raise InvalidParameterError.channels(ch, min_=min_, max_=max_)
    return ch


def validate_range(min_val: Any, max_val: Any) -> None:
    """Validate min/max range and finiteness."""
    validate_finite("min", min_val)
    validate_finite("max", max_val)
    if float(max_val) <= float(min_val):
        raise InvalidParameterError.value_range(min_val, max_val)


def validate_delta(delta: Any | None) -> None:
    """Validate delta (optional, must be >= 0 and finite)."""
    if delta is None:
        return
    validate_finite("delta", delta)
    if float(delta) < 0:
        raise InvalidParameterError.param("delta", delta, "delta must be >= 0")


def validate_probability(name: str, p: Any) -> None:
    """Validate probability in (0, 1]."""
    validate_finite(name, p)
    if not (0.0 < float(p) <= 1.0):
        raise InvalidParameterError.param(name, p, f"{name} must be within (0, 1]")


def validate_value_in_range(value: Any, min_val: Any, max_val: Any) -> None:
    """Validate a value is finite and inside [min, max]."""
    validate_finite("value", value)
    if float(value) < float(min_val) or float(value) > float(max_val):
        raise OutOfRangeError(value, min_val, max_val)

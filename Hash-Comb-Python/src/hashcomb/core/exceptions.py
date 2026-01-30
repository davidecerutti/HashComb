"""Exception hierarchy for HashComb."""
from __future__ import annotations
from typing import Any, Dict, Optional

class HashCombError(Exception):
    """
    Base exception for hashcomb.

    Attributes
    ----------
    message : str
        Human-readable error message.
    code : str
        Stable machine identifier for the error type (useful for logs/CLI).
    ctx : dict
        Optional context (e.g., channels, vmin, vmax, path, token).
    """
    code = "hashcomb_error"

    def __init__(
        self,
        message: str = "",
        *,
        ctx: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.message = message or self.__class__.__name__
        self.ctx = ctx or {}
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        if self.ctx:
            parts = ", ".join(f"{k}={v}" for k, v in self.ctx.items())
            return f"{self.message} [{parts}]"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "context": self.ctx or {},
            "cause": repr(self.__cause__) if hasattr(self, "__cause__") else None,
        }


# -------- Domain-specific exceptions --------

class InvalidParameterError(HashCombError):
    """Raised when user-supplied parameters are invalid."""
    code = "invalid_parameter"

    @classmethod
    def channels(cls, channels: Any, *, min_: int = 1, max_: int = 30) -> InvalidParameterError:
        return cls(
            "channels must be within the allowed range",
            ctx={"param": "channels", "value": channels, "min": min_, "max": max_},
        )

    @classmethod
    def value_range(cls, vmin: Any, vmax: Any) -> InvalidParameterError:
        return cls("max must be greater than min", ctx={"min": vmin, "max": vmax})

    @classmethod
    def param(cls, name: str, value: Any, message: str) -> InvalidParameterError:
        return cls(message, ctx={"param": name, "value": value})


class OutOfRangeError(HashCombError):
    """Raised when a value to encode is outside [vmin, vmax]."""
    code = "out_of_range"

    def __init__(self, value: Any, vmin: Any, vmax: Any) -> None:
        super().__init__("Value is outside the encodable range", ctx={"value": value, "min": vmin, "max": vmax})


class UnknownTokenError(HashCombError):
    """Raised when a token cannot be resolved by the decoder."""
    code = "unknown_token"

    def __init__(self, token: Any) -> None:
        super().__init__("Token not present in the decoder table", ctx={"token": token})


class HashTableError(HashCombError):
    """Generic error around hash table construction/consistency."""
    code = "hash_table_error"


class PathLengthMismatch(HashTableError):
    """Raised when the computed path length does not match configured channels."""
    code = "path_length_mismatch"

    def __init__(self, channels: int, path_len: int) -> None:
        super().__init__(
            "Path length is not consistent with channels",
            ctx={"channels": channels, "path_len": path_len},
        )


class ConfigIOError(HashCombError):
    """Raised on generic I/O failures while reading/writing a config."""
    code = "config_io_error"

    def __init__(self, path: str, op: str, *, cause: Optional[BaseException] = None) -> None:
        super().__init__(f"I/O error while trying to {op} config", ctx={"path": path, "op": op}, cause=cause)


class ConfigNotFoundError(HashCombError):
    """Raised when a config file cannot be found."""
    code = "config_not_found"

    def __init__(self, path: str, *, cause: Optional[BaseException] = None) -> None:
        super().__init__("Config file not found", ctx={"path": path}, cause=cause)


class InvalidConfigError(HashCombError):
    """Raised when a config exists but is invalid/corrupted."""
    code = "invalid_config"

    def __init__(self, path: str, *, cause: Optional[BaseException] = None) -> None:
        super().__init__("Config content is invalid or corrupted", ctx={"path": path}, cause=cause)


class CSVError(HashCombError):
    """Generic CSV processing error."""
    code = "csv_error"


class MissingColumnError(CSVError):
    """Raised when a required CSV column is missing."""
    code = "csv_missing_column"

    def __init__(self, column: str) -> None:
        super().__init__("Required CSV column is missing", ctx={"column": column})


class CSVFormatError(CSVError):
    """Raised when CSV format/sniffing/parsing fails."""
    code = "csv_format_error"

    def __init__(
        self,
        message: str,
        *,
        ctx: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message, ctx=ctx, cause=cause)

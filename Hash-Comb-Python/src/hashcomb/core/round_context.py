from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import secrets


@dataclass(frozen=True)
class RoundContext:
    """Per-round shared context (e.g., salt and RNG seed)."""
    salt: Optional[str] = None
    seed: Optional[int] = None

    @staticmethod
    def generate(*, salt_bytes: int = 4, seed: Optional[int] = None) -> RoundContext:
        """Generate a random salt (hex) and optional seed for a round."""
        salt = secrets.token_hex(salt_bytes) if salt_bytes > 0 else None
        return RoundContext(salt=salt, seed=seed)

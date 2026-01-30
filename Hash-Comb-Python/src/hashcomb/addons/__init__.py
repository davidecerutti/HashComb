"""Optional add-ons for HashComb (aggregation, serialization)."""

from .aggregation import aggregate_ciphertexts
from .serialization import serialize_path, deserialize_path

__all__ = ["aggregate_ciphertexts", "serialize_path", "deserialize_path"]
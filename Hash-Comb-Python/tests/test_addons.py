import pytest

from hashcomb.addons.aggregation import aggregate_ciphertexts
from hashcomb.addons.serialization import serialize_path, deserialize_path


def test_aggregate_ciphertexts_sum():
    items = [
        ("a", 1),
        ("b", 2),
        ("a", 3),
    ]
    out = aggregate_ciphertexts(items)
    assert out == {"a": 4, "b": 2}


def test_serialize_deserialize_roundtrip():
    tokens = ["12|34", "a:b", "xyz"]
    s = serialize_path(tokens)
    parsed = deserialize_path(s)
    assert parsed == tokens


def test_deserialize_invalid_prefix():
    with pytest.raises(ValueError):
        deserialize_path("v0|1|1:a")
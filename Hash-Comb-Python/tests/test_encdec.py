# tests/test_encdec.py
import os
import random
import math
import pytest

from src.encoder import Encoder
from src.decoder import Decoder

def test_encode_decode_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enc = Encoder(4, 15.5, 0.0)
    value = 12.34
    h = enc.encode(value)
    dec = Decoder()
    x = dec.decode(h)
    assert h.isdigit()
    assert 0.0 <= x <= 15.5

def test_encode_decode_coherence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enc = Encoder(4, 15.5, 0.0)
    dec = Decoder()
    value = 12.34
    h = enc.encode(value)
    decoded = dec.decode(h)
    leaf_width = (enc.max - enc.min) / (2 ** enc.channels)
    assert abs(decoded - value) <= leaf_width / 2, (
        f"Decoded {decoded} troppo lontano da {value}, leaf_width={leaf_width}"
    )


def test_monotonic_and_path_length(tmp_path, monkeypatch):
    channels, vmin, vmax = 6, 0.0, 100.0
    random.seed(42)
    monkeypatch.chdir(tmp_path)
    enc = Encoder(channels, vmax, vmin)
    dec = Decoder()
    xs = sorted(random.uniform(vmin, vmax) for _ in range(200))
    decoded_vals = []
    for x in xs:
        hs = enc.tree.getHValues(x, True)
        assert len(hs) == channels
        decoded_vals.append(dec.decode(hs[-1]))
    for a, b in zip(decoded_vals, decoded_vals[1:]):
        assert a <= b + 1e-12


@pytest.mark.parametrize("channels", [4, 6, 8])
@pytest.mark.parametrize("value", [0.0, 0.1, 3.7, 7.25, 12.34, 15.5])
def test_encode_decode_many_values(tmp_path, monkeypatch, channels, value):
    monkeypatch.chdir(tmp_path)
    enc = Encoder(channels, 15.5, 0.0)
    dec = Decoder()
    h = enc.encode(value)
    decoded = dec.decode(h)
    leaf_width = (enc.max - enc.min) / (2 ** enc.channels)
    assert abs(decoded - value) <= leaf_width / 2 + 1e-12
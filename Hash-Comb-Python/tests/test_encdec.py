# tests/test_encdec.py
import os
import random
import math
import numpy as np
import pytest

import hashcomb as hc

def test_encode_decode_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enc = hc.Encoder(4, 15.5, 0.0)
    value = 12.34
    h = enc.encode(value)
    dec = hc.Decoder()
    x = dec.decode(h)
    assert h.isdigit()
    assert 0.0 <= x <= 15.5

def test_encode_decode_coherence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enc = hc.Encoder(4, 15.5, 0.0)
    dec = hc.Decoder()
    value = 12.34
    h = enc.encode(value)
    decoded = dec.decode(h)
    leaf_width = (enc.max - enc.min) / (2 ** enc.channels)
    assert abs(decoded - value) <= leaf_width / 2, (
        f"Decoded {decoded} to much difference from {value}, leaf_width={leaf_width}"
    )


def test_monotonic_and_path_length(tmp_path, monkeypatch):
    channels, vmin, vmax = 6, 0.0, 100.0
    random.seed(42)
    monkeypatch.chdir(tmp_path)
    enc = hc.Encoder(channels, vmax, vmin)
    dec = hc.Decoder()
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
    enc = hc.Encoder(channels, 15.5, 0.0)
    dec = hc.Decoder()
    h = enc.encode(value)
    decoded = dec.decode(h)
    leaf_width = (enc.max - enc.min) / (2 ** enc.channels)
    assert abs(decoded - value) <= leaf_width / 2 + 1e-12




def test_encode_decode_array(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    
    channels, vmin, vmax = 8, 0.0, 15.5
    enc = hc.Encoder(channels, vmax, vmin)
    dec = hc.Decoder()

    x = np.linspace(vmin, vmax, 257, dtype=np.float64) 
    tok = enc.encodeArray(x)
    y = dec.decodeArray(tok)

    assert isinstance(tok, np.ndarray) and tok.ndim == 1
    assert isinstance(y, np.ndarray) and y.ndim == 1 and y.dtype == np.float64
    assert y.shape == x.shape
    assert all(isinstance(t, (str, np.str_)) and t.isdigit() for t in tok)

    leaf_width = (enc.max - enc.min) / (2 ** enc.channels)
    assert y.min() >= vmin - 1e-12
    assert y.max() <= vmax + 1e-12
    assert np.all(np.abs(y - x) <= (leaf_width / 2 + 1e-12))

    x_list = x[:10].tolist()
    tok_list = enc.encodeArray(x_list)
    y_list = dec.decodeArray(tok_list)
    assert np.allclose(y_list, dec.decodeArray(tok_list))
    y_scalar = np.array([dec.decode(t) for t in tok_list], dtype=np.float64)
    assert np.allclose(y_list, y_scalar)
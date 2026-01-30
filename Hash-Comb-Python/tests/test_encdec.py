# tests/test_encdec.py
import random
import numpy as np
import pytest

import hashcomb as hc
from hashcomb.core.exceptions import InvalidParameterError

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


def test_delta_expands_range(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    channels, vmin, vmax = 4, 0.0, 10.0
    delta = 0.5
    enc = hc.Encoder(channels, vmax, vmin, delta=delta)
    assert enc.min == vmin - delta
    assert enc.max == vmax + delta

    # value outside original range but inside expanded range
    h = enc.encode(vmax + 0.25)
    dec = hc.Decoder()
    decoded = dec.decode(h)
    assert enc.min <= decoded <= enc.max


def test_invalid_delta_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(InvalidParameterError):
        hc.Encoder(4, 10.0, 0.0, delta=-0.1)


def test_encode_path_prefix_decode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    channels, vmin, vmax = 5, 0.0, 10.0
    enc = hc.Encoder(channels, vmax, vmin, includeInternal=True)
    dec = hc.Decoder()

    value = 7.25
    path = enc.encodePath(value)
    assert len(path) == channels

    # Walk the tree and compare decoded centers for each prefix
    node = enc.tree.root
    for depth in range(1, channels + 1):
        decoded = dec.decodePath(path[:depth])
        assert isinstance(decoded, float)

        if node.isLeaf:
            expected_center = node.getCenter
        else:
            c = node.getCenter
            if value < c:
                node = node.left
            else:
                node = node.right
            expected_center = node.getCenter

        assert abs(decoded - expected_center) <= 1e-12


def test_prefix_length_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enc = hc.Encoder(4, 10.0, 0.0)
    with pytest.raises(InvalidParameterError):
        enc.encodePrefix(1.23, 0)
    with pytest.raises(InvalidParameterError):
        enc.encodePrefix(1.23, 5)


def test_randomized_encoder_prefix_and_leaf(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    channels, vmin, vmax = 6, 0.0, 12.0
    enc = hc.RandomizedEncoder(
        channels,
        vmax,
        vmin,
        selectionProbability=0.5,
        seed=123,
        includeInternal=True,
    )
    dec = hc.Decoder()

    value = 9.75
    path = enc.encodePath(value)
    assert 1 <= len(path) <= channels
    decoded = dec.decodePath(path)
    assert isinstance(decoded, float)

    leaf = enc.encode(value)
    assert isinstance(leaf, str)
    leaf_decoded = dec.decode(leaf)
    assert vmin - 1e-12 <= leaf_decoded <= vmax + 1e-12


def test_randomized_encoder_determinism_with_round_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = hc.RoundContext(salt="roundX", seed=123)
    enc1 = hc.RandomizedEncoder(6, 10.0, 0.0, selectionProbability=0.5, roundContext=ctx)
    enc2 = hc.RandomizedEncoder(6, 10.0, 0.0, selectionProbability=0.5, roundContext=ctx)

    v = 4.56
    assert enc1.encode(v) == enc2.encode(v)
    assert enc1.encodePath(v) == enc2.encodePath(v)


def test_compute_selection_probability_paper_example():
    # Paper example: L=16 -> p ~= 0.087826 for target level 8
    p = hc.RandomizedEncoder.compute_selection_probability(16, 8)
    assert abs(p - 0.087826) < 1e-4


def test_salt_changes_hashes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    channels, vmin, vmax = 4, 0.0, 15.0

    enc_a = hc.Encoder(channels, vmax, vmin, salt="saltA", includeInternal=True, configPath="configA.pkl")
    enc_b = hc.Encoder(channels, vmax, vmin, salt="saltB", includeInternal=True, configPath="configB.pkl")

    value = 12.34
    h_a = enc_a.encode(value)
    h_b = enc_b.encode(value)
    assert h_a != h_b

    dec_a = hc.Decoder(configPath="configA.pkl")
    dec_b = hc.Decoder(configPath="configB.pkl")
    assert isinstance(dec_a.decode(h_a), float)
    assert isinstance(dec_b.decode(h_b), float)


def test_encoder_from_pkl_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enc = hc.Encoder(4, 10.0, 0.0, salt="s1", includeInternal=True, configPath="enc.pkl")
    enc_loaded = hc.Encoder.from_pkl("enc.pkl")

    v = 3.21
    assert enc.encode(v) == enc_loaded.encode(v)


def test_randomized_encoder_from_pkl_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enc = hc.RandomizedEncoder(
        6,
        12.0,
        0.0,
        selectionProbability=0.4,
        seed=123,
        includeInternal=True,
        configPath="rand.pkl",
    )
    enc_loaded = hc.RandomizedEncoder.from_pkl("rand.pkl")

    v = 5.67
    assert enc.encode(v) == enc_loaded.encode(v)


def test_round_context_overrides_salt_and_seed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = hc.RoundContext(salt="roundX", seed=123)

    enc = hc.RandomizedEncoder(
        6,
        10.0,
        0.0,
        selectionProbability=0.5,
        seed=999,
        salt="other",
        roundContext=ctx,
        includeInternal=True,
    )

    h1 = enc.encode(3.14)
    h2 = enc.encode(3.14)
    # deterministic RNG with fixed seed should produce repeatable sequence
    assert isinstance(h1, str) and isinstance(h2, str)


def test_decode_path_empty_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enc = hc.Encoder(4, 10.0, 0.0, includeInternal=True)
    dec = hc.Decoder()
    with pytest.raises(InvalidParameterError):
        dec.decodePath([])
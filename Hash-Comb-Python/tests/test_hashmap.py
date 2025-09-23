from src.encoder import Encoder
from src.decoder import Decoder

def test_hashmap_size_and_uniqueness(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    channels, vmin, vmax = 6, 0.0, 15.5
    enc = Encoder(channels, vmax, vmin)
    dec = Decoder()
    expected_size = 2 ** channels
    assert len(enc.hashMap) == expected_size
    assert len(set(enc.hashMap.keys())) == expected_size

def test_all_hashes_point_to_valid_centers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enc = Encoder(4, 15.5, 0.0)
    dec = Decoder()
    for h, node in enc.hashMap.items():
        decoded = dec.decode(h)
        assert node.getMin() <= decoded <= node.getMax()
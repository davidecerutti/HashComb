import hashcomb as hc

def test_hashmap_size_and_uniqueness(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    channels, vmin, vmax = 6, 0.0, 15.5
    enc = hc.Encoder(channels, vmax, vmin)
    dec = hc.Decoder()
    expected_size = 2 ** channels
    assert len(enc.hashMap) == expected_size
    assert len(set(enc.hashMap.keys())) == expected_size

def test_all_hashes_point_to_valid_centers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enc = hc.Encoder(4, 15.5, 0.0)
    dec = hc.Decoder()
    for h, node in enc.hashMap.items():
        decoded = dec.decode(h)
        assert node.min <= decoded <= node.max
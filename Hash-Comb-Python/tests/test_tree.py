import os, sys, io, contextlib

import hashcomb as hc

def test_build_and_traverse_hash(capsys):
    t = hc.Tree(4, 15.5, 0)
    count = t.traverseLevelOrder(True)
    assert count > 0
    captured = capsys.readouterr().out


def test_get_hvalues_in_range():
    t = hc.Tree(4, 15.5, 0)
    hs = t.getHValues(12.344578, True)
    assert isinstance(hs, list) and len(hs) > 0
    assert all(isinstance(h, str) and h.isdigit() for h in hs)


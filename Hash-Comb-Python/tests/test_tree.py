import pytest

import hashcomb as hc
from hashcomb.core.exceptions import OutOfRangeError

def test_build_and_traverse_hash(capsys):
    t = hc.Tree(4, 15.5, 0)
    count = t.traverseLevelOrder(True)
    assert count > 0
    capsys.readouterr()


def test_get_hvalues_in_range():
    t = hc.Tree(4, 15.5, 0)
    hs = t.getHValues(12.344578, True)
    assert isinstance(hs, list) and len(hs) > 0
    assert all(isinstance(h, str) and h.isdigit() for h in hs)


def test_get_hvalues_with_salt():
    t = hc.Tree(4, 15.5, 0)
    hs = t.getHValues(12.34, True, salt="round1")
    assert isinstance(hs, list) and len(hs) > 0
    assert all(isinstance(h, str) and h.isdigit() for h in hs)


def test_get_hvalues_out_of_range():
    t = hc.Tree(4, 15.5, 0)
    with pytest.raises(OutOfRangeError):
        t.getHValues(99.0, True)


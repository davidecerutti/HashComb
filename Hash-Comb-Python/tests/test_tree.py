import os, sys, io, contextlib

from src.node import Node
from src.tree import Tree

def test_build_and_traverse_hash(capsys):
    t = Tree(4, 15.5, 0)
    count = t.traverseLevelOrder(True)
    assert count > 0
    captured = capsys.readouterr().out


def test_build_and_traverse_non_hash():
    t = Tree(4, 15.5, 0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        count = t.traverseLevelOrder(False)
    out = buf.getvalue()
    assert count > 0
    assert "[" in out and "]" in out, f"output inatteso: {out!r}"


def test_get_hvalues_in_range():
    t = Tree(4, 15.5, 0)
    hs = t.getHValues(12.344578, True)  # (hash Java & 0x0FFFFFFF)
    assert isinstance(hs, list) and len(hs) > 0
    assert all(isinstance(h, str) and h.isdigit() for h in hs)

def test_out_of_range_exits():
    t = Tree(4, 15.5, 0)
    try:
        t.getHValues(99.0, True)
        assert False, "Doveva lanciare SystemExit"
    except SystemExit as e:
        assert e.code == -1

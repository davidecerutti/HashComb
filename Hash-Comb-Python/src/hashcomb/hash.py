from __future__ import annotations
import io, os, csv
from typing import Dict, Optional, List, Iterable, Tuple
import hashlib

from .node import Node
from .tree import Tree

class Hash:

    @staticmethod
    def buildHashTable(tree: Tree) -> Dict[str, Node]:
        table: Dict[str, Node] = {}
        def collect(n: Optional[Node]) -> None:
            if n is None:
                return
            if n.channel == tree.channels:
                key = n.getValue(True)
                table[key] = n
                return
            collect(n.left)
            collect(n.right)
        collect(tree.root)
        return table
    
    @staticmethod
    def sha3_256_int64(s: str) -> int:
        d = hashlib.sha3_256(s.encode('utf-8')).digest()
        return int.from_bytes(d[:8], 'big', signed=False)
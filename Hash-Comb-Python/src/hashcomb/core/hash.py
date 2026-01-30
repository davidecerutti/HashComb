from __future__ import annotations
"""Hash helpers for HashComb tokens and lookup tables."""
from typing import Dict, Optional
import hashlib

from .node import Node
from .tree import Tree

class Hash:

    @staticmethod
    def buildHashTable(tree: Tree, include_internal: bool = False, salt: str | None = None) -> Dict[str, Node]:
        """Build a map from hash token to node (leaf-only by default)."""
        table: Dict[str, Node] = {}
        def collect(n: Optional[Node]) -> None:
            if n is None:
                return
            if include_internal or n.channel == tree.channels:
                key = n.getValue(True, salt)
                table[key] = n
            if n.channel == tree.channels:
                return
            collect(n.left)
            collect(n.right)
        collect(tree.root)
        return table
    
    @staticmethod
    def sha3_256_int64(s: str) -> int:
        """Hash a string with SHA3-256 and return the first 64 bits as int."""
        d = hashlib.sha3_256(s.encode('utf-8')).digest()
        return int.from_bytes(d[:8], 'big', signed=False)

    @staticmethod
    def hash_token(s: str, salt: str | None = None) -> str:
        """Return a compact decimal token (28-bit) for a string + optional salt."""
        if salt is not None:
            s = f"{s}|{salt}"
        return str(Hash.sha3_256_int64(s) & 0x0FFFFFFF)
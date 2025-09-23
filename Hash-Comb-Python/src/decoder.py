from __future__ import annotations
from typing import Optional, List, Any, Union, Dict

from src.node import Node
from src.tree import Tree
from src.utils import Utils

class Decoder:
    hashMap: Dict[str, Node]

    def __init__(self, configPath: str = "configuration.pkl") -> None:
        self.hashMap = Utils.readHashTable2File(configPath)

    def decode(self, encValue:str) -> float:
        node : Node = self.hashMap.get(encValue)
        if node is None: raise KeyError(f"hash {encValue} not found")
        return float(node.getCenter())

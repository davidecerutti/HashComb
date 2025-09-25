from __future__ import annotations
from typing import Optional, List, Any, Union, Dict, Sequence
import numpy as np

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

    def decodeArray(self, encValues: Union[Sequence[str], np.ndarray] ) -> np.ndarray:
        decodedStrings = np.empty(len(encValues), dtype=np.float64)
        for index, value in enumerate(encValues):
            decodedStrings[index] = self.decode(str(value))
        return decodedStrings
    
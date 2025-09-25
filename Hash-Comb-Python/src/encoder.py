from __future__ import annotations
from typing import Optional, List, Any, Union, Dict, Sequence
import numpy as np

from src.node import Node
from src.tree import Tree
from src.utils import Utils

class Encoder : 
    channels : int
    min : float
    max : float
    tree : Tree
    hashMap : Dict[str, Node]

    def __init__(self, channels: int , maxValue: float, minValue: float, configPath: str = "configuration.pkl") -> None :
        Encoder.channels = int(channels)
        Encoder.min = float(minValue)
        Encoder.max = float(maxValue)
        self.tree = Tree(channels, maxValue, minValue)
        Utils.writeHashTable2File(configPath, self.tree)
        self.hashMap = Utils.readHashTable2File(configPath)

    def encode(self, value: float) -> str:
        hashStrings: List[str] = self.tree.getHValues(value, True)
        if Encoder.channels < 1 or Encoder.channels > len(hashStrings):
            raise ValueError(f"path length {len(hashStrings)} mismatched channels {Encoder.channels}")
        return hashStrings[Encoder.channels - 1]
    
    def encodeArray(self, values: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        hashesStrings = np.empty(values.shape[0], dtype=object)
        for index, value in enumerate(values):
            hashesStrings[index] = self.encode(float(value))
        return hashesStrings
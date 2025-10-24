from __future__ import annotations
from typing import Optional, List, Any, Union, Dict, Sequence
import numpy as np
import logging

from .node import Node
from .tree import Tree
from .io import CsvIO, PklIO
from .hash import Hash
from .exceptions import InvalidParameterError, PathLengthMismatch


logger = logging.getLogger(__name__)

class Encoder : 
    channels : int
    min : float
    max : float
    tree : Tree
    hashMap : Dict[str, Node]

    def __init__(self, channels: int , maxValue: float, minValue: float, configPath: str = "configuration.pkl") -> None :
        if maxValue <= minValue:
            raise InvalidParameterError.value_range(minValue, maxValue)
        if not (1 <= channels):
            raise InvalidParameterError.channels(channels, min_=1)
        self.channels = int(channels)
        self.min = float(minValue)
        self.max = float(maxValue)
        self.tree = Tree(channels, maxValue, minValue)
        self.hashMap = Hash.buildHashTable(self.tree)
        PklIO.savePickle(configPath, self.hashMap)
        logger.debug(f"Encoder initialized with channels={channels}, min={minValue}, max={maxValue}, configPath='{configPath}'")

    def encode(self, value: float) -> str:
        hashStrings: List[str] = self.tree.getHValues(value, True)
        if self.channels < 1 or self.channels > len(hashStrings):
            raise PathLengthMismatch(self.channels, len(hashStrings))
        return hashStrings[self.channels - 1]
    
    def encodeArray(self, values: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        hashesStrings = np.empty(values.shape[0], dtype=object)
        for index, value in enumerate(values):
            hashesStrings[index] = self.encode(float(value))
        return hashesStrings
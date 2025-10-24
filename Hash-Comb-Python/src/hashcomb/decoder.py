from __future__ import annotations
import logging
from typing import Optional, List, Any, Union, Dict, Sequence
import numpy as np


from .node import Node
from .tree import Tree
from .io import CsvIO, PklIO
from .exceptions import UnknownTokenError

logger = logging.getLogger(__name__)

class Decoder:
    hashMap: Dict[str, Node]

    def __init__(self, configPath: str = "configuration.pkl") -> None:
        self.hashMap = PklIO.loadPickle(configPath)
        logger.debug(f"Decoder initialized with configPath='{configPath}'")

    def decode(self, encValue:str) -> float:
        node : Node = self.hashMap.get(encValue)
        if node is None: 
            raise UnknownTokenError(encValue)
        return float(node.getCenter)

    def decodeArray(self, encValues: Union[Sequence[str], np.ndarray] ) -> np.ndarray:
        decodedStrings = np.empty(len(encValues), dtype=np.float64)
        for index, value in enumerate(encValues):
            decodedStrings[index] = self.decode(str(value))
        return decodedStrings
    
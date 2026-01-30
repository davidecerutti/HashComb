from __future__ import annotations
"""Decoder utilities for HashComb tokens."""
import logging
from typing import Union, Dict, Sequence
import numpy as np

from ..core.node import Node
from ..io.io import PklIO
from ..core.hash import Hash
from ..core.exceptions import UnknownTokenError, InvalidParameterError, InvalidConfigError

logger = logging.getLogger(__name__)

class Decoder:
    """Decode hash tokens into quantized values (centers)."""
    hashMap: Dict[str, Node]

    def __init__(self, configPath: str = "configuration.pkl") -> None:
        config = PklIO.loadConfig(configPath)
        hash_map = config.get("hashMap")
        if hash_map is None:
            tree = config.get("tree")
            if tree is None:
                raise InvalidConfigError(configPath, cause=ValueError("missing hashMap and tree"))
            include_internal = bool(config.get("params", {}).get("includeInternal", False))
            salt = config.get("salt")
            hash_map = Hash.buildHashTable(tree, include_internal=include_internal, salt=salt)
        self.hashMap = hash_map
        logger.debug(f"Decoder initialized with configPath='{configPath}'")

    def decode(self, encValue:str) -> float:
        """Decode a single hash token to the bin center."""
        node : Node = self.hashMap.get(encValue)
        if node is None: 
            raise UnknownTokenError(encValue)
        return float(node.getCenter)

    def decodePath(self, encPath: Sequence[str]) -> float:
        """Decode a path/prefix by using its last token."""
        if not encPath:
            raise InvalidParameterError.param("encPath", encPath, "encPath must contain at least one hash")
        return self.decode(str(encPath[-1]))

    def decodeArray(self, encValues: Union[Sequence[str], np.ndarray] ) -> np.ndarray:
        """Vectorized decode for leaf tokens."""
        decodedStrings = np.empty(len(encValues), dtype=np.float64)
        for index, value in enumerate(encValues):
            decodedStrings[index] = self.decode(str(value))
        return decodedStrings

    def decodePathArray(self, encPaths: Sequence[Sequence[str]]) -> np.ndarray:
        """Vectorized decode for paths/prefixes (uses last token)."""
        decodedStrings = np.empty(len(encPaths), dtype=np.float64)
        for index, path in enumerate(encPaths):
            decodedStrings[index] = self.decodePath(path)
        return decodedStrings
    
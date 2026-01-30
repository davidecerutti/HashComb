from __future__ import annotations
"""Deterministic HashComb encoder."""
from typing import List, Any, Union, Dict, Sequence
from pathlib import Path
import numpy as np
import logging

from ..core.node import Node
from ..core.tree import Tree
from ..core.round_context import RoundContext
from ..io.io import PklIO
from ..core.hash import Hash
from ..core.exceptions import InvalidParameterError, PathLengthMismatch, InvalidConfigError


logger = logging.getLogger(__name__)

class Encoder:
    """Encode values using a fixed quantization tree (deterministic)."""
    channels : int
    min : float
    max : float
    tree : Tree
    hashMap : Dict[str, Node]

    def __init__(
        self,
        channels: int,
        maxValue: float,
        minValue: float,
        configPath: str = "configuration.pkl",
        includeInternal: bool = False,
        delta: float | None = None,
        salt: str | None = None,
        roundContext: RoundContext | None = None,
    ) -> None :
        if maxValue <= minValue:
            raise InvalidParameterError.value_range(minValue, maxValue)
        if not (1 <= channels):
            raise InvalidParameterError.channels(channels, min_=1)
        if delta is not None and delta < 0:
            raise InvalidParameterError(
                "delta must be >= 0",
                ctx={"param": "delta", "value": delta},
            )
        self.channels = int(channels)
        if delta is not None:
            self.min = float(minValue) - float(delta)
            self.max = float(maxValue) + float(delta)
        else:
            self.min = float(minValue)
            self.max = float(maxValue)
        self.salt = roundContext.salt if roundContext is not None else salt
        self.tree = Tree(channels, self.max, self.min)
        self.hashMap = Hash.buildHashTable(self.tree, include_internal=includeInternal, salt=self.salt)
        config = {
            "schema": "hashcomb.config.v1",
            "encoder": "deterministic",
            "tree": self.tree,
            "hashMap": self.hashMap,
            "params": {
                "channels": self.channels,
                "min": self.min,
                "max": self.max,
                "includeInternal": includeInternal,
                "delta": delta,
            },
            "salt": self.salt,
        }
        PklIO.saveConfig(configPath, config)
        logger.debug(f"Encoder initialized with channels={channels}, min={self.min}, max={self.max}, configPath='{configPath}', includeInternal={includeInternal}, delta={delta}, salt={self.salt}")

    @classmethod
    def from_pkl(cls, path: str | Path) -> Encoder:
        """Instantiate an Encoder from a saved configuration file."""
        config = PklIO.loadConfig(path)
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Encoder:
        """Instantiate an Encoder from an in-memory configuration dict."""
        params = config.get("params", {}) if isinstance(config, dict) else {}
        tree = config.get("tree") if isinstance(config, dict) else None
        if tree is None:
            try:
                channels = int(params["channels"])
                min_val = float(params["min"])
                max_val = float(params["max"])
            except Exception as e:
                raise InvalidConfigError("<in-memory>", cause=e) from e
            tree = Tree(channels, max_val, min_val)

        hash_map = config.get("hashMap") if isinstance(config, dict) else None
        salt = config.get("salt") if isinstance(config, dict) else None
        if hash_map is None:
            include_internal = bool(params.get("includeInternal", False))
            hash_map = Hash.buildHashTable(tree, include_internal=include_internal, salt=salt)

        self = cls.__new__(cls)
        self.channels = int(params.get("channels", tree.channels))
        self.min = float(params.get("min", tree.min))
        self.max = float(params.get("max", tree.max))
        self.salt = salt
        self.tree = tree
        self.hashMap = hash_map
        return self

    def encode(self, value: float) -> str:
        """Encode a single value, returning the leaf hash token."""
        hashStrings: List[str] = self.tree.getHValues(value, True, self.salt)
        if self.channels < 1 or self.channels > len(hashStrings):
            raise PathLengthMismatch(self.channels, len(hashStrings))
        return hashStrings[self.channels - 1]

    def encodePath(self, value: float) -> List[str]:
        """Encode a value and return the full path of hashes (root→leaf)."""
        hashStrings: List[str] = self.tree.getHValues(value, True, self.salt)
        if self.channels < 1 or self.channels > len(hashStrings):
            raise PathLengthMismatch(self.channels, len(hashStrings))
        return hashStrings[: self.channels]

    def encodePrefix(self, value: float, length: int) -> List[str]:
        """Encode a value and return a prefix of the hash path of length k."""
        if not (1 <= int(length) <= self.channels):
            raise InvalidParameterError.channels(length, min_=1, max_=self.channels)
        hashStrings: List[str] = self.tree.getHValues(value, True, self.salt)
        if len(hashStrings) < int(length):
            raise PathLengthMismatch(int(length), len(hashStrings))
        return hashStrings[: int(length)]
    
    def encodeArray(self, values: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Vectorized encode for a list/array of values (leaf hash tokens)."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        hashesStrings = np.empty(values.shape[0], dtype=object)
        for index, value in enumerate(values):
            hashesStrings[index] = self.encode(float(value))
        return hashesStrings

    def encodePathArray(self, values: Union[Sequence[float], np.ndarray]) -> List[List[str]]:
        """Vectorized encode for full hash paths."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        out: List[List[str]] = []
        for value in values:
            out.append(self.encodePath(float(value)))
        return out

    def encodePrefixArray(self, values: Union[Sequence[float], np.ndarray], length: int) -> List[List[str]]:
        """Vectorized encode for prefix paths of fixed length."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        out: List[List[str]] = []
        for value in values:
            out.append(self.encodePrefix(float(value), length))
        return out
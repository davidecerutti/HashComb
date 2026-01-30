from __future__ import annotations
"""Randomized HashComb encoder (paper mode)."""
from typing import Optional, List, Union, Dict, Sequence, Any
from pathlib import Path
import numpy as np
import logging
import random

from ..core.tree import Tree
from ..core.node import Node
from ..core.hash import Hash
from ..io.io import PklIO
from ..core.round_context import RoundContext
from ..core.exceptions import InvalidParameterError, PathLengthMismatch, InvalidConfigError

logger = logging.getLogger(__name__)


class RandomizedEncoder:
    """Encode values with randomized level selection (last-head rule)."""
    channels: int
    min: float
    max: float
    tree: Tree
    hashMap: Dict[str, Node]

    def __init__(
        self,
        channels: int,
        maxValue: float,
        minValue: float,
        *,
        delta: float | None = None,
        selectionProbability: float = 0.5,
        seed: Optional[int] = None,
        configPath: str = "configuration.pkl",
        includeInternal: bool = True,
        salt: str | None = None,
        roundContext: RoundContext | None = None,
    ) -> None:
        if maxValue <= minValue:
            raise InvalidParameterError.value_range(minValue, maxValue)
        if not (1 <= channels):
            raise InvalidParameterError.channels(channels, min_=1)
        if delta is not None and delta < 0:
            raise InvalidParameterError(
                "delta must be >= 0",
                ctx={"param": "delta", "value": delta},
            )
        if not (0.0 < selectionProbability <= 1.0):
            raise InvalidParameterError(
                "selectionProbability must be within (0, 1]",
                ctx={"param": "selectionProbability", "value": selectionProbability},
            )

        self.channels = int(channels)
        if delta is not None:
            self.min = float(minValue) - float(delta)
            self.max = float(maxValue) + float(delta)
        else:
            self.min = float(minValue)
            self.max = float(maxValue)
        self.selectionProbability = float(selectionProbability)
        ctx_seed = roundContext.seed if roundContext is not None else seed
        self._rng = random.Random(ctx_seed)
        self.salt = roundContext.salt if roundContext is not None else salt

        self.tree = Tree(self.channels, self.max, self.min)
        self.hashMap = Hash.buildHashTable(self.tree, include_internal=includeInternal, salt=self.salt)
        config = {
            "schema": "hashcomb.config.v1",
            "encoder": "randomized",
            "tree": self.tree,
            "hashMap": self.hashMap,
            "params": {
                "channels": self.channels,
                "min": self.min,
                "max": self.max,
                "includeInternal": includeInternal,
                "delta": delta,
                "selectionProbability": self.selectionProbability,
                "seed": ctx_seed,
            },
            "salt": self.salt,
        }
        PklIO.saveConfig(configPath, config)

        logger.debug(
            "RandomizedEncoder initialized with channels=%s, min=%s, max=%s, p=%s, seed=%s, configPath='%s', includeInternal=%s",
            channels,
            self.min,
            self.max,
            selectionProbability,
            seed,
            configPath,
            includeInternal,
        )

    @classmethod
    def from_pkl(cls, path: str | Path) -> RandomizedEncoder:
        """Instantiate a RandomizedEncoder from a saved configuration file."""
        config = PklIO.loadConfig(path)
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> RandomizedEncoder:
        """Instantiate a RandomizedEncoder from an in-memory configuration dict."""
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

        selection_probability = params.get("selectionProbability")
        if selection_probability is None:
            raise InvalidConfigError("<in-memory>", cause=ValueError("selectionProbability missing"))

        hash_map = config.get("hashMap") if isinstance(config, dict) else None
        salt = config.get("salt") if isinstance(config, dict) else None
        if hash_map is None:
            include_internal = bool(params.get("includeInternal", False))
            hash_map = Hash.buildHashTable(tree, include_internal=include_internal, salt=salt)

        seed = params.get("seed")
        self = cls.__new__(cls)
        self.channels = int(params.get("channels", tree.channels))
        self.min = float(params.get("min", tree.min))
        self.max = float(params.get("max", tree.max))
        self.selectionProbability = float(selection_probability)
        self._rng = random.Random(seed)
        self.salt = salt
        self.tree = tree
        self.hashMap = hash_map
        return self

    @staticmethod
    def expected_level(channels: int, selectionProbability: float) -> float:
        """
        Expected last-head position using the paper's Eq. (14):
        sum_{i=0}^{L-1} (L - i) * p * (1 - p)^i
        """
        L = int(channels)
        p = float(selectionProbability)
        if L < 1:
            raise InvalidParameterError.channels(L, min_=1)
        if not (0.0 < p <= 1.0):
            raise InvalidParameterError(
                "selectionProbability must be within (0, 1]",
                ctx={"param": "selectionProbability", "value": selectionProbability},
            )
        q = 1.0 - p
        total = 0.0
        for i in range(0, L):
            total += (L - i) * p * (q ** i)
        return total

    @staticmethod
    def compute_selection_probability(
        channels: int,
        targetLevel: float,
        *,
        tol: float = 1e-9,
        max_iter: int = 200,
    ) -> float:
        """
        Solve Eq. (14) for p in (0, 1]:
        sum_{i=0}^{L-1} (L - i) * p * (1 - p)^i = targetLevel
        """
        L = int(channels)
        if L < 1:
            raise InvalidParameterError.channels(L, min_=1)
        if not (0.0 < float(targetLevel) <= float(L)):
            raise InvalidParameterError(
                "targetLevel must be within (0, L]",
                ctx={"param": "targetLevel", "value": targetLevel, "L": L},
            )

        low, high = 1e-12, 1.0
        for _ in range(max_iter):
            mid = (low + high) / 2.0
            val = RandomizedEncoder.expected_level(L, mid)
            if abs(val - targetLevel) <= tol:
                return mid
            if val < targetLevel:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    def _sample_level(self) -> int:
        """
        Sample a quantization level using the 'last head in L tosses' rule.
        Ensures at least one head (resamples otherwise).
        """
        L = self.channels
        p = self.selectionProbability
        last_head = 0
        while last_head == 0:
            for i in range(1, L + 1):
                if self._rng.random() < p:
                    last_head = i
        return last_head

    def encode(self, value: float) -> str:
        """Encode a value with randomized level, returning a single hash token."""
        hashStrings: List[str] = self.tree.getHValues(value, True, self.salt)
        if self.channels < 1 or self.channels > len(hashStrings):
            raise PathLengthMismatch(self.channels, len(hashStrings))
        k = self._sample_level()
        return hashStrings[k - 1]

    def encodePath(self, value: float) -> List[str]:
        """Encode a value and return a randomized-length hash path."""
        hashStrings: List[str] = self.tree.getHValues(value, True, self.salt)
        if self.channels < 1 or self.channels > len(hashStrings):
            raise PathLengthMismatch(self.channels, len(hashStrings))
        k = self._sample_level()
        return hashStrings[:k]

    def encodePrefix(self, value: float, length: int) -> List[str]:
        """Encode a value and return a deterministic prefix length."""
        if not (1 <= int(length) <= self.channels):
            raise InvalidParameterError.channels(length, min_=1, max_=self.channels)
        hashStrings: List[str] = self.tree.getHValues(value, True, self.salt)
        if len(hashStrings) < int(length):
            raise PathLengthMismatch(int(length), len(hashStrings))
        return hashStrings[: int(length)]

    def encodeArray(self, values: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Vectorized encode for leaf tokens with randomized levels."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        hashesStrings = np.empty(values.shape[0], dtype=object)
        for index, value in enumerate(values):
            hashesStrings[index] = self.encode(float(value))
        return hashesStrings

    def encodePathArray(self, values: Union[Sequence[float], np.ndarray]) -> List[List[str]]:
        """Vectorized encode for randomized-length paths."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        out: List[List[str]] = []
        for value in values:
            out.append(self.encodePath(float(value)))
        return out

    def encodePrefixArray(self, values: Union[Sequence[float], np.ndarray], length: int) -> List[List[str]]:
        """Vectorized encode for deterministic prefix length."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        out: List[List[str]] = []
        for value in values:
            out.append(self.encodePrefix(float(value), length))
        return out
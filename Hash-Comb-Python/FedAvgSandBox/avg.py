from __future__ import annotations
from typing import Dict, Optional, List, Iterable, Tuple
import numpy as np
import torch

class Avg:

    @staticmethod
    def fedAvg(updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not updates:
            raise ValueError("FedAvg no updates")
        keys = list(updates[0].keys())
        out = {k: torch.zeros_like(updates[0][k]) for k in keys}
        w = 1.0 / len(updates)
        for sd in updates:
            for k in keys:
                out[k] += sd[k]*w
        return out
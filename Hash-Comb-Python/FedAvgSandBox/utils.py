from __future__ import annotations
from typing import Dict, Optional, List, Iterable, Tuple
import numpy as np
import torch

class Utils:

    @staticmethod
    def flattenDict(sd: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, List[int], List[str]]:
        arrs: List[np.ndarray] = []
        lens: List[int] = []
        order: List[str] = []
        for k, v in sd.items():
            a = v.detach().cpu().numpy().reshape(-1).astype(np.float64)
            arrs.append(a); lens.append(a.size); order.append(k)
        flat = np.concatenate(arrs) if arrs else np.array([], dtype=np.float64)
        return flat, lens, order
    
    @staticmethod
    def unflattenDict(flat: np.ndarray, lens: List[int], order: List[str], template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        index = 0
        for k, n in zip(order, lens):
            t = template[k]
            out[k] = torch.from_numpy(flat[index:index+n]).reshape(t.shape).to(t.dtype)
            index = index + n
        return out
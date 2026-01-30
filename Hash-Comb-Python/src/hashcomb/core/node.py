from __future__ import annotations
"""Node structure for HashComb binary quantization tree."""
from typing import Optional, List, Any, Union


class Node:
    """Binary tree node representing a quantization interval."""
    min : float
    max : float
    channel : int
    left : Optional[Node]
    right : Optional[Node]

    def __init__(self, minValue: float, maxValue: float, channelValue: int) -> None:
        self.min: float = float(minValue)
        self.max: float = float(maxValue)
        self.channel: int = int(channelValue)
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None    

    @property
    def getCenter(self) -> float:
        """Return the midpoint of the interval."""
        span = self.max - self.min
        return self.min + span / 2.0
    
    @property
    def isLeaf(self) -> bool:
        """Return True if the node has no children."""
        return (self.left is None and self.right is None)
    
    def __str__(self) -> str:
        return f"Min: {self.min}    Max: {self.max}"

    def getValue(self, *args: Any) -> Union[str, List[str], None]:
        from .hash import Hash
        # Signature 1: getValue(isHashed[, salt]) -> token for this node
        if len(args) in (1, 2) and isinstance(args[0], bool):
            isHashed: bool = args[0]
            salt = args[1] if len(args) == 2 else None
            out = f"{self.channel}[{self.min}  {self.max}]"
            if isHashed:
                return Hash.hash_token(out, salt)
            return out
        # Signature 2: getValue(number, isHashed[, salt]) -> path tokens to leaf
        if len(args) in (2, 3) and isinstance(args[0], (int, float)) and isinstance(args[1], bool):
            number: float = float(args[0])
            isHashed: bool = args[1]
            salt = args[2] if len(args) == 3 else None
            if not self.isLeaf:
                out: List[str] = []
                left_min = self.left.min
                left_max = self.left.max
                if (number >= left_min) and (number < left_max):
                    out.append(self.left.getValue(isHashed, salt))
                    deeper = self.left.getValue(number, isHashed, salt)
                else:
                    out.append(self.right.getValue(isHashed, salt))
                    deeper = self.right.getValue(number, isHashed, salt)
                if deeper is not None:
                    out.extend(deeper)
                return out
            else:
                return None
        raise TypeError("node.py/getValue : expected (bool[, salt]) or (float, bool[, salt])")
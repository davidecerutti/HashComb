from __future__ import annotations
from typing import Optional, List, Any, Union

def _java_string_hashcode(s: str) -> int:
    # Replica di Java String.hashCode() in Python.
    h = 0
    for ch in s:
        h = (31 * h + ord(ch)) & 0xFFFFFFFF 
    return h 


class Node:
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

    def getCenter(self) -> float:
        span = self.max - self.min
        return self.min + span / 2.0
    
    def isLeaf(self) -> bool:
        return (self.left is None and self.right is None)
    
    def getLeft(self) -> Optional[Node]:
        return self.left
    
    def getRight(self) -> Optional[Node]:
        return self.right
    
    def getChannel(self) -> int:
        return self.channel
    
    def getMin(self) -> float:
        return self.min

    def getMax(self) -> float:
        return self.max
    
    def toString(self) -> str:
        return f"Min: {self.min}    Max: {self.max}"
    
    def __str__(self) -> str:
        return self.toString()
    
    #overload due opertaori in java "getValue" ... emulato con args e type checking

    #getValue(boolean isHashed)
    # def getValue(self, isHashed: bool) -> str:
    #     out = f"{self.channel}[{self.min}  {self.max}]"
    #     if isHashed:
    #         h = _java_string_hashcode(out) & 0x0FFFFFFF
    #         return str(h)
    #     return out

    #getValue(double number, boolean isHashed)
    # def getValue(self, number: float, isHashed: bool) -> Optional[List[str]] | None:
    #     if not self.isLeaf():
    #         out: List[str] = []
    #         left_min = self.left.min
    #         left_max = self.left.max
    #         if (number >= left_min) and (number < left_max):
    #             out.append(self.left.getValue(isHashed))
    #             deeper = self.left.getValue_for_number(number, isHashed)
    #             if deeper is not None:
    #                 out.extend(deeper)
    #         else:
    #             out.append(self.right.getValue(isHashed))
    #             deeper = self.right.getValue_for_number(number, isHashed)
    #             if deeper is not None:
    #                 out.extend(deeper)
    #         return out
    #     else:
    #         return None

    def getValue(self, *args: Any) -> Union[str, List[str], None]:
        if len(args) == 1 and isinstance(args[0], bool):
            isHashed: bool = args[0]
            out = f"{self.channel}[{self.min}  {self.max}]"
            if isHashed:
                return str(_java_string_hashcode(out) & 0x0FFFFFFF)
            return out

        if len(args) == 2 and isinstance(args[0], (int, float)) and isinstance(args[1], bool):
            number: float = float(args[0])
            isHashed: bool = args[1]
            if not self.isLeaf():
                out: List[str] = []
                left_min = self.left.min
                left_max = self.left.max
                if (number >= left_min) and (number < left_max):
                    out.append(self.left.getValue(isHashed))
                    deeper = self.left.getValue(number, isHashed)
                else:
                    out.append(self.right.getValue(isHashed))
                    deeper = self.right.getValue(number, isHashed)
                if deeper is not None:
                    out.extend(deeper)
                return out
            else:
                return None

        raise TypeError("node.py/getValue : expected (bool) or (float, bool)")
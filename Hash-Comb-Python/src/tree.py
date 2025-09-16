from __future__ import annotations
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional, List, Any, Union

from src.node import Node

class Tree:
    channels : int
    min : float
    max : float
    root : Node
    places : int = 3
    isRounded : bool = False


    #doppio costruttore Tree(int channels, double max, double min) or Tree(int channels, double max, double min, int rounds)
    def __init__(self, channels: int, maxValue: float, minValue: float, rounds: Optional[int] = None) -> None:
        self.channels = int(channels)
        self.isRounded = False
        self.places = 3

        if rounds is None:
            self.max = float(maxValue)
            self.min = float(minValue)
        else:
            self.isRounded = True
            self.places = int(rounds)
            self.max = self.round(maxValue, self.places)
            self.min = self.round(minValue, self.places)

        self.root = Node(self.min, self.max, 0)
        self.insert(self.root)

    @staticmethod
    def round(value: float, places: int) -> float:
        if places < 0:
            raise ValueError("tree.py/round : places must be >= 0")
        q = Decimal("1").scaleb(-places)
        d = Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP)
        return float(d)
    
    def insert(self, node: Node) -> None:
        currentChannel = node.getChannel()
        if currentChannel != self.channels :
            center = node.getCenter()
            if self.isRounded:
                center = Tree.round(center, self.places)
            leftChild = Node(node.min, center, (currentChannel + 1))
            rightChild = Node(center, node.max, (currentChannel + 1))
            node.left = leftChild
            node.right = rightChild
            self.insert(node.left)
            self.insert(node.right)       

    def traverseLevelOrder(self, isHashed: bool) -> int:
        count = 0
        return self.traverseInOrder(self.root, isHashed, count)

    def traverseInOrder(self, node: Optional[Node], isHashed: bool, count: int) -> int:
        if node is not None:
            count = count + 1
            a = self.traverseInOrder(node.left, isHashed, 0)
            print(" " + node.getValue(isHashed))
            b = self.traverseInOrder(node.right, isHashed, 0)
            count = count + a + b
        return count
    
    def getHValues(self, num: float, isHashed: bool) -> list[str]:
        if (num < self.min) or (num > self.max):
            print(f"Weight {num} is Exceeding Limits --> ({self.min}, {self.max} )")
            raise SystemExit(-1)
        out = self.root.getValue(num, isHashed)
        return out or []
    
    def getMin(self) -> float:
        return self.min 
    
    def getMax(self) -> float:
        return self.max
    
    def getRoot(self) -> Node:
        return self.root
    
    def getChannels(self) -> int:
        return self.channels
from __future__ import annotations
from typing import Dict, Optional, List, Iterable, Tuple
import argparse, time, json
import numpy as np
import torch

from FedAvgSandBox.utils import Utils
from FedAvgSandBox.avg import Avg

from src.encoder import Encoder
from src.decoder import Decoder

class ModelFedAvg(torch.nn.Module):

    def __init__(self, inDim=4, hidden=32, outDim=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(inDim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, outDim),
        )


    def forward(self, x): 
        return self.net(x)
    

    @staticmethod
    def trainOneEpoch(model: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor, lr=0.05, iters=20) -> None:
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        lossf = torch.nn.CrossEntropyLoss()
        for _ in range(iters):
            opt.zero_grad()
            out = model(X)
            loss = lossf(out, Y)
            loss.backward()
            opt.step()

    @staticmethod
    def estimatePlainBytes(sd: Dict[str, torch.Tensor]) -> int:
        tot = 0
        for v in sd.values():
            tot = tot + v.numel() * 4
        return tot
    
    @staticmethod
    def estimateTokenBytes(tokens: np.ndarray) -> int:
        return int(
            sum(len(str(token)) for token in tokens)
        )
    
    @staticmethod
    def clientUpload(stateDict, encoder: Encoder | None):
        if encoder is None:
            return {"plain": stateDict}
        flat, lens, order = Utils.flattenDict(stateDict)
        tokens = encoder.encodeArray(flat) 
        return {"order": order, "lengths": lens, "tokens": tokens.tolist()}
    
    @staticmethod
    def serverAggregate(payloads: List[Dict], decoder: Decoder | None, globalModel: torch.nn.Module):
        template = globalModel.state_dict()
        if decoder is None:
            sds = [p["plain"] for p in payloads]
            newSd = Avg.fedAvg(sds)
        else:
            client_sds = []
            for p in payloads:
                toks = np.array(p["tokens"], dtype=object)
                flat = decoder.decodeArray(toks)
                sd = Utils.unflattenDict(flat, p["lengths"], p["order"], template)
                client_sds.append(sd)
            newSd = Avg.fedAvg(client_sds)
        globalModel.load_state_dict(newSd)
        return newSd
    
    @staticmethod
    def evaluateAcc(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            pred = model(X).argmax(1)
            return float(
                (pred == y).float().mean().item()
                )
#   python -m FedAvgSandBox.main --clients 10 --rounds 50 
#   python -m FedAvgSandBox.main --clients 10 --rounds 50 --encoded --channels 8 --vmin -1.0 --vmax 1.0


from __future__ import annotations
import argparse, time
import numpy as np
import torch


from FedAvgSandBox.mlp import ModelFedAvg

from src.encoder import Encoder
from src.decoder import Decoder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=3, help="number of client in federated scenario")
    ap.add_argument("--rounds", type=int, default=5, help="number of round evaluated by clients")
    ap.add_argument("--encoded", action="store_true", help="uses HashComb encode/decode")
    ap.add_argument("--channels", type=int, default=8, help="depth of the Hash tree")
    ap.add_argument("--vmin", type=float, default=-1.0, help="min value for a hashcomb bucket")
    ap.add_argument("--vmax", type=float, default=1.0, help="max value for a hashcomb bucket")
    ap.add_argument("--seed", type=int, default=50, help="torch random seed")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    X = torch.randn(256, 4)
    y = torch.randint(0, 3, (256,))

    G = ModelFedAvg()

    encoder = Encoder(args.channels, args.vmax, args.vmin) if args.encoded else None
    decoder = Decoder() if args.encoded else None

    total_plain_bytes = 0
    total_token_bytes = 0
    t_start = time.time()

    for r in range(1, args.rounds + 1):
        clients = []
        for _ in range(args.clients):
            m = ModelFedAvg()
            m.load_state_dict(G.state_dict())
            clients.append(m)

        for m in clients:
            ModelFedAvg.trainOneEpoch(m, X, y, lr=0.05, iters=20)

        payloads = []
        round_plain, round_tok = 0, 0
        for m in clients:
            sd = m.state_dict()
            if encoder is None:
                payload = ModelFedAvg.clientUpload(sd, None)
                round_plain += ModelFedAvg.estimatePlainBytes(sd)
            else:
                payload = ModelFedAvg.clientUpload(sd, encoder)
                toks = np.array(payload["tokens"], dtype=object)
                round_tok += ModelFedAvg.estimateTokenBytes(toks)
            payloads.append(payload)

        _ = ModelFedAvg.serverAggregate(payloads, decoder, G)

        acc = ModelFedAvg.evaluateAcc(G, X, y)

        total_plain_bytes += round_plain
        total_token_bytes += round_tok

        if encoder is None:
            print(f"[Round {r}] acc={acc:.3f}  uplink≈{round_plain/1e6:.3f} MB (plain)")
        else:
            print(f"[Round {r}] acc={acc:.3f}  uplink≈{round_tok/1e6:.3f} MB (encoded)")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.2f}s")
    if encoder is None:
        print(f"Total uplink (plain) ≈ {total_plain_bytes/1e6:.3f} MB")
    else:
        print(f"Total uplink (encoded) ≈ {total_token_bytes/1e6:.3f} MB")

if __name__ == "__main__":
    main()
#   python -m src.cli encode --channels 8 --min 0 --max 15.5 --config configurationDemo.pkl --value 12.34
#   python -m src.cli decode --config configurationDemo.pkl --hash 190681195 

import argparse
import sys
from src.encoder import Encoder
from src.decoder import Decoder

def cmd_train(a: argparse.Namespace) -> int:
    Encoder(a.channels, a.max, a.min, configPath=a.config)
    print(f"Model saved: {a.config}")
    return 0

def cmd_encode(a: argparse.Namespace) -> int:
    enc = Encoder(a.channels, a.max, a.min, configPath=a.config)
    print(enc.encode(a.value))
    return 0

def cmd_decode(a: argparse.Namespace) -> int:
    dec = Decoder(configPath=a.config)
    print(dec.decode(a.hash))
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hashcomb", description="HashComb CLI (minimal)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    t = sub.add_parser("train", help="Crea un nuovo modello (.pkl)")
    t.add_argument("--channels", type=int, required=True)
    t.add_argument("--min", type=float, required=True)
    t.add_argument("--max", type=float, required=True)
    t.add_argument("--config", type=str, default="configuration.pkl")
    t.set_defaults(func=cmd_train)

    # encode
    e = sub.add_parser("encode", help="Encoda un valore")
    e.add_argument("--channels", type=int, required=True)
    e.add_argument("--min", type=float, required=True)
    e.add_argument("--max", type=float, required=True)
    e.add_argument("--config", type=str, default="configuration.pkl")
    e.add_argument("--value", type=float, required=True)
    e.set_defaults(func=cmd_encode)

    # decode
    d = sub.add_parser("decode", help="Decoda un hash")
    d.add_argument("--config", type=str, default="configuration.pkl")
    d.add_argument("--hash", type=str, required=True)
    d.set_defaults(func=cmd_decode)

    return p

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())

#i think that a wrapper for encodeArray and decodeArray is useless...
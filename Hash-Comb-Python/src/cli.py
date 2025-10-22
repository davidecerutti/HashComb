#
# CLI wrapper usage:
# python -m src.cli encode --channels <int> --min <float> --max <float> [--config <file.pkl>] --value <float>
# python -m src.cli decode [--config <file.pkl>] --hash <str>
#
# python -m src.cli decode [--config <file.pkl>] --hash $(& python -m src.cli encode --channels <int> --min <float> --max <float> [--config <file.pkl>] --value <float> 2>&1 | Select-Object -First 1).ToString().Trim()

import argparse
import sys
from src.encoder import Encoder
from src.decoder import Decoder

def cmd_encode(a: argparse.Namespace) -> int:
    enc = Encoder(a.channels, a.max, a.min, configPath=a.config)
    return(enc.encode(a.value))

def cmd_decode(a: argparse.Namespace) -> int:
    dec = Decoder(configPath=a.config)
    return(dec.decode(a.hash))

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hashcomb", description="HashComb CLI (minimal)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # encode
    e = sub.add_parser("encode", help="Encode a plaintext value")
    e.add_argument("--channels", type=int, required=True, help="Number of channels of the tree")
    e.add_argument("--min", type=float, required=True, help="Minimum value to encode")
    e.add_argument("--max", type=float, required=True, help="Maximum value to encode")
    e.add_argument("--config", type=str, default="configuration.pkl", help="Path to save the configuration file")
    e.add_argument("--value", type=float, required=True, help="Plaintext value to encode")
    e.set_defaults(func=cmd_encode)

    # decode
    d = sub.add_parser("decode", help="Decode a ciphertext value")
    d.add_argument("--config", type=str, default="configuration.pkl", help="Path to load the configuration file")
    d.add_argument("--hash", type=str, required=True, help="Ciphertext value to decode")
    d.set_defaults(func=cmd_decode)

    return p

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())

#i think that a wrapper for encodeArray and decodeArray is useless...
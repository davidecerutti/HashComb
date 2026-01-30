"""CLI wrapper for HashComb encode/decode operations."""

# Usage examples:
# python -m hashcomb.cli encode --channels <int> --min <float> --max <float> [--config <file.pkl>] --value <float>
# python -m hashcomb.cli decode [--config <file.pkl>] --hash <str>
# python -m hashcomb.cli encode --channels <int> --min <float> --max <float> --value <float> --mode path

import argparse
import sys
from ..encoders.encoder import Encoder
from ..encoders.randomized_encoder import RandomizedEncoder
from ..decoders.decoder import Decoder

def _build_encoder(a: argparse.Namespace):
    if a.randomized:
        if a.target_level is not None and a.p is None:
            p = RandomizedEncoder.compute_selection_probability(a.channels, a.target_level)
        else:
            p = a.p
        if p is None:
            p = 0.5
        return RandomizedEncoder(
            a.channels,
            a.max,
            a.min,
            delta=a.delta,
            selectionProbability=p,
            seed=a.seed,
            configPath=a.config,
            includeInternal=a.include_internal,
            salt=a.salt,
        )
    return Encoder(a.channels, a.max, a.min, configPath=a.config, includeInternal=a.include_internal, delta=a.delta, salt=a.salt)


def cmd_encode(a: argparse.Namespace) -> int:
    enc = _build_encoder(a)
    mode = a.mode
    if mode == "leaf":
        print(enc.encode(a.value))
        return 0
    if mode == "path":
        path = enc.encodePath(a.value)
        print(",".join(path))
        return 0
    if mode == "prefix":
        if a.prefix_length is None:
            raise SystemExit("--prefix-length is required when --mode prefix")
        prefix = enc.encodePrefix(a.value, a.prefix_length)
        print(",".join(prefix))
        return 0
    raise SystemExit("Invalid --mode")

def cmd_decode(a: argparse.Namespace) -> int:
    dec = Decoder(configPath=a.config)
    if a.path is not None:
        enc_path = [p for p in a.path.split(",") if p]
        print(dec.decodePath(enc_path))
        return 0
    if a.hash is None:
        raise SystemExit("Provide --hash or --path")
    print(dec.decode(a.hash))
    return 0

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
    e.add_argument("--delta", type=float, help="Range enlargement delta (min-delta, max+delta)")
    e.add_argument("--salt", type=str, help="Optional salt value for hashing (per round)")
    e.add_argument("--mode", choices=["leaf", "path", "prefix"], default="leaf", help="Encoding mode")
    e.add_argument("--prefix-length", type=int, help="Prefix length for --mode prefix")
    e.add_argument("--include-internal", action="store_true", help="Include internal nodes in the hash table")
    e.add_argument("--randomized", action="store_true", help="Use randomized encoder (paper mode)")
    e.add_argument("--p", type=float, help="Selection probability p (paper Eq. 14)")
    e.add_argument("--target-level", type=float, help="Target expected level l (computes p automatically)")
    e.add_argument("--seed", type=int, help="Random seed for randomized encoder")
    e.set_defaults(func=cmd_encode)

    # decode
    d = sub.add_parser("decode", help="Decode a ciphertext value")
    d.add_argument("--config", type=str, default="configuration.pkl", help="Path to load the configuration file")
    d.add_argument("--hash", type=str, help="Ciphertext value to decode (leaf)")
    d.add_argument("--path", type=str, help="Comma-separated hash path/prefix to decode")
    d.set_defaults(func=cmd_decode)

    return p

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())

#i think that a wrapper for encodeArray and decodeArray is useless...
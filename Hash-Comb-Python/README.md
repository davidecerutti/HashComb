# HashComb Python

[![PyPI](https://img.shields.io/pypi/v/hashcomb.svg)](https://pypi.org/project/hashcomb/)
[![Python Versions](https://img.shields.io/pypi/pyversions/hashcomb.svg)](https://pypi.org/project/hashcomb/)
[![License](https://img.shields.io/pypi/l/hashcomb.svg)](https://pypi.org/project/hashcomb/)
[![Homepage](https://img.shields.io/badge/homepage-github-blue)](https://github.com/davidecerutti/HashCombPython/tree/main/Hash-Comb-Python)

HashComb is a quantization-based hashing technique for privacy‑preserving distributed learning. It builds a balanced binary tree over a numeric range and maps values to hash tokens derived from tree nodes. Tokens can be used as compact, privacy‑preserving representations for aggregation, statistics, and clustering, as described in the HashComb paper.

> **Reference**: This implementation follows the HashComb paper (see the PDF in the repository root).

---

## Why HashComb

- **Quantization via tree**: values are mapped to bins defined by a balanced binary tree.
- **Hash tokens**: each tree node can be hashed into a compact token.
- **Three signatures**:
  - **Leaf hash**: single token for the leaf bin.
  - **Prefix multihash**: first $k$ tokens from the path.
  - **Full‑path multihash**: tokens from root to leaf.
- **Aggregation‑ready**: server can aggregate counts without seeing raw values.

---

## Installation

From PyPI:

```bash
pip install hashcomb
```

Optional dependencies for notebooks:

```bash
pip install "hashcomb[notebooks]"
```

From source:

```bash
pip install -e .
```

Optional dependencies for notebooks:

```bash
pip install -e .[notebooks]
```

---

## Quickstart

```python
from hashcomb import Encoder, Decoder

enc = Encoder(channels=4, maxValue=10.0, minValue=0.0, configPath="artifacts/config.pkl")
dec = Decoder(configPath="artifacts/config.pkl")

v = 3.7
leaf = enc.encode(v)
center = dec.decode(leaf)

leaf, center
```

---

## Core concepts (from the paper)

1. **Quantization tree**: splits the range into $2^L$ bins (leaf nodes), where $L$ is `channels`.
2. **Tokenization**: each node is hashed into a compact token (optionally salted).
3. **Encoding**: a value maps to a path of tokens (root→leaf).
4. **Decoding**: tokens map to bin centers for approximate reconstruction.

---

## Public API (classes + methods)

### `Encoder`
Deterministic encoder using a fixed tree.

**Constructor**
```python
from hashcomb import Encoder
enc = Encoder(channels=4, maxValue=10.0, minValue=0.0, configPath="artifacts/enc.pkl")
```

**Methods**
```python
leaf = enc.encode(3.7)                 # leaf token
path = enc.encodePath(3.7)             # full path tokens
prefix = enc.encodePrefix(3.7, 2)      # first k tokens

arr_leaf = enc.encodeArray([1.0, 2.0])
arr_path = enc.encodePathArray([1.0, 2.0])
arr_pref = enc.encodePrefixArray([1.0, 2.0], length=2)
```

**Factory methods**
```python
from hashcomb import PklIO
enc2 = Encoder.from_pkl("artifacts/enc.pkl")
enc3 = Encoder.from_config(PklIO.loadConfig("artifacts/enc.pkl"))
```

---

### `RandomizedEncoder`
Randomized encoder (paper mode) using the “last‑head in $L$ tosses” rule.

**Constructor**
```python
from hashcomb import RandomizedEncoder, RoundContext
ctx = RoundContext(salt="roundA", seed=123)
enc = RandomizedEncoder(
    channels=4,
    maxValue=10.0,
    minValue=0.0,
    selectionProbability=0.6,
    roundContext=ctx,
    configPath="artifacts/rand.pkl",
)
```

**Methods**
```python
leaf = enc.encode(3.7)
path = enc.encodePath(3.7)
prefix = enc.encodePrefix(3.7, 2)

arr_leaf = enc.encodeArray([1.0, 2.0])
arr_path = enc.encodePathArray([1.0, 2.0])
arr_pref = enc.encodePrefixArray([1.0, 2.0], length=2)
```

**Probability helpers**
```python
p = RandomizedEncoder.compute_selection_probability(channels=4, targetLevel=2.5)
exp = RandomizedEncoder.expected_level(channels=4, selectionProbability=p)
```

**Factory methods**
```python
enc2 = RandomizedEncoder.from_pkl("artifacts/rand.pkl")
```

---

### `Decoder`
Decode tokens to bin centers.

```python
from hashcomb import Decoder

dec = Decoder(configPath="artifacts/enc.pkl")
center = dec.decode(leaf)
center2 = dec.decodePath(prefix)

centers = dec.decodeArray([leaf])
centers2 = dec.decodePathArray([path])
```

---

### `Tree`
Balanced binary tree defining the quantization bins.

```python
from hashcomb import Tree

tr = Tree(channels=3, maxValue=10.0, minValue=0.0)
path_tokens = tr.getHValues(3.7, True)
rounded = Tree.round(1.2345, 2)
```

---

### `Node`
Tree node with interval and helpers.

```python
from hashcomb import Node

n = Node(0.0, 1.0, 0)
center = n.getCenter
is_leaf = n.isLeaf
as_str = str(n)
node_token = n.getValue(True)
node_path = n.getValue(0.2, True)
```

---

### `Hash`
Tokenization helpers.

```python
from hashcomb.core.hash import Hash

hmap = Hash.buildHashTable(tr, include_internal=True)
sha = Hash.sha3_256_int64("abc")
tok = Hash.hash_token("abc", "salt")
```

---

### `RoundContext`
Per‑round shared salt and RNG seed.

```python
from hashcomb import RoundContext

ctx = RoundContext.generate(salt_bytes=2, seed=7)
```

---

### `PklIO`
Read/write configs and pickles.

```python
from hashcomb import PklIO

PklIO.savePickle("artifacts/obj.pkl", {"a": 1})
obj = PklIO.loadPickle("artifacts/obj.pkl")

PklIO.saveConfig("artifacts/config.pkl", {"schema": "hashcomb.config.v1", "params": {}})
cfg = PklIO.loadConfig("artifacts/config.pkl")
```

---

### `CsvIO`
Encode/decode a CSV column with HashComb.

```python
from hashcomb import CsvIO

# input CSV has header with a "value" column
CsvIO.encodeCsv("data.csv", "data_encoded.csv", enc, valueCol="value", hashCol="hash")
CsvIO.decodeCsv("data_encoded.csv", "data_decoded.csv", dec, hashCol="hash", decodedValueCol="decoded_value")
```

---

### Add‑ons
Optional utilities (not required for core usage).

```python
from hashcomb.addons import aggregate_ciphertexts, serialize_path, deserialize_path

agg = aggregate_ciphertexts([("a", 1), ("a", 2), ("b", 5)])
ser = serialize_path(["x", "y"])
rest = deserialize_path(ser)
```

---

## CLI

Minimal CLI wrapper:

```bash
# encode leaf
python -m hashcomb.cli encode --channels 4 --min 0 --max 10 --value 3.7

# encode full path
python -m hashcomb.cli encode --channels 4 --min 0 --max 10 --value 3.7 --mode path

# encode prefix
python -m hashcomb.cli encode --channels 4 --min 0 --max 10 --value 3.7 --mode prefix --prefix-length 2

# decode leaf
python -m hashcomb.cli decode --config artifacts/enc.pkl --hash <token>

# decode path/prefix
python -m hashcomb.cli decode --config artifacts/enc.pkl --path token1,token2
```

---

## Notebooks

- **01_basics.ipynb**: API basics, signatures, randomization.
- **02_statistics.ipynb**: mean estimation and error.
- **03_clustering.ipynb**: clustering + K‑means demos.
- **04_mpc_addons.ipynb**: add‑on utilities.
- **hashcomb.ipynb**: this reference notebook (overview + API examples).

---

## License

MIT

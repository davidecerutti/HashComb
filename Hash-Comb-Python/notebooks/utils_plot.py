"""Shared plotting helpers for HashComb notebooks."""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except Exception:  # pragma: no cover - fallback when IPython is unavailable
    def display(obj) -> None:  # type: ignore[no-redef]
        print(obj)

from hashcomb.core.node import Node
from hashcomb.core.hash import Hash
from hashcomb.io.io import PklIO


def collect_leaves(root: Node) -> List[Node]:
    """Collect leaf nodes in ascending order."""
    leaves: List[Node] = []

    def rec(n: Node | None) -> None:
        if n is None:
            return
        if n.isLeaf:
            leaves.append(n)
        else:
            rec(n.left)
            rec(n.right)

    rec(root)
    leaves.sort(key=lambda n: n.min)
    return leaves


def plot_leaves_strip(leaves: Sequence[Node], inputs: Iterable[float] | None = None, title: str | None = None) -> None:
    """Plot leaf intervals on a single axis with optional input markers."""
    if title is None:
        title = f"Leaf intervals — {len(leaves)} leaves"

    fig, ax = plt.subplots(figsize=(10, 2.8))
    y = 0.0
    for n in leaves:
        ax.plot([n.min, n.max], [y, y], linewidth=4)
        ax.plot([n.getCenter], [y], marker="o")

    if inputs:
        for v in inputs:
            ax.plot([v], [y], marker="x", markersize=10)
            ax.annotate(f"{v:.3g}", xy=(v, y), xytext=(0, 10), textcoords="offset points", ha="center")

    ax.set_yticks([])
    ax.set_xlabel("value")
    ax.set_title(title)
    min_x = leaves[0].min
    max_x = leaves[-1].max
    if inputs:
        inputs = list(inputs)
        min_x = min(min_x, min(inputs))
        max_x = max(max_x, max(inputs))
    ax.set_xlim(min_x, max_x)
    fig.tight_layout()
    plt.show()


def path_for_value(root: Node, v: float) -> List[Node]:
    """Return the node path (root→leaf) for a value."""
    path: List[Node] = []
    n = root
    while n is not None:
        path.append(n)
        if n.isLeaf:
            break
        c = n.getCenter
        n = n.left if v < c else n.right
    return path


def nodes_by_level(root: Node) -> List[List[Node]]:
    """Collect nodes by level (BFS)."""
    levels: List[List[Node]] = []

    def rec(n: Node | None, depth: int) -> None:
        if n is None:
            return
        if depth == len(levels):
            levels.append([])
        levels[depth].append(n)
        if not n.isLeaf:
            rec(n.left, depth + 1)
            rec(n.right, depth + 1)

    rec(root, 0)
    return levels


def plot_tree_levels(root: Node, highlight_path: Sequence[Node] | None = None, title: str | None = None) -> None:
    """Plot a binary tree with optional highlighted path."""
    lvls = nodes_by_level(root)
    if title is None:
        title = f"Binary tree — depth={len(lvls) - 1}"

    coords: dict[Node, tuple[int, int]] = {}
    for d, nodes in enumerate(lvls):
        for i, n in enumerate(nodes):
            coords[n] = (i, -d)

    fig, ax = plt.subplots(figsize=(max(8, len(lvls[-1])), len(lvls) + 2))
    for n, (x, y) in coords.items():
        if n.left:
            ax.plot([x, coords[n.left][0]], [y, coords[n.left][1]], linewidth=1, alpha=0.5)
        if n.right:
            ax.plot([x, coords[n.right][0]], [y, coords[n.right][1]], linewidth=1, alpha=0.5)

    for n, (x, y) in coords.items():
        ax.plot(x, y, "o", ms=8, color="#666")
        ax.annotate(
            f"[{n.min:.3g}, {n.max:.3g}]",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    if highlight_path:
        xs = [coords[n][0] for n in highlight_path]
        ys = [coords[n][1] for n in highlight_path]
        ax.plot(xs, ys, "-o", color="C3", linewidth=2.5, ms=10)
        ax.plot(xs[-1], ys[-1], "o", ms=14, mfc="none", mec="C3", mew=2.5)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1, len(lvls[-1]))
    ax.invert_yaxis()
    fig.tight_layout()
    plt.show()


def visualize_config_pkl(config_path: str, show_table: bool = True) -> pd.DataFrame:
    """Load a config file and visualize the leaf intervals."""
    config = PklIO.loadConfig(config_path)
    hash_map = config.get("hashMap")
    if hash_map is None:
        tree = config.get("tree")
        if tree is None:
            raise ValueError("Config missing hashMap and tree")
        include_internal = bool(config.get("params", {}).get("includeInternal", False))
        salt = config.get("salt")
        hash_map = Hash.buildHashTable(tree, include_internal=include_internal, salt=salt)

    rows = []
    for h, n in hash_map.items():
        rows.append({
            "hash": h,
            "min": n.min,
            "center": n.getCenter,
            "max": n.max,
            "width": n.max - n.min,
        })
    df = pd.DataFrame(rows).sort_values("min").reset_index(drop=True)

    if show_table:
        display(df.head(20))

    fig, ax = plt.subplots(figsize=(10, 2.8))
    y = 0.0
    for _, r in df.iterrows():
        ax.plot([r["min"], r["max"]], [y, y], linewidth=4)
        ax.plot([r["center"]], [y], marker="o")
    ax.set_yticks([])
    ax.set_xlabel("value")
    ax.set_title(f"Leaf intervals — {len(df)} leaves")
    plt.show()

    return df


def plot_hash_distribution(
    client_values: Sequence[Sequence[float]],
    counts: dict[str, int],
    decoder,
    mean_plain: float,
    mean_hash_server: float,
    title: str | None = None,
) -> None:
    """Plot plaintext distribution vs hashed bin frequencies."""
    all_values = np.concatenate([np.asarray(v) for v in client_values])

    centers = np.array([decoder.decode(h) for h in counts.keys()])
    freqs = np.array(list(counts.values()), dtype=float)
    freqs = freqs / freqs.sum()

    if len(centers) > 1:
        width = np.abs(centers[1] - centers[0])
    else:
        width = 1.0

    plt.figure(figsize=(8, 4))
    plt.hist(
        all_values,
        bins=40,
        density=True,
        alpha=0.45,
        color="gray",
        edgecolor="white",
        label="original values",
        zorder=1,
    )
    plt.bar(
        centers,
        height=(freqs / width),
        width=width,
        alpha=0.35,
        color="royalblue",
        edgecolor="navy",
        linewidth=0.5,
        label="hash frequencies",
        zorder=2,
    )

    plt.axvline(mean_plain, color="black", linestyle="--", label=f"meanPlain={mean_plain:.3f}")
    plt.axvline(mean_hash_server, color="red", linestyle="-.", label=f"meanHash={mean_hash_server:.3f}")

    plt.xlabel("value")
    plt.ylabel("Density / norm. frequency")
    if title is None:
        title = "Original distribution vs quantized (HashComb)"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

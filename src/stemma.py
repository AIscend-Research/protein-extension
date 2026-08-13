"""Stemma construction and ancestral sequence reconstruction.

Self-contained ML ancestral reconstruction under an F81/Poisson protein model:
neighbour-joining on Poisson-corrected distances, midpoint rooting, then
Felsenstein pruning for marginal posteriors at any node.

Two properties matter for this project:

  * It is *standard* — the point of the paper is that ordinary ASR returns a
    confident chimera on contaminated data, so the reconstruction must be the
    ordinary kind, with per-site marginal posteriors and no joint reasoning.
  * `reconstruct(..., taxa=[...])` can be restricted to a subset of witnesses,
    which is how [repair] builds two internally coherent sub-ancestors from the
    two sub-histories instead of one mosaic.

`parse_iqtree_state` reads IQ-TREE 2's `--ancestral` output for real families,
so the same downstream code works on simulated and empirical data. (FastML's
web server is unreliable; IQ-TREE's `.state` file carries the same marginal
posteriors.)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(CANONICAL)}
PSEUDOCOUNT = 0.5


# ----------------------------------------------------------------- alignment


def read_fasta(path: str | Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(chunks)
            name = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if name is not None:
        seqs[name] = "".join(chunks)
    return seqs


def encode(seqs: Sequence[str]) -> np.ndarray:
    """(n_taxa, L) integer matrix; unknown residues become -1."""
    return np.array([[AA_TO_IDX.get(c, -1) for c in s] for s in seqs], dtype=np.int64)


def equilibrium(matrix: np.ndarray) -> np.ndarray:
    counts = np.full(20, PSEUDOCOUNT)
    valid = matrix[matrix >= 0]
    counts += np.bincount(valid, minlength=20)
    return counts / counts.sum()


def poisson_distance(matrix: np.ndarray) -> np.ndarray:
    """Poisson-corrected protein distance, the standard NJ input."""
    n = matrix.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            both = (matrix[i] >= 0) & (matrix[j] >= 0)
            if not both.any():
                dist = 1.0
            else:
                p = float(np.mean(matrix[i][both] != matrix[j][both]))
                p = min(p, 0.94)  # keep the log finite at saturation
                dist = -np.log(1.0 - p) if p > 0 else 0.0
            d[i, j] = d[j, i] = dist
    return d


# ---------------------------------------------------------------------- tree


@dataclass(eq=False)  # identity-based equality/hash: nodes are graph vertices
class TreeNode:
    name: str = ""
    length: float = 0.0
    parent: "TreeNode | None" = None
    children: list["TreeNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def add(self, child: "TreeNode", length: float) -> "TreeNode":
        child.parent = self
        child.length = max(length, 0.0)
        self.children.append(child)
        return child

    def leaves(self) -> list["TreeNode"]:
        return [self] if self.is_leaf else [l for c in self.children for l in c.leaves()]

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def newick(self) -> str:
        if self.is_leaf:
            return f"{self.name}:{self.length:.6f}"
        return f"({','.join(c.newick() for c in self.children)}){self.name}:{self.length:.6f}"


def _adjacency(tree: TreeNode) -> dict[TreeNode, list[tuple[TreeNode, float]]]:
    """Undirected view of the tree: NJ topologies carry no meaningful root."""
    adj: dict[TreeNode, list[tuple[TreeNode, float]]] = {}
    for node in tree.walk():
        adj.setdefault(node, [])
        for child in node.children:
            adj.setdefault(child, [])
            adj[node].append((child, child.length))
            adj[child].append((node, child.length))
    return adj


def _distances_from(start: TreeNode, adj) -> dict[TreeNode, tuple[float, TreeNode | None]]:
    """Path length and predecessor for every node, walking the undirected tree."""
    seen = {start: (0.0, None)}
    stack = [start]
    while stack:
        current = stack.pop()
        base = seen[current][0]
        for neighbour, length in adj[current]:
            if neighbour not in seen:
                seen[neighbour] = (base + length, current)
                stack.append(neighbour)
    return seen


def midpoint_root(tree: TreeNode) -> TreeNode:
    """Re-root at the midpoint of the longest leaf-to-leaf path.

    Neighbour-joining produces an unrooted tree; taking its last join as the root
    puts the "ancestor" at an arbitrary point inside one clade, which silently
    turns the reconstruction into that clade's ancestor and makes the two-way
    clade split meaningless. Midpoint rooting is the standard fix.
    """
    adj = _adjacency(tree)
    leaves = [n for n in adj if n.is_leaf or len(adj[n]) == 1]
    if len(leaves) < 2:
        return tree

    # Double sweep: the farthest leaf from any leaf is an endpoint of the diameter.
    first = _distances_from(leaves[0], adj)
    far_a = max(leaves, key=lambda n: first[n][0])
    dist_a = _distances_from(far_a, adj)
    far_b = max((n for n in leaves), key=lambda n: dist_a[n][0])
    diameter = dist_a[far_b][0]
    if diameter <= 0:
        return tree

    # Walk back from far_b toward far_a until half the diameter is covered.
    target = diameter / 2.0
    node = far_b
    while True:
        parent = dist_a[node][1]
        if parent is None:
            return tree
        remaining = dist_a[node][0] - target
        edge = dist_a[node][0] - dist_a[parent][0]
        if remaining <= edge:
            below, above = node, parent
            below_len = max(remaining, 0.0)
            above_len = max(edge - remaining, 0.0)
            break
        node = parent

    def build(current: TreeNode, came_from: TreeNode) -> TreeNode:
        clone = TreeNode(current.name)
        for neighbour, length in adj[current]:
            if neighbour is came_from:
                continue
            clone.add(build(neighbour, current), length)
        return clone

    root = TreeNode("ROOT")
    root.add(build(below, above), below_len)
    root.add(build(above, below), above_len)
    return root


def neighbour_joining(names: Sequence[str], dist: np.ndarray) -> TreeNode:
    """Saitou-Nei NJ, returning a tree rooted at the final joined node."""
    n = len(names)
    if n < 3:
        root = TreeNode("ROOT")
        for i, nm in enumerate(names):
            root.add(TreeNode(nm), dist[0, 1] / 2 if n == 2 else 0.0)
        return root

    nodes: list[TreeNode | None] = [TreeNode(nm) for nm in names]
    d = dist.astype(float).copy()
    active = list(range(n))
    counter = 0

    while len(active) > 2:
        m = len(active)
        sub = d[np.ix_(active, active)]
        totals = sub.sum(axis=1)
        q = (m - 2) * sub - totals[:, None] - totals[None, :]
        np.fill_diagonal(q, np.inf)
        flat = int(np.argmin(q))
        i_local, j_local = divmod(flat, m)
        i, j = active[i_local], active[j_local]

        d_ij = d[i, j]
        delta = (totals[i_local] - totals[j_local]) / (m - 2)
        li = max(0.5 * (d_ij + delta), 1e-6)
        lj = max(d_ij - li, 1e-6)

        counter += 1
        parent = TreeNode(f"N{counter}")
        parent.add(nodes[i], li)
        parent.add(nodes[j], lj)

        new_row = np.array([0.5 * (d[i, k] + d[j, k] - d_ij) for k in range(d.shape[0])])
        d = np.pad(d, ((0, 1), (0, 1)), constant_values=0.0)
        new_index = d.shape[0] - 1
        d[new_index, :-1] = new_row
        d[:-1, new_index] = new_row
        nodes.append(parent)
        active = [k for k in active if k not in (i, j)] + [new_index]

    i, j = active
    counter += 1
    root = TreeNode(f"N{counter}")
    half = max(d[i, j] / 2, 1e-6)
    root.add(nodes[i], half)
    root.add(nodes[j], half)
    return root


# ------------------------------------------------------- ancestral inference


@dataclass
class Reconstruction:
    """Marginal ASR at one node."""

    node: str
    sequence: str
    posteriors: np.ndarray  # (L, 20)
    taxa: list[str]
    tree_newick: str

    @property
    def max_posterior(self) -> np.ndarray:
        return self.posteriors.max(axis=1)

    def posterior_table(self) -> dict[int, tuple[str, float, dict[str, float]]]:
        """Same shape [apparatus] expects from a FastML table."""
        out = {}
        for i in range(self.posteriors.shape[0]):
            row = {aa: float(self.posteriors[i, k]) for k, aa in enumerate(CANONICAL)}
            best = max(row, key=row.get)
            out[i + 1] = (best, row[best], row)
        return out

    def write_posteriors(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Node", "Pos", *CANONICAL])
            for i in range(self.posteriors.shape[0]):
                writer.writerow([self.node, i + 1, *[f"{p:.6f}" for p in self.posteriors[i]]])
        return path


def _transition(length: float, pi: np.ndarray) -> np.ndarray:
    """F81 transition matrix for one branch."""
    beta = 1.0 / max(1.0 - float((pi ** 2).sum()), 1e-9)
    stay = np.exp(-beta * max(length, 0.0))
    return stay * np.eye(20) + (1.0 - stay) * pi[None, :]


def _partials(node: TreeNode, data: dict[str, np.ndarray], pi: np.ndarray, L: int) -> np.ndarray:
    """Felsenstein down-pass: (L, 20) conditional likelihoods below `node`."""
    if node.is_leaf:
        obs = data[node.name]
        out = np.zeros((L, 20))
        known = obs >= 0
        out[known, obs[known]] = 1.0
        out[~known] = 1.0
        return out
    total = np.ones((L, 20))
    for child in node.children:
        below = _partials(child, data, pi, L)
        total *= below @ _transition(child.length, pi).T
    return total


def _reroot(node: TreeNode) -> TreeNode:
    """Return a copy of the tree re-rooted at `node` (its parent becomes a child)."""
    def copy_away(current: TreeNode, came_from: TreeNode | None) -> TreeNode:
        clone = TreeNode(current.name)
        for child in current.children:
            if child is came_from:
                continue
            clone.add(copy_away(child, current), child.length)
        parent = current.parent
        if parent is not None and parent is not came_from:
            clone.add(copy_away(parent, current), current.length)
        return clone

    return copy_away(node, None)


def reconstruct(
    seqs: dict[str, str],
    *,
    taxa: Iterable[str] | None = None,
    node: str | None = None,
    tree: TreeNode | None = None,
) -> Reconstruction:
    """Marginal ML ancestral reconstruction at the root (or at a named node).

    `taxa` restricts the reconstruction to a subset of witnesses — the operation
    that turns one contaminated family into two clean sub-families.
    """
    names = list(taxa) if taxa is not None else list(seqs)
    missing = [n for n in names if n not in seqs]
    if missing:
        raise KeyError(f"unknown taxa: {missing}")
    if len(names) < 2:
        raise ValueError("need at least two witnesses")

    matrix = encode([seqs[n] for n in names])
    L = matrix.shape[1]
    pi = equilibrium(matrix)
    data = {n: matrix[i] for i, n in enumerate(names)}

    if tree is None:
        tree = midpoint_root(neighbour_joining(names, poisson_distance(matrix)))

    target = tree
    if node is not None:
        found = next((x for x in tree.walk() if x.name == node), None)
        if found is None:
            raise KeyError(f"node {node!r} not in tree")
        target = _reroot(found)

    partials = _partials(target, data, pi, L)
    post = partials * pi[None, :]
    post /= np.clip(post.sum(axis=1, keepdims=True), 1e-300, None)
    sequence = "".join(CANONICAL[k] for k in post.argmax(axis=1))
    return Reconstruction(
        node=node or "ROOT",
        sequence=sequence,
        posteriors=post,
        taxa=names,
        tree_newick=tree.newick() + ";",
    )


def split_clades(tree: TreeNode) -> tuple[list[str], list[str]]:
    """Leaf names on either side of the root split — the two candidate sub-histories."""
    if len(tree.children) < 2:
        raise ValueError("root has fewer than two children")
    if len(tree.children) == 2:
        left, right = tree.children
        return ([l.name for l in left.leaves()], [l.name for l in right.leaves()])
    groups = sorted(tree.children, key=lambda c: len(c.leaves()), reverse=True)
    first = [l.name for l in groups[0].leaves()]
    rest = [l.name for g in groups[1:] for l in g.leaves()]
    return first, rest


def parse_iqtree_state(path: str | Path, node: str | None = None) -> dict[str, np.ndarray]:
    """Read IQ-TREE 2's `.state` file into {node_name: (L, 20) posteriors}."""
    rows: dict[str, dict[int, np.ndarray]] = {}
    header: list[str] | None = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if header is None and parts[0].lower() in {"node", "#node"}:
            header = [p.strip() for p in parts]
            continue
        if header is None:
            continue
        name, site = parts[0], int(parts[1])
        if node is not None and name != node:
            continue
        probs = np.array([float(x) for x in parts[3:23]])
        rows.setdefault(name, {})[site] = probs
    out = {}
    for name, sites in rows.items():
        ordered = [sites[k] for k in sorted(sites)]
        out[name] = np.array(ordered)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Reconstruct an ancestor from a FASTA of witnesses.")
    ap.add_argument("fasta")
    ap.add_argument("--out", default=None, help="write per-site posteriors here")
    args = ap.parse_args()

    seqs = read_fasta(args.fasta)
    rec = reconstruct(seqs)
    left, right = split_clades(neighbour_joining(list(seqs), poisson_distance(encode(list(seqs.values())))))
    print(f"{len(seqs)} witnesses, {len(rec.sequence)} sites")
    print(f"clades: {len(left)} | {len(right)}")
    print(f"ancestor : {rec.sequence}")
    print(f"mean max-posterior : {rec.max_posterior.mean():.4f}")
    if args.out:
        print(f"wrote {rec.write_posteriors(args.out)}")

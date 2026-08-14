"""Simulate a homolog family on a fixed backbone, with and without contamination.

The evaluation needs families whose true history is known by construction. Every
witness is simulated on one shared backbone — a family of structural homologs is
assumed to keep its fold, which is what makes a single ancestral backbone
meaningful in the first place — so no structure prediction enters the loop and
the whole experiment runs on CPU.

Two evolution models, deliberately:

  f81        Site-independent substitution toward a per-site equilibrium profile
             taken from ProteinMPNN's structure-only distribution. No epistasis.
  selection  Gibbs sampling from ProteinMPNN's own joint distribution over the
             backbone: a site is resampled from p(s_i | all other sites). Lineages
             drift to internally coherent, *epistatically coupled* solutions.

The contrast is the point. The contamination detector in [conflict] is supposed
to work by sensing broken joint compatibility, so it should fire on `selection`
data and not on `f81` data. If it fires on both, it is detecting composition
differences rather than structural incoherence; if it fires on neither, MPNN's
receptive field is too local for the signal. Running both is the early
sensitivity check that decides whether the method is viable at all.

Contamination is injected the way a scribe does it: a witness copies one segment
of its sequence from a second exemplar in the other lineage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from mpnn_api import ALPHABET, CANONICAL, MPNNScorer, idx_to_seq, seq_to_idx


# ---------------------------------------------------------------------- trees


@dataclass
class Node:
    name: str
    length: float = 0.0
    parent: "Node | None" = None
    children: list["Node"] = field(default_factory=list)
    seq: str | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def add(self, child: "Node") -> "Node":
        child.parent = self
        self.children.append(child)
        return child

    def leaves(self) -> list["Node"]:
        if self.is_leaf:
            return [self]
        return [leaf for child in self.children for leaf in child.leaves()]

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    def newick(self) -> str:
        if self.is_leaf:
            return f"{self.name}:{self.length:.6f}"
        inner = ",".join(c.newick() for c in self.children)
        return f"({inner}){self.name}:{self.length:.6f}"


def two_clade_tree(
    n_per_clade: int = 6,
    *,
    stem: float = 2.0,
    leaf_depth: float = 0.4,
) -> Node:
    """Root splits into clades A and B; each clade fans out into `n_per_clade` leaves.

    Branch "lengths" are in units the evolvers interpret as substitution
    opportunity (F81 time, or number of Gibbs sweeps). `stem` must exceed
    `leaf_depth` by a good margin: the two lineages can only be told apart at
    sites where they actually differ, so a family whose between-clade divergence
    is no larger than its within-clade divergence has almost no diagnostic sites
    and nothing to detect.
    """
    root = Node("ROOT")
    for clade in ("A", "B"):
        stem_node = root.add(Node(f"{clade}anc", length=stem))
        for i in range(n_per_clade):
            stem_node.add(Node(f"{clade}{i}", length=leaf_depth))
    return root


# ------------------------------------------------------------------- evolvers


class F81Evolver:
    """Site-independent evolution toward per-site equilibrium profiles.

    P(t)_{ab} = e^{-mu t} delta_{ab} + (1 - e^{-mu t}) pi_b, applied per site.
    """

    name = "f81"

    def __init__(self, profiles: np.ndarray, mu: float = 1.0) -> None:
        self.pi = profiles / profiles.sum(axis=1, keepdims=True)  # (L, 20)
        self.mu = mu

    def evolve(self, seq: str, length: float, rng: np.random.Generator) -> str:
        idx = seq_to_idx(seq).copy()
        stay = np.exp(-self.mu * length)
        L = len(idx)
        resample = rng.random(L) > stay
        for i in np.flatnonzero(resample):
            idx[i] = rng.choice(20, p=self.pi[i])
        return idx_to_seq(idx)


class GibbsEvolver:
    """Evolution under structural constraint: Gibbs sampling of MPNN's joint model.

    One "sweep" resamples `sites_per_sweep` positions from
    p(s_i | all other current residues, backbone), tempered by `temperature`.
    Because every site is drawn conditional on the rest, compensatory pairs
    accumulate and lineages become internally coherent — the epistasis the
    contamination detector is meant to sense.
    """

    name = "selection"

    def __init__(
        self,
        scorer: MPNNScorer,
        *,
        temperature: float = 0.6,
        sweeps_per_unit: int = 8,
        sites_per_sweep: int = 6,
    ) -> None:
        self.scorer = scorer
        self.temperature = temperature
        self.sweeps_per_unit = sweeps_per_unit
        self.sites_per_sweep = sites_per_sweep

    def evolve(self, seq: str, length: float, rng: np.random.Generator) -> str:
        n_sweeps = int(round(self.sweeps_per_unit * length))
        current = seq
        for _ in range(n_sweeps):
            cond = self.scorer.full_context_log_probs(current)[:, :20]
            logits = cond / max(self.temperature, 1e-6)
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            sites = rng.choice(len(current), size=min(self.sites_per_sweep, len(current)), replace=False)
            chars = list(current)
            for i in sites:
                chars[i] = CANONICAL[rng.choice(20, p=probs[i])]
            current = "".join(chars)
        return current


# ---------------------------------------------------------------- simulation


@dataclass
class SimulatedFamily:
    """Witnesses plus the history that produced them."""

    tree: Node
    leaf_seqs: dict[str, str]
    true_root: str
    true_clade_ancestors: dict[str, str]
    contaminated: list[str]
    breakpoint: tuple[int, int] | None
    model: str
    donor_clade: str | None = None

    @property
    def names(self) -> list[str]:
        return list(self.leaf_seqs)

    def clade_of(self, name: str) -> str:
        return name[0]

    def true_source(self) -> np.ndarray | None:
        """Per-site indicator: 1 where the contaminated segment sits, else 0."""
        if self.breakpoint is None:
            return None
        L = len(self.true_root)
        mark = np.zeros(L, dtype=int)
        mark[self.breakpoint[0] : self.breakpoint[1]] = 1
        return mark

    def to_fasta(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for name, seq in self.leaf_seqs.items():
                handle.write(f">{name}\n{seq}\n")
        return path

    def metadata(self) -> dict:
        return {
            "model": self.model,
            "n_taxa": len(self.leaf_seqs),
            "length": len(self.true_root),
            "true_root": self.true_root,
            "true_clade_ancestors": self.true_clade_ancestors,
            "contaminated": self.contaminated,
            "breakpoint": list(self.breakpoint) if self.breakpoint else None,
            "donor_clade": self.donor_clade,
            "newick": self.tree.newick() + ";",
        }


def simulate_family(
    evolver: F81Evolver | GibbsEvolver,
    root_seq: str,
    *,
    n_per_clade: int = 6,
    stem: float = 2.0,
    leaf_depth: float = 0.4,
    n_contaminated: int = 3,
    breakpoint: tuple[int, int] | None = None,
    donor_clade: str = "B",
    seed: int = 0,
) -> SimulatedFamily:
    """Evolve a two-clade family, then optionally contaminate some witnesses.

    `breakpoint=None` produces clean, strictly tree-like data — the null.
    Otherwise `n_contaminated` witnesses of the *other* clade have the segment
    [start, stop) overwritten from a donor in `donor_clade`, which is exactly a
    scribe copying part of the text from a second exemplar.
    """
    rng = np.random.default_rng(seed)
    tree = two_clade_tree(n_per_clade, stem=stem, leaf_depth=leaf_depth)
    tree.seq = root_seq

    for node in tree.walk():
        if node.parent is not None:
            node.seq = evolver.evolve(node.parent.seq, node.length, rng)

    leaf_seqs = {leaf.name: leaf.seq for leaf in tree.leaves()}
    clade_ancestors = {
        node.name[0]: node.seq for node in tree.children if node.seq is not None
    }

    contaminated: list[str] = []
    if breakpoint is not None and n_contaminated > 0:
        recipient_clade = "A" if donor_clade == "B" else "B"
        donors = [n for n in leaf_seqs if n.startswith(donor_clade)]
        recipients = [n for n in leaf_seqs if n.startswith(recipient_clade)]
        chosen = rng.choice(recipients, size=min(n_contaminated, len(recipients)), replace=False)
        start, stop = breakpoint
        for name in chosen:
            donor = leaf_seqs[str(rng.choice(donors))]
            seq = leaf_seqs[name]
            leaf_seqs[name] = seq[:start] + donor[start:stop] + seq[stop:]
            contaminated.append(str(name))

    return SimulatedFamily(
        tree=tree,
        leaf_seqs=leaf_seqs,
        true_root=root_seq,
        true_clade_ancestors=clade_ancestors,
        contaminated=sorted(contaminated),
        breakpoint=breakpoint,
        model=evolver.name,
        donor_clade=donor_clade if breakpoint is not None else None,
    )


def contaminate(
    family: SimulatedFamily,
    breakpoint: tuple[int, int],
    *,
    n_contaminated: int = 3,
    donor_clade: str = "B",
    seed: int = 0,
) -> SimulatedFamily:
    """Copy one segment into some witnesses from a donor in the other clade.

    Splitting simulation from contamination lets every experiment reuse the same
    evolved lineages: the history is held fixed and only the contamination varies,
    which is both cheaper and a cleaner comparison than re-evolving each time.
    """
    rng = np.random.default_rng(seed)
    recipient_clade = "A" if donor_clade == "B" else "B"
    donors = [n for n in family.leaf_seqs if n.startswith(donor_clade)]
    recipients = [n for n in family.leaf_seqs if n.startswith(recipient_clade)]
    if not donors or not recipients:
        raise ValueError("need witnesses in both clades to contaminate")

    leaf_seqs = dict(family.leaf_seqs)
    contaminated: list[str] = []
    start, stop = breakpoint
    if n_contaminated > 0:
        chosen = rng.choice(recipients, size=min(n_contaminated, len(recipients)), replace=False)
        for name in chosen:
            donor = leaf_seqs[str(rng.choice(donors))]
            seq = leaf_seqs[name]
            leaf_seqs[name] = seq[:start] + donor[start:stop] + seq[stop:]
            contaminated.append(str(name))

    return SimulatedFamily(
        tree=family.tree,
        leaf_seqs=leaf_seqs,
        true_root=family.true_root,
        true_clade_ancestors=family.true_clade_ancestors,
        contaminated=sorted(contaminated),
        breakpoint=breakpoint if contaminated else None,
        model=family.model,
        donor_clade=donor_clade if contaminated else None,
    )


def contaminate_positions(
    family: SimulatedFamily,
    positions: np.ndarray | Sequence[int],
    *,
    n_contaminated: int = 3,
    donor_clade: str = "B",
    seed: int = 0,
) -> SimulatedFamily:
    """Copy an arbitrary *set* of positions into some witnesses from a donor.

    `contaminate` swaps a contiguous sequence interval, which is what ordinary
    recombination does to a gene. This takes a position set instead, so the
    swapped residues can be contiguous on the folded backbone while scattered
    across the sequence — gene conversion of a structural element, an exchanged
    binding face. The distinction matters because a scan that searches over
    sequence position cannot represent the second case at all.

    `breakpoint` is left None because there is no interval to record; the
    positions are carried on `contaminated_positions` instead.
    """
    rng = np.random.default_rng(seed)
    positions = np.asarray(sorted(int(p) for p in positions), dtype=int)
    recipient_clade = "A" if donor_clade == "B" else "B"
    donors = [n for n in family.leaf_seqs if n.startswith(donor_clade)]
    recipients = [n for n in family.leaf_seqs if n.startswith(recipient_clade)]
    if not donors or not recipients:
        raise ValueError("need witnesses in both clades to contaminate")

    leaf_seqs = dict(family.leaf_seqs)
    contaminated: list[str] = []
    if n_contaminated > 0:
        chosen = rng.choice(recipients, size=min(n_contaminated, len(recipients)), replace=False)
        for name in chosen:
            donor = leaf_seqs[str(rng.choice(donors))]
            seq = list(leaf_seqs[name])
            for p in positions:
                seq[p] = donor[p]
            leaf_seqs[name] = "".join(seq)
            contaminated.append(str(name))

    out = SimulatedFamily(
        tree=family.tree,
        leaf_seqs=leaf_seqs,
        true_root=family.true_root,
        true_clade_ancestors=family.true_clade_ancestors,
        contaminated=sorted(contaminated),
        breakpoint=None,
        model=family.model,
        donor_clade=donor_clade if contaminated else None,
    )
    out.contaminated_positions = positions.tolist() if contaminated else []
    return out


def make_evolver(
    kind: str, scorer: MPNNScorer, *, mu: float = 1.0, temperature: float = 0.6, sweeps_per_unit: int = 8
) -> F81Evolver | GibbsEvolver:
    if kind == "f81":
        profiles = np.exp(scorer.structure_only_log_probs()[:, :20])
        return F81Evolver(profiles, mu=mu)
    if kind == "selection":
        return GibbsEvolver(scorer, temperature=temperature, sweeps_per_unit=sweeps_per_unit)
    raise ValueError(f"unknown evolver {kind!r}; use f81|selection")


def save_family(family: SimulatedFamily, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    family.to_fasta(out_dir / "witnesses.fasta")
    (out_dir / "truth.json").write_text(json.dumps(family.metadata(), indent=2))
    return out_dir


if __name__ == "__main__":
    import argparse

    from mpnn_api import MPNN_DIR

    ap = argparse.ArgumentParser(description="Simulate one homolog family.")
    ap.add_argument("--pdb", default=str(MPNN_DIR / "inputs/PDB_monomers/pdbs/5L33.pdb"))
    ap.add_argument("--model", choices=["f81", "selection"], default="f81")
    ap.add_argument("--n-per-clade", type=int, default=6)
    ap.add_argument("--contaminate", type=int, default=3)
    ap.add_argument("--breakpoint", default=None, help="e.g. 30,60; omit for clean data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/interim/sim")
    args = ap.parse_args()

    scorer = MPNNScorer(args.pdb)
    evolver = make_evolver(args.model, scorer)
    bp = tuple(int(x) for x in args.breakpoint.split(",")) if args.breakpoint else None
    family = simulate_family(
        evolver, scorer.native_seq, n_per_clade=args.n_per_clade,
        n_contaminated=args.contaminate, breakpoint=bp, seed=args.seed,
    )
    out = save_family(family, args.out)
    ident = np.mean([
        np.mean([a == b for a, b in zip(seq, family.true_root)])
        for seq in family.leaf_seqs.values()
    ])
    print(f"{len(family.leaf_seqs)} witnesses, mean identity to true root {ident:.3f}")
    print(f"contaminated: {family.contaminated or 'none'} breakpoint={family.breakpoint}")
    print(f"wrote {out}")

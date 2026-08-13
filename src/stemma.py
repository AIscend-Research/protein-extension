"""Phylogeny and ancestral reconstruction — the "stemma codicum".

Builds a tree from the aligned witness family, runs FastML (or an equivalent) for
marginal ancestral state reconstruction, and exposes the per-node posterior
tables that `apparatus.load_fastml_posteriors` reads.

Planned surface:
    infer_tree(alignment_path, ...) -> Path
    run_fastml(alignment_path, tree_path, out_dir) -> FastMLResult
    node_posteriors(result, node) -> dict[int, dict[str, float]]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FastMLResult:
    """Where FastML left its outputs for one family."""

    out_dir: Path
    tree_path: Path | None = None
    marginal_prob_csv: Path | None = None
    ancestral_fasta: Path | None = None

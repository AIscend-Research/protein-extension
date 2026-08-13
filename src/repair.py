"""Don't just detect contamination — separate the sub-histories and rebuild.

A contaminated tradition has no single archetype. The constructive move is the
textual critic's: stop trying to reconstruct one text, identify the hyparchetypes,
and edit each separately. Here that means reconstructing an ancestor from each
sub-history alone and designing from each, instead of from a mosaic that never
folded in any organism.

The falsifiable prediction: the mosaic should score *worse* under ProteinMPNN's
joint model than either coherent reconstruction, and the gap should widen as
contamination increases. If it holds, some part of the reported stability of
ancestral reconstructions is being suppressed by undetected tree violation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mpnn_api import MPNNScorer
from stemma import (
    Reconstruction,
    encode,
    neighbour_joining,
    poisson_distance,
    reconstruct,
    split_clades,
)


@dataclass
class Scores:
    """Two views of how comfortable a sequence is on the backbone."""

    pseudo_log_likelihood: float
    autoregressive: float

    def as_dict(self) -> dict[str, float]:
        return {
            "pseudo_log_likelihood": round(self.pseudo_log_likelihood, 4),
            "autoregressive": round(self.autoregressive, 4),
        }


def score_sequence(scorer: MPNNScorer, seq: str, *, n_orders: int = 16) -> Scores:
    return Scores(
        pseudo_log_likelihood=scorer.pseudo_log_likelihood(seq),
        autoregressive=scorer.autoregressive_score(seq, n_orders=n_orders),
    )


@dataclass
class RepairResult:
    """The mosaic archetype next to the two coherent sub-ancestors."""

    mosaic: Reconstruction
    sub_a: Reconstruction
    sub_b: Reconstruction
    clade_a: list[str]
    clade_b: list[str]
    mosaic_scores: Scores
    sub_a_scores: Scores
    sub_b_scores: Scores

    @property
    def gap(self) -> float:
        """Best coherent reconstruction minus the mosaic (positive = repair helped)."""
        best = max(self.sub_a_scores.pseudo_log_likelihood, self.sub_b_scores.pseudo_log_likelihood)
        return best - self.mosaic_scores.pseudo_log_likelihood

    @property
    def mean_gap(self) -> float:
        mean_sub = 0.5 * (
            self.sub_a_scores.pseudo_log_likelihood + self.sub_b_scores.pseudo_log_likelihood
        )
        return mean_sub - self.mosaic_scores.pseudo_log_likelihood

    def summary(self) -> dict[str, object]:
        return {
            "clade_sizes": [len(self.clade_a), len(self.clade_b)],
            "mosaic": self.mosaic_scores.as_dict(),
            "sub_a": self.sub_a_scores.as_dict(),
            "sub_b": self.sub_b_scores.as_dict(),
            "gap_best": round(self.gap, 4),
            "gap_mean": round(self.mean_gap, 4),
            "mosaic_mean_posterior": round(float(self.mosaic.max_posterior.mean()), 4),
            "sub_a_mean_posterior": round(float(self.sub_a.max_posterior.mean()), 4),
            "sub_b_mean_posterior": round(float(self.sub_b.max_posterior.mean()), 4),
        }


def clade_split(seqs: dict[str, str]) -> tuple[list[str], list[str]]:
    """The two sub-histories implied by the stemma, without using any ground truth."""
    names = list(seqs)
    tree = neighbour_joining(names, poisson_distance(encode([seqs[n] for n in names])))
    return split_clades(tree)


def repair_family(
    scorer: MPNNScorer,
    seqs: dict[str, str],
    *,
    clade_a: Sequence[str] | None = None,
    clade_b: Sequence[str] | None = None,
    n_orders: int = 16,
) -> RepairResult:
    """Reconstruct the mosaic and both coherent sub-ancestors, and score all three."""
    if clade_a is None or clade_b is None:
        clade_a, clade_b = clade_split(seqs)
    clade_a, clade_b = list(clade_a), list(clade_b)
    if len(clade_a) < 2 or len(clade_b) < 2:
        raise ValueError(f"clades too small to reconstruct: {len(clade_a)} | {len(clade_b)}")

    mosaic = reconstruct(seqs)
    sub_a = reconstruct(seqs, taxa=clade_a)
    sub_b = reconstruct(seqs, taxa=clade_b)

    return RepairResult(
        mosaic=mosaic,
        sub_a=sub_a,
        sub_b=sub_b,
        clade_a=clade_a,
        clade_b=clade_b,
        mosaic_scores=score_sequence(scorer, mosaic.sequence, n_orders=n_orders),
        sub_a_scores=score_sequence(scorer, sub_a.sequence, n_orders=n_orders),
        sub_b_scores=score_sequence(scorer, sub_b.sequence, n_orders=n_orders),
    )


def excise(seqs: dict[str, str], drop: Sequence[str]) -> dict[str, str]:
    """Reconstruct from the tradition minus the witnesses judged contaminated."""
    return {k: v for k, v in seqs.items() if k not in set(drop)}


def identity(a: str, b: str) -> float:
    return float(np.mean([x == y for x, y in zip(a, b)]))

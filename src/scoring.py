"""How comfortable a sequence is on a backbone, and how well it was recovered.

One home for every sequence-level number the project reports, so [repair],
[pipeline] and the experiments all quote the same quantities.

Two scores, because they answer different questions:

  pseudo_log_likelihood  mean over sites of log p(s_i | *all* other sites). Every
                         residue is judged in full context, which is the score
                         that responds to joint incoherence — the one a mosaic
                         archetype should lose on.
  autoregressive         ProteinMPNN's own score: mean log-likelihood over random
                         decoding orders, where each residue sees only part of the
                         sequence. Comparable to published MPNN numbers.

Neither is a ΔΔG. They are structure-conditioned likelihoods used as a
stability *proxy*, and the paper should say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from mpnn_api import MPNNScorer, seq_to_idx


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


def recovery(seq: str, reference: str) -> float:
    """Fraction of positions matching a reference sequence."""
    if len(seq) != len(reference):
        raise ValueError(f"length mismatch: {len(seq)} vs {len(reference)}")
    return float(np.mean([a == b for a, b in zip(seq, reference)]))


def perplexity(scorer: MPNNScorer, seq: str) -> float:
    """exp(-mean log p(s_i | rest)) — lower is a better fit to the backbone."""
    return float(np.exp(-scorer.pseudo_log_likelihood(seq)))


def per_site_log_prob(scorer: MPNNScorer, seq: str) -> np.ndarray:
    """(L,) log p(s_i | all other sites) — the per-residue view of the above."""
    return scorer.site_log_prob(seq)


def recovery_by_label(
    seq: str, reference: str, labels: Sequence[str]
) -> dict[str, float]:
    """Recovery split by apparatus label (certain / probable / conjectural).

    The question this answers: is the reconstruction's accuracy actually
    concentrated where the apparatus claims confidence? If `certain` positions
    do not recover better than `conjectural` ones, the labels are decorative.
    """
    if not (len(seq) == len(reference) == len(labels)):
        raise ValueError("sequence, reference and labels must be the same length")
    out: dict[str, list[float]] = {}
    for residue, ref, label in zip(seq, reference, labels):
        out.setdefault(label, []).append(float(residue == ref))
    return {label: float(np.mean(hits)) for label, hits in sorted(out.items())}


def score_by_region(
    scorer: MPNNScorer, seq: str, segment: tuple[int, int]
) -> dict[str, float]:
    """Mean per-site log-probability inside vs outside a segment.

    Used to ask whether a flagged contaminated block is genuinely less at home on
    the backbone than the rest of the reconstruction.
    """
    site = scorer.site_log_prob(seq)
    inside = np.zeros(len(seq), dtype=bool)
    inside[segment[0] : segment[1]] = True
    return {
        "inside": round(float(site[inside].mean()), 4),
        "outside": round(float(site[~inside].mean()), 4),
        "difference": round(float(site[inside].mean() - site[~inside].mean()), 4),
    }


def compare(scorer: MPNNScorer, named: Mapping[str, str], *, n_orders: int = 16) -> dict[str, dict]:
    """Score several candidate sequences side by side."""
    return {name: score_sequence(scorer, seq, n_orders=n_orders).as_dict()
            for name, seq in named.items()}

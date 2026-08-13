"""End-to-end analysis of one family: stemma -> ASR -> conflict -> repair.

Everything the experiments need in one call, so simulated and empirical families
go through exactly the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from conflict import ConflictResult, auc, detect_contamination, jaccard
from mpnn_api import MPNNScorer
from repair import RepairResult, identity, repair_family


@dataclass
class FamilyAnalysis:
    conflict: ConflictResult
    repair: RepairResult
    metrics: dict[str, object]

    def summary(self) -> dict[str, object]:
        return {**self.conflict.summary(), **self.repair.summary(), **self.metrics}


def analyze(
    scorer: MPNNScorer,
    seqs: dict[str, str],
    *,
    truth: dict | None = None,
    clade_a: Sequence[str] | None = None,
    clade_b: Sequence[str] | None = None,
    n_perm: int = 200,
    alpha: float = 0.05,
    n_orders: int = 64,
    min_len: int = 3,
    seed: int = 0,
) -> FamilyAnalysis:
    """Reconstruct, probe for contamination, and repair — no ground truth required.

    `truth` (the `truth.json` written by [evolve]) only adds evaluation metrics;
    nothing in the detection path consults it.
    """
    rep = repair_family(scorer, seqs, clade_a=clade_a, clade_b=clade_b)
    con = detect_contamination(
        scorer,
        rep.mosaic.sequence,
        rep.sub_a.sequence,
        rep.sub_b.sequence,
        n_perm=n_perm,
        alpha=alpha,
        min_len=min_len,
        n_orders=n_orders,
        seed=seed,
    )

    metrics: dict[str, object] = {
        "n_taxa": len(seqs),
        "length": len(rep.mosaic.sequence),
        "mosaic_vs_sub_a_identity": round(identity(rep.mosaic.sequence, rep.sub_a.sequence), 4),
        "mosaic_vs_sub_b_identity": round(identity(rep.mosaic.sequence, rep.sub_b.sequence), 4),
    }

    if truth is not None:
        true_bp = tuple(truth["breakpoint"]) if truth.get("breakpoint") else None
        root = truth["true_root"]
        L = len(root)
        labels = np.zeros(L, dtype=bool)
        if true_bp:
            labels[true_bp[0] : true_bp[1]] = True

        # Only sites where the two sub-ancestors differ carry ancestry information,
        # so site-level accuracy is measured there; including the rest would score
        # the detector on positions that are uninformative by construction.
        diag = con.diff_sites if len(con.diff_sites) else np.arange(L)
        oriented = con.delta
        if con.segment is not None:
            inside = np.zeros(L, dtype=bool)
            inside[con.segment[0] : con.segment[1]] = True
            if oriented[inside].mean() > oriented[~inside].mean():
                oriented = -oriented
        metrics.update(
            {
                "true_breakpoint": list(true_bp) if true_bp else None,
                "contaminated_witnesses": truth.get("contaminated", []),
                "evolution_model": truth.get("model"),
                "asr_accuracy_vs_true_root": round(identity(rep.mosaic.sequence, root), 4),
                "asr_mean_max_posterior": round(float(rep.mosaic.max_posterior.mean()), 4),
                "segment_jaccard": round(jaccard(con.segment, true_bp), 4),
                "n_diagnostic": int(len(con.diff_sites)),
                "site_auc": (round(auc(-oriented[diag], labels[diag]), 4) if true_bp else None),
                "instability_auc": (
                    round(auc(con.instability[diag], labels[diag]), 4) if true_bp else None
                ),
                "false_positive": bool(con.detected and true_bp is None),
            }
        )

    return FamilyAnalysis(conflict=con, repair=rep, metrics=metrics)

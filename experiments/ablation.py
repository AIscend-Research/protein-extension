"""Ablate the instrument: does the structural model add anything?

The claim under test is that contamination is visible to a *joint structural*
model. The first thing a reader should demand is the cheap alternative: at each
diagnostic site, simply ask which sub-ancestor the mosaic matches.

    identity_i = [a_i == subA_i] - [a_i == subB_i]

That is a pure sequence statistic. It uses no structure, no backbone, no neural
network — nothing but string comparison — and it is exactly the signal classical
recombination detectors look for. Run it through the *same* segment scan and the
same permutation test as the MPNN conflict score and the comparison is direct.

Three outcomes, all informative:

  identity wins or ties   the structural claim is unsupported; MPNN is an
                          expensive way to compute string equality
  MPNN wins               the joint model sees something identity cannot, which
                          is the paper's thesis
  both fail               the family is below the diagnostic-site floor

A second ablation is included for the case where identity is uninformative by
construction: `scrambled` keeps ProteinMPNN but permutes the backbone's residue
coordinates, destroying the structural neighbourhood while preserving the model,
the alphabet and the marginal composition. Signal that survives scrambling is not
structural signal.

    python experiments/ablation.py --model selection --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from conflict import (  # noqa: E402
    auc, detect_contamination, jaccard, oriented_delta, scan_segment,
    permutation_test, smooth,
)
from evolve import contaminate  # noqa: E402
from mpnn_api import MPNN_DIR, MPNNScorer  # noqa: E402
from repair import repair_family  # noqa: E402
from sweeps import _rebuild, clean_family  # noqa: E402

DEFAULT_PDB = MPNN_DIR / "inputs/PDB_monomers/pdbs/5L33.pdb"
RESULTS = REPO_ROOT / "experiments" / "results"


def identity_delta(ancestor: str, context_a: str, context_b: str) -> np.ndarray:
    """+1 where the mosaic matches sub-ancestor A, -1 where it matches B.

    The sequence-only stand-in for the context-swap score, with the same sign
    convention so it can go through the identical downstream machinery.
    """
    a = np.array(list(ancestor))
    return (a == np.array(list(context_a))).astype(float) - (
        a == np.array(list(context_b))).astype(float)


def scan_and_test(delta: np.ndarray, diff_sites: np.ndarray, *, n_perm: int, alpha: float,
                  min_len: int = 3, window: int = 5, seed: int = 0) -> dict:
    """The segment scan and permutation test from [conflict], on any per-site score.

    Factored out so the ablated score is judged by exactly the same procedure as
    the real one — otherwise a difference in outcome could be a difference in
    test rather than in signal.
    """
    if len(diff_sites) < max(12, 2 * min_len):
        return {"detected": False, "p_value": 1.0, "segment": None, "best_segment": None}
    signs = np.sign(delta[diff_sites])
    (start, stop), statistic = scan_segment(smooth(signs, window), min_len=min_len)
    segment = (int(diff_sites[start]), int(diff_sites[stop - 1]) + 1)
    p_value, _, _ = permutation_test(signs, n_perm=n_perm, alpha=alpha, seed=seed,
                                     min_len=min_len, window=window)
    return {"detected": bool(p_value <= alpha), "p_value": float(p_value),
            "segment": segment if p_value <= alpha else None, "best_segment": segment,
            "statistic": float(statistic)}


class _Oriented:
    """Minimal duck-type so `oriented_delta` can score an ablated statistic."""

    def __init__(self, delta, diff_sites, best_segment, segment):
        self.delta, self.diff_sites = delta, diff_sites
        self.best_segment, self.segment = best_segment, segment


def evaluate_all(scorer, scrambled, leaf_seqs, truth, *, n_perm, n_orders, alpha, seed) -> dict:
    """Run the real probe and both ablations on one contaminated family."""
    rep = repair_family(scorer, leaf_seqs)
    mosaic, sub_a, sub_b = rep.mosaic.sequence, rep.sub_a.sequence, rep.sub_b.sequence
    diff_sites = np.flatnonzero(np.array(list(sub_a)) != np.array(list(sub_b)))

    bp = tuple(truth["breakpoint"]) if truth.get("breakpoint") else None
    labels = np.zeros(len(mosaic), dtype=bool)
    if bp:
        labels[bp[0]:bp[1]] = True
    diag = diff_sites if len(diff_sites) else np.arange(len(mosaic))

    arms: dict[str, dict] = {}

    # 1. the real thing
    con = detect_contamination(scorer, mosaic, sub_a, sub_b, n_perm=n_perm,
                               alpha=alpha, min_len=3, n_orders=n_orders, seed=seed)
    arms["mpnn"] = {
        "detected": bool(con.detected), "p_value": float(con.p_value),
        "segment": list(con.segment) if con.segment else None,
        "site_auc": float(auc(oriented_delta(con)[diag], labels[diag])) if bp else None,
        "segment_jaccard": float(jaccard(con.segment, bp)) if bp else None,
    }

    # 2. sequence identity only — no structure, no network
    d_ident = identity_delta(mosaic, sub_a, sub_b)
    res = scan_and_test(d_ident, diff_sites, n_perm=n_perm, alpha=alpha, seed=seed)
    stub = _Oriented(d_ident, diff_sites, res["best_segment"], res["segment"])
    arms["identity"] = {
        **{k: res[k] for k in ("detected", "p_value")},
        "segment": list(res["segment"]) if res["segment"] else None,
        "site_auc": float(auc(oriented_delta(stub)[diag], labels[diag])) if bp else None,
        "segment_jaccard": float(jaccard(res["segment"], bp)) if bp else None,
    }

    # 3. same network, structural neighbourhood destroyed
    con_s = detect_contamination(scrambled, mosaic, sub_a, sub_b, n_perm=n_perm,
                                 alpha=alpha, min_len=3, n_orders=n_orders, seed=seed)
    arms["scrambled"] = {
        "detected": bool(con_s.detected), "p_value": float(con_s.p_value),
        "segment": list(con_s.segment) if con_s.segment else None,
        "site_auc": float(auc(oriented_delta(con_s)[diag], labels[diag])) if bp else None,
        "segment_jaccard": float(jaccard(con_s.segment, bp)) if bp else None,
    }

    return {"n_diagnostic": int(len(diff_sites)),
            "n_diagnostic_in_segment": int(labels[diag].sum()) if bp else 0,
            "true_breakpoint": list(bp) if bp else None, "arms": arms}


def scrambled_scorer(pdb: str, device: str, seed: int) -> MPNNScorer:
    """A scorer whose backbone coordinates have been permuted across residues.

    Same model, same sequence alphabet, same composition — but residue i's
    structural neighbourhood is now some other residue's. Any conflict signal
    that survives this is not coming from structure.
    """
    scorer = MPNNScorer(pdb, device=device)
    rng = np.random.default_rng(seed)
    order = rng.permutation(scorer.L)
    backbone = scorer.backbone
    backbone.X = backbone.X[:, order, :, :].contiguous()
    return scorer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="selection", choices=["f81", "selection"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pdb", default=str(DEFAULT_PDB))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-per-clade", type=int, default=6)
    ap.add_argument("--contaminated", type=int, default=3)
    ap.add_argument("--widths", type=int, nargs="+", default=[30, 50])
    ap.add_argument("--start", type=int, default=55)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-orders", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    scorer = MPNNScorer(args.pdb, device=args.device)
    scrambled = scrambled_scorer(args.pdb, args.device, seed=1234)
    print(f"device={scorer.device} L={scorer.L}", flush=True)

    rows = []
    for seed in range(args.seeds):
        stored = clean_family(scorer, args.model, seed, n_per_clade=args.n_per_clade)
        fam = _rebuild(stored)
        for width in args.widths:
            stop = min(args.start + width, len(stored["true_root"]))
            cont = contaminate(fam, (args.start, stop),
                               n_contaminated=args.contaminated, seed=seed)
            t0 = time.time()
            out = evaluate_all(scorer, scrambled, cont.leaf_seqs, cont.metadata(),
                               n_perm=args.n_perm, n_orders=args.n_orders,
                               alpha=args.alpha, seed=seed)
            row = {"model": args.model, "seed": seed, "width": width,
                   "seconds": round(time.time() - t0, 1), **out}
            rows.append(row)
            line = f"  [{args.model} s{seed}] w={width:3} ndiag={out['n_diagnostic_in_segment']:3}"
            for arm in ("mpnn", "identity", "scrambled"):
                a = out["arms"][arm]
                j = a["segment_jaccard"]
                line += (f" | {arm}: p={a['p_value']:.3f} "
                         f"J={(j if j is not None else float('nan')):.2f} "
                         f"AUC={(a['site_auc'] if a['site_auc'] is not None else float('nan')):.2f}")
            print(line, flush=True)

    out_path = Path(args.out) / f"ablation_{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {out_path}")

    print("\n" + "=" * 70)
    for arm in ("mpnn", "identity", "scrambled"):
        vals = [r["arms"][arm] for r in rows]
        jac = [v["segment_jaccard"] for v in vals if v["segment_jaccard"] is not None]
        aucs = [v["site_auc"] for v in vals if v["site_auc"] is not None and not np.isnan(v["site_auc"])]
        fired = [v for v in vals if v["detected"]]
        print(f"  {arm:<10} fired {len(fired)}/{len(vals)}   "
              f"mean Jaccard {np.mean(jac):.3f}   mean site AUC {np.mean(aucs):.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

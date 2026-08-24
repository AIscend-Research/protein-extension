"""
Donor-distance dose-response: near sibling vs deep outgroup.

RESULTS.md is explicit that the floor is established but the dose-response is
not -- "Nothing fires below ~20 diagnostic sites ... The floor is established;
the dose-response is not." A single detection is easy to explain away; a
monotone curve of detection rate against how foreign the donor is, is not.

The existing `divergence` sweep varies `stem` (whole between-clade separation),
which conflates two things: how foreign the donor is AND how separable the
detector's two contexts are. This isolates the first. It holds `stem` fixed --
so clade A and clade B are equally distinguishable in every condition -- and
varies only `donor_depth`: the branch length of a SEPARATE donor lineage grown
from the root, whose segment is spliced into clade-A recipients. Small depth =
a near-sibling donor whose block barely differs; large depth = a deep outgroup
whose block is conspicuously foreign.

Read the curve as detection-rate (with a Wilson CI) and mean segment-Jaccard
against `donor_depth`, and against the diagnostic-site count the depth induces --
the count is the mechanism, the depth is the knob.

    python experiments/donor_distance.py --model selection --seeds 12 \
        --donor-depths 0.25 0.5 1.0 2.0 4.0 8.0

Interpretation caveat, stated up front: the detector's context_b is clade B, not
the true (outgroup) donor, so it senses "this block does not fit the clade-A
context it sits in," not "this block belongs to B." Detection rate and Jaccard
are therefore the primary readouts; the oriented site-AUC is reported but is
noisier here than in the sibling-donor setup and should be read as secondary.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from evolve import make_evolver  # noqa: E402
from mpnn_api import MPNN_DIR, MPNNScorer  # noqa: E402
from sweeps import clean_family, evaluate  # noqa: E402

DEFAULT_PDB = MPNN_DIR / "inputs/PDB_monomers/pdbs/5L33.pdb"
RESULTS = REPO_ROOT / "experiments" / "results"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a detection rate -- honest for small n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def evolve_donor(scorer, model: str, root_seq: str, depth: float, *, seed: int,
                 temperature=0.5, mu=1.0, sweeps=8) -> str:
    """Grow ONE donor lineage from the root to a controlled distance.

    Cheap relative to a family: it is a single branch, not a whole tree. Under
    `selection` it is a handful of Gibbs sweeps; under `f81` it is instant.
    """
    evolver = make_evolver(model, scorer, mu=mu, temperature=temperature, sweeps_per_unit=sweeps)
    rng = np.random.default_rng(10_000 + seed)
    return evolver.evolve(root_seq, depth, rng)


def splice_from_donor(leaf_seqs: dict[str, str], donor_seq: str, bp: tuple[int, int],
                      *, n_contaminated: int, recipient_clade: str = "A",
                      seed: int) -> dict[str, str]:
    """Overwrite [start:stop) of some recipient-clade witnesses from the donor.

    Mirrors evolve.contaminate, but the donor is an external outgroup sequence
    rather than a witness from the sibling clade.
    """
    rng = np.random.default_rng(seed)
    start, stop = bp
    recipients = [n for n in leaf_seqs if n.startswith(recipient_clade)]
    chosen = rng.choice(recipients, size=min(n_contaminated, len(recipients)), replace=False)
    out = dict(leaf_seqs)
    for name in chosen:
        s = out[name]
        out[name] = s[:start] + donor_seq[start:stop] + s[stop:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="selection", choices=["f81", "selection"])
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--pdb", default=str(DEFAULT_PDB))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--donor-depths", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
                    help="branch length of the donor lineage: the near-sibling -> outgroup axis")
    ap.add_argument("--stem", type=float, default=2.0,
                    help="HELD FIXED, so context separability is constant across depths")
    ap.add_argument("--n-per-clade", type=int, default=6)
    ap.add_argument("--contaminated", type=int, default=3)
    ap.add_argument("--start", type=int, default=55)
    ap.add_argument("--width", type=int, default=30)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-orders", type=int, default=32)
    ap.add_argument("--out", default=str(RESULTS / "donor_distance.json"))
    args = ap.parse_args()

    scorer = MPNNScorer(args.pdb, device=args.device)
    rows: list[dict] = []

    for seed in range(args.seeds):
        stored = clean_family(scorer, args.model, seed,
                              n_per_clade=args.n_per_clade, stem=args.stem)
        root = stored["true_root"]
        stop = min(args.start + args.width, len(root))
        bp = (args.start, stop)
        for depth in args.donor_depths:
            donor = evolve_donor(scorer, args.model, root, depth, seed=seed)
            leaf_seqs = splice_from_donor(stored["leaf_seqs"], donor, bp,
                                          n_contaminated=args.contaminated,
                                          recipient_clade="A", seed=seed)
            # reuse the shipped evaluate by handing it the true block as 'truth'
            t0 = time.time()
            res = evaluate(scorer, leaf_seqs, {"breakpoint": [bp[0], bp[1]]},
                           n_perm=args.n_perm, n_orders=args.n_orders, seed=seed)
            row = {"model": args.model, "seed": seed, "donor_depth": depth,
                   "stem": args.stem, "seconds": round(time.time() - t0, 1), **res}
            rows.append(row)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(rows, indent=2, default=str))
            print(f"  [s{seed} depth={depth:>4}] det={res['detected']!s:5} "
                  f"p={res['p_value']:.3f} J={res.get('segment_jaccard')} "
                  f"ndiag={res['n_diagnostic']} (in-block {res['n_diagnostic_in_segment']})",
                  flush=True)

    # --- the dose-response curve ---
    print("\n=== detection rate vs donor distance (stem held fixed) ===")
    print(f"{'depth':>7} {'det_rate':>9} {'95% CI':>15} {'mean_J':>7} {'mean_ndiag':>11}")
    curve = []
    for depth in args.donor_depths:
        sub = [r for r in rows if r["donor_depth"] == depth]
        k = sum(r["detected"] for r in sub)
        lo, hi = wilson_ci(k, len(sub))
        mj = float(np.mean([r.get("segment_jaccard") or 0 for r in sub]))
        nd = float(np.mean([r["n_diagnostic"] for r in sub]))
        curve.append({"donor_depth": depth, "n": len(sub), "detected": k,
                      "detection_rate": round(k / len(sub), 3),
                      "ci95": [round(lo, 3), round(hi, 3)],
                      "mean_jaccard": round(mj, 3), "mean_n_diagnostic": round(nd, 1)})
        print(f"{depth:>7} {k/len(sub):>9.2f} {f'[{lo:.2f},{hi:.2f}]':>15} {mj:>7.2f} {nd:>11.1f}")

    Path(args.out).write_text(json.dumps({"rows": rows, "curve": curve}, indent=2, default=str))
    print(f"\nwrote {args.out}")
    print("A monotone rise here is the dose-response the README says is missing; a "
          "flat curve says depth doesn't matter and detection is set by something else.")


if __name__ == "__main__":
    main()

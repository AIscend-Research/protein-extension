"""
Multiple breakpoints and gene conversion -- not just one clean swap.

The shipped `contaminate` splices a single contiguous block (two breakpoints).
Real contamination is messier: several independent transfers, or gene
conversion of a scattered set of sites. Two questions:

  1. Does fragmenting the SAME amount of contamination into more blocks make it
     harder to detect? (n_blocks = 1, 2, 3 at equal total width.)
  2. Does gene conversion -- a set of sites contiguous on the fold but scattered
     in sequence -- survive the sequence-space scan at all?

Metric. `detect_contamination` returns one best window, so segment-Jaccard
against the true block is the wrong yardstick once there are several blocks.
The honest readout is a per-site AUC over diagnostic sites, with the label
"this site sits in ANY contaminated block/position." Because every block here
comes from the same donor clade, all contaminated sites lean the same way, so
the detector's own `oriented_delta` orientation is valid across all of them --
no per-block orientation needed. Detection rate (the permutation test) is
reported alongside, but per-site AUC is the primary number for fragmented
contamination.

    python experiments/multi_breakpoint.py --model selection --seeds 8 \
        --total-width 45 --n-blocks 1 2 3

`contaminate_positions` (already in evolve.py) provides the gene-conversion
injection; this script adds the multi-block injection and the multi-label
scoring around the shipped detector, and touches nothing in conflict.py.
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

from conflict import auc, detect_contamination, oriented_delta  # noqa: E402
from evolve import contaminate_positions  # noqa: E402
from mpnn_api import MPNN_DIR, MPNNScorer  # noqa: E402
from repair import repair_family  # noqa: E402
from sweeps import _rebuild, clean_family  # noqa: E402

DEFAULT_PDB = MPNN_DIR / "inputs/PDB_monomers/pdbs/5L33.pdb"
RESULTS = REPO_ROOT / "experiments" / "results"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def even_blocks(start: int, total_width: int, n_blocks: int, gap: int, L: int) -> list[tuple[int, int]]:
    """Split `total_width` contaminated residues into `n_blocks` disjoint blocks,
    separated by `gap`, so total contaminated amount is held constant and only
    fragmentation changes."""
    per = total_width // n_blocks
    blocks, pos = [], start
    for _ in range(n_blocks):
        stop = min(pos + per, L)
        blocks.append((pos, stop))
        pos = stop + gap
        if pos >= L:
            break
    return blocks


def contaminate_blocks(fam, blocks: list[tuple[int, int]], *, n_contaminated: int,
                       donor_clade: str = "B", seed: int) -> tuple[dict[str, str], np.ndarray]:
    """Splice several disjoint donor blocks into the SAME recipient witnesses.

    Recipients and donor are chosen once and reused across blocks, so the result
    is one coherent multi-transfer contamination rather than several unrelated
    ones. Returns (leaf_seqs, contaminated_mask over positions).
    """
    rng = np.random.default_rng(seed)
    recipient_clade = "A" if donor_clade == "B" else "B"
    donors = [n for n in fam.leaf_seqs if n.startswith(donor_clade)]
    recipients = [n for n in fam.leaf_seqs if n.startswith(recipient_clade)]
    chosen = rng.choice(recipients, size=min(n_contaminated, len(recipients)), replace=False)
    leaf_seqs = dict(fam.leaf_seqs)
    L = len(next(iter(leaf_seqs.values())))
    mask = np.zeros(L, dtype=bool)
    for name in chosen:
        donor = leaf_seqs[str(rng.choice(donors))]
        s = list(leaf_seqs[name])
        for (start, stop) in blocks:
            s[start:stop] = list(donor[start:stop])
            mask[start:stop] = True
        leaf_seqs[name] = "".join(s)
    return leaf_seqs, mask


def evaluate_multi(scorer, leaf_seqs: dict[str, str], contaminated_mask: np.ndarray,
                   *, n_perm: int, n_orders: int, seed: int) -> dict:
    """Run the shipped detector; score it against a multi-block label mask."""
    rep = repair_family(scorer, leaf_seqs)
    con = detect_contamination(scorer, rep.mosaic.sequence, rep.sub_a.sequence,
                               rep.sub_b.sequence, n_perm=n_perm, n_orders=n_orders,
                               min_len=3, seed=seed)
    L = len(rep.mosaic.sequence)
    diag = con.diff_sites if len(con.diff_sites) else np.arange(L)
    labels = contaminated_mask[diag]
    scores = oriented_delta(con)[diag]  # same orientation the shipped sweeps use
    site_auc = auc(scores, labels) if labels.any() and not labels.all() else None
    return {
        "detected": bool(con.detected),
        "p_value": round(con.p_value, 5),
        "found_segment": list(con.best_segment) if con.best_segment else None,
        "n_diagnostic": int(len(con.diff_sites)),
        "n_diagnostic_contaminated": int(labels.sum()),
        "site_auc": round(site_auc, 4) if site_auc is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="selection", choices=["f81", "selection"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--pdb", default=str(DEFAULT_PDB))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-blocks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--total-width", type=int, default=45,
                    help="total contaminated residues, held constant across n_blocks")
    ap.add_argument("--gap", type=int, default=8, help="clean residues between blocks")
    ap.add_argument("--start", type=int, default=40)
    ap.add_argument("--gene-conversion", action="store_true",
                    help="also run a scattered (3D-contiguous, sequence-scattered) tract")
    ap.add_argument("--gc-stride", type=int, default=3, help="scatter step for gene conversion")
    ap.add_argument("--n-per-clade", type=int, default=6)
    ap.add_argument("--contaminated", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-orders", type=int, default=32)
    ap.add_argument("--out", default=str(RESULTS / "multi_breakpoint.json"))
    args = ap.parse_args()

    scorer = MPNNScorer(args.pdb, device=args.device)
    rows: list[dict] = []

    conditions = [("blocks", nb) for nb in args.n_blocks]
    if args.gene_conversion:
        conditions.append(("gene_conversion", None))

    for seed in range(args.seeds):
        stored = clean_family(scorer, args.model, seed, n_per_clade=args.n_per_clade)
        fam = _rebuild(stored)
        L = len(stored["true_root"])

        for kind, nb in conditions:
            if kind == "blocks":
                blocks = even_blocks(args.start, args.total_width, nb, args.gap, L)
                leaf_seqs, mask = contaminate_blocks(
                    fam, blocks, n_contaminated=args.contaminated, seed=seed)
                label = f"blocks={nb}"
            else:
                positions = np.arange(args.start, min(args.start + args.total_width * args.gc_stride, L),
                                      args.gc_stride)
                cont = contaminate_positions(fam, positions,
                                             n_contaminated=args.contaminated, seed=seed)
                leaf_seqs = cont.leaf_seqs
                mask = np.zeros(L, dtype=bool)
                mask[positions] = True
                label = "gene_conversion"

            t0 = time.time()
            res = evaluate_multi(scorer, leaf_seqs, mask,
                                 n_perm=args.n_perm, n_orders=args.n_orders, seed=seed)
            row = {"model": args.model, "seed": seed, "condition": label,
                   "n_blocks": nb, "total_width": int(mask.sum()),
                   "seconds": round(time.time() - t0, 1), **res}
            rows.append(row)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(rows, indent=2, default=str))
            print(f"  [s{seed} {label:>15}] det={res['detected']!s:5} p={res['p_value']:.3f} "
                  f"AUC={res['site_auc']} ndiag={res['n_diagnostic']} "
                  f"(contam {res['n_diagnostic_contaminated']})", flush=True)

    # --- summary: does fragmentation hurt? ---
    print("\n=== detection rate & site-AUC by fragmentation ===")
    print(f"{'condition':>16} {'det_rate':>9} {'95% CI':>15} {'mean_AUC':>9}")
    labels = [f"blocks={nb}" for nb in args.n_blocks] + (["gene_conversion"] if args.gene_conversion else [])
    for label in labels:
        sub = [r for r in rows if r["condition"] == label]
        if not sub:
            continue
        k = sum(r["detected"] for r in sub)
        lo, hi = wilson_ci(k, len(sub))
        aucs = [r["site_auc"] for r in sub if r["site_auc"] is not None]
        ma = float(np.mean(aucs)) if aucs else float("nan")
        print(f"{label:>16} {k/len(sub):>9.2f} {f'[{lo:.2f},{hi:.2f}]':>15} {ma:>9.3f}")
    print(f"\nwrote {args.out}")
    print("If site-AUC holds as n_blocks rises, the detector reads fragmented "
          "contamination fine and only the single-window verdict degrades; if AUC "
          "falls, fragmentation genuinely hurts and that is a stated limitation.")


if __name__ == "__main__":
    main()

"""Does the detector survive more than one clean swap?

Every contamination condition run so far — main experiment, segment sweep,
divergence grid, ablation, spatial patches — copies a single contiguous run
from one donor. That is the easy case for `conflict.scan_segment`: it looks
for exactly one contiguous window and nothing in the simulator has ever asked
it to find more than that.

Real recombination does not stop at one crossover. Multiple breakpoints
between the same two parents (A-B-A-B-A...) and multi-tract gene conversion
both produce a mosaic whose contaminated positions are several separate runs
in sequence, not one. This holds the total number of contaminated residues
fixed (50, the width already shown in RESULTS.md §12 to sit above the
diagnostic-site floor for a single block) and varies only how many separate
runs it is split into — so any change in detection is attributable to
fragmentation, not to swapping less sequence overall.

No new detection code: `evolve.contaminate_positions` already takes an
arbitrary position *set* rather than an interval, and the ground truth is
scored with `spatial.patch_jaccard` (set Jaccard) against the shipped 1D scan
— the same machinery `spatial_contamination.py` uses for a patch that is
discontiguous for a different reason. The scan itself is not touched; the
question is how much a single-window scan degrades against a target it was
never designed to represent.

    python experiments/multi_breakpoint.py --seeds 3
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

from ablation import identity_delta  # noqa: E402
from evolve import contaminate_positions  # noqa: E402
from mpnn_api import MPNN_DIR, MPNNScorer  # noqa: E402
from repair import repair_family  # noqa: E402
from spatial import patch_jaccard, permutation_test_1d, scan_1d  # noqa: E402

DEFAULT_PDB = MPNN_DIR / "inputs/PDB_monomers/pdbs/5L33.pdb"
RESULTS = REPO_ROOT / "experiments" / "results"


def multi_block_positions(
    L: int, n_blocks: int, total_size: int, *, margin: int = 8
) -> np.ndarray:
    """`n_blocks` contiguous runs of ~total_size/n_blocks residues, evenly spaced.

    `n_blocks=1` reproduces an ordinary single-block swap. `total_size` is held
    *exactly* constant across conditions (the remainder from an uneven split
    is distributed across the first few blocks, one extra residue each), so
    the sweep varies fragmentation, not how much sequence changes hands. An
    earlier version floor-divided `total_size // n_blocks` and dropped the
    remainder, so n_blocks=3/4 actually swapped 48 residues against 50 for
    n_blocks=1/2 — a real, if small, confound on the "same total budget"
    claim this sweep depends on.
    """
    base, extra = divmod(total_size, n_blocks)
    sizes = [base + (1 if i < extra else 0) for i in range(n_blocks)]
    span = L - 2 * margin
    gap_space = span - total_size
    if n_blocks == 1:
        start = margin + gap_space // 2
        return np.arange(start, start + sizes[0])
    gap = gap_space / (n_blocks - 1)
    positions: list[int] = []
    pos = float(margin)
    for size in sizes:
        start = min(int(round(pos)), L - margin - size)
        positions.extend(range(start, start + size))
        pos = start + size + gap
    return np.array(sorted(set(positions)))


def evaluate(scorer, seqs, true_positions, *, n_perm, seed, alpha=0.05) -> dict:
    """MPNN and identity scores, both through the shipped 1D scan."""
    rep = repair_family(scorer, seqs, n_orders=8)
    m, a, b = rep.mosaic.sequence, rep.sub_a.sequence, rep.sub_b.sequence
    diag = np.flatnonzero(np.array(list(a)) != np.array(list(b)))
    if len(diag) < 12:
        return {}

    scores = {
        "mpnn": scorer.site_log_prob(m, context_seq=a) - scorer.site_log_prob(m, context_seq=b),
        "identity": identity_delta(m, a, b),
    }
    truth = set(int(p) for p in true_positions)
    out: dict[str, dict] = {}
    for name, values in scores.items():
        found, _ = scan_1d(values, diag)
        p_value = permutation_test_1d(values, diag, n_perm=n_perm, seed=seed)
        out[name] = {
            "p_value": float(p_value),
            "detected": bool(p_value <= alpha),
            "jaccard": round(patch_jaccard(found, sorted(truth)), 4),
            "n_found": int(len(found)),
        }
    out["n_diagnostic"] = int(len(diag))
    out["n_diagnostic_in_truth"] = int(sum(1 for d in diag if int(d) in truth))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pdb", default=str(DEFAULT_PDB))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--total-size", type=int, default=50, help="contaminated residues, summed across blocks")
    ap.add_argument("--breakpoint-counts", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--contaminated", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    from sweeps import _rebuild, clean_family

    scorer = MPNNScorer(args.pdb, device=args.device)
    L = scorer.L
    print(f"device={scorer.device} L={L}", flush=True)

    out_path = Path(args.out) / "multi_breakpoint.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["seed"], r["n_blocks"]) for r in rows}
    if done:
        print(f"resuming: {len(done)} cells already on disk", flush=True)

    for seed in range(args.seeds):
        fam = None
        for n_blocks in args.breakpoint_counts:
            if (seed, n_blocks) in done:
                continue
            if fam is None:
                fam = _rebuild(clean_family(scorer, "selection", seed, n_per_clade=6))
            positions = multi_block_positions(L, n_blocks, args.total_size)
            fam_c = contaminate_positions(
                fam, positions, n_contaminated=args.contaminated, seed=seed
            )
            t0 = time.time()
            res = evaluate(scorer, fam_c.leaf_seqs, positions, n_perm=args.n_perm, seed=seed)
            if not res:
                print(f"  [s{seed}] n_blocks={n_blocks}: too few diagnostic sites", flush=True)
                continue
            row = {
                "seed": seed, "n_blocks": n_blocks, "total_size": int(len(positions)),
                "true_positions": positions.tolist(),
                "seconds": round(time.time() - t0, 1), **res,
            }
            rows.append(row)
            out_path.write_text(json.dumps(rows, indent=2, default=str))
            cell = lambda k: f"p={row[k]['p_value']:.3f} J={row[k]['jaccard']:.2f} n={row[k]['n_found']}"
            print(f"  [s{seed}] n_blocks={n_blocks} size={row['total_size']:3} "
                  f"diag={res['n_diagnostic']:3} in_truth={res['n_diagnostic_in_truth']:3} | "
                  f"mpnn {cell('mpnn')} | identity {cell('identity')}", flush=True)

    print(f"\nwrote {out_path}  ({len(rows)} cells)")

    print("\n" + "=" * 78)
    print(f"{'n_blocks':>9}{'score':>10}{'fired':>9}{'mean Jaccard':>14}{'mean n_found':>14}")
    for n_blocks in args.breakpoint_counts:
        sub = [r for r in rows if r["n_blocks"] == n_blocks]
        if not sub:
            continue
        for score in ("mpnn", "identity"):
            c = [r[score] for r in sub]
            print(f"{n_blocks:>9}{score:>10}"
                  f"{sum(x['detected'] for x in c):>5}/{len(c):<3}"
                  f"{np.mean([x['jaccard'] for x in c]):>14.3f}"
                  f"{np.mean([x['n_found'] for x in c]):>14.1f}")
    print("=" * 78)


if __name__ == "__main__":
    main()

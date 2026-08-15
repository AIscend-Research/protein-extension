"""Can a structural scan find contamination that a sequence scan cannot?

Every previous experiment swapped a contiguous *sequence* block and searched for
it by scanning *sequence* position. A string comparison wins that game by
construction. This asks the question the other way round.

A 2x2x2, because attributing the result correctly requires it:

    contamination  block  (contiguous in sequence, as recombination does it)
                   patch  (contiguous on the backbone, scattered in sequence,
                           as gene conversion of a structural element does it)

    score          mpnn      log p(a_i | ctx=A) - log p(a_i | ctx=B)
                   identity  [a_i == subA_i] - [a_i == subB_i]

    scan           1d  contiguous runs in diagnostic-site space  (the shipped scan)
                   3d  spatial neighbourhoods among diagnostic sites

The predictions worth distinguishing:

    both scores, 1d scan, patch contamination  ->  should fail. The target does
        not exist in the coordinate being searched.
    mpnn + 3d beats identity + 3d              ->  the structural *model* earns
        its place.
    mpnn + 3d == identity + 3d, both beat 1d   ->  the structural *scan* earns
        its place and the model still does not. Different conclusion, and the
        more likely one given everything measured so far.

Only the second outcome rescues ProteinMPNN. The third would still be a real
finding — it would say the useful structural idea is *where you look*, not what
you look with.

    python experiments/spatial_contamination.py --seeds 3
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
from evolve import contaminate, contaminate_positions  # noqa: E402
from mpnn_api import MPNN_DIR, MPNNScorer  # noqa: E402
from repair import repair_family  # noqa: E402
from spatial import (  # noqa: E402
    ca_coords, neighbour_sets, patch_jaccard, permutation_test_1d,
    scan_1d, spatial_patch, spatial_permutation_test, spatial_scan, sequence_runs,
)

DEFAULT_PDB = MPNN_DIR / "inputs/PDB_monomers/pdbs/5L33.pdb"
RESULTS = REPO_ROOT / "experiments" / "results"


def evaluate(scorer, coords, seqs, true_positions, *, n_perm, seed, alpha=0.05,
             k_neighbours=10) -> dict:
    """Both scores through both scans, on one contaminated family."""
    rep = repair_family(scorer, seqs, n_orders=8)
    m, a, b = rep.mosaic.sequence, rep.sub_a.sequence, rep.sub_b.sequence
    diag = np.flatnonzero(np.array(list(a)) != np.array(list(b)))
    if len(diag) < 12:
        return {}

    # Spatial neighbourhoods among the diagnostic sites only, so the 3D scan
    # searches the same set of informative positions the 1D scan does. Anything
    # else would compare the scans on different data.
    nb_local = neighbour_sets(coords[diag], k=min(k_neighbours, len(diag)))

    scores = {
        "mpnn": scorer.site_log_prob(m, context_seq=a) - scorer.site_log_prob(m, context_seq=b),
        "identity": identity_delta(m, a, b),
    }
    truth = set(int(p) for p in true_positions)
    out: dict[str, dict] = {}
    for score_name, values in scores.items():
        # --- 1D: contiguous runs in diagnostic-site space (the shipped scan)
        found_1d, _ = scan_1d(values, diag)
        p_1d = permutation_test_1d(values, diag, n_perm=n_perm, seed=seed)
        # --- 3D: spatial neighbourhoods among the same diagnostic sites
        members, _ = spatial_scan(np.sign(values[diag]), nb_local, min_size=4)
        found_3d = diag[members]
        p_3d, _ = spatial_permutation_test(np.sign(values[diag]), nb_local,
                                           n_perm=n_perm, seed=seed, min_size=4)
        out[score_name] = {
            "1d": {"p_value": float(p_1d), "detected": bool(p_1d <= alpha),
                   "jaccard": round(patch_jaccard(found_1d, sorted(truth)), 4),
                   "n_found": int(len(found_1d))},
            "3d": {"p_value": float(p_3d), "detected": bool(p_3d <= alpha),
                   "jaccard": round(patch_jaccard(found_3d, sorted(truth)), 4),
                   "n_found": int(len(found_3d))},
        }
    out["n_diagnostic"] = int(len(diag))
    out["n_diagnostic_in_truth"] = int(sum(1 for d in diag if int(d) in truth))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pdb", default=str(DEFAULT_PDB))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--size", type=int, default=30, help="residues swapped, both modes")
    ap.add_argument("--centres", type=int, nargs="+", default=[30, 55, 80])
    ap.add_argument("--contaminated", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    from sweeps import _rebuild, clean_family

    scorer = MPNNScorer(args.pdb, device=args.device)
    coords = ca_coords(scorer)
    L = scorer.L
    print(f"device={scorer.device} L={L}", flush=True)

    out_path = Path(args.out) / "spatial_contamination.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume, and checkpoint after every cell. An earlier version of this script
    # only wrote at the end; the run was reaped an hour in and the results
    # survived only because they could be parsed back out of stdout. A long
    # experiment that keeps its results in memory is one signal away from
    # having produced nothing.
    rows = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["seed"], r["mode"], r["centre"]) for r in rows}
    if done:
        print(f"resuming: {len(done)} cells already on disk", flush=True)

    for seed in range(args.seeds):
        fam = None
        for centre in args.centres:
            if all((seed, m, centre) in done for m in ("block", "patch")):
                continue
            if fam is None:
                fam = _rebuild(clean_family(scorer, "selection", seed, n_per_clade=6))
            # block: contiguous in sequence, centred on the same residue
            start = max(0, min(centre - args.size // 2, L - args.size))
            block = np.arange(start, start + args.size)
            patch = spatial_patch(coords, centre, args.size)
            for mode, positions in (("block", block), ("patch", patch)):
                if (seed, mode, centre) in done:
                    continue
                fam_c = contaminate_positions(fam, positions,
                                              n_contaminated=args.contaminated, seed=seed)
                t0 = time.time()
                res = evaluate(scorer, coords, fam_c.leaf_seqs, positions,
                               n_perm=args.n_perm, seed=seed)
                if not res:
                    print(f"  [s{seed}] {mode} c={centre}: too few diagnostic sites", flush=True)
                    continue
                runs = len(sequence_runs(positions))
                row = {"seed": seed, "mode": mode, "centre": centre,
                       "size": int(len(positions)), "sequence_runs": runs,
                       "seconds": round(time.time() - t0, 1), **res}
                rows.append(row)
                out_path.write_text(json.dumps(rows, indent=2, default=str))
                cell = lambda s, k: (f"p={row[s][k]['p_value']:.3f} "
                                     f"J={row[s][k]['jaccard']:.2f}")
                print(f"  [s{seed}] {mode:5} c={centre:3} runs={runs} "
                      f"diag={res['n_diagnostic']:3} | "
                      f"mpnn 1d {cell('mpnn','1d')}  3d {cell('mpnn','3d')} | "
                      f"ident 1d {cell('identity','1d')}  3d {cell('identity','3d')}",
                      flush=True)

    print(f"\nwrote {out_path}  ({len(rows)} cells)")

    print("\n" + "=" * 78)
    print(f"{'contamination':>14}{'score':>10}{'scan':>6}{'fired':>9}{'mean Jaccard':>14}")
    for mode in ("block", "patch"):
        sub = [r for r in rows if r["mode"] == mode]
        if not sub:
            continue
        for score in ("mpnn", "identity"):
            for scan in ("1d", "3d"):
                cells = [r[score][scan] for r in sub]
                print(f"{mode:>14}{score:>10}{scan:>6}"
                      f"{sum(c['detected'] for c in cells):>5}/{len(cells):<3}"
                      f"{np.mean([c['jaccard'] for c in cells]):>14.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()

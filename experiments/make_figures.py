"""Generate every figure from one end-to-end family.

Runs the whole pipeline on a simulated family with a known breakpoint, then
renders the stemma, the conflict profile, the permutation null, the apparatus,
the structure views and the contact junction — plus summary panels from any
results already in `experiments/results/`.

    python experiments/make_figures.py --model selection --out figures/
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import viz  # noqa: E402
from apparatus import annotate_contamination, build_apparatus  # noqa: E402
from conflict import permutation_test, smooth  # noqa: E402
from evolve import contaminate, make_evolver, simulate_family  # noqa: E402
from mpnn_api import MPNN_DIR, MPNNScorer  # noqa: E402
from pipeline import analyze  # noqa: E402
from scoring import recovery_by_label  # noqa: E402
from stemma import encode, midpoint_root, neighbour_joining, poisson_distance  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default=str(MPNN_DIR / "inputs/PDB_monomers/pdbs/5L33.pdb"))
    ap.add_argument("--model", choices=["f81", "selection"], default="selection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-per-clade", type=int, default=6)
    ap.add_argument("--breakpoint", default="55,85")
    ap.add_argument("--contaminated", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default=str(REPO_ROOT / "figures"))
    ap.add_argument("--cache", default=str(REPO_ROOT / "scratch" / "figure_family.json"),
                    help="reuse a simulated family so figure tweaks skip the slow simulation")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--device", default="cpu",
                    help="cpu is the default here: MPS has been observed returning "
                         "corrupted tensors on long runs without raising")
    args = ap.parse_args()

    out = Path(args.out)
    start, stop = (int(x) for x in args.breakpoint.split(","))
    scorer = MPNNScorer(args.pdb, device=args.device)
    print(f"device={scorer.device} L={scorer.L}", flush=True)

    signature = {"model": args.model, "seed": args.seed, "n_per_clade": args.n_per_clade,
                 "breakpoint": [start, stop], "contaminated": args.contaminated,
                 "pdb": Path(args.pdb).name}
    cache = Path(args.cache)
    truth = None
    if not args.no_cache and cache.exists():
        stored = json.loads(cache.read_text())
        if stored.get("signature") == signature:
            truth = stored["truth"]
            leaf_seqs = stored["leaf_seqs"]
            print(f"reusing simulated family from {cache}", flush=True)

    if truth is None:
        print("simulating...", flush=True)
        evolver = make_evolver(args.model, scorer)
        clean = simulate_family(evolver, scorer.native_seq, n_per_clade=args.n_per_clade,
                                breakpoint=None, n_contaminated=0, seed=args.seed)
        family = contaminate(clean, (start, stop), n_contaminated=args.contaminated,
                             seed=args.seed)
        truth, leaf_seqs = family.metadata(), family.leaf_seqs
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(
            {"signature": signature, "truth": truth, "leaf_seqs": leaf_seqs}, indent=2))

    print("analysing...", flush=True)
    result = analyze(scorer, leaf_seqs, truth=truth, n_perm=args.n_perm)
    conflict = result.conflict
    print(f"  detected={conflict.detected} p={conflict.p_value:.3f} "
          f"segment={conflict.segment} AUC={result.metrics.get('site_auc')}", flush=True)

    paths: list[Path] = []

    names = list(leaf_seqs)
    tree = midpoint_root(neighbour_joining(
        names, poisson_distance(encode([leaf_seqs[n] for n in names]))))
    paths.append(viz.plot_stemma(tree, contaminated=truth["contaminated"],
                                 clade_a=result.repair.clade_a, path=out / "stemma.png"))

    paths.append(viz.plot_conflict_profile(conflict, true_segment=(start, stop),
                                           path=out / "conflict_profile.png"))

    signs = np.sign(conflict.delta[conflict.diff_sites])
    _, _, null = permutation_test(signs, n_perm=args.n_perm, window=5, min_len=3)
    observed = max(conflict.statistic, float(null.max()) * 0.0 + conflict.statistic)
    paths.append(viz.plot_permutation_null(null, observed, conflict.p_value,
                                           path=out / "permutation_null.png"))

    probs = np.exp(scorer.full_context_log_probs(result.repair.mosaic.sequence))
    calls = annotate_contamination(
        build_apparatus(probs, result.repair.mosaic.posterior_table()), conflict)
    paths.append(viz.plot_apparatus(calls, path=out / "apparatus.png"))

    paths.append(viz.plot_structure_conflict(
        scorer, conflict.delta, segment=conflict.segment or (start, stop),
        path=out / "structure_conflict.png"))
    paths.append(viz.plot_structure_conflict(
        scorer, conflict.instability, diverging=False, segment=None,
        path=out / "structure_instability.png",
        title="Sensitivity to decoding order",
        label="fraction of orders that change the call"))
    paths.append(viz.plot_contact_junction(scorer, conflict.segment or (start, stop),
                                           path=out / "contact_junction.png"))

    paths.append(viz.write_bfactor_pdb(scorer, conflict.delta, out / "conflict_bfactor.pdb"))

    calibration = recovery_by_label(
        result.repair.mosaic.sequence, truth["true_root"], [c.label for c in calls])
    paths.append(viz.plot_label_calibration(calibration, path=out / "label_calibration.png"))

    rows = [row for f in sorted(glob.glob(str(REPO_ROOT / "experiments/results/*.json")))
            for row in json.load(open(f))]
    if rows:
        paths.append(viz.plot_detection_summary(rows, path=out / "detection_summary.png"))

    (out / "figure_context.json").write_text(json.dumps({
        "model": args.model, "seed": args.seed, "breakpoint": [start, stop],
        "detected": conflict.detected, "p_value": conflict.p_value,
        "segment": list(conflict.segment) if conflict.segment else None,
        "metrics": result.metrics, "label_recovery": calibration,
    }, indent=2, default=str))

    print(f"\nwrote {len(paths)} files to {out}")
    for path in paths:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()

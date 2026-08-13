"""Run the three evaluations on simulated families with known histories.

    detect   Can the conflict score recover a known recombination breakpoint?
    null     How often does it fire on clean, strictly tree-like data?
    repair   Does the mosaic archetype score worse than the two coherent
             sub-ancestors, and does the gap scale with contamination?

Each (model, seed) pair evolves its lineages once and is then contaminated in
several ways, so the history is held fixed while contamination varies — cheaper,
and a cleaner comparison than re-evolving per condition.

`--model f81` is the control: site-independent evolution has no epistasis for
the detector to sense, so a detector that fires there is reading composition,
not structural incoherence.

    python experiments/run_experiments.py --exp detect --model selection --seeds 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evolve import contaminate, make_evolver, simulate_family  # noqa: E402
from mpnn_api import MPNN_DIR, MPNNScorer  # noqa: E402
from pipeline import analyze  # noqa: E402

DEFAULT_PDB = MPNN_DIR / "inputs/PDB_monomers/pdbs/5L33.pdb"
RESULTS = REPO_ROOT / "experiments" / "results"


def build_family(scorer: MPNNScorer, model: str, seed: int, args) -> object:
    """Evolve one clean two-clade family under the chosen model."""
    evolver = make_evolver(
        model, scorer, mu=args.mu, temperature=args.temperature, sweeps_per_unit=args.sweeps
    )
    return simulate_family(
        evolver,
        scorer.native_seq,
        n_per_clade=args.n_per_clade,
        stem=args.stem,
        leaf_depth=args.leaf_depth,
        breakpoint=None,
        n_contaminated=0,
        seed=seed,
    )


def run(args) -> list[dict]:
    scorer = MPNNScorer(args.pdb, device=args.device)
    L = scorer.L
    rows: list[dict] = []

    for model in args.models:
        for seed in range(args.seeds):
            t0 = time.time()
            clean = build_family(scorer, model, seed, args)
            sim_time = time.time() - t0
            print(f"[{model} seed={seed}] simulated {len(clean.leaf_seqs)} witnesses "
                  f"in {sim_time:.1f}s", flush=True)

            conditions: list[tuple[str, object]] = []
            if args.exp in ("detect", "all"):
                for start, width in args.segments:
                    stop = min(start + width, L)
                    conditions.append((
                        f"detect_bp{start}-{stop}",
                        contaminate(clean, (start, stop), n_contaminated=args.contaminated, seed=seed),
                    ))
            if args.exp in ("null", "all"):
                conditions.append(("null_clean", clean))
            if args.exp in ("repair", "all"):
                start = L // 4
                stop = min(start + args.repair_width, L)
                for n_cont in args.contamination_levels:
                    tag = f"repair_n{n_cont}"
                    fam = clean if n_cont == 0 else contaminate(
                        clean, (start, stop), n_contaminated=n_cont, seed=seed
                    )
                    conditions.append((tag, fam))

            for tag, family in conditions:
                t1 = time.time()
                try:
                    result = analyze(
                        scorer,
                        family.leaf_seqs,
                        truth=family.metadata(),
                        n_perm=args.n_perm,
                        alpha=args.alpha,
                        n_orders=args.n_orders,
                        seed=seed,
                    )
                except ValueError as exc:  # degenerate clade split
                    print(f"  {tag}: skipped ({exc})", flush=True)
                    continue

                row = {
                    "experiment": tag.split("_")[0],
                    "condition": tag,
                    "model": model,
                    "seed": seed,
                    "n_contaminated": len(family.contaminated),
                    "seconds": round(time.time() - t1, 1),
                    **result.summary(),
                }
                rows.append(row)
                print(
                    f"  {tag:22} detected={row['detected']!s:5} p={row['p_value']:.3f} "
                    f"seg={row['segment']} J={row.get('segment_jaccard')} "
                    f"AUC={row.get('site_auc')} gap={row['gap_best']:+.3f} "
                    f"ASR_acc={row.get('asr_accuracy_vs_true_root')} "
                    f"post={row.get('asr_mean_max_posterior')}",
                    flush=True,
                )
    return rows


def write_results(rows: list[dict], out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(rows, indent=2, default=str))
    if rows:
        flat = [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()}
                for r in rows]
        fields = list({k: None for row in flat for k in row})
        with (out_dir / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)
    print(f"\nwrote {out_dir / (name + '.json')}")


def report(rows: list[dict]) -> None:
    """Headline numbers, grouped the way the paper reports them."""
    def group(pred):
        return [r for r in rows if pred(r)]

    print("\n" + "=" * 72)
    for model in sorted({r["model"] for r in rows}):
        det = group(lambda r: r["model"] == model and r["experiment"] == "detect")
        null = group(lambda r: r["model"] == model and r["experiment"] == "null")
        rep = group(lambda r: r["model"] == model and r["experiment"] == "repair")
        print(f"\n[{model}]")
        if det:
            aucs = [r["site_auc"] for r in det if r.get("site_auc") is not None]
            jac = [r["segment_jaccard"] for r in det if r.get("segment_jaccard") is not None]
            print(f"  detection power   : {sum(r['detected'] for r in det)}/{len(det)} "
                  f"({np.mean([r['detected'] for r in det]):.0%})")
            print(f"  site AUC          : {np.mean(aucs):.3f} +- {np.std(aucs):.3f}")
            print(f"  segment Jaccard   : {np.mean(jac):.3f} +- {np.std(jac):.3f}")
        if null:
            fp = np.mean([r["detected"] for r in null])
            print(f"  false positives   : {sum(r['detected'] for r in null)}/{len(null)} ({fp:.0%})")
        if rep:
            print("  repair gap by contamination level:")
            for n in sorted({r["n_contaminated"] for r in rep}):
                sub = [r for r in rep if r["n_contaminated"] == n]
                gaps = [r["gap_best"] for r in sub]
                accs = [r["asr_accuracy_vs_true_root"] for r in sub]
                print(f"    n={n}: gap={np.mean(gaps):+.4f} +- {np.std(gaps):.4f}   "
                      f"ASR acc={np.mean(accs):.3f}")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", choices=["detect", "null", "repair", "all"], default="all")
    ap.add_argument("--models", nargs="+", default=["selection"], choices=["f81", "selection"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pdb", default=str(DEFAULT_PDB))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-per-clade", type=int, default=6)
    ap.add_argument("--stem", type=float, default=2.0)
    ap.add_argument("--leaf-depth", type=float, default=0.4)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--sweeps", type=int, default=8, help="Gibbs sweeps per unit branch length")
    ap.add_argument("--contaminated", type=int, default=3, help="witnesses contaminated in `detect`")
    ap.add_argument("--contamination-levels", type=int, nargs="+", default=[0, 1, 3, 5])
    ap.add_argument("--repair-width", type=int, default=30)
    ap.add_argument("--segments", default="25:30,55:30",
                    help="comma-separated start:width pairs for the detect experiment")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-orders", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    args.segments = [tuple(int(x) for x in pair.split(":")) for pair in args.segments.split(",")]
    name = args.name or f"{args.exp}_{'-'.join(args.models)}_s{args.seeds}"

    started = time.time()
    rows = run(args)
    report(rows)
    write_results(rows, Path(args.out), name)
    print(f"total wall clock: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()

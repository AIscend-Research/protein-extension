"""
Head-to-head: the structural detector vs a sequence-based recombination method
(GARD) on the SAME families.

The README's orthogonality claim -- "a structural detector would be orthogonal
signal in the regime where the sequence methods run out (deep divergence,
saturated sites, short genes)" -- is currently asserted, never measured. This
measures it. For each simulated family we run BOTH detectors on the identical
alignment and record, per divergence bin:

    MPNN:  detected? / segment Jaccard against the true block
    GARD:  detected? / did it place a breakpoint near the true block edges?

Orthogonality is the specific pattern where one recovers the block in a regime
the other misses -- e.g. GARD collapsing at high divergence while MPNN holds, or
(the RESULTS.md prior) the sequence signal staying strong throughout and MPNN
adding nothing. Either way it stops being an assertion.

RDP note: RDP5 is a Windows GUI and not scriptable cross-platform, so this uses
GARD (HyPhy), which is CLI-driven and installable via `conda install -c
bioconda hyphy` or a HyPhy build. If `hyphy` is not on PATH the script says so
and stops, the same way real_family.py does for mafft/iqtree.

    python experiments/compare_recombination.py --model selection --seeds 5 \
        --stems 0.5 1.0 2.0 4.0

Alignments here are gap-free (every witness is evolved on one backbone, no
indels), so no aligner step is needed before GARD.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from conflict import jaccard  # noqa: E402
from evolve import contaminate  # noqa: E402
from mpnn_api import MPNN_DIR, MPNNScorer  # noqa: E402
from sweeps import _rebuild, clean_family, evaluate  # noqa: E402  (reuse the shipped pipeline)

DEFAULT_PDB = MPNN_DIR / "inputs/PDB_monomers/pdbs/5L33.pdb"
RESULTS = REPO_ROOT / "experiments" / "results"
HYPHY = shutil.which("hyphy") or shutil.which("HYPHYMP")


# ------------------------------------------------------------------- GARD side


def write_fasta(leaf_seqs: dict[str, str], path: Path) -> None:
    with path.open("w") as fh:
        for name, seq in leaf_seqs.items():
            fh.write(f">{name}\n{seq}\n")


def run_gard(fasta: Path, out_json: Path, *, kind: str = "protein", timeout: int = 3600) -> Path:
    """Shell out to HyPhy GARD. Returns the path to its JSON result.

    GARD writes `<alignment>.GARD.json` by default; we pin it with --output so
    the audit knows exactly where to read. GARD is not fast -- a small family is
    minutes, and it scales badly with witness count -- hence the timeout.
    """
    if HYPHY is None:
        raise SystemExit(
            "hyphy not found on PATH. Install with `conda install -c bioconda hyphy` "
            "(or build HyPhy) -- this experiment measures the structural detector "
            "against GARD, so it cannot run without it."
        )
    cmd = [HYPHY, "gard", "--alignment", str(fasta), "--type", kind]
    subprocess.run(cmd, check=True, timeout=timeout,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return fasta.with_suffix(fasta.suffix + ".GARD.json")


def parse_gard(out_json: Path) -> dict:
    """Extract GARD's CONCLUSION (schema: HyPhy 2.5.x).

    GARD's verdict lives in `improvements`, keyed by breakpoint-count model.
    Key "0" is the no-breakpoint baseline (breakpoints=null). A real detection
    is a higher-key model whose breakpoints improve on baseline by c-AIC. If the
    best model is the baseline (bestModelAICc == singleTreeAICc), GARD found no
    recombination -- the positions printed during the GA search were explored
    and rejected, and must NOT be read as detections.
    """
    data = json.loads(out_json.read_text())
    improvements = data.get("improvements") or {}

    breakpoints: list[int] = []
    for key, entry in improvements.items():
        if key == "0":
            continue
        bps = entry.get("breakpoints")
        if bps:
            for bp in bps:
                pos = bp[0] if isinstance(bp, (list, tuple)) else bp
                breakpoints.append(int(pos))

    best = data.get("bestModelAICc")
    baseline = data.get("singleTreeAICc") or data.get("baselineScore")
    improved = bool(best is not None and baseline is not None and best < baseline - 1e-3)

    return {"n_breakpoints": len(breakpoints),
            "breakpoints": sorted(breakpoints),
            "gard_detected": improved and len(breakpoints) > 0}


def gard_vs_truth(breakpoints: list[int], true_bp: tuple[int, int], tol: int = 5) -> dict:
    """Did GARD place a breakpoint near either edge of the true block?"""
    if not breakpoints:
        return {"gard_edge_hit": False, "gard_min_dist": None, "gard_interval_jaccard": 0.0}
    start, stop = true_bp
    dists = [min(abs(b - start), abs(b - stop)) for b in breakpoints]
    edge_hit = min(dists) <= tol
    # crude interval recovery: nearest GARD-delimited segment to the true block
    edges = sorted(set([0] + breakpoints))
    best_j = 0.0
    for lo, hi in zip(edges, edges[1:]):
        best_j = max(best_j, jaccard((lo, hi), true_bp))
    return {"gard_edge_hit": bool(edge_hit), "gard_min_dist": int(min(dists)),
            "gard_interval_jaccard": round(best_j, 4)}


# ------------------------------------------------------------------- main sweep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="selection", choices=["f81", "selection"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pdb", default=str(DEFAULT_PDB))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stems", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0],
                    help="between-clade divergence = the 'sequence methods run out' axis")
    ap.add_argument("--n-per-clade", type=int, default=6)
    ap.add_argument("--contaminated", type=int, default=3)
    ap.add_argument("--start", type=int, default=55)
    ap.add_argument("--width", type=int, default=30)
    ap.add_argument("--tol", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-orders", type=int, default=32)
    ap.add_argument("--gard-type", default="amino-acid", choices=["amino-acid", "nucleotide", "codon"])
    ap.add_argument("--out", default=str(RESULTS / "compare_recombination.json"))
    args = ap.parse_args()

    scorer = MPNNScorer(args.pdb, device=args.device)
    work = RESULTS / "gard_work"
    work.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for seed in range(args.seeds):
        for stem in args.stems:
            stored = clean_family(scorer, args.model, seed,
                                  n_per_clade=args.n_per_clade, stem=stem)
            fam = _rebuild(stored)
            stop = min(args.start + args.width, len(stored["true_root"]))
            bp = (args.start, stop)
            cont = contaminate(fam, bp, n_contaminated=args.contaminated, seed=seed)

            # --- structural detector (identical call to the shipped sweeps) ---
            mpnn = evaluate(scorer, cont.leaf_seqs, cont.metadata(),
                            n_perm=args.n_perm, n_orders=args.n_orders, seed=seed)

            # --- GARD on the same alignment ---
            fasta = work / f"{args.model}_s{seed}_stem{stem}.fasta"
            gjson = fasta.with_suffix(fasta.suffix + ".GARD.json")
            write_fasta(cont.leaf_seqs, fasta)
            t0 = time.time()
            try:
                run_gard(fasta, gjson, kind=args.gard_type)
                gard = parse_gard(gjson)
                gard.update(gard_vs_truth(gard["breakpoints"], bp, tol=args.tol))
                gard["gard_error"] = None
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                gard = {"gard_detected": None, "gard_error": str(e)[:200]}
            gard_secs = round(time.time() - t0, 1)

            row = {"model": args.model, "seed": seed, "stem": stem,
                   "true_breakpoint": list(bp), "gard_seconds": gard_secs,
                   "mpnn_detected": mpnn["detected"],
                   "mpnn_segment_jaccard": mpnn.get("segment_jaccard"),
                   "mpnn_site_auc": mpnn.get("site_auc_segment"),
                   "n_diagnostic": mpnn["n_diagnostic"], **gard}
            rows.append(row)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(rows, indent=2, default=str))
            print(f"  [s{seed} stem={stem}] MPNN det={row['mpnn_detected']!s:5} "
                  f"J={row['mpnn_segment_jaccard']} | GARD det={row.get('gard_detected')!s:5} "
                  f"edge_hit={row.get('gard_edge_hit')} bpJ={row.get('gard_interval_jaccard')} "
                  f"ndiag={row['n_diagnostic']}", flush=True)

    # --- orthogonality summary: per divergence bin, who recovers the block ---
    print("\n=== recovery by divergence (the orthogonality test) ===")
    print(f"{'stem':>6} {'MPNN det':>9} {'GARD det':>9} {'MPNN J':>7} {'GARD hit':>9} {'ndiag':>6}")
    for stem in args.stems:
        sub = [r for r in rows if r["stem"] == stem]
        mdet = np.mean([r["mpnn_detected"] for r in sub])
        gdet = np.mean([bool(r.get("gard_detected")) for r in sub])
        mj = np.mean([r["mpnn_segment_jaccard"] or 0 for r in sub])
        ghit = np.mean([bool(r.get("gard_edge_hit")) for r in sub])
        nd = np.mean([r["n_diagnostic"] for r in sub])
        print(f"{stem:>6} {mdet:>9.2f} {gdet:>9.2f} {mj:>7.2f} {ghit:>9.2f} {nd:>6.1f}")
    print(f"\nwrote {args.out}")
    print("Orthogonality = a divergence bin where one column recovers the block and "
          "the other does not. If GARD tracks or beats MPNN at every stem, the "
          "structural signal is redundant and the README claim should be retracted.")


if __name__ == "__main__":
    main()

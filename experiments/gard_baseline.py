"""Head-to-head against a published recombination detector.

Every comparison so far has been internal: MPNN against a sequence-identity
statistic built for this pipeline (`ablation.py`), never against a method
someone else published. `identity` is not GARD or RDP — it consumes the two
sub-ancestors this pipeline reconstructs, so it is a control on the
instrument, not the orthogonality test the README's thesis actually needs.

GARD (Kosakovsky Pond et al. 2006) is that test: a genetic algorithm over
alignment partitions, searching for phylogenetic incongruence via a
substitution model fit to each candidate partition, with model choice by
c-AIC. HyPhy 2.5 ships GARD for amino-acid alignments directly (`--type
amino-acid`, JTT), so it runs on exactly the same protein data as everything
else in this project — no nucleotide or codon layer exists here to feed it,
which is a weaker use of GARD than its usual deployment (GARD is typically
run on coding-sequence alignments, where synonymous substitution adds power
a pure amino-acid fit does not have).

Runs on the *same six conditions* already reported for mpnn / identity /
scrambled in RESULTS.md section 8 (seeds 0-2, widths 30 and 50, breakpoint
starting at 55) — no new simulation code, so the new column drops straight
into the existing table. GARD's step-up procedure returns the breakpoint
count and locations of whichever model wins by c-AIC; when that model has
exactly two breakpoints (three partitions), the middle partition is scored
against the true block with the same `conflict.jaccard` used everywhere
else. Other breakpoint counts are reported as detected/not but are not
forced into a two-sided window for Jaccard — the true contamination here is
always a single block, so a GARD answer that is not "two breakpoints" is not
being unfairly penalized, just not comparable on that one metric.

    python experiments/gard_baseline.py --model selection --seeds 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from conflict import jaccard  # noqa: E402
from evolve import contaminate  # noqa: E402
from mpnn_api import MPNN_DIR, MPNNScorer  # noqa: E402
from sweeps import _rebuild, clean_family  # noqa: E402

DEFAULT_PDB = MPNN_DIR / "inputs/PDB_monomers/pdbs/5L33.pdb"
RESULTS = REPO_ROOT / "experiments" / "results"


def run_gard(leaf_seqs: dict[str, str], *, work_dir: Path, model: str = "JTT",
            timeout: int = 600) -> dict:
    """Invoke `hyphy gard` on one alignment and parse the winning model.

    Every simulated witness in this project is evolved on one shared backbone
    with no indels, so `leaf_seqs` is already an alignment — no MAFFT step.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    fasta = work_dir / "aln.fasta"
    out_json = work_dir / "gard.json"
    with fasta.open("w") as handle:
        for name, seq in leaf_seqs.items():
            handle.write(f">{name}\n{seq}\n")

    # Every return path carries the same keys (detected/n_breakpoints/found_segment
    # default to a not-detected shape on failure) so a consumer that reads a row
    # without checking for "error" first — summarize.py, most obviously — gets a
    # well-formed dict instead of a KeyError several sweeps into a checkpointed run.
    failure = {"detected": False, "n_breakpoints": None, "found_segment": None}

    t0 = time.time()
    try:
        proc = subprocess.run(
            ["hyphy", "gard", "--type", "amino-acid", "--alignment", str(fasta),
             "--model", model, "--output", str(out_json)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {**failure, "error": f"hyphy gard timed out after {timeout}s",
                "seconds": round(time.time() - t0, 1)}
    seconds = round(time.time() - t0, 1)
    # HyPhy creates an empty JSON file even on an assertion failure (e.g. the
    # alignment is too short relative to taxon count for c-AIC comparison),
    # so an empty/unparseable file is a run failure, not a missing one.
    raw = out_json.read_text() if out_json.exists() else ""
    if not raw.strip():
        message = (proc.stdout + proc.stderr)
        assertion = [l for l in message.splitlines() if "ASSERTION FAILED" in l or "Need at least" in l]
        return {**failure, "error": "\n".join(assertion) or message[-2000:], "seconds": seconds}

    d = json.loads(raw)
    partitions = d["breakpointData"]
    n_bp = len(partitions) - 1
    detected = n_bp > 0

    found_segment = None
    if n_bp == 2:
        # three partitions; the middle one is the candidate intruding block.
        start_1idx, stop_1idx = partitions["1"]["bps"][0]
        found_segment = (int(start_1idx) - 1, int(stop_1idx))

    return {
        "seconds": seconds,
        "baseline_aicc": d["baselineScore"],
        "best_aicc": d["bestModelAICc"],
        "delta_aicc": round(d["baselineScore"] - d["bestModelAICc"], 4),
        "n_breakpoints": n_bp,
        "detected": detected,
        "found_segment": list(found_segment) if found_segment else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="selection", choices=["f81", "selection"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pdb", default=str(DEFAULT_PDB))
    ap.add_argument("--n-per-clade", type=int, default=6)
    ap.add_argument("--contaminated", type=int, default=3)
    ap.add_argument("--widths", type=int, nargs="+", default=[30, 50])
    ap.add_argument("--start", type=int, default=55)
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if shutil.which("hyphy") is None:
        raise SystemExit("hyphy not found on PATH — `brew install hyphy`")

    # A dummy MPNNScorer is only needed to reuse `clean_family`'s cache path,
    # which stores families independent of any GARD-specific state.
    scorer = MPNNScorer(args.pdb, device="cpu")
    print(f"L={scorer.L}", flush=True)

    suffix = f"_{args.tag}" if args.tag else ""
    out_path = Path(args.out) / f"gard_{args.model}{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    done: set[tuple[int, int]] = set()
    if out_path.exists():
        rows = json.loads(out_path.read_text())
        done = {(r["seed"], r["width"]) for r in rows}
        if done:
            print(f"  resuming: {len(done)} (seed, width) pairs already on disk", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for seed in range(args.seeds):
            if all((seed, w) in done for w in args.widths):
                continue
            stored = clean_family(scorer, args.model, seed, n_per_clade=args.n_per_clade)
            fam = _rebuild(stored)
            for width in args.widths:
                if (seed, width) in done:
                    continue
                stop = min(args.start + width, len(stored["true_root"]))
                cont = contaminate(fam, (args.start, stop),
                                   n_contaminated=args.contaminated, seed=seed)
                true_bp = (args.start, stop)
                res = run_gard(cont.leaf_seqs, work_dir=tmp_dir / f"s{seed}_w{width}")
                row = {"model": args.model, "seed": seed, "width": width,
                       "true_breakpoint": list(true_bp), **res}
                if res.get("found_segment"):
                    row["segment_jaccard"] = round(
                        jaccard(tuple(res["found_segment"]), true_bp), 4)
                else:
                    row["segment_jaccard"] = None
                rows.append(row)
                out_path.write_text(json.dumps(rows, indent=2, default=str))
                print(f"  [{args.model} s{seed}] w={width:3} n_bp={res.get('n_breakpoints')} "
                      f"detected={res.get('detected')} found={res.get('found_segment')} "
                      f"J={row['segment_jaccard']} deltaAICc={res.get('delta_aicc')} "
                      f"({res['seconds']}s)", flush=True)

    print(f"\nwrote {out_path}")
    print("\n" + "=" * 70)
    fired = [r for r in rows if r.get("detected")]
    jac = [r["segment_jaccard"] for r in rows if r.get("segment_jaccard") is not None]
    print(f"  gard  fired {len(fired)}/{len(rows)}   "
          f"mean Jaccard (2-breakpoint calls only) "
          f"{(sum(jac) / len(jac)) if jac else float('nan'):.3f} (n={len(jac)})")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
The Bedier audit: run the detector across many published reconstructions and
report the DISTRIBUTION of conflict scores, not one verdict.

Named for Joseph Bedier, who distrusted any stemmatic method that always found
exactly two branches. The analogue here: a contamination detector that fires on
a suspicious fraction of ordinary families would be suspect. So run the SAME
empirical pipeline (real_family.py: UniProt -> MAFFT -> IQ-TREE -> the detector)
over a panel of families and look at where the p-values land.

This is a DRIVER, not a reimplementation. It subprocess-calls real_family.py
once per family so "the same detector" stays literally true, then collects each
family's result JSON and assembles the distribution.

The one reading rule that must not be skipped: a p-value from a family near the
diagnostic-site floor is inconclusive, not negative. RESULTS.md makes this point
about the single 3FTx family (p=0.42 "because the family has only 32-38
diagnostic sites ... near the floor where the method has no power"). Pooling
underpowered families with powered ones is exactly the misread to avoid, so the
audit splits the distribution on `n_diff_sites` and reports the two separately.

Manifest (JSON list). `backbone` is the folded ancestor PDB for that family --
the same site-for-site correspondence real_family.py requires:

    [
      {"name": "3ftx",     "query": "\"three-finger toxin\" AND reviewed:true AND database:pdb",
       "backbone": "data/raw/pdb/3EBX.pdb", "split": "midpoint"},
      {"name": "lysozyme", "query": "lysozyme C AND reviewed:true AND database:pdb",
       "backbone": "data/raw/pdb/xxxx.pdb"}
    ]

    python experiments/bedier_audit.py --manifest experiments/audit_families.json
    python experiments/bedier_audit.py --manifest ... --dry-run   # print the commands only

Needs mafft and iqtree3 on PATH (real_family.py enforces this per family).
"""The Bedier audit: run the detector across several published, structurally
characterized protein families, not just one.

Section 11 (RESULTS.md) reports the detector on a single empirical family
(three-finger toxins) and is explicit that one non-detection near the
diagnostic-site floor is inconclusive, not evidence either way. This is the
obvious next question: what does the conflict-score distribution look like
across several such families? Not a claim that any of these families is
actually contaminated — there is no ground truth for any of them, same as
3FTx — but a report of where the method lands on real, independently
published reconstructions rather than one hand-picked case.

Each family goes through exactly [real_family]'s staged pipeline (fetch ->
align -> trim -> tree+ASR, independently per clade) with one addition: the
mosaic ancestor this pipeline reconstructs is folded with ColabFold so the
detector has a backbone that corresponds to it residue-for-residue, then
[real_family] is invoked a second time with that backbone to run detection —
the same two-phase workflow section 11 used for 3FTx, just automated across
a family list. `check_fold.read_atoms` gives mean pLDDT as a fold-confidence
gate; the family-specific disulfide-topology check 3FTx got is not run here,
since it assumes 3FTx's own canonical connectivity and has no equivalent for
an arbitrary family without curating one per family, which this audit does
not do.

    python experiments/bedier_audit.py --families rnase_a lysozyme_c cytochrome_c
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"
REAL_FAMILY = REPO_ROOT / "experiments" / "real_family.py"


def run_one(entry: dict, *, limit: int, device: str, n_perm: int, dry_run: bool) -> dict | None:
    """Drive real_family.py for one family; return its collected result dict."""
    name = entry["name"]
    tag = name  # real_family writes results/real_3ftx_<tag>.json
    work = RESULTS / "audit" / name
    cmd = [
        sys.executable, str(REAL_FAMILY),
        "--query", entry["query"],
        "--backbone", entry["backbone"],
        "--split", entry.get("split", "midpoint"),
        "--work", str(work),
        "--raw", str(work / "raw.fasta"),
        "--tag", tag,
        "--limit", str(entry.get("limit", limit)),
        "--n-perm", str(n_perm),
        "--device", device,
    ]
    if dry_run:
        print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return None

    work.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  [{name}] real_family.py failed (rc={e.returncode}) -- skipping", flush=True)
        return {"name": name, "error": f"real_family rc={e.returncode}"}

    out_json = RESULTS / f"real_3ftx_{tag}.json"
    if not out_json.exists():
        print(f"  [{name}] no result at {out_json} (did the detect stage run? "
              "backbone length must match the ancestor) -- skipping", flush=True)
        return {"name": name, "error": "no result json"}

    d = json.loads(out_json.read_text())
    return {
        "name": name,
        "detected": d.get("detected"),
        "p_value": d.get("p_value"),
        "n_diff_sites": d.get("n_diff_sites"),
        "n_witnesses": d.get("n_witnesses"),
        "segment": d.get("segment"),
        "mosaic_mean_max_posterior": d.get("mosaic_mean_max_posterior"),
        "split_rule": d.get("split_rule"),
    }


def distribution(collected: list[dict], floor: int) -> dict:
    """Split the p-value distribution on the diagnostic-site floor."""
    ok = [r for r in collected if r.get("p_value") is not None]
    powered = [r for r in ok if (r.get("n_diff_sites") or 0) >= floor]
    under = [r for r in ok if (r.get("n_diff_sites") or 0) < floor]

    def block(rows):
        if not rows:
            return {"n": 0}
        p = np.array([r["p_value"] for r in rows], dtype=float)
        return {
            "n": len(rows),
            "n_detected": int(sum(bool(r["detected"]) for r in rows)),
            "p_median": round(float(np.median(p)), 4),
            "p_min": round(float(p.min()), 4),
            "p_max": round(float(p.max()), 4),
            "frac_below_0.05": round(float((p <= 0.05).mean()), 3),
        }

    return {"floor": floor, "powered": block(powered), "underpowered": block(under),
            "n_failed": sum("error" in r for r in collected)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="JSON list of {name, query, backbone, split?}")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--floor", type=int, default=20,
                    help="diagnostic-site count below which a family is underpowered "
                         "(RESULTS.md: nothing fires below ~20)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(RESULTS / "bedier_audit.json"))
    args = ap.parse_args()

    families = json.loads(Path(args.manifest).read_text())
    print(f"auditing {len(families)} families\n")

    collected: list[dict] = []
    for entry in families:
        r = run_one(entry, limit=args.limit, device=args.device,
                    n_perm=args.n_perm, dry_run=args.dry_run)
        if r is None:
            continue
        collected.append(r)
        if "error" not in r:
            print(f"  [{r['name']:>14}] p={r['p_value']:.3f} "
                  f"ndiag={r['n_diff_sites']} det={r['detected']} "
                  f"maxP={r['mosaic_mean_max_posterior']}", flush=True)

    if args.dry_run:
        return

    dist = distribution(collected, args.floor)
    Path(args.out).write_text(json.dumps({"families": collected, "distribution": dist},
                                         indent=2, default=str))

    print("\n=== conflict-score distribution, split on the diagnostic-site floor ===")
    for band in ("powered", "underpowered"):
        b = dist[band]
        if b.get("n"):
            print(f"  {band:>12} (n={b['n']:>2}): detected {b['n_detected']}/{b['n']}, "
                  f"p median {b['p_median']}, range [{b['p_min']}, {b['p_max']}], "
                  f"frac p<=0.05 {b['frac_below_0.05']}")
        else:
            print(f"  {band:>12}: none")
    if dist["n_failed"]:
        print(f"  ({dist['n_failed']} families failed to produce a result)")
    print(f"\nwrote {args.out}")
    print("Read the POWERED band as the audit result; the underpowered band is "
          "reported for completeness but carries no verdict, by construction. A "
          "detector that fired across many powered ordinary families would be the "
          "warning sign -- Bedier's two branches every time.")
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "experiments" / "results"
COLABFOLD = shutil.which("colabfold_batch") or str(REPO_ROOT / ".venv" / "bin" / "colabfold_batch")

# Small (fast to fold on CPU), structurally well-characterized, independently
# published ASR/phylogenetics subjects — not chosen for a favorable outcome,
# chosen for being small enough to fold in minutes and having enough
# structure-backed reviewed entries to form a family at all.
FAMILIES = {
    "rnase_a": '"ribonuclease A" AND reviewed:true AND database:pdb AND length:[100 TO 160]',
    "lysozyme_c": '"lysozyme C" AND reviewed:true AND database:pdb AND length:[110 TO 150]',
    "cytochrome_c": 'cytochrome c AND reviewed:true AND database:pdb AND length:[90 TO 115]',
}


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def fold_ancestor(name: str, sequence: str, work: Path) -> tuple[Path, float]:
    """ColabFold the mosaic ancestor; return (pdb path, mean pLDDT)."""
    from check_fold import read_atoms

    fold_dir = work / "fold"
    fold_dir.mkdir(parents=True, exist_ok=True)
    fasta = fold_dir / "ancestor.fasta"
    fasta.write_text(f">{name}_mosaic_ancestor\n{sequence}\n")
    out_dir = fold_dir / "out"
    already_folded = out_dir.exists() and any(out_dir.glob("*rank_001*.pdb"))
    if not already_folded:
        run([COLABFOLD, "--num-models", "1", "--num-recycle", "3",
             str(fasta), str(out_dir)])
    pdbs = sorted(out_dir.glob("*rank_001*.pdb"))
    if not pdbs:
        raise SystemExit(f"colabfold produced no rank_001 PDB in {out_dir}")
    pdb = pdbs[0]
    _, bfac = read_atoms(pdb)
    plddt = sum(bfac.values()) / len(bfac) if bfac else float("nan")
    return pdb, plddt


def run_family(name: str, query: str, *, seed: int, threads: str, device: str,
              n_perm: int, n_orders: int, limit: int) -> dict:
    work = DATA / "interim" / name
    raw = DATA / "raw" / f"{name}_all.fasta"
    work.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {name} ===  phase 1: fetch/align/trim/tree", flush=True)
    ancestors_path = work / "ancestors_midpoint.json"
    if not ancestors_path.exists():
        run([sys.executable, "experiments/real_family.py",
             "--query", query, "--limit", str(limit),
             "--work", str(work), "--raw", str(raw),
             "--seed", str(seed), "--threads", threads, "--split", "midpoint"],
            cwd=str(REPO_ROOT))
    else:
        print(f"  reusing {ancestors_path}", flush=True)
    ancestors = json.loads(ancestors_path.read_text())
    mosaic_seq = ancestors["mosaic"]
    clade_sizes = [len(ancestors["clade_a"]), len(ancestors["clade_b"])]
    print(f"  mosaic ancestor: {len(mosaic_seq)} aa, clades {clade_sizes}", flush=True)

    print(f"=== {name} ===  phase 2: fold the ancestor", flush=True)
    t0 = time.time()
    pdb, plddt = fold_ancestor(name, mosaic_seq, work)
    print(f"  {pdb.name}  mean pLDDT {plddt:.1f}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"=== {name} ===  phase 3: detect", flush=True)
    out_path = RESULTS / f"real_{name}_midpoint.json"
    if not out_path.exists():
        run([sys.executable, "experiments/real_family.py",
             "--query", query, "--limit", str(limit),
             "--work", str(work), "--raw", str(raw),
             "--seed", str(seed), "--threads", threads, "--split", "midpoint",
             "--backbone", str(pdb), "--n-perm", str(n_perm),
             "--n-orders", str(n_orders), "--device", device],
            cwd=str(REPO_ROOT))
    result = json.loads(out_path.read_text())
    result["family"] = name
    result["plddt"] = round(plddt, 2)
    result["query"] = query
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", default="1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-orders", type=int, default=32)
    ap.add_argument("--out", default=str(RESULTS / "bedier_audit.json"))
    args = ap.parse_args()

    if shutil.which("mafft") is None:
        raise SystemExit("mafft not found (brew install mafft)")
    if shutil.which("iqtree3") is None and shutil.which("iqtree") is None:
        raise SystemExit("iqtree3 not found (brew install iqtree3)")
    if not Path(COLABFOLD).exists():
        raise SystemExit("colabfold_batch not found (.venv/bin/pip install colabfold)")

    out_path = Path(args.out)
    rows: list[dict] = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["family"] for r in rows}

    for name in args.families:
        if name in done:
            print(f"skipping {name}: already in {out_path}", flush=True)
            continue
        row = run_family(name, FAMILIES[name], seed=args.seed, threads=args.threads,
                         device=args.device, n_perm=args.n_perm, n_orders=args.n_orders,
                         limit=args.limit)
        rows.append(row)
        out_path.write_text(json.dumps(rows, indent=2, default=str))

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

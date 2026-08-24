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

"""Verify a folded ancestor before anything downstream is allowed to trust it.

An ancestral sequence is a hypothesis, and a reconstruction that does not fold
into its family's topology is a failed one however sharp its posteriors were.
For three-finger toxins the check is unusually clean: the fold is defined by a
disulfide-stapled beta-sheet core, and the canonical connectivity in ordinal
cysteine terms is

    C1-C3   C2-C4   C5-C6   C7-C8

(erabutoxin numbering 3-24, 17-41, 43-54, 55-60). A reconstruction that pairs
its cysteines any other way is not a three-finger toxin, and the detector's
verdict on it would be a statement about a molecule that does not exist.

    python experiments/check_fold.py data/interim/3ftx/fold/out/*_relaxed_*.pdb
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

CANONICAL_3FTX = [(1, 3), (2, 4), (5, 6), (7, 8)]
SS_BOND_MAX = 2.5  # angstrom; a real S-S sits at ~2.05


def read_atoms(path: Path, chain: str | None = None) -> tuple[list[dict], dict[int, float]]:
    """CYS SG atoms, plus mean per-residue B-factor (pLDDT in AlphaFold output).

    Restricted to a single chain — the first one seen unless `chain` says
    otherwise. Pooling chains would invent inter-chain "disulfides" between
    copies of the same residue in a homodimer, which is how a perfectly ordinary
    crystal form turns into a non-canonical topology report. Alternate locations
    beyond the first are skipped for the same reason.
    """
    sg: list[dict] = []
    bfac: dict[int, list[float]] = {}
    seen_altloc: set[tuple[int, str]] = set()
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        chain_id = line[21]
        if chain is None:
            chain = chain_id
        if chain_id != chain:
            continue
        altloc = line[16]
        name = line[12:16].strip()
        resi = int(line[22:26])
        if altloc not in (" ", "A"):
            continue
        if (resi, name) in seen_altloc:
            continue
        seen_altloc.add((resi, name))
        resn = line[17:20].strip()
        bfac.setdefault(resi, []).append(float(line[60:66]))
        if resn == "CYS" and name == "SG":
            sg.append({
                "resi": resi,
                "xyz": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            })
    return sg, {k: float(np.mean(v)) for k, v in bfac.items()}


def disulfides(sg: list[dict]) -> list[tuple[int, int, float]]:
    """Greedy nearest-partner pairing among SG atoms within bonding distance.

    Greedy on distance rather than all-pairs optimal: a real disulfide is ~2.05 A
    and the next-nearest non-bonded SG pair is far outside the cutoff, so the
    assignment is not actually ambiguous when the fold is good. When it *is*
    ambiguous the fold is bad, which is what the check is for.
    """
    pairs = sorted(
        (
            (float(np.linalg.norm(a["xyz"] - b["xyz"])), a["resi"], b["resi"])
            for a, b in itertools.combinations(sg, 2)
        )
    )
    used: set[int] = set()
    bonds = []
    for dist, i, j in pairs:
        if dist > SS_BOND_MAX or i in used or j in used:
            continue
        bonds.append((i, j, round(dist, 2)))
        used.update((i, j))
    return sorted(bonds)


def report(path: Path) -> dict:
    sg, bfac = read_atoms(path)
    all_cys = sorted(a["resi"] for a in sg)
    bonds = disulfides(sg)
    plddt = float(np.mean(list(bfac.values()))) if bfac else float("nan")

    bonded = sorted({r for i, j, _ in bonds for r in (i, j)})
    unpaired = [r for r in all_cys if r not in set(bonded)]
    # Rank among *bonded* cysteines only. A supernumerary free cysteine does not
    # change which staples hold the fold together, but it does shift every raw
    # ordinal by one, which turns a canonical topology into an apparent mismatch.
    rank = {r: k + 1 for k, r in enumerate(bonded)}
    observed = sorted(tuple(sorted((rank[i], rank[j]))) for i, j, _ in bonds)
    matches = observed == CANONICAL_3FTX
    strained = [(i, j, d) for i, j, d in bonds if d < 1.8 or d > 2.3]

    print(f"\n{path.name}")
    print(f"  residues              : {len(bfac)}")
    print(f"  mean pLDDT            : {plddt:.1f}")
    print(f"  cysteines             : {len(sg)} at {all_cys}")
    print(f"  disulfides found      : {len(bonds)}")
    for i, j, d in bonds:
        flag = "  <- strained" if (d < 1.8 or d > 2.3) else ""
        print(f"      {i:3}-{j:<3}  (C{rank[i]}-C{rank[j]} of bonded)  {d} A{flag}")
    if unpaired:
        print(f"  unpaired cysteines    : {unpaired}")
    print(f"  topology (bonded rank): {observed}")
    print(f"  canonical 3FTx core   : {CANONICAL_3FTX}")
    print(f"  VERDICT               : {'CANONICAL' if matches else 'NON-CANONICAL'}")
    if strained:
        print(f"  geometry warning      : {len(strained)} bond(s) outside 1.8-2.3 A; "
              f"connectivity is still unambiguous but run --amber before quoting lengths")
    return {
        "file": str(path), "mean_plddt": round(plddt, 2), "n_cys": len(sg),
        "disulfides": [[i, j, d] for i, j, d in bonds],
        "topology_bonded_rank": [list(p) for p in observed],
        "canonical": bool(matches), "unpaired": unpaired,
        "strained_bonds": [[i, j, d] for i, j, d in strained],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb", nargs="+")
    ap.add_argument("--out", default=None,
                    help="write the verdicts here. Worth doing: the folded model itself "
                         "lives under data/, which is gitignored, so without this the "
                         "check is the only part of the result that survives.")
    args = ap.parse_args()
    verdicts = [report(Path(p)) for p in args.pdb]
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(verdicts, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

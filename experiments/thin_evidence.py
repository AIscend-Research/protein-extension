"""Does the structural model win when the sequence evidence runs out?

This is the experiment the README's central claim implies and that nothing in
the repository has tested. The claim is that a structural detector is orthogonal
signal "in the regime where the sequence methods run out — deep divergence,
saturated sites, short genes". Every condition measured so far had six healthy
witnesses per clade, which is the opposite regime: sequence evidence is abundant
there, the marginal reconstruction is sharp, and 95% of diagnostic sites end up
holding one sub-ancestor's residue verbatim. A string comparison reads the
answer straight off, and no model can beat a lookup at reading a lookup.

So vary the one thing the identity baseline depends on entirely — how well the
two sub-ancestors are reconstructed — and watch both arms degrade:

    witnesses per clade  6 -> 4 -> 3 -> 2

Identity has no information beyond those two reconstructions, so it should fall
apart as they get noisy. ProteinMPNN conditions on a backbone that does not
degrade at all, so if the structural prior is worth anything the two curves
should cross somewhere.

    crossing      the structural claim survives, in a regime that can be stated
    no crossing   the claim is dead and the negative is clean

Subsampling costs nothing: the clean families are already cached, so this reuses
them rather than re-running Gibbs.

    python experiments/thin_evidence.py --seeds 3 --width 50
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

from ablation import identity_delta, scan_and_test, _Oriented  # noqa: E402
from conflict import auc, detect_contamination, jaccard, oriented_delta  # noqa: E402
from evolve import contaminate  # noqa: E402
from mpnn_api import MPNN_DIR, MPNNScorer  # noqa: E402
from repair import repair_family  # noqa: E402
from sweeps import _rebuild, clean_family  # noqa: E402

DEFAULT_PDB = MPNN_DIR / "inputs/PDB_monomers/pdbs/5L33.pdb"
RESULTS = REPO_ROOT / "experiments" / "results"


def subsample(leaf_seqs: dict[str, str], contaminated: list[str], k: int,
              rng: np.random.Generator) -> dict[str, str]:
    """Keep k witnesses per clade, preserving the contaminated fraction.

    Dropping witnesses at random would sometimes drop every contaminated one and
    silently turn a detection condition into a clean one, so the contaminated
    share is held as close to its original proportion as k allows.
    """
    out: dict[str, str] = {}
    for clade in ("A", "B"):
        names = sorted(n for n in leaf_seqs if n.startswith(clade))
        bad = [n for n in names if n in set(contaminated)]
        good = [n for n in names if n not in set(contaminated)]
        want_bad = min(len(bad), max(1, round(k * len(bad) / max(len(names), 1)))) if bad else 0
        chosen = list(rng.choice(bad, want_bad, replace=False)) if want_bad else []
        chosen += list(rng.choice(good, min(k - len(chosen), len(good)), replace=False))
        out.update({str(n): leaf_seqs[str(n)] for n in chosen})
    return out


def both_arms(scorer, seqs, truth, *, n_perm, n_orders, seed) -> dict:
    """Run the structural probe and the sequence-identity control on one family.

    The clade split is taken from the simulation rather than inferred. That is a
    use of ground truth and it is deliberate: below four witnesses per clade,
    neighbour-joining plus midpoint rooting stops recovering the true split at
    all (it returned 5 | 1 on a 3 | 3 family), so an inferred split would end the
    run before the question here — how the two arms degrade as the sub-ancestors
    get noisy — could be asked. Both arms receive the identical split, so the
    comparison between them is unaffected; only the absolute difficulty is.
    """
    clade_a = sorted(n for n in seqs if n.startswith("A"))
    clade_b = sorted(n for n in seqs if n.startswith("B"))
    rep = repair_family(scorer, seqs, clade_a=clade_a, clade_b=clade_b, n_orders=8)
    m, a, b = rep.mosaic.sequence, rep.sub_a.sequence, rep.sub_b.sequence
    diff = np.flatnonzero(np.array(list(a)) != np.array(list(b)))
    bp = tuple(truth["breakpoint"])
    labels = np.zeros(len(m), dtype=bool)
    labels[bp[0]:bp[1]] = True
    diag = diff if len(diff) else np.arange(len(m))

    con = detect_contamination(scorer, m, a, b, n_perm=n_perm, n_orders=n_orders,
                               min_len=3, seed=seed)
    mpnn = {"detected": bool(con.detected), "p_value": float(con.p_value),
            "segment_jaccard": float(jaccard(con.segment, bp)),
            "site_auc": float(auc(oriented_delta(con)[diag], labels[diag]))}

    di = identity_delta(m, a, b)
    res = scan_and_test(di, diff, n_perm=n_perm, alpha=0.05, seed=seed)
    stub = _Oriented(di, diff, res["best_segment"], res["segment"])
    ident = {"detected": bool(res["detected"]), "p_value": float(res["p_value"]),
             "segment_jaccard": float(jaccard(res["segment"], bp)),
             "site_auc": float(auc(oriented_delta(stub)[diag], labels[diag]))}

    # How much of the answer is sitting in the string? This is the quantity that
    # decides whether identity can win, so it is recorded per condition.
    verbatim = sum(1 for i in diff if m[i] == a[i] or m[i] == b[i])
    return {"n_diagnostic": int(len(diff)),
            "n_diagnostic_in_segment": int(labels[diag].sum()),
            "verbatim_fraction": round(verbatim / max(len(diff), 1), 4),
            "arms": {"mpnn": mpnn, "identity": ident}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pdb", default=str(DEFAULT_PDB))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clade-sizes", type=int, nargs="+", default=[6, 4, 3, 2])
    ap.add_argument("--start", type=int, default=55)
    ap.add_argument("--width", type=int, default=50,
                    help="50 by default: narrower blocks sit below the diagnostic-site "
                         "floor and cannot fire for either arm, so they measure nothing")
    ap.add_argument("--contaminated", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-orders", type=int, default=16)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    scorer = MPNNScorer(args.pdb, device=args.device)
    print(f"device={scorer.device} L={scorer.L}", flush=True)
    rows = []
    for seed in range(args.seeds):
        stored = clean_family(scorer, "selection", seed, n_per_clade=6)
        fam = _rebuild(stored)
        stop = min(args.start + args.width, len(stored["true_root"]))
        cont = contaminate(fam, (args.start, stop), n_contaminated=args.contaminated, seed=seed)
        truth = cont.metadata()
        for k in args.clade_sizes:
            rng = np.random.default_rng(1000 + seed)
            seqs = subsample(cont.leaf_seqs, cont.contaminated, k, rng)
            if len(seqs) < 4:
                continue
            t0 = time.time()
            try:
                out = both_arms(scorer, seqs, truth, n_perm=args.n_perm,
                                n_orders=args.n_orders, seed=seed)
            except ValueError as exc:
                print(f"  [s{seed}] k={k}: skipped ({exc})", flush=True)
                continue
            row = {"seed": seed, "n_per_clade": k, "n_witnesses": len(seqs),
                   "seconds": round(time.time() - t0, 1), **out}
            rows.append(row)
            mp, idt = out["arms"]["mpnn"], out["arms"]["identity"]
            print(f"  [s{seed}] n/clade={k} witnesses={len(seqs):2} "
                  f"diag={out['n_diagnostic']:3} verbatim={out['verbatim_fraction']:.0%} | "
                  f"mpnn p={mp['p_value']:.3f} J={mp['segment_jaccard']:.2f} "
                  f"AUC={mp['site_auc']:.2f} | "
                  f"identity p={idt['p_value']:.3f} J={idt['segment_jaccard']:.2f} "
                  f"AUC={idt['site_auc']:.2f}", flush=True)

    out_path = Path(args.out) / "thin_evidence_selection.json"
    out_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {out_path}")

    print("\n" + "=" * 74)
    print(f"{'n/clade':>8}{'verbatim':>11}{'mpnn J':>10}{'ident J':>10}"
          f"{'mpnn AUC':>11}{'ident AUC':>11}{'mpnn fired':>12}{'ident fired':>12}")
    for k in sorted({r["n_per_clade"] for r in rows}, reverse=True):
        s = [r for r in rows if r["n_per_clade"] == k]
        g = lambda arm, f: np.mean([r["arms"][arm][f] for r in s])
        print(f"{k:>8}{np.mean([r['verbatim_fraction'] for r in s]):>10.0%}"
              f"{g('mpnn','segment_jaccard'):>10.2f}{g('identity','segment_jaccard'):>10.2f}"
              f"{g('mpnn','site_auc'):>11.2f}{g('identity','site_auc'):>11.2f}"
              f"{sum(r['arms']['mpnn']['detected'] for r in s):>8}/{len(s):<3}"
              f"{sum(r['arms']['identity']['detected'] for r in s):>9}/{len(s):<3}")
    print("=" * 74)
    print("A crossing as n/clade falls rescues the structural claim in a stated regime.")
    print("No crossing means the claim is dead and the negative is clean.")


if __name__ == "__main__":
    main()

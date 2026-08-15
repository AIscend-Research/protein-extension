"""Sensitivity sweeps: where does the detector stop working?

A detection reported at one segment length and one divergence is an anecdote.
What makes the claim checkable is the shape of the curve — where AUC falls to
chance, and how many diagnostic sites it takes to get off the floor.

The economy that makes these affordable: evolving a clean family is the only
expensive step (Gibbs sampling under `selection` costs ~11 min on CPU), and it
does not depend on the contamination. So each clean family is simulated once,
cached to disk, and then contaminated many different ways for free.

    python experiments/sweeps.py --sweep segment  --model selection --seeds 3
    python experiments/sweeps.py --sweep divergence --model selection --seeds 3
    python experiments/sweeps.py --sweep orientation --model selection --seeds 1
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

from conflict import auc, detect_contamination, oriented_delta  # noqa: E402
from evolve import contaminate, make_evolver, simulate_family  # noqa: E402
from mpnn_api import MPNN_DIR, MPNNScorer  # noqa: E402
from repair import repair_family  # noqa: E402

DEFAULT_PDB = MPNN_DIR / "inputs/PDB_monomers/pdbs/5L33.pdb"
CACHE = REPO_ROOT / "scratch" / "families"
RESULTS = REPO_ROOT / "experiments" / "results"


# ------------------------------------------------------------- family cache


def clean_family(scorer, model: str, seed: int, *, n_per_clade=6, stem=2.0, leaf_depth=0.4,
                 mu=1.0, temperature=0.5, sweeps=8):
    """Evolve one clean two-clade family, reusing a cached copy when identical."""
    key = f"{model}_s{seed}_n{n_per_clade}_stem{stem}_leaf{leaf_depth}_mu{mu}_T{temperature}"
    path = CACHE / f"{key}.json"
    if path.exists():
        stored = json.loads(path.read_text())
        return stored
    t0 = time.time()
    evolver = make_evolver(model, scorer, mu=mu, temperature=temperature, sweeps_per_unit=sweeps)
    fam = simulate_family(evolver, scorer.native_seq, n_per_clade=n_per_clade, stem=stem,
                          leaf_depth=leaf_depth, breakpoint=None, n_contaminated=0, seed=seed)
    stored = {"leaf_seqs": fam.leaf_seqs, "truth": fam.metadata(),
              "true_root": fam.true_root, "model": model, "seed": seed}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored, indent=2))
    print(f"  simulated {key} in {time.time() - t0:.0f}s", flush=True)
    return stored


class _CachedTree:
    """Stands in for the simulated tree: `metadata()` only ever asks for newick."""

    def __init__(self, newick: str):
        self._newick = newick.rstrip(";")

    def newick(self) -> str:
        return self._newick


def _rebuild(stored: dict):
    """A SimulatedFamily-shaped object from the cache, enough for `contaminate`."""
    from evolve import SimulatedFamily

    return SimulatedFamily(
        tree=_CachedTree(stored["truth"]["newick"]),
        leaf_seqs=dict(stored["leaf_seqs"]),
        true_root=stored["true_root"],
        true_clade_ancestors=stored["truth"].get("true_clade_ancestors", {}),
        contaminated=[],
        breakpoint=None,
        model=stored["model"],
        donor_clade=None,
    )


# -------------------------------------------------------------- orientation


def oriented_scores(con, rule: str) -> np.ndarray:
    """Per-site contamination score under a label-free orientation rule.

    The scan is sign-symmetric: the intruding block and its complement are both
    "the window whose mean differs most", so `delta` says which context a site
    prefers but not which side is the intrusion. Resolving that from ground
    truth would score a coin flip as skill, so the rule has to be label-free —
    but it also has to be *stated*, because a bad rule inverts the AUC and the
    inversion looks exactly like a failed method.

    minority   the intrusion is the rarer sign among diagnostic sites.
    segment    the intrusion is the window the scan already picked out.
    """
    diag = con.diff_sites if len(con.diff_sites) else np.arange(len(con.delta))
    if rule == "minority":
        d = con.delta[diag]
        flip = np.sum(d < 0) > np.sum(d > 0)
        return -con.delta if not flip else con.delta
    if rule == "segment":
        return oriented_delta(con)  # the shipped rule, defined once in [conflict]
    raise ValueError(f"unknown orientation rule {rule!r}")


def evaluate(scorer, leaf_seqs, truth, *, n_perm=200, n_orders=32, seed=0) -> dict:
    """Detect on one contaminated family and score it under both orientations."""
    rep = repair_family(scorer, leaf_seqs)
    con = detect_contamination(scorer, rep.mosaic.sequence, rep.sub_a.sequence,
                               rep.sub_b.sequence, n_perm=n_perm, n_orders=n_orders,
                               min_len=3, seed=seed)
    L = len(rep.mosaic.sequence)
    bp = tuple(truth["breakpoint"]) if truth.get("breakpoint") else None
    labels = np.zeros(L, dtype=bool)
    if bp:
        labels[bp[0]:bp[1]] = True
    diag = con.diff_sites if len(con.diff_sites) else np.arange(L)
    out = {
        "detected": bool(con.detected),
        "p_value": round(con.p_value, 5),
        "segment": list(con.segment) if con.segment else None,
        "best_segment": list(con.best_segment) if con.best_segment else None,
        "n_diagnostic": int(len(con.diff_sites)),
        "n_diagnostic_in_segment": int(labels[diag].sum()) if bp else 0,
    }
    if bp:
        from conflict import jaccard
        out["true_breakpoint"] = list(bp)
        out["segment_jaccard"] = round(jaccard(con.segment, bp), 4)
        for rule in ("minority", "segment"):
            out[f"site_auc_{rule}"] = round(auc(oriented_scores(con, rule)[diag], labels[diag]), 4)
    return out


# ------------------------------------------------------------------ sweeps


def sweep_segment(scorer, args, out_path: Path | None = None) -> list[dict]:
    """AUC as a function of how long the contaminated block is.

    Checkpointed when `out_path` is given: existing (seed, width) pairs already
    on disk are skipped, and every new row is flushed to disk immediately. A
    20-seed run is long enough that a crash partway through should cost minutes,
    not the whole run — the same reasoning that motivated the checkpointing in
    `spatial_contamination.py`.
    """
    rows: list[dict] = []
    done: set[tuple[int, int]] = set()
    if out_path is not None and out_path.exists():
        rows = json.loads(out_path.read_text())
        done = {(r["seed"], r["width"]) for r in rows}
        if done:
            print(f"  resuming: {len(done)} (seed, width) pairs already on disk", flush=True)

    for seed in range(args.seeds):
        if all((seed, w) in done for w in args.widths):
            continue
        stored = clean_family(scorer, args.model, seed, n_per_clade=args.n_per_clade)
        fam = _rebuild(stored)
        for width in args.widths:
            if (seed, width) in done:
                continue
            start = args.start
            stop = min(start + width, len(stored["true_root"]))
            cont = contaminate(fam, (start, stop), n_contaminated=args.contaminated, seed=seed)
            t0 = time.time()
            row = {"sweep": "segment", "model": args.model, "seed": seed, "width": width,
                   "start": start, **evaluate(scorer, cont.leaf_seqs, cont.metadata(),
                                              n_perm=args.n_perm, n_orders=args.n_orders, seed=seed)}
            row["seconds"] = round(time.time() - t0, 1)
            rows.append(row)
            if out_path is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(rows, indent=2, default=str))
            print(f"  [{args.model} s{seed}] width={width:3} det={row['detected']!s:5} "
                  f"p={row['p_value']:.3f} J={row.get('segment_jaccard')} "
                  f"AUC_seg={row.get('site_auc_segment')} AUC_min={row.get('site_auc_minority')} "
                  f"ndiag={row['n_diagnostic']}", flush=True)
    return rows


def sweep_divergence(scorer, args) -> list[dict]:
    """AUC against between-clade divergence and witness count.

    `stem` is the branch separating the two clades, so it sets how many
    diagnostic sites exist at all — the floor the detector cannot go below.
    """
    rows = []
    for seed in range(args.seeds):
        for stem in args.stems:
            for n_per_clade in args.clade_sizes:
                stored = clean_family(scorer, args.model, seed,
                                      n_per_clade=n_per_clade, stem=stem)
                fam = _rebuild(stored)
                stop = min(args.start + args.width, len(stored["true_root"]))
                n_cont = min(args.contaminated, n_per_clade - 1)
                cont = contaminate(fam, (args.start, stop), n_contaminated=n_cont, seed=seed)
                t0 = time.time()
                row = {"sweep": "divergence", "model": args.model, "seed": seed,
                       "stem": stem, "n_per_clade": n_per_clade,
                       **evaluate(scorer, cont.leaf_seqs, cont.metadata(),
                                  n_perm=args.n_perm, n_orders=args.n_orders, seed=seed)}
                row["seconds"] = round(time.time() - t0, 1)
                rows.append(row)
                print(f"  [{args.model} s{seed}] stem={stem} n={n_per_clade} "
                      f"det={row['detected']!s:5} p={row['p_value']:.3f} "
                      f"AUC_seg={row.get('site_auc_segment')} ndiag={row['n_diagnostic']}",
                      flush=True)
    return rows


def sweep_orientation(scorer, args) -> list[dict]:
    """Both orientation rules on the headline conditions, to settle which to use."""
    rows = []
    for seed in range(args.seeds):
        stored = clean_family(scorer, args.model, seed, n_per_clade=args.n_per_clade)
        fam = _rebuild(stored)
        for start, width in [(25, 30), (55, 30)]:
            stop = min(start + width, len(stored["true_root"]))
            cont = contaminate(fam, (start, stop), n_contaminated=args.contaminated, seed=seed)
            row = {"sweep": "orientation", "model": args.model, "seed": seed,
                   "start": start, "width": width,
                   **evaluate(scorer, cont.leaf_seqs, cont.metadata(),
                              n_perm=args.n_perm, n_orders=args.n_orders, seed=seed)}
            rows.append(row)
            print(f"  [{args.model} s{seed}] bp={start}-{stop} det={row['detected']!s:5} "
                  f"J={row.get('segment_jaccard')} "
                  f"AUC_minority={row.get('site_auc_minority')} "
                  f"AUC_segment={row.get('site_auc_segment')}", flush=True)
    return rows


def sweep_repair(scorer, args) -> list[dict]:
    """The repair prediction, with sample size held constant.

    The prediction is that the mosaic archetype scores worse under the joint
    model than a coherent sub-ancestor. As measured so far it does not — but the
    comparison is confounded: the mosaic is reconstructed from every witness and
    each sub-ancestor from half of them, so "coherent" and "reconstructed from
    fewer sequences" are varied together. A reconstruction from more data sits
    closer to the family consensus, which is exactly where a joint model assigns
    high likelihood, and that pull is in the opposite direction to the effect
    being tested.

    So this contrasts two reconstructions built from the *same number* of
    witnesses:

        clade   n witnesses drawn from one clade  -> one coherent history
        mixed   n witnesses split across clades   -> spans the root

    If joint compatibility is what the score is sensing, `clade` should beat
    `mixed` at equal n. If it does not, the repair prediction fails for a reason
    that has nothing to do with sample size.
    """
    from stemma import reconstruct

    rows = []
    for seed in range(args.seeds):
        stored = clean_family(scorer, args.model, seed, n_per_clade=args.n_per_clade)
        fam = _rebuild(stored)
        stop = min(args.start + args.width, len(stored["true_root"]))
        for n_cont in args.contamination_levels:
            cont = fam if n_cont == 0 else contaminate(
                fam, (args.start, stop), n_contaminated=n_cont, seed=seed)
            seqs = cont.leaf_seqs
            names = sorted(seqs)
            a = [n for n in names if n.startswith("A")]
            b = [n for n in names if n.startswith("B")]
            half = min(len(a), len(b))
            rng = np.random.default_rng(seed)

            groups = {
                "mosaic_all": names,
                "clade_a": a,
                "clade_b": b,
                # matched-n incoherent control: half from each clade
                "mixed": sorted(list(rng.choice(a, half // 2 + half % 2, replace=False))
                                + list(rng.choice(b, half // 2, replace=False))),
            }
            scored = {}
            for tag, taxa in groups.items():
                rec = reconstruct(seqs, taxa=list(taxa))
                from scoring import score_sequence
                scored[tag] = {
                    "n": len(taxa),
                    "pll": round(score_sequence(scorer, rec.sequence, n_orders=16)
                                 .pseudo_log_likelihood, 4),
                }
            row = {"sweep": "repair", "model": args.model, "seed": seed,
                   "n_contaminated": len(cont.contaminated), **{
                       f"{k}_{f}": v[f] for k, v in scored.items() for f in ("n", "pll")}}
            row["gap_uncontrolled"] = round(
                max(scored["clade_a"]["pll"], scored["clade_b"]["pll"])
                - scored["mosaic_all"]["pll"], 4)
            row["gap_matched_n"] = round(
                max(scored["clade_a"]["pll"], scored["clade_b"]["pll"])
                - scored["mixed"]["pll"], 4)
            rows.append(row)
            print(f"  [{args.model} s{seed}] n_cont={row['n_contaminated']} "
                  f"mosaic(n={scored['mosaic_all']['n']})={scored['mosaic_all']['pll']:+.4f} "
                  f"cladeA(n={scored['clade_a']['n']})={scored['clade_a']['pll']:+.4f} "
                  f"mixed(n={scored['mixed']['n']})={scored['mixed']['pll']:+.4f}  "
                  f"gap_uncontrolled={row['gap_uncontrolled']:+.4f} "
                  f"gap_matched_n={row['gap_matched_n']:+.4f}", flush=True)
    return rows


SWEEPS = {"segment": sweep_segment, "divergence": sweep_divergence,
          "orientation": sweep_orientation, "repair": sweep_repair}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", choices=sorted(SWEEPS), required=True)
    ap.add_argument("--model", default="selection", choices=["f81", "selection"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pdb", default=str(DEFAULT_PDB))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-per-clade", type=int, default=6)
    ap.add_argument("--contaminated", type=int, default=3)
    ap.add_argument("--widths", type=int, nargs="+", default=[10, 20, 30, 50])
    ap.add_argument("--start", type=int, default=55)
    ap.add_argument("--width", type=int, default=30)
    ap.add_argument("--stems", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--clade-sizes", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--contamination-levels", type=int, nargs="+", default=[0, 1, 3, 5])
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-orders", type=int, default=32)
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--tag", default="",
                    help="filename suffix, so a wider-seed rerun does not silently "
                         "overwrite an earlier result at the same (sweep, model)")
    args = ap.parse_args()

    scorer = MPNNScorer(args.pdb, device=args.device)
    print(f"device={scorer.device} L={scorer.L} sweep={args.sweep}", flush=True)
    started = time.time()
    suffix = f"_{args.tag}" if args.tag else ""
    out = Path(args.out) / f"sweep_{args.sweep}_{args.model}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.sweep == "segment":
        rows = sweep_segment(scorer, args, out_path=out)
    else:
        rows = SWEEPS[args.sweep](scorer, args)
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {out}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()

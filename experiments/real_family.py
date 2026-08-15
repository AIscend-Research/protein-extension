"""Assemble an empirical family and run the detector on it.

The simulated experiments hold the history fixed and vary contamination, which is
the only way to measure detection against ground truth — but the `selection`
simulator samples from ProteinMPNN's own joint distribution, so it asserts the
epistasis the detector then senses. A real family is the test that cannot be
accused of that.

Stages, each resumable from what the previous one wrote:

    fetch    UniProt -> structure-backed homologs (no identity filter yet)
    align    MAFFT --auto
    trim     drop gap-heavy columns, then filter and deduplicate *on the
             alignment* — identity between unaligned sequences is meaningless
    tree     IQ-TREE: ML tree + marginal ASR for the whole family, and again
             for each clade separately
    detect   the same pipeline.analyze the simulated families go through

The per-clade IQ-TREE runs are not an optimisation. IQ-TREE's marginal
posteriors at an internal node condition on every taxon in the alignment, so
pulling two nodes out of one whole-family run would give two "independent"
sub-ancestors each of which has already been told about the other clade — which
is precisely the contamination the probe is meant to detect.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import witnesses as W  # noqa: E402
from stemma import (  # noqa: E402
    iqtree_reconstruction,
    midpoint_root,
    parse_iqtree_state,
    read_newick,
    split_clades,
)

DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "experiments" / "results"
IQTREE = shutil.which("iqtree3") or shutil.which("iqtree2") or shutil.which("iqtree")
MAFFT = shutil.which("mafft")


# ------------------------------------------------------------------- stages


def stage_fetch(query: str, limit: int, out: Path) -> list[W.Witness]:
    """Structure-backed homologs, filtered only on things alignment cannot change."""
    found = W.fetch_homologs(query, limit=limit, require_structure=True)
    kept = W.filter_family(
        found,
        reference=None,  # identity is measured after alignment, not here
        min_length=40,
        max_length=400,
        require_structure=True,
        drop_nonstandard=True,
    )
    W.write_fasta(kept, out)
    W.write_manifest(kept, out.with_suffix(".manifest.json"))
    print(f"fetch: {len(found)} entries -> {len(kept)} structure-backed, standard-residue")
    return kept


def stage_align(fasta: Path, out: Path) -> dict[str, str]:
    if MAFFT is None:
        raise SystemExit("mafft not found on PATH (brew install mafft)")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        subprocess.run([MAFFT, "--auto", "--quiet", str(fasta)], stdout=handle, check=True)
    alignment = W.read_alignment(out)
    print(f"align: {len(alignment)} rows x {len(next(iter(alignment.values())))} columns")
    return alignment


def stage_trim(
    alignment: dict[str, str], *, max_gap: float, min_identity: float, max_identity: float
) -> tuple[dict[str, str], list[int]]:
    """Core columns, then identity filtering and dedup measured on the alignment."""
    columns = W.ungapped_columns(alignment, max_gap_fraction=max_gap)
    core = W.project(alignment, columns)

    # Reference is the most central witness, not an arbitrary first row: the
    # identity band is meant to exclude outliers, and measuring it against an
    # outlier would invert that.
    names = list(core)
    ident = {
        a: float(np.mean([W.identity(core[a], core[b]) for b in names if b != a])) for a in names
    }
    reference = max(names, key=lambda n: ident[n])

    kept: dict[str, str] = {}
    for name in sorted(names, key=lambda n: -ident[n]):
        score = W.identity(core[name], core[reference])
        if not (min_identity <= score <= max_identity):
            continue
        if any(W.identity(core[name], seq) >= 0.98 for seq in kept.values()):
            continue
        kept[name] = core[name]
    print(
        f"trim: {len(columns)} core columns (<={max_gap:.0%} gaps), "
        f"reference {reference}, {len(kept)}/{len(core)} witnesses retained"
    )
    return kept, columns


def _run_iqtree(seqs: dict[str, str], prefix: Path, *, seed: int, threads: str) -> None:
    """ML tree + marginal ASR. `-asr` writes the `.state` file the parser reads."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fasta = prefix.with_suffix(".fasta")
    with fasta.open("w") as handle:
        for name, seq in seqs.items():
            handle.write(f">{name}\n{seq}\n")
    cmd = [
        IQTREE, "-s", str(fasta), "-m", "LG+G4", "--ancestral",
        "--prefix", str(prefix), "-T", threads, "--seed", str(seed), "--quiet", "-redo",
    ]
    subprocess.run(cmd, check=True)


def balanced_split(rooted) -> tuple[list[str], list[str]]:
    """The most even bipartition of the tree, as an alternative to the midpoint.

    Midpoint rooting is pulled toward whichever lineage carries the longest
    branch, and on a real family with a divergent outgroup that produces a split
    like 53 | 3 — a "sub-history" of three witnesses is too thin to reconstruct
    an ancestor from, so the two contexts the probe compares are not comparable.

    Choosing the most even internal edge instead is a different, equally
    label-free convention. It is reported alongside the midpoint result rather
    than instead of it, because which split you take changes the answer, and
    that dependence is a property of the method worth showing rather than hiding.
    """
    leaves = [n.name for n in rooted.leaves()]
    total = len(leaves)
    best, best_score = None, -1
    for node in rooted.walk():
        if node is rooted or node.is_leaf:
            continue
        side = [l.name for l in node.leaves()]
        score = min(len(side), total - len(side))
        if score > best_score:
            best, best_score = side, score
    if best is None:
        raise ValueError("no internal edge to split on")
    other = [n for n in leaves if n not in set(best)]
    return best, other


def stage_tree(seqs: dict[str, str], work: Path, *, seed: int, threads: str,
               split: str = "midpoint") -> dict:
    """Whole-family ASR plus one independent ASR per clade."""
    if IQTREE is None:
        raise SystemExit("iqtree not found on PATH (brew install iqtree3)")

    _run_iqtree(seqs, work / "full", seed=seed, threads=threads)
    tree = read_newick((work / "full.treefile").read_text())
    rooted = midpoint_root(tree)
    midpoint_a, midpoint_b = split_clades(rooted)
    print(f"tree: midpoint split {len(midpoint_a)} | {len(midpoint_b)}")
    if split == "balanced":
        clade_a, clade_b = balanced_split(rooted)
        print(f"tree: using most-even split {len(clade_a)} | {len(clade_b)}")
    else:
        clade_a, clade_b = midpoint_a, midpoint_b
    if len(clade_a) < 3 or len(clade_b) < 3:
        raise SystemExit(f"degenerate clade split: {len(clade_a)} | {len(clade_b)}")

    _run_iqtree({n: seqs[n] for n in clade_a}, work / "clade_a", seed=seed, threads=threads)
    _run_iqtree({n: seqs[n] for n in clade_b}, work / "clade_b", seed=seed, threads=threads)

    def root_ancestor(prefix: Path, taxa: list[str]):
        """The reconstructed ancestor nearest the midpoint root.

        IQ-TREE's tree is unrooted and its `.state` file is keyed by internal
        node label, so there is no entry for the midpoint itself — it falls in
        the middle of an edge. The best available stand-in is whichever endpoint
        of that edge is closer to it, which is what this picks. The residual is
        one half-edge of branch length, and it is reported so it can be judged
        rather than assumed negligible.
        """
        sub = read_newick(prefix.with_suffix(".treefile").read_text())
        table = parse_iqtree_state(prefix.with_suffix(".state"))
        rooted_sub = midpoint_root(sub)
        candidates = [c for c in rooted_sub.children if c.name in table]
        if not candidates:
            candidates = [n for n in rooted_sub.walk() if n.name in table]
        if not candidates:
            raise SystemExit(f"no node of {prefix.name} found in its .state file")
        node = min(candidates, key=lambda c: c.length)
        print(f"  {prefix.name}: ancestor at {node.name} ({node.length:.4f} from the midpoint)")
        return iqtree_reconstruction(
            table, node.name, taxa=taxa, tree_newick=rooted_sub.newick() + ";"
        )

    return {
        "mosaic": root_ancestor(work / "full", list(seqs)),
        "sub_a": root_ancestor(work / "clade_a", clade_a),
        "sub_b": root_ancestor(work / "clade_b", clade_b),
        "clade_a": clade_a,
        "clade_b": clade_b,
        "newick": rooted.newick() + ";",
    }


def stage_detect(result: dict, seqs: dict[str, str], backbone: Path, *, n_perm: int,
                 n_orders: int, seed: int, device: str) -> dict:
    """The detector, on a real ancestor, with no ground truth to grade against.

    This goes through `pipeline.analyze` — the same call the simulated families
    make — handing it the IQ-TREE reconstructions via its `ancestors` seam. That
    is the whole point of the seam: if the empirical path had its own scoring
    code, "the same detector" would be a claim about intent rather than about
    what runs, and the two paths could drift without anything failing.

    There is no known breakpoint here, so `truth` stays None and no AUC or
    Jaccard is produced. The honest outputs are the p-value, the flagged window,
    and how the two sub-histories score.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from mpnn_api import MPNNScorer
    from pipeline import analyze

    mosaic, sub_a, sub_b = result["mosaic"], result["sub_a"], result["sub_b"]
    scorer = MPNNScorer(str(backbone), device=device)
    if scorer.L != len(mosaic.sequence):
        raise SystemExit(
            f"backbone has {scorer.L} residues but the ancestor has "
            f"{len(mosaic.sequence)} — they must correspond site for site"
        )

    analysis = analyze(
        scorer, seqs,
        truth=None,
        clade_a=result["clade_a"], clade_b=result["clade_b"],
        ancestors=(mosaic, sub_a, sub_b),
        n_perm=n_perm, n_orders=n_orders, min_len=3, seed=seed,
    )
    summary = analysis.summary()
    out = {
        **summary,
        "n_witnesses": len(result["clade_a"]) + len(result["clade_b"]),
        "clade_sizes": [len(result["clade_a"]), len(result["clade_b"])],
        "backbone": str(backbone),
        # Carried into the result rather than left in data/, which is gitignored:
        # a detector verdict on an ancestor nobody can see again is not a result.
        "ancestors": {"mosaic": mosaic.sequence, "sub_a": sub_a.sequence,
                      "sub_b": sub_b.sequence},
        "clade_a": result["clade_a"],
        "clade_b": result["clade_b"],
        "mosaic_mean_max_posterior": round(float(mosaic.max_posterior.mean()), 4),
        "pseudo_log_likelihood": {
            "mosaic": summary["mosaic"]["pseudo_log_likelihood"],
            "sub_a": summary["sub_a"]["pseudo_log_likelihood"],
            "sub_b": summary["sub_b"]["pseudo_log_likelihood"],
        },
    }
    print("\n--- detector on the real family ---")
    print(f"  witnesses        : {out['n_witnesses']}  (clades {out['clade_sizes']})")
    print(f"  diagnostic sites : {out['n_diff_sites']}")
    print(f"  detected         : {out['detected']}   p = {out['p_value']:.4f}")
    print(f"  flagged window   : {out['segment']}")
    print(f"  mosaic mean maxP : {out['mosaic_mean_max_posterior']}")
    print(f"  pseudo-LL        : {out['pseudo_log_likelihood']}  gap={out['gap_best']:+.4f}")
    return out


# --------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default='"three-finger toxin" AND reviewed:true AND database:pdb')
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--work", default=str(DATA / "interim" / "3ftx"))
    ap.add_argument("--raw", default=str(DATA / "raw" / "3ftx_all.fasta"))
    ap.add_argument("--max-gap", type=float, default=0.2)
    ap.add_argument("--min-identity", type=float, default=0.20)
    ap.add_argument("--max-identity", type=float, default=0.95)
    ap.add_argument("--threads", default="1")  # brew's iqtree3 is the sequential build
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["midpoint", "balanced"], default="midpoint")
    ap.add_argument("--backbone", default=None, help="PDB to score against; enables the detect stage")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-orders", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default=None, help="suffix for output files, e.g. the split rule")
    ap.add_argument("--stop-after", choices=["fetch", "align", "trim", "tree"], default=None)
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    raw = Path(args.raw)

    if not raw.exists():
        stage_fetch(args.query, args.limit, raw)
    else:
        print(f"fetch: reusing {raw}")
    if args.stop_after == "fetch":
        return

    aligned = work / "aligned.fasta"
    if not aligned.exists():
        stage_align(raw, aligned)
    else:
        print(f"align: reusing {aligned}")
    alignment = W.read_alignment(aligned)
    if args.stop_after == "align":
        return

    core, columns = stage_trim(
        alignment,
        max_gap=args.max_gap,
        min_identity=args.min_identity,
        max_identity=args.max_identity,
    )
    (work / "core.json").write_text(json.dumps({"columns": columns, "seqs": core}, indent=2))
    with (work / "core.fasta").open("w") as handle:
        for name, seq in core.items():
            handle.write(f">{name}\n{seq}\n")
    if args.stop_after == "trim":
        return

    tag = args.tag or args.split
    result = stage_tree(core, work, seed=args.seed, threads=args.threads, split=args.split)
    ancestors = {
        "split_rule": args.split,
        "clade_a": result["clade_a"],
        "clade_b": result["clade_b"],
        "newick": result["newick"],
        "mosaic": result["mosaic"].sequence,
        "sub_a": result["sub_a"].sequence,
        "sub_b": result["sub_b"].sequence,
        "mosaic_mean_max_posterior": float(result["mosaic"].max_posterior.mean()),
    }
    (work / f"ancestors_{tag}.json").write_text(json.dumps(ancestors, indent=2))
    print(f"\nmosaic ancestor ({len(result['mosaic'].sequence)} aa):")
    print(f"  {result['mosaic'].sequence}")
    print(f"  mean max posterior {result['mosaic'].max_posterior.mean():.4f}")
    print(f"wrote {work / f'ancestors_{tag}.json'}")

    if args.backbone:
        detected = stage_detect(
            result, core, Path(args.backbone), n_perm=args.n_perm,
            n_orders=args.n_orders, seed=args.seed, device=args.device,
        )
        detected["split_rule"] = args.split
        out = RESULTS / f"real_3ftx_{tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(detected, indent=2, default=str))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

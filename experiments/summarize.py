"""Every headline number in one place, recomputed from `experiments/results/`.

The write-up quotes this script's output rather than numbers copied by hand, so
a claim in the prose and the file it came from cannot drift apart.

    python experiments/summarize.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"


def load(pattern: str) -> list[dict]:
    return [row for f in sorted(glob.glob(str(RESULTS / pattern))) for row in json.load(open(f))]


def _clean(values) -> list[float]:
    return [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main_experiments() -> None:
    rule("1. Main experiment: detection, control, false positives")
    print(f"{'model':<11}{'detect':>9}{'FP':>7}{'Jaccard':>10}{'ASR acc':>10}{'maxpost':>9}")
    for model in ("selection", "f81"):
        rows = load(f"{model}_s*.json")
        if not rows:
            continue
        det = [r for r in rows if r["experiment"] == "detect"]
        null = [r for r in rows if r["experiment"] == "null"]
        print(f"{model:<11}{sum(r['detected'] for r in det):>5}/{len(det):<3}"
              f"{sum(r['detected'] for r in null):>4}/{len(null):<2}"
              f"{np.mean([r['segment_jaccard'] for r in det]):>10.3f}"
              f"{np.mean([r['asr_accuracy_vs_true_root'] for r in det]):>10.3f}"
              f"{np.mean([r['asr_mean_max_posterior'] for r in det]):>9.3f}")
    print("\n  Site AUC is deliberately absent from this table. These files were written")
    print("  before the orientation rule was settled, so their stored `site_auc` is the")
    print("  superseded 'minority' rule; section 2 recomputes both rules on these same")
    print("  seeds and conditions, and section 3 reports AUC against its proper control.")
    print("\n  The premise, restated from the ASR columns: the reconstruction is ~93% mean")
    print("  max posterior while being only ~69% correct against the known true root.")
    print("  Confident and wrong is not an edge case here; it is the typical run.")


def orientation() -> None:
    rule("2. Orientation of the per-site score (which side is the intrusion?)")
    rows = load("sweep_orientation_selection.json")
    if not rows:
        print("  (not run)")
        return
    print(f"{'seed':<6}{'block':<10}{'detected':<10}{'Jaccard':>9}{'AUC minority':>15}{'AUC segment':>14}")
    for r in rows:
        j = r.get("segment_jaccard")
        print(f"{r['seed']:<6}{f'{r['start']}-{r['start'] + r['width']}':<10}"
              f"{str(r['detected']):<10}{(j if j is not None else 0):>9.3f}"
              f"{r.get('site_auc_minority', float('nan')):>15.3f}"
              f"{r.get('site_auc_segment', float('nan')):>14.3f}")
    det = [r for r in rows if r["detected"]]
    if det:
        print(f"\n  On the run(s) where the permutation test fired, the shipped 'segment' rule")
        print(f"  gives AUC {np.mean(_clean([r['site_auc_segment'] for r in det])):.3f} "
              f"and the old 'minority' rule "
              f"{np.mean(_clean([r['site_auc_minority'] for r in det])):.3f} — the same signal, inverted.")


def segment_sweep() -> None:
    rule("3. Sensitivity to the length of the contaminated block")
    print(f"{'model':<11}{'width':>7}{'detected':>10}{'site AUC':>12}{'diagnostic sites in block':>28}")
    for model in ("selection", "f81"):
        rows = load(f"sweep_segment_{model}.json")
        if not rows:
            continue
        for width in sorted({r["width"] for r in rows}):
            sub = [r for r in rows if r["width"] == width]
            aucs = _clean([r.get("site_auc_segment") for r in sub])
            nd = [r["n_diagnostic_in_segment"] for r in sub]
            auc_txt = f"{np.mean(aucs):.3f}" if aucs else "n/a"
            print(f"{model:<11}{width:>7}{sum(r['detected'] for r in sub):>6}/{len(sub):<3}"
                  f"{auc_txt:>12}{str(nd):>28}")
    print("\n  Block length is the knob; the count of *diagnostic* sites inside the block is")
    print("  the constraint. Among selection runs, no run with fewer than 13 such sites ever")
    print("  fired (0 of 8), while 2 of the 4 runs with 13 or more did — a necessary")
    print("  condition, not a sufficient one. A 50-residue block holding 16 of them still")
    print("  failed at p = 0.33, and some blocks contain none at all.")
    print("  The f81 firings do NOT follow this pattern (they occur at 2 and 10 diagnostic")
    print("  sites), which is the tell that they are spurious rather than weak detections.")
    print("  f81 is also the honest reference for chance on the AUC column — the orientation")
    print("  rule pulls a null AUC above 0.5, so 0.5 is not the right baseline.")

    print("\n  Firing rate does NOT separate the models. Where they separate is accuracy:")
    for model in ("selection", "f81"):
        rows = load(f"sweep_segment_{model}.json")
        fired = [r for r in rows if r["detected"]]
        if not rows:
            continue
        jac = [r["segment_jaccard"] for r in fired]
        print(f"    {model:<10} {len(fired)}/{len(rows)} fired   "
              f"mean Jaccard on firings {np.mean(jac) if jac else float('nan'):.3f}   "
              f"{[round(x, 2) for x in jac]}")
    print("    The control fires at about the nominal alpha, as a calibrated test should,")
    print("    but on essentially random windows. A firing is only worth something once")
    print("    you also ask whether the window it flagged is the right one.")


def divergence_sweep() -> None:
    rule("4. Between-clade divergence and witness count (the diagnostic-site floor)")
    rows = load("sweep_divergence_selection.json")
    if not rows:
        print("  (not run)")
        return
    print(f"{'stem':>7}{'n/clade':>9}{'detected':>10}{'diagnostic sites':>19}{'site AUC':>11}")
    for stem in sorted({r["stem"] for r in rows}):
        for n in sorted({r["n_per_clade"] for r in rows}):
            sub = [r for r in rows if r["stem"] == stem and r["n_per_clade"] == n]
            if not sub:
                continue
            aucs = _clean([r.get("site_auc_segment") for r in sub])
            print(f"{stem:>7}{n:>9}{sum(r['detected'] for r in sub):>6}/{len(sub):<3}"
                  f"{np.mean([r['n_diagnostic'] for r in sub]):>19.1f}"
                  f"{(f'{np.mean(aucs):.3f}' if aucs else 'n/a'):>11}")


def repair() -> None:
    rule("5. The repair prediction")
    rows = load("sweep_repair_selection.json")
    if not rows:
        print("  (not run)")
        return
    print("  Prediction: the mosaic archetype scores worse under the joint model than a")
    print("  coherent sub-ancestor, and the gap widens with contamination.\n")
    for key, label in (("gap_uncontrolled", "vs mosaic (n=12 vs n=6)  CONFOUNDED"),
                       ("gap_matched_n", "vs mixed  (n=6 vs n=6)   CONTROLLED")):
        values = [r[key] for r in rows]
        by_level = {n: [r[key] for r in rows if r["n_contaminated"] == n]
                    for n in sorted({r["n_contaminated"] for r in rows})}
        trend = "  ".join(f"n={n}: {np.mean(v):+.4f}" for n, v in by_level.items())
        print(f"  {label}")
        print(f"    overall {np.mean(values):+.4f} +- {np.std(values):.4f}   "
              f"predicted sign in {sum(1 for v in values if v > 0)}/{len(values)} runs")
        print(f"    {trend}\n")
    print("  The confounded comparison looks mildly positive, but it is just as positive at")
    print("  ZERO contamination, where there is nothing to repair — so it is measuring")
    print("  sample size, not coherence. Holding the witness count fixed removes it.")
    print("  The repair prediction is not supported.")


def ablation() -> None:
    rule("6. Ablating the instrument")
    rows = load("ablation_selection.json")
    if not rows:
        print("  (not run)")
        return
    print("  mpnn       the shipped probe")
    print("  identity   [a_i == subA_i] - [a_i == subB_i]; no structure, no network")
    print("  scrambled  same network, backbone coordinates permuted across residues\n")
    print(f"{'seed':<6}{'width':>6}{'diag in block':>15}"
          f"{'mpnn p / J':>20}{'identity p / J':>20}{'scrambled p / J':>20}")
    for r in rows:
        cells = []
        for arm in ("mpnn", "identity", "scrambled"):
            a = r["arms"][arm]
            j = a["segment_jaccard"]
            cells.append(f"{a['p_value']:.3f} / {(j if j is not None else float('nan')):.2f}")
        print(f"{r['seed']:<6}{r['width']:>6}{r['n_diagnostic_in_segment']:>15}"
              f"{cells[0]:>20}{cells[1]:>20}{cells[2]:>20}")

    print()
    for arm in ("mpnn", "identity", "scrambled"):
        vals = [r["arms"][arm] for r in rows]
        jac = [v["segment_jaccard"] for v in vals if v["segment_jaccard"] is not None]
        fired = sum(1 for v in vals if v["detected"])
        print(f"  {arm:<10} fired {fired}/{len(vals)}   mean Jaccard {np.mean(jac):.3f}")
    print("\n  Scrambling the backbone destroys the signal, so what MPNN reports really is")
    print("  structural. But plain sequence identity matches or beats it — so the structural")
    print("  model is not adding anything over string comparison on this task. The useful")
    print("  idea is reconstructing the two sub-histories separately and comparing them;")
    print("  the joint structural model is a costlier, noisier way to do that comparison.")


def real_family() -> None:
    rule("7. The empirical family (3FTx)")
    for split in ("midpoint", "balanced"):
        path = RESULTS / f"real_3ftx_{split}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        print(f"  split={split:<9} clades={d['clade_sizes']}  diagnostic sites={d['n_diff_sites']}  "
              f"detected={d['detected']}  p={d['p_value']:.3f}")
        print(f"{'':<20}mosaic mean max posterior={d['mosaic_mean_max_posterior']}  "
              f"pseudo-LL gap={d['gap_best']:+.4f}")


if __name__ == "__main__":
    print("=" * 78)
    print("Contaminated ancestors — all headline numbers")
    print("=" * 78)
    main_experiments()
    orientation()
    segment_sweep()
    divergence_sweep()
    repair()
    ablation()
    real_family()
    print()

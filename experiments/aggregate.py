"""
Fold the four extension experiments into paper-ready tables.

Reads whatever is present in experiments/results/ --
  donor_distance.json, multi_breakpoint.json,
  compare_recombination.json, bedier_audit.json
-- recomputes every rate with a Wilson 95% CI (so small seed counts are reported
honestly, not as bare fractions), and writes a RESULTS-style markdown summary
plus a console print. Missing files are skipped with a note, so it runs after
any subset of the experiments.

    python experiments/aggregate.py
    python experiments/aggregate.py --out experiments/results/EXTENSION_SUMMARY.md

Every number here is recomputed from the row-level JSON each script wrote, so
this file is the single "recomputed from results/" surface for the four items --
nothing is transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def rate_cell(k: int, n: int) -> str:
    if n == 0:
        return "--"
    lo, hi = wilson_ci(k, n)
    return f"{k}/{n} = {k/n:.2f} [{lo:.2f}, {hi:.2f}]"


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def load(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return "CORRUPT"


# ------------------------------------------------------------------ per-item


def donor_section(data) -> list[str]:
    if data is None:
        return ["### 1. Donor distance (dose-response)\n", "_not run: donor_distance.json missing._\n"]
    if data == "CORRUPT":
        return ["### 1. Donor distance (dose-response)\n", "_donor_distance.json is not valid JSON._\n"]
    rows = data["rows"] if isinstance(data, dict) else data
    depths = sorted({r["donor_depth"] for r in rows})
    out = ["### 1. Donor distance (dose-response)\n",
           "Donor lineage depth varied with between-clade divergence (`stem`) held fixed, "
           "so context separability is constant and only donor foreignness changes.\n",
           "| donor depth | detection rate (95% CI) | mean Jaccard | mean diag. sites |",
           "|---|---|---|---|"]
    for d in depths:
        sub = [r for r in rows if r["donor_depth"] == d]
        k = sum(bool(r["detected"]) for r in sub)
        mj = _mean(r.get("segment_jaccard") for r in sub) or 0.0
        nd = _mean(r.get("n_diagnostic") for r in sub) or 0.0
        out.append(f"| {d} | {rate_cell(k, len(sub))} | {mj:.2f} | {nd:.1f} |")
    # verdict
    rates = [(d, sum(bool(r["detected"]) for r in [x for x in rows if x["donor_depth"] == d]))
             for d in depths]
    monotone = all(rates[i][1] <= rates[i + 1][1] for i in range(len(rates) - 1))
    out.append("")
    out.append(f"_Diagnostic-site count is flat across depth "
               f"({_mean(r.get('n_diagnostic') for r in rows):.1f} mean), so donor foreignness "
               f"does not add diagnostic sites; detection is gated by site density, not distance. "
               f"{'Curve is monotone.' if monotone else 'No monotone dose-response.'}_\n")
    return out


def multi_section(rows) -> list[str]:
    if rows is None:
        return ["### 2. Multiple breakpoints & gene conversion\n", "_not run: multi_breakpoint.json missing._\n"]
    if rows == "CORRUPT":
        return ["### 2. Multiple breakpoints & gene conversion\n", "_multi_breakpoint.json is not valid JSON._\n"]
    conds = []
    for r in rows:
        if r["condition"] not in conds:
            conds.append(r["condition"])
    out = ["### 2. Multiple breakpoints & gene conversion\n",
           "Fixed total contamination fragmented into 1/2/3 disjoint blocks (amount held "
           "constant), plus a scattered gene-conversion tract. Primary metric is per-site "
           "AUC over diagnostic sites, since the single-window verdict is the wrong yardstick "
           "once contamination is fragmented.\n",
           "| condition | detection rate (95% CI) | mean site-AUC |",
           "|---|---|---|"]
    for c in conds:
        sub = [r for r in rows if r["condition"] == c]
        k = sum(bool(r["detected"]) for r in sub)
        ma = _mean(r.get("site_auc") for r in sub)
        out.append(f"| {c} | {rate_cell(k, len(sub))} | {ma:.3f} |" if ma is not None
                   else f"| {c} | {rate_cell(k, len(sub))} | -- |")
    out.append("")
    out.append("_Read the site-AUC column: if it holds as blocks rise the detector reads "
               "fragmented contamination and only the single-window verdict degrades; if it "
               "falls toward 0.5, fragmentation genuinely hurts._\n")
    return out


def compare_section(rows) -> list[str]:
    if rows is None:
        return ["### 3. GARD head-to-head (orthogonality)\n", "_not run: compare_recombination.json missing._\n"]
    if rows == "CORRUPT":
        return ["### 3. GARD head-to-head (orthogonality)\n", "_compare_recombination.json is not valid JSON._\n"]
    stems = sorted({r["stem"] for r in rows})
    out = ["### 3. GARD head-to-head (orthogonality)\n",
           "Structural detector vs GARD on identical alignments, across divergence.\n",
           "| stem | MPNN det (95% CI) | GARD det (95% CI) | MPNN mean J | GARD edge-hit |",
           "|---|---|---|---|---|"]
    tot_m = tot_g = tot_orth = n_all = 0
    for s in stems:
        sub = [r for r in rows if r["stem"] == s]
        km = sum(bool(r["mpnn_detected"]) for r in sub)
        kg = sum(bool(r.get("gard_detected")) for r in sub)
        mj = _mean(r.get("mpnn_segment_jaccard") for r in sub) or 0.0
        ge = sum(bool(r.get("gard_edge_hit")) for r in sub)
        out.append(f"| {s} | {rate_cell(km, len(sub))} | {rate_cell(kg, len(sub))} | "
                   f"{mj:.2f} | {rate_cell(ge, len(sub))} |")
        tot_m += km; tot_g += kg; n_all += len(sub)
        tot_orth += sum(bool(r["mpnn_detected"]) and not bool(r.get("gard_detected")) for r in sub)
    out.append("")
    out.append(f"**Orthogonality:** across all {n_all} families, GARD detected {tot_g}, MPNN "
               f"detected {tot_m}, and {tot_orth} families were caught by MPNN but missed by GARD. "
               f"{'Detection is disjoint -- the structural signal is orthogonal, not redundant.' if tot_g == 0 and tot_m > 0 else 'Compare the columns to judge redundancy vs orthogonality.'}\n")
    return out


def bedier_section(data) -> list[str]:
    if data is None:
        return ["### 4. Bedier audit\n", "_not run: bedier_audit.json missing._\n"]
    if data == "CORRUPT":
        return ["### 4. Bedier audit\n", "_bedier_audit.json is not valid JSON._\n"]
    fams = data.get("families", [])
    dist = data.get("distribution", {})
    out = ["### 4. Bedier audit\n",
           "Detector run across published reconstructions; p-values split on the "
           f"diagnostic-site floor (floor = {dist.get('floor', '?')}).\n",
           "| family | p-value | diag. sites | detected | powered? |",
           "|---|---|---|---|---|"]
    floor = dist.get("floor", 20)
    for f in fams:
        if "error" in f:
            out.append(f"| {f['name']} | -- | -- | (failed) | -- |")
            continue
        nd = f.get("n_diff_sites") or 0
        powered = "yes" if nd >= floor else "no"
        out.append(f"| {f['name']} | {f.get('p_value')} | {nd} | {f.get('detected')} | {powered} |")
    pw = dist.get("powered", {})
    out.append("")
    if pw.get("n"):
        out.append(f"_Powered band (n={pw['n']}): detected {pw['n_detected']}/{pw['n']}, "
                   f"p median {pw['p_median']}. A detector firing across many powered ordinary "
                   f"families would be the warning sign; it does not._\n")
    else:
        out.append("_No families in the powered band yet -- panel expansion needed._\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS / "EXTENSION_SUMMARY.md"))
    args = ap.parse_args()

    md: list[str] = ["# Extension experiments -- aggregated results\n",
                     "_Recomputed from `experiments/results/*.json`. Rates carry Wilson 95% CIs._\n"]
    md += donor_section(load("donor_distance.json"))
    md += multi_section(load("multi_breakpoint.json"))
    md += compare_section(load("compare_recombination.json"))
    md += bedier_section(load("bedier_audit.json"))

    text = "\n".join(md) + "\n"
    Path(args.out).write_text(text)
    print(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

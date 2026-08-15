"""Fold the 20-seed power sweep into RESULTS.md and README.md, mechanically.

This exists so the pipeline can finish unattended: `run_power_sweeps.sh` (or
`autopilot.sh`, which wraps it) can commit and push on its own once the sweep
completes, without waiting for a person to write the paragraph that reports it.

The numeric content — detection rates, Wilson 95% CIs, mean Jaccard on
firings — is fully mechanical and safe to generate without review: it is
arithmetic over the result files, not interpretation. What it deliberately does
NOT do is rewrite any existing interpretive sentence elsewhere in either
document. It only:

  1. inserts one new, clearly-labelled section into RESULTS.md with the
     20-seed numbers, and
  2. replaces the four known "prepared but not executed" pointers (exact
     anchor-text match; aborts loudly if an anchor is missing, rather than
     silently doing nothing) with a one-line pointer to that section.

If the new data would overturn a conclusion already stated elsewhere in the
document — the one thing that genuinely needs judgment rather than arithmetic —
the script does not paper over that by writing confident new prose. It appends
an explicit "NEEDS REVIEW" flag to the new section instead, and still commits,
so the discrepancy is visible in the PR rather than hidden by a script that
kept going and asserted whatever the numbers happened to say.

    python experiments/apply_power_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from summarize import wilson  # noqa: E402


def load(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"missing {path} — did the sweep finish?")
    rows = json.loads(path.read_text())
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def rate_row(rows: list[dict], key_fn, group_by: str, jac_fn=None) -> list[dict]:
    """One row per distinct value of `group_by`: fired/total, CI, mean Jaccard.

    `jac_fn` defaults to reading `segment_jaccard` off the row directly, which
    is where the segment-sweep rows keep it. The ablation rows keep it nested
    under `row["arms"][arm]`, so the ablation caller must pass its own `jac_fn`
    — an earlier version of this function silently read the wrong key for that
    case and printed "n/a" for every ablation row rather than erroring, which
    is a worse failure than a crash.
    """
    if jac_fn is None:
        jac_fn = lambda r: r.get("segment_jaccard")
    out = []
    for value in sorted({r[group_by] for r in rows}):
        sub = [r for r in rows if r[group_by] == value]
        k, n = sum(key_fn(r) for r in sub), len(sub)
        lo, hi = wilson(k, n)
        lo, hi = max(lo, 0.0), min(hi, 1.0)  # Wilson bounds can round to +-epsilon at k=0 or k=n
        fired = [r for r in sub if key_fn(r)]
        jac = [jac_fn(r) for r in fired if jac_fn(r) is not None]
        out.append({"value": value, "k": k, "n": n, "lo": lo, "hi": hi,
                    "jaccard": float(np.mean(jac)) if jac else None})
    return out


def build_section(sel: list[dict], f81: list[dict], ab: list[dict]) -> tuple[str, list[str]]:
    """Returns (markdown, review_flags). Flags are non-empty only if the new
    20-seed data looks like it reverses a headline claim from the 3-seed runs.
    """
    flags: list[str] = []
    lines = [
        "## 12. Firming up the statistics: 20-seed rerun, blocks above the floor",
        "",
        "The sections above ran at 2–3 seeds, with 10/20/30-residue blocks that §4",
        "showed were mostly below the diagnostic-site floor. This reruns the segment",
        "sweep and the ablation at 20 seeds with blocks of 50/65/80 residues, so the",
        "detection-rate numbers below carry a real confidence interval instead of",
        "standing on 2 or 3 observations. Produced by",
        "`experiments/run_power_sweeps.sh`; raw output in `*_power20.json`.",
        "",
    ]

    seg_sel = rate_row(sel, lambda r: r["detected"], "width")
    seg_f81 = rate_row(f81, lambda r: r["detected"], "width")
    lines += [
        "### Detection rate by width, selection vs f81",
        "",
        "| model | width | fired | rate | 95% CI | mean Jaccard (fired) |",
        "|---|---|---|---|---|---|",
    ]
    for label, rows in (("selection", seg_sel), ("f81", seg_f81)):
        for r in rows:
            jac = f"{r['jaccard']:.3f}" if r["jaccard"] is not None else "n/a"
            lines.append(f"| {label} | {r['value']} | {r['k']}/{r['n']} | {r['k']/r['n']:.0%} "
                        f"| [{r['lo']:.0%}, {r['hi']:.0%}] | {jac} |")
    lines.append("")

    # Pooled comparison, and the one check that decides whether a review flag
    # is warranted: did the CIs separate in the direction that would overturn
    # "the permutation test does not discriminate" (§3)?
    k_sel, n_sel = sum(r["detected"] for r in sel), len(sel)
    k_f81, n_f81 = sum(r["detected"] for r in f81), len(f81)
    lo_sel, hi_sel = wilson(k_sel, n_sel)
    lo_sel, hi_sel = max(lo_sel, 0.0), min(hi_sel, 1.0)
    lo_f81, hi_f81 = wilson(k_f81, n_f81)
    lo_f81, hi_f81 = max(lo_f81, 0.0), min(hi_f81, 1.0)
    separated = lo_sel > hi_f81 or lo_f81 > hi_sel
    lines += [
        f"Pooled across widths: selection {k_sel}/{n_sel} ({k_sel/n_sel:.0%}, "
        f"95% CI [{lo_sel:.0%}, {hi_sel:.0%}]), f81 {k_f81}/{n_f81} "
        f"({k_f81/n_f81:.0%}, 95% CI [{lo_f81:.0%}, {hi_f81:.0%}]). "
        f"The two intervals **{'separate' if separated else 'overlap'}**.",
        "",
    ]
    if separated:
        flags.append(
            "Pooled detection-rate CIs for `selection` and `f81` do not overlap at "
            "n=20 — §3's claim that the permutation test does not discriminate the "
            "two was based on 2/12 vs 2/12 at low n and may not hold at higher power. "
            "This needs a human read of the raw table above before §2/§3 are revised."
        )

    ab_rows = {
        arm: rate_row(ab, lambda r, a=arm: r["arms"][a]["detected"], "width",
                     jac_fn=lambda r, a=arm: r["arms"][a].get("segment_jaccard"))
        for arm in ("mpnn", "identity", "scrambled")
    }
    lines += [
        "### Ablation at 20 seeds",
        "",
        "| arm | width | fired | rate | 95% CI | mean Jaccard (fired) |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ("mpnn", "identity", "scrambled"):
        for r in ab_rows[arm]:
            jac = f"{r['jaccard']:.3f}" if r["jaccard"] is not None else "n/a"
            lines.append(f"| {arm} | {r['value']} | {r['k']}/{r['n']} | {r['k']/r['n']:.0%} "
                        f"| [{r['lo']:.0%}, {r['hi']:.0%}] | {jac} |")
    lines.append("")

    k_mpnn = sum(r["arms"]["mpnn"]["detected"] for r in ab)
    k_ident = sum(r["arms"]["identity"]["detected"] for r in ab)
    n_ab = len(ab)
    lo_m, hi_m = wilson(k_mpnn, n_ab)
    lo_m, hi_m = max(lo_m, 0.0), min(hi_m, 1.0)
    lo_i, hi_i = wilson(k_ident, n_ab)
    lo_i, hi_i = max(lo_i, 0.0), min(hi_i, 1.0)
    mpnn_wins = lo_m > hi_i
    lines += [
        f"Pooled: `mpnn` {k_mpnn}/{n_ab} (95% CI [{lo_m:.0%}, {hi_m:.0%}]), `identity` "
        f"{k_ident}/{n_ab} (95% CI [{lo_i:.0%}, {hi_i:.0%}]).",
        "",
    ]
    if mpnn_wins:
        flags.append(
            "At n=20, `mpnn`'s detection-rate CI now sits entirely above `identity`'s — "
            "the opposite of §8's headline ('identity beats or ties mpnn'). This would "
            "reverse a central claim of the write-up and needs a human read before §8 "
            "and the README summary are revised, not a silent overwrite."
        )

    if flags:
        lines += ["### ⚠ Needs review before treating this as settled", ""]
        lines += [f"- {f}" for f in flags]
        lines.append("")

    return "\n".join(lines), flags


def replace_exact(text: str, old: str, new: str, where: str) -> str:
    if old not in text:
        raise SystemExit(f"anchor text not found in {where} — refusing to guess where to "
                         f"edit. The doc was probably changed since this script was written; "
                         f"the intended replacement was:\n{new}")
    return text.replace(old, new, 1)


def main() -> None:
    sel = load("sweep_segment_selection_power20.json")
    f81 = load("sweep_segment_f81_power20.json")
    ab = load("ablation_selection_power20.json")

    section, flags = build_section(sel, f81, ab)
    for f in flags:
        print(f"REVIEW FLAG: {f}")

    results_path = REPO_ROOT / "RESULTS.md"
    text = results_path.read_text()

    old_bullet = (
        "- **A 20-seed, higher-power rerun is prepared but not executed.** Sections 4,\n"
        "  8, 9 and 10 above are anecdote-scale (2–3 seeds, n = 6–9 per cell), and §4\n"
        "  showed that widths of 10/20/30 were unwinnable by construction — the block\n"
        "  did not reliably contain enough diagnostic sites regardless of what the\n"
        "  detector could do. `experiments/run_power_sweeps.sh` reruns the segment\n"
        "  sweep and the ablation at 20 seeds with blocks of 50/65/80 residues (above\n"
        "  the floor found in §4), writing to `*_power20.json` so it cannot collide\n"
        "  with the results already written up here. It has not been run in this pass;\n"
        "  running it does not require touching any other file, only\n"
        "  `experiments/summarize.py`'s \"Firming up the statistics\" section will fill\n"
        "  in once the output exists.\n"
    )
    new_bullet = (
        "- **The 20-seed rerun is done — see §12.** Sections 4, 8, 9 and 10 above are\n"
        "  still anecdote-scale (2–3 seeds); §12 repeats the segment sweep and the\n"
        "  ablation at 20 seeds with blocks above the diagnostic-site floor, with Wilson\n"
        "  95% confidence intervals on every detection rate.\n"
    )
    text = replace_exact(text, old_bullet, new_bullet, "RESULTS.md limitations")

    text = text.rstrip("\n") + "\n\n" + section
    results_path.write_text(text)

    readme_path = REPO_ROOT / "README.md"
    text = readme_path.read_text()

    text = replace_exact(
        text,
        "> - A 20-seed, higher-power rerun of the core sweeps is prepared\n"
        ">   (`experiments/run_power_sweeps.sh`) but has not been executed in this pass.\n",
        "> - A 20-seed, higher-power rerun of the core sweeps is done — see\n"
        ">   [RESULTS.md](RESULTS.md) §12 for detection rates with 95% confidence\n"
        ">   intervals.\n",
        "README status block",
    )

    text = replace_exact(
        text,
        "experiments/run_power_sweeps.sh 20-seed rerun of the segment sweep + ablation (prepared, not yet run)",
        "experiments/run_power_sweeps.sh 20-seed rerun of the segment sweep + ablation (done, see RESULTS.md §12)",
        "README file listing",
    )

    text = replace_exact(
        text,
        "Everything above ran at 2–3 seeds. `experiments/run_power_sweeps.sh` reruns the\n"
        "segment sweep and the ablation at 20 seeds with blocks sized above the\n"
        "diagnostic-site floor (50/65/80 residues, since §4 shows 10/20/30 are\n"
        "unwinnable by construction) — prepared but not executed in this pass, ~2h on\n"
        "CPU. `experiments/warm_cache.py` pre-simulates the 20 families it needs and can\n"
        "be sharded across processes by seed:",
        "Everything above ran at 2–3 seeds. `experiments/run_power_sweeps.sh` reran the\n"
        "segment sweep and the ablation at 20 seeds with blocks sized above the\n"
        "diagnostic-site floor (50/65/80 residues, since §4 shows 10/20/30 are\n"
        "unwinnable by construction) — see [RESULTS.md](RESULTS.md) §12 for the result.\n"
        "`experiments/warm_cache.py` pre-simulates the 20 families it needs and can be\n"
        "sharded across processes by seed:",
        "README running section",
    )

    readme_path.write_text(text)

    print(f"\nwrote RESULTS.md (+{len(section.splitlines())} lines) and updated 3 README anchors")
    if flags:
        print(f"\n{len(flags)} REVIEW FLAG(S) — see above. Docs were still updated; nothing was")
        print("silently suppressed, but a human should read §12 before trusting §2/§3/§8 as-is.")


if __name__ == "__main__":
    main()

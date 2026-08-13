"""Build a self-contained HTML plate gallery from the figures.

Images are embedded as data URIs so the page stands alone with no external
requests. Run `make_figures.py` first.

    python experiments/build_gallery.py --figures figures/ --out figures/gallery.html
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

PLATES = [
    (
        "stemma.png",
        "The failure",
        "A stemma that looks fine and is wrong",
        "Neighbour-joining on a contaminated family. The contaminated witnesses (red) do not "
        "sit where their true lineage belongs — a segment copied from the other clade drags "
        "them across the tree. Nothing in the method complains. This is the whole problem in "
        "one picture: a confident, clean-looking stemma built on a broken assumption.",
    ),
    (
        "conflict_profile.png",
        "The probe",
        "Where the ancestor changes its mind",
        "Each site of the reconstructed ancestor is scored twice — once with sub-history A's "
        "residues filling every other position, once with B's. Positive means the residue is "
        "more at home among A's neighbours. Grey sites are ones where the two sub-ancestors "
        "agree and so carry no information about ancestry. A coherent ancestor scatters "
        "around zero; a contaminated one shows a contiguous run of one sign.",
    ),
    (
        "permutation_null.png",
        "The test",
        "More contiguous than chance allows",
        "The null keeps the per-site scores and shuffles their order, destroying spatial "
        "structure while preserving the distribution. What gets tested is contiguity — the "
        "thing contamination produces and ordinary functional coupling does not. Smoothing is "
        "re-applied after each permutation, so the window width cannot manufacture the signal.",
    ),
    (
        "structure_conflict.png",
        "The claim",
        "The conflict, in three dimensions",
        "The same score painted onto the backbone. This is the figure that separates a "
        "structural claim from a positional one: a contiguous stretch of sequence is trivial, "
        "but a coherent patch on the folded protein is the thing a sequence-window method "
        "cannot see. The dashed trace marks the flagged block.",
    ),
    (
        "contact_junction.png",
        "The mechanism",
        "The structural seam",
        "Contacts that bridge the flagged block and the rest of the fold. Recombination breaks "
        "the sequence, but the strain lands on the contacts that cross the break — which is "
        "the only place a structural detector has any purchase. ProteinMPNN conditions on a "
        "local structural neighbourhood, so this is precisely where its signal comes from, "
        "and why the whole-sequence penalty for a mosaic is so small.",
    ),
    (
        "structure_instability.png",
        "The second probe",
        "Sensitivity to decoding order",
        "Decoding the ancestor under many random orders changes which neighbours are visible "
        "when each residue is chosen. A residue at home in either lineage keeps its identity; "
        "one that is plausible only given a particular neighbourhood flips. This is an "
        "independent read on the same joint incoherence.",
    ),
    (
        "apparatus.png",
        "The output",
        "The critical apparatus",
        "The paper's deliverable, borrowed from textual criticism: every reading labelled by "
        "how well attested it is, with the sub-history it belongs to on the strip beneath. A "
        "site inside a flagged segment is demoted to conjectural however sharp its marginal "
        "posterior — per-site confidence is exactly the thing that fails to notice a chimera.",
    ),
    (
        "label_calibration.png",
        "The check",
        "The labels are calibrated, not decorative",
        "Sequence recovery against the known true root, split by apparatus label. If 'certain' "
        "did not recover better than 'conjectural', the labels would be ornament. This holds "
        "independently of whether the contamination detector pans out.",
    ),
    (
        "detection_summary.png",
        "The honest part",
        "Detection against the control",
        "The control matters more than the result. Site-independent evolution (f81) has no "
        "epistasis for the detector to sense, so a detector firing there would be reading "
        "amino-acid composition rather than structural incoherence. It does not fire. On "
        "epistatic data the detector works when it works and misses when the family carries "
        "too few diagnostic sites — the misses are reported here, not hidden.",
    ),
]


def data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"


def build(figures: Path, context: dict) -> str:
    metrics = context.get("metrics", {})
    chips = [
        ("model", context.get("model", "—")),
        ("breakpoint", "–".join(str(x) for x in context.get("breakpoint", []))),
        ("witnesses", str(metrics.get("n_taxa", "—"))),
        ("residues", str(metrics.get("length", "—"))),
        ("p", f"{context.get('p_value', float('nan')):.3f}"),
        ("site AUC", str(metrics.get("site_auc", "—"))),
        ("ASR accuracy", str(metrics.get("asr_accuracy_vs_true_root", "—"))),
        ("mean posterior", str(metrics.get("asr_mean_max_posterior", "—"))),
    ]
    chip_html = "\n".join(
        f'<div class="chip"><span class="chip-key">{key}</span>'
        f'<span class="chip-val">{value}</span></div>'
        for key, value in chips
    )

    plates = []
    for index, (filename, eyebrow, heading, caption) in enumerate(PLATES, start=1):
        path = figures / filename
        if not path.exists():
            continue
        plates.append(f"""
    <figure class="plate">
      <div class="plate-head">
        <p class="eyebrow">{eyebrow}</p>
        <h2>{heading}</h2>
      </div>
      <div class="plate-img"><img src="{data_uri(path)}" alt="{heading}" loading="lazy" /></div>
      <figcaption>{caption}</figcaption>
    </figure>""")

    return f"""<title>Contaminated Ancestors</title>
<style>
  :root {{
    color-scheme: light;
    --paper:   #f7f8f9;
    --surface: #ffffff;
    --ink:     #14161a;
    --muted:   #5c626b;
    --rule:    #dfe3e8;
    --hair:    #eceff2;
    --a:       #2a78d6;
    --b:       #eb6834;
    --flag:    #d03b3b;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --paper:   #101215;
      --surface: #171a1e;
      --ink:     #eef1f4;
      --muted:   #98a0aa;
      --rule:    #272c33;
      --hair:    #1f2329;
      --a:       #3987e5;
      --b:       #e06a34;
      --flag:    #e05a5a;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --paper:   #101215;
    --surface: #171a1e;
    --ink:     #eef1f4;
    --muted:   #98a0aa;
    --rule:    #272c33;
    --hair:    #1f2329;
    --a:       #3987e5;
    --b:       #e06a34;
    --flag:    #e05a5a;
  }}

  body {{
    background: var(--paper);
    color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height: 1.62;
    margin: 0;
    padding: 0 24px 96px;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 1080px; margin: 0 auto; }}
  .prose {{ max-width: 66ch; }}

  h1, h2, .eyebrow {{ font-family: "Iowan Old Style", Palatino, "Palatino Linotype", Georgia, serif; }}
  h1 {{
    font-size: clamp(2.1rem, 5vw, 3.1rem);
    line-height: 1.1; letter-spacing: -0.015em;
    text-wrap: balance; margin: 0 0 4px;
  }}
  h2 {{ font-size: 1.32rem; line-height: 1.25; margin: 0; text-wrap: balance; font-weight: 600; }}
  .lede {{ font-size: 1.06rem; color: var(--muted); margin: 0 0 28px; }}
  p {{ margin: 0 0 14px; }}

  header.masthead {{ padding: 72px 0 30px; border-bottom: 1px solid var(--rule); }}
  .kicker {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--muted); margin: 0 0 18px;
  }}
  .kicker b {{ color: var(--a); font-weight: 600; }}
  .kicker i {{ color: var(--b); font-style: normal; font-weight: 600; }}

  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 26px; }}
  .chip {{
    display: flex; align-items: baseline; gap: 7px;
    border: 1px solid var(--rule); border-radius: 2px;
    padding: 5px 10px; background: var(--surface);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.75rem;
    font-variant-numeric: tabular-nums;
  }}
  .chip-key {{ color: var(--muted); }}
  .chip-val {{ color: var(--ink); font-weight: 600; }}

  .plate {{ margin: 0; padding: 46px 0; border-bottom: 1px solid var(--hair); }}
  .plate-head {{ display: flex; flex-direction: column; gap: 4px; margin-bottom: 18px; }}
  .eyebrow {{
    font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--b); margin: 0; font-weight: 600;
  }}
  /* Plates keep a light ground in both themes: the figures are rendered on a
     light surface, and inverting the page around them would leave each one
     glowing in its own rectangle. */
  .plate-img {{
    background: #fcfcfb; border: 1px solid var(--rule); border-radius: 3px;
    padding: 10px; overflow-x: auto;
  }}
  .plate-img img {{ display: block; width: 100%; height: auto; max-width: 100%; }}
  figcaption {{ max-width: 68ch; color: var(--muted); font-size: 0.94rem; margin-top: 16px; }}

  section.note {{ padding: 46px 0 0; }}
  section.note h2 {{ margin-bottom: 14px; }}
  ul {{ padding-left: 20px; margin: 0 0 14px; }}
  li {{ margin-bottom: 9px; color: var(--muted); max-width: 66ch; }}
  li b {{ color: var(--ink); font-weight: 600; }}
  a {{ color: var(--a); }}
  .foot {{
    margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--rule);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.74rem; color: var(--muted);
  }}
</style>

<div class="wrap">
  <header class="masthead">
    <p class="kicker">stemmatics &middot; <b>sub-history A</b> &middot; <i>sub-history B</i></p>
    <h1>Contaminated ancestors</h1>
    <p class="lede prose">
      The hardest problem in stemmatics is not building a family tree of manuscripts — it is
      contamination. A scribe with two exemplars copies from both, and the resulting witness has
      no single parent. Ancestral sequence reconstruction has the same failure mode and no
      equivalent alarm: recombination makes the tree assumption false, and maximum-likelihood ASR
      responds by returning a chimeric ancestor with high per-site posteriors. Every site is
      individually fine. The pathology is joint, and every standard diagnostic is marginal.
    </p>
    <p class="lede prose">
      ProteinMPNN's decoder scores each residue given the other residues and the backbone, which
      makes it a joint-compatibility model rather than a decorative one. Fix its decoding order
      and you can ask a site whether it prefers one lineage's neighbours to the other's.
    </p>
    <div class="chips">{chip_html}</div>
  </header>
{"".join(plates)}

  <section class="note prose">
    <h2>What is not established</h2>
    <ul>
      <li><b>Detection is inconsistent.</b> On epistatic data the detector fires on some
        conditions and not others. The clean hits are strong; the misses track families with
        too few diagnostic sites.</li>
      <li><b>Short segments are out of reach.</b> A 10-residue swap gives an AUC near chance,
        against roughly 0.87 for a 30-residue swap. Detection needs a block long enough to
        contain enough informative sites.</li>
      <li><b>The repair prediction is unsupported so far.</b> The mosaic was expected to score
        worse than either coherent sub-ancestor, with the gap widening as contamination rises.
        Measured gaps are small and do not scale.</li>
      <li><b>The simulation assumes what the detector senses.</b> The epistatic simulator draws
        from ProteinMPNN's own joint distribution. The f81 control is what keeps this honest;
        a real protein family is what would settle it.</li>
      <li><b>Everything here is in silico</b>, on one backbone, with stability as a
        pseudo-log-likelihood proxy rather than a measured &Delta;&Delta;G.</li>
    </ul>
  </section>

  <p class="foot">
    Simulated family on a fixed backbone &middot; ProteinMPNN (Dauparas et al. 2022) as the joint
    model &middot; palettes validated for colour-vision deficiency rather than chosen by eye
  </p>
</div>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--out", default="figures/gallery.html")
    args = ap.parse_args()

    figures = Path(args.figures)
    context_path = figures / "figure_context.json"
    context = json.loads(context_path.read_text()) if context_path.exists() else {}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(figures, context))
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

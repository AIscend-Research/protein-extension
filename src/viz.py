"""Figures for the contaminated-ancestors paper.

Every figure answers one question, and the encoding is chosen for that question
rather than for decoration:

  stemma            Which witnesses are contaminated, and where do they sit in
                    the tree that the method believes?
  conflict profile  Where along the sequence does the ancestor stop preferring
                    one sub-history and start preferring the other?
  permutation null  Is that segment more contiguous than chance allows?
  apparatus         Which readings are certain, probable, conjectural — and which
                    sub-history does each belong to?
  structure         Where does the conflict sit in three dimensions? (This is the
                    picture that shows the signal is structural, not positional.)
  contact junction  Which residue pairs bridge the two sub-histories in space —
                    the structural seam a sequence method cannot see.

Colour follows the data's job. The conflict score is *polar* (prefers A / prefers
B / indifferent), so it is diverging blue-red with a neutral grey midpoint. The
apparatus labels are *ordinal* (certain > probable > conjectural), so they are one
blue hue, light to dark. Clade identity is *categorical*, so it takes the first
two fixed slots. All three palettes were checked with the validator rather than
chosen by eye, including for colour-vision deficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

# ------------------------------------------------------------------- palette


@dataclass(frozen=True)
class Theme:
    """Validated palette slots. Swap these for a different design system."""

    surface: str = "#fcfcfb"
    ink: str = "#0b0b0b"
    ink_secondary: str = "#52514e"
    ink_muted: str = "#898781"
    grid: str = "#e1e0d9"
    axis: str = "#c3c2b7"

    # categorical: the two sub-histories
    clade_a: str = "#2a78d6"  # blue
    clade_b: str = "#eb6834"  # orange
    accent: str = "#1baf7a"  # aqua — third slot, used for "detected" marks

    # ordinal: apparatus labels. A neutral ink ramp, not a blue one — the
    # sub-history strips are blue and orange, and two encodings sharing a hue is
    # how a reader ends up conflating "well attested" with "belongs to clade A".
    conjectural: str = "#a9a8a2"
    probable: str = "#6e6d68"
    certain: str = "#2f2e2b"

    # status
    critical: str = "#d03b3b"
    good: str = "#0ca30c"

    # diverging midpoint
    neutral: str = "#f0efec"


THEME = Theme()

#: Diverging ramp for the conflict score: blue = prefers sub-history A,
#: red = prefers sub-history B, neutral grey = indifferent.
CONFLICT_CMAP = LinearSegmentedColormap.from_list(
    "conflict", [THEME.clade_a, "#9ec5f4", THEME.neutral, "#f0a08d", THEME.critical]
)

#: Sequential ramp for magnitudes that have no sign (instability, contact density).
MAGNITUDE_CMAP = LinearSegmentedColormap.from_list(
    "magnitude", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"]
)

LABEL_COLORS = {
    "certain": THEME.certain,
    "probable": THEME.probable,
    "conjectural": THEME.conjectural,
}


def _style(ax, *, grid_axis: str | None = "y") -> None:
    """Recessive chrome: hairline grid, muted ticks, no box."""
    ax.set_facecolor(THEME.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(THEME.axis)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=THEME.ink_muted, labelsize=8, length=3, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(THEME.ink_secondary)
    if grid_axis:
        ax.grid(axis=grid_axis, color=THEME.grid, linewidth=0.7, alpha=1.0)
        ax.set_axisbelow(True)


def _figure(width: float, height: float):
    fig = plt.figure(figsize=(width, height), facecolor=THEME.surface)
    return fig


def _save(fig, path: str | Path, *, dpi: int = 200) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=THEME.surface, bbox_inches="tight")
    plt.close(fig)
    return path


def _title(ax, title: str, subtitle: str | None = None) -> None:
    pad = 26 if subtitle else 12
    ax.set_title(title, color=THEME.ink, fontsize=11, fontweight="bold", loc="left", pad=pad)
    if subtitle:
        ax.text(
            0.0, 1.012, subtitle, transform=ax.transAxes, color=THEME.ink_secondary,
            fontsize=8.5, va="bottom", ha="left",
        )


# --------------------------------------------------------------------- stemma


def _layout_tree(tree, contaminated: set[str]) -> tuple[dict, float]:
    """Assign (x = distance from root, y = leaf order) to every node."""
    positions: dict = {}
    counter = [0.0]

    def walk(node, depth: float):
        if node.is_leaf:
            y = counter[0]
            counter[0] += 1.0
            positions[id(node)] = (depth, y, node)
            return y
        ys = [walk(child, depth + max(child.length, 0.01)) for child in node.children]
        y = float(np.mean(ys))
        positions[id(node)] = (depth, y, node)
        return y

    walk(tree, 0.0)
    return positions, counter[0]


def plot_stemma(
    tree,
    *,
    contaminated: Sequence[str] = (),
    clade_a: Sequence[str] = (),
    path: str | Path = "figures/stemma.png",
    title: str = "The stemma the method believes",
) -> Path:
    """Draw the inferred tree, marking witnesses known to be contaminated.

    The point of the picture: a contaminated witness does not sit where its true
    lineage says it should. Lachmannian method returns this tree without
    complaint, which is exactly the failure the paper is about.
    """
    contaminated = set(contaminated)
    clade_a = set(clade_a)
    positions, n_leaves = _layout_tree(tree, contaminated)

    fig = _figure(7.2, max(3.2, 0.34 * n_leaves + 1.2))
    ax = fig.add_subplot(111)

    for depth, y, node in positions.values():
        if node.parent is None:
            continue
        parent_depth, parent_y, _ = positions[id(node.parent)]
        colour = THEME.ink_muted
        leaves_below = [leaf.name for leaf in node.leaves()]
        if leaves_below and all(name in clade_a for name in leaves_below):
            colour = THEME.clade_a
        elif leaves_below and all(name not in clade_a for name in leaves_below):
            colour = THEME.clade_b
        ax.plot([parent_depth, parent_depth], [parent_y, y], color=colour, linewidth=2.0,
                solid_capstyle="round", zorder=2)
        ax.plot([parent_depth, depth], [y, y], color=colour, linewidth=2.0,
                solid_capstyle="round", zorder=2)

    for depth, y, node in positions.values():
        if not node.is_leaf:
            continue
        is_bad = node.name in contaminated
        ax.scatter([depth], [y], s=64 if is_bad else 34,
                   facecolor=THEME.critical if is_bad else (
                       THEME.clade_a if node.name in clade_a else THEME.clade_b),
                   edgecolor=THEME.surface, linewidth=1.6, zorder=4)
        ax.text(depth + 0.012, y, node.name, va="center", ha="left", fontsize=8.5,
                color=THEME.critical if is_bad else THEME.ink_secondary,
                fontweight="bold" if is_bad else "normal")

    ax.set_ylim(-0.8, n_leaves - 0.2)
    ax.set_xlabel("substitutions per site", color=THEME.ink_secondary, fontsize=9)
    ax.set_yticks([])
    _style(ax, grid_axis="x")
    _title(ax, title, "contaminated witnesses in red — note where they fall in the tree")

    handles = [
        Line2D([], [], color=THEME.clade_a, linewidth=2.4, label="sub-history A"),
        Line2D([], [], color=THEME.clade_b, linewidth=2.4, label="sub-history B"),
        Line2D([], [], color=THEME.critical, marker="o", linestyle="none",
               markersize=7, label="contaminated"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, ncol=3,
              loc="upper left", bbox_to_anchor=(0.0, -0.10),
              labelcolor=THEME.ink_secondary)
    return _save(fig, path)


# ----------------------------------------------------------- conflict profile


def plot_conflict_profile(
    conflict,
    *,
    true_segment: tuple[int, int] | None = None,
    path: str | Path = "figures/conflict_profile.png",
    title: str = "Where the ancestor changes its mind",
) -> Path:
    """The central figure: per-site context preference along the sequence.

    Positive = the residue is more at home among sub-history A's neighbours;
    negative = among B's. A coherent ancestor scatters around zero. A
    contaminated one shows a contiguous run of one sign.
    """
    delta = np.asarray(conflict.delta, dtype=float)
    L = len(delta)
    diagnostic = np.asarray(getattr(conflict, "diff_sites", []), dtype=int)
    positions = np.arange(1, L + 1)

    fig = _figure(9.0, 3.6)
    ax = fig.add_subplot(111)

    if true_segment is not None:
        ax.axvspan(true_segment[0] + 1, true_segment[1], color=THEME.grid, alpha=0.9, zorder=0)

    limit = float(np.max(np.abs(delta))) or 1.0
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    ax.axhline(0.0, color=THEME.axis, linewidth=1.0, zorder=1)

    # Uninformative sites recede; diagnostic sites carry the signal.
    mask = np.zeros(L, dtype=bool)
    mask[diagnostic] = True
    ax.vlines(positions[~mask], 0, delta[~mask], color=THEME.grid, linewidth=1.4, zorder=2)
    ax.vlines(positions[mask], 0, delta[mask],
              color=CONFLICT_CMAP(norm(delta[mask])), linewidth=2.0, zorder=3)
    ax.scatter(positions[mask], delta[mask], s=18,
               color=CONFLICT_CMAP(norm(delta[mask])), edgecolor=THEME.surface,
               linewidth=0.8, zorder=4)

    # Reserve a band beneath the data for the detection bracket, so the label
    # never lands on top of a residue.
    low, high = ax.get_ylim()
    ax.set_ylim(low - 0.30 * (high - low), high)

    segment = getattr(conflict, "segment", None)
    if segment is not None:
        floor = ax.get_ylim()[0]
        bracket = floor + 0.06 * (high - floor)
        ax.plot([segment[0] + 1, segment[1]], [bracket, bracket],
                color=THEME.accent, linewidth=3.0, solid_capstyle="round", zorder=5)
        ax.text((segment[0] + segment[1]) / 2, bracket + 0.02 * (high - floor),
                f"detected  p = {conflict.p_value:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=THEME.accent, fontweight="bold")

    ax.set_xlabel("residue position", color=THEME.ink_secondary, fontsize=9)
    ax.set_ylabel("prefers A  ←→  prefers B", color=THEME.ink_secondary, fontsize=9)
    ax.set_xlim(0, L + 1)
    if true_segment is not None:
        ax.annotate("true insertion",
                    xy=((true_segment[0] + true_segment[1]) / 2, 0.985),
                    xycoords=("data", "axes fraction"), ha="center", va="top",
                    fontsize=8, color=THEME.ink_muted)
    _style(ax, grid_axis="y")
    _title(ax, title,
           f"{len(diagnostic)} diagnostic sites (grey = the two sub-ancestors agree, "
           f"so the site says nothing)")
    return _save(fig, path)


def plot_permutation_null(
    null: np.ndarray,
    observed: float,
    p_value: float,
    *,
    path: str | Path = "figures/permutation_null.png",
    title: str = "Is the segment more contiguous than chance?",
) -> Path:
    """The significance test, drawn: null distribution with the observed statistic."""
    fig = _figure(6.4, 3.2)
    ax = fig.add_subplot(111)

    ax.hist(null, bins=28, color=THEME.grid, edgecolor=THEME.surface, linewidth=0.8, zorder=2)
    ax.axvline(observed, color=THEME.critical, linewidth=2.0, zorder=3)
    ax.text(observed, ax.get_ylim()[1] * 0.95,
            f"  observed {observed:.2f}\n  p = {p_value:.3f}",
            color=THEME.critical, fontsize=9, va="top", ha="left", fontweight="bold")

    ax.set_xlabel("segment statistic", color=THEME.ink_secondary, fontsize=9)
    ax.set_ylabel("permutations", color=THEME.ink_secondary, fontsize=9)
    _style(ax)
    _title(ax, title, "null = the same per-site scores, arranged at random")
    return _save(fig, path)


# ------------------------------------------------------------------ apparatus


def plot_apparatus(
    calls: Sequence,
    *,
    path: str | Path = "figures/apparatus.png",
    title: str = "The critical apparatus",
    max_per_row: int = 60,
) -> Path:
    """The paper's output format: every reading labelled and attributed.

    Laid out like a text with an apparatus beneath it — the reconstructed
    sequence in rows, each residue tinted by how well attested it is, with the
    sub-history attribution as a strip underneath.
    """
    n = len(calls)
    rows = int(np.ceil(n / max_per_row))
    fig = _figure(10.0, 0.62 * rows + 1.35)
    ax = fig.add_subplot(111)

    for index, call in enumerate(calls):
        row, column = divmod(index, max_per_row)
        y = -row * 1.0
        colour = LABEL_COLORS.get(call.label, THEME.conjectural)

        ax.add_patch(Rectangle((column, y), 0.92, 0.5, facecolor=colour,
                               edgecolor="none", zorder=2))
        ax.text(column + 0.46, y + 0.25, call.mpnn_aa, ha="center", va="center",
                fontsize=7.2, color=THEME.ink if call.label == "conjectural" else "#ffffff",
                fontweight="bold", zorder=3, family="monospace")

        sub = getattr(call, "sub_history", None)
        if sub is not None:
            strip = THEME.clade_a if sub == "A" else THEME.clade_b
            ax.add_patch(Rectangle((column, y - 0.22), 0.92, 0.16, facecolor=strip,
                                   edgecolor="none", zorder=2))
        if getattr(call, "contaminated", False):
            ax.add_patch(Rectangle((column, y - 0.30), 0.92, 0.06,
                                   facecolor=THEME.critical, edgecolor="none", zorder=3))

    for row in range(rows):
        ax.text(-1.2, -row * 1.0 + 0.25, f"{row * max_per_row + 1}", ha="right", va="center",
                fontsize=7.5, color=THEME.ink_muted, family="monospace")

    ax.set_xlim(-3.0, max_per_row + 0.5)
    ax.set_ylim(-rows * 1.0 - 0.2, 0.9)
    ax.axis("off")
    ax.set_title(title, color=THEME.ink, fontsize=11, fontweight="bold", loc="left", pad=16)
    ax.text(0.0, 1.008,
            "residue tint = how well attested · strip = which sub-history · red = flagged contaminated",
            transform=ax.transAxes, color=THEME.ink_secondary, fontsize=8.5, va="bottom")

    handles = [
        Patch(facecolor=THEME.certain, label="certain"),
        Patch(facecolor=THEME.probable, label="probable"),
        Patch(facecolor=THEME.conjectural, label="conjectural"),
        Patch(facecolor=THEME.clade_a, label="sub-history A"),
        Patch(facecolor=THEME.clade_b, label="sub-history B"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, ncol=5,
              loc="lower left", bbox_to_anchor=(0.0, -0.16), labelcolor=THEME.ink_secondary)
    return _save(fig, path)


# ------------------------------------------------------------------ structure


def ca_coordinates(scorer) -> np.ndarray:
    """(L, 3) alpha-carbon trace from the featurised backbone."""
    return scorer.backbone.X[0, :, 1, :].detach().cpu().numpy()


def _equal_aspect(ax, coords: np.ndarray) -> None:
    centre = coords.mean(axis=0)
    span = float(np.max(np.abs(coords - centre))) * 0.92
    ax.set_xlim(centre[0] - span, centre[0] + span)
    ax.set_ylim(centre[1] - span, centre[1] + span)
    ax.set_zlim(centre[2] - span, centre[2] + span)
    ax.set_box_aspect((1, 1, 1), zoom=1.45)
    ax.set_axis_off()
    ax.set_facecolor(THEME.surface)


def plot_structure_conflict(
    scorer,
    values: np.ndarray,
    *,
    diverging: bool = True,
    segment: tuple[int, int] | None = None,
    path: str | Path = "figures/structure_conflict.png",
    title: str = "The conflict, in three dimensions",
    views: Sequence[tuple[float, float]] = ((18, 35), (18, 125), (72, 35)),
    label: str = "prefers A  ←→  prefers B",
) -> Path:
    """Colour the backbone trace by a per-residue score, from several viewpoints.

    This is the figure that distinguishes a structural claim from a positional
    one. If the flagged residues form a contiguous *patch on the fold* rather
    than merely a contiguous stretch of sequence, the signal is what the method
    says it is.
    """
    coords = ca_coordinates(scorer)
    values = np.asarray(values, dtype=float)
    if len(values) != len(coords):
        raise ValueError(f"{len(values)} values for {len(coords)} residues")

    if diverging:
        # Robust limits: a couple of extreme residues would otherwise flatten the
        # whole fold to the neutral midpoint and hide the pattern being claimed.
        limit = float(np.percentile(np.abs(values), 92)) or 1.0
        norm, cmap = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), CONFLICT_CMAP
    else:
        low, high = np.percentile(values, [4, 96])
        norm, cmap = Normalize(vmin=float(low), vmax=float(high)), MAGNITUDE_CMAP

    fig = _figure(3.9 * len(views), 4.4)
    for index, (elevation, azimuth) in enumerate(views):
        ax = fig.add_subplot(1, len(views), index + 1, projection="3d")
        ax.view_init(elev=elevation, azim=azimuth)

        # Backbone as coloured segments, with a pale casing so the fold reads.
        for i in range(len(coords) - 1):
            piece = coords[i : i + 2]
            ax.plot(piece[:, 0], piece[:, 1], piece[:, 2],
                    color=THEME.surface, linewidth=9.0, solid_capstyle="round", zorder=1)
        for i in range(len(coords) - 1):
            piece = coords[i : i + 2]
            shade = cmap(norm(np.clip(values[i : i + 2].mean(), norm.vmin, norm.vmax)))
            ax.plot(piece[:, 0], piece[:, 1], piece[:, 2],
                    color=shade, linewidth=5.4, solid_capstyle="round", zorder=2)

        if segment is not None:
            # Trace the flagged block as a dark casing rather than dotting every
            # residue — the eye should read one continuous run, not 30 marks.
            block = coords[segment[0] : segment[1]]
            ax.plot(block[:, 0], block[:, 1], block[:, 2], color=THEME.ink,
                    linewidth=1.4, linestyle=(0, (2, 2)), zorder=3)

        ax.scatter(*coords[0], s=34, color=THEME.ink, zorder=4)
        ax.text(*coords[0], "  N", color=THEME.ink_secondary, fontsize=8)
        _equal_aspect(ax, coords)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    bar = fig.colorbar(mappable, ax=fig.axes, fraction=0.02, pad=0.02, aspect=28)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=THEME.ink_muted, labelsize=8)
    bar.set_label(label, color=THEME.ink_secondary, fontsize=9)

    fig.suptitle(title, color=THEME.ink, fontsize=11.5, fontweight="bold", x=0.09, ha="left", y=0.97)
    fig.text(0.09, 0.915,
             "same fold, three viewpoints — a structural signal forms a patch, not a stripe",
             color=THEME.ink_secondary, fontsize=8.5, ha="left")
    return _save(fig, path)


def contact_map(scorer, *, cutoff: float = 8.0) -> np.ndarray:
    """(L, L) boolean CA-CA contact map."""
    coords = ca_coordinates(scorer)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return distances <= cutoff


def plot_contact_junction(
    scorer,
    segment: tuple[int, int],
    *,
    cutoff: float = 8.0,
    path: str | Path = "figures/contact_junction.png",
    title: str = "The structural seam",
) -> Path:
    """Contact map with the contaminated block marked, and its spatial neighbours.

    Recombination is a break in *sequence*, but the strain it causes is at the
    contacts that cross the break. Off-diagonal contacts between the block and
    the rest of the fold are where a structural detector has any purchase at all
    — and where a sequence-window method has none.
    """
    contacts = contact_map(scorer, cutoff=cutoff)
    L = contacts.shape[0]
    start, stop = segment

    inside = np.zeros(L, dtype=bool)
    inside[start:stop] = True
    crossing = contacts & (inside[:, None] != inside[None, :])

    fig = _figure(6.6, 6.0)
    ax = fig.add_subplot(111)

    ys, xs = np.nonzero(contacts & ~crossing)
    ax.scatter(xs + 1, ys + 1, s=2.2, color=THEME.grid, zorder=2, marker="s")
    ys, xs = np.nonzero(crossing)
    ax.scatter(xs + 1, ys + 1, s=4.0, color=THEME.critical, zorder=3, marker="s")

    for edge in (start + 1, stop):
        ax.axvline(edge, color=THEME.ink_muted, linewidth=0.9, linestyle=(0, (4, 3)), zorder=4)
        ax.axhline(edge, color=THEME.ink_muted, linewidth=0.9, linestyle=(0, (4, 3)), zorder=4)

    n_crossing = int(crossing.sum() // 2)
    ax.set_xlim(0, L + 1)
    ax.set_ylim(L + 1, 0)
    ax.set_xlabel("residue", color=THEME.ink_secondary, fontsize=9)
    ax.set_ylabel("residue", color=THEME.ink_secondary, fontsize=9)
    _style(ax, grid_axis=None)
    _title(ax, title,
           f"{n_crossing} contacts bridge the flagged block and the rest of the fold "
           f"(CA-CA ≤ {cutoff:.0f} Å)")

    handles = [
        Line2D([], [], marker="s", linestyle="none", color=THEME.grid, markersize=7,
               label="contact"),
        Line2D([], [], marker="s", linestyle="none", color=THEME.critical, markersize=7,
               label="crosses the breakpoint"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, ncol=2,
              loc="upper left", bbox_to_anchor=(0.0, -0.08),
              labelcolor=THEME.ink_secondary)
    return _save(fig, path)


def write_bfactor_pdb(
    scorer, values: np.ndarray, path: str | Path, *, scale: float = 100.0
) -> Path:
    """Write the backbone with a per-residue score in the B-factor column.

    Lets the same numbers be rendered in PyMOL or ChimeraX for a publication
    figure (`spectrum b`), without this project depending on either.
    """
    coords = scorer.backbone.X[0].detach().cpu().numpy()  # (L, 4, 3) N, CA, C, O
    values = np.asarray(values, dtype=float)
    span = float(np.max(np.abs(values))) or 1.0
    scaled = values / span * scale

    lines: list[str] = []
    atom_id = 1
    for residue_index in range(coords.shape[0]):
        for atom_index, atom_name in enumerate(("N", "CA", "C", "O")):
            x, y, z = coords[residue_index, atom_index]
            lines.append(
                f"ATOM  {atom_id:>5} {atom_name:<4}GLY A{residue_index + 1:>4}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00{scaled[residue_index]:>6.2f}"
                f"           {atom_name[0]:>2}"
            )
            atom_id += 1
    lines.append("END")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


# -------------------------------------------------------------- result panels


def plot_detection_summary(
    rows: Sequence[Mapping],
    *,
    path: str | Path = "figures/detection_summary.png",
    title: str = "Firing rate says nothing; where it points says everything",
) -> Path:
    """The central comparison: epistatic data versus the no-epistasis control.

    Deliberately *not* a plot of detection rate alone. On matched conditions the
    detector fires as often on `f81` as on `selection`, so a rate-only panel
    invites exactly the wrong conclusion. What separates the two is whether the
    flagged window is the true breakpoint, so the rate is shown beside the
    Jaccard of the windows each model actually flagged.
    """
    models = sorted({row["model"] for row in rows}, reverse=True)  # selection first
    colours = {m: (THEME.clade_a if m == "selection" else THEME.clade_b) for m in models}

    fig = _figure(8.8, 3.8)
    axes = fig.subplots(1, 2)

    ax = axes[0]
    rates = []
    for model in models:
        sub = [r for r in rows if r["model"] == model]
        rates.append(float(np.mean([r["detected"] for r in sub])) if sub else 0.0)
    ax.bar(range(len(models)), rates, width=0.5,
           color=[colours[m] for m in models], zorder=3)
    for index, model in enumerate(models):
        sub = [r for r in rows if r["model"] == model]
        ax.text(index, rates[index] + 0.02,
                f"{rates[index]:.0%}   ({sum(r['detected'] for r in sub)}/{len(sub)})",
                ha="center", fontsize=9, color=THEME.ink, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 0.45)
    ax.set_ylabel("detection rate", color=THEME.ink_secondary, fontsize=9)
    _style(ax)
    _title(ax, "Detection rate", "indistinguishable — this panel is the trap")

    ax = axes[1]
    for index, model in enumerate(models):
        fired = [r for r in rows if r["model"] == model and r["detected"]]
        values = [r["segment_jaccard"] for r in fired
                  if r.get("segment_jaccard") is not None]
        if not values:
            continue
        jitter = (np.random.default_rng(0).random(len(values)) - 0.5) * 0.16
        ax.scatter(np.full(len(values), index) + jitter, values, s=64,
                   color=colours[model], edgecolor=THEME.surface, linewidth=1.2, zorder=4)
        ax.plot([index - 0.2, index + 0.2], [np.mean(values)] * 2,
                color=THEME.ink, linewidth=2.0, zorder=5)
        # Beside the mean rule, not above it: the jittered points sit above.
        ax.text(index + 0.25, np.mean(values), f"mean {np.mean(values):.2f}",
                ha="left", va="center", fontsize=8.5, color=THEME.ink, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Jaccard vs the true breakpoint", color=THEME.ink_secondary, fontsize=9)
    _style(ax)
    _title(ax, "Accuracy of the flagged window", "only the runs that fired")

    fig.suptitle(title, color=THEME.ink, fontsize=12, fontweight="bold", x=0.02, ha="left",
                 y=1.07)
    fig.subplots_adjust(wspace=0.3)
    return _save(fig, path)


ABLATION_COLORS = {
    "mpnn": THEME.clade_a,
    "identity": THEME.clade_b,
    "scrambled": THEME.accent,
}

ABLATION_LABELS = {
    "mpnn": "ProteinMPNN",
    "identity": "sequence\nidentity",
    "scrambled": "scrambled\nbackbone",
}


def plot_ablation(
    rows: Sequence[Mapping],
    *,
    path: str | Path = "figures/ablation.png",
    title: str = "The structural model does not beat string comparison",
) -> Path:
    """Does the joint structural model earn its place against two controls?

    Arm identity is carried by x position and the tick labels, so hue here is
    redundant encoding rather than the only channel — which is what lets three
    categorical colours sit this close together safely.
    """
    arms = [a for a in ("mpnn", "identity", "scrambled")
            if any(a in r.get("arms", {}) for r in rows)]

    fig = _figure(9.0, 4.2)
    axes = fig.subplots(1, 2)

    ax = axes[0]
    for index, arm in enumerate(arms):
        values = [r["arms"][arm]["segment_jaccard"] for r in rows
                  if r["arms"][arm].get("segment_jaccard") is not None]
        if not values:
            continue
        jitter = (np.random.default_rng(1).random(len(values)) - 0.5) * 0.22
        ax.scatter(np.full(len(values), index) + jitter, values, s=52,
                   color=ABLATION_COLORS[arm], alpha=0.75,
                   edgecolor=THEME.surface, linewidth=1.1, zorder=4)
        ax.plot([index - 0.25, index + 0.25], [np.mean(values)] * 2,
                color=THEME.ink, linewidth=2.2, zorder=5)
        ax.text(index + 0.3, np.mean(values), f"{np.mean(values):.2f}",
                ha="left", va="center", fontsize=9, color=THEME.ink, fontweight="bold")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ABLATION_LABELS[a] for a in arms], fontsize=8)
    ax.set_xlim(-0.6, len(arms) - 0.25)
    ax.set_ylim(-0.03, 1.08)
    ax.set_ylabel("Jaccard vs the true breakpoint", color=THEME.ink_secondary, fontsize=9)
    _style(ax)
    _title(ax, "Recovery of the contaminated block",
           "one point per run; rule is the mean")

    ax = axes[1]
    rates, counts = [], []
    for arm in arms:
        fired = [r for r in rows if r["arms"][arm]["detected"]]
        rates.append(len(fired) / max(len(rows), 1))
        counts.append((len(fired), len(rows)))
    ax.bar(range(len(arms)), rates, width=0.5,
           color=[ABLATION_COLORS[a] for a in arms], zorder=3)
    for index, (fired, total) in enumerate(counts):
        ax.text(index, rates[index] + 0.025, f"{fired}/{total}", ha="center",
                fontsize=9.5, color=THEME.ink, fontweight="bold")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ABLATION_LABELS[a] for a in arms], fontsize=8)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("detection rate", color=THEME.ink_secondary, fontsize=9)
    _style(ax)
    _title(ax, "How often each arm fired",
           "sequence identity uses no structure and no network")

    fig.suptitle(title, color=THEME.ink, fontsize=12, fontweight="bold", x=0.02, ha="left",
                 y=1.06)
    fig.subplots_adjust(wspace=0.32)
    return _save(fig, path)


def plot_sensitivity_curve(
    rows: Sequence[Mapping],
    *,
    path: str | Path = "figures/sensitivity_curve.png",
    metric: str = "site_auc_segment",
    title: str = "Where detection dies",
) -> Path:
    """Site AUC and detection rate against the length of the contaminated block.

    The honest figure in the set: it shows the regime where the method stops
    working rather than the single condition where it does. The `f81` control is
    plotted on the same axes because that — not the 0.5 line — is what chance
    actually looks like for this estimator on this data, inflation included.
    """
    models = sorted({r["model"] for r in rows}, reverse=True)  # selection first
    widths = sorted({r["width"] for r in rows})
    # The design system's categorical order, used in order. No sub-history is
    # encoded in this figure, so the clade hues are free here.
    colours = {m: (THEME.clade_a if m == "selection" else THEME.clade_b) for m in models}

    def clean(sub):
        return [r[metric] for r in sub
                if r.get(metric) is not None and not np.isnan(r[metric])]

    fig = _figure(9.2, 4.0)
    axes = fig.subplots(1, 2)

    # -- left: the requested curve, AUC against block length -----------------
    ax = axes[0]
    # Deliberately NOT labelled "chance": the orientation rule pulls a null AUC
    # above 0.5, so the f81 series is the empirical null and 0.5 is only a ruler.
    ax.axhline(0.5, color=THEME.axis, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    for model in models:
        means, seen = [], []
        for width in widths:
            values = clean([r for r in rows if r["model"] == model and r["width"] == width])
            if not values:
                continue
            seen.append(width)
            means.append(float(np.mean(values)))
            ax.scatter(np.full(len(values), width), values, s=26, color=colours[model],
                       alpha=0.4, edgecolor=THEME.surface, linewidth=0.8, zorder=3)
        if not seen:
            continue
        ax.plot(seen, means, color=colours[model], linewidth=2.0, zorder=4,
                marker="o", markersize=5, markeredgecolor=THEME.surface, markeredgewidth=1.2)
        note = "  (empirical null)" if model != "selection" else ""
        # Keep the direct label off the 0.5 rule line when the series ends on it.
        offset = -0.062 if abs(means[-1] - 0.5) < 0.05 else 0.0
        ax.text(seen[-1] + 1.5, means[-1] + offset, model + note, color=THEME.ink_secondary,
                fontsize=8.5, va="center", ha="left")
    ax.set_xticks(widths)
    ax.set_xlim(widths[0] - 4, widths[-1] + 26)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("contaminated block length (residues)", color=THEME.ink_secondary, fontsize=9)
    ax.set_ylabel("site AUC", color=THEME.ink_secondary, fontsize=9)
    _style(ax)
    _title(ax, "AUC against block length", "one point per seed; line is the mean")

    # -- right: the variable that actually governs it ------------------------
    ax = axes[1]
    ax.axhline(0.5, color=THEME.axis, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    for model in models:
        sub = [r for r in rows if r["model"] == model
               and r.get(metric) is not None and not np.isnan(r[metric])
               and r.get("n_diagnostic_in_segment") is not None]
        if not sub:
            continue
        quiet = [r for r in sub if not r["detected"]]
        fired = [r for r in sub if r["detected"]]
        ax.scatter([r["n_diagnostic_in_segment"] for r in quiet], [r[metric] for r in quiet],
                   s=34, color=colours[model], alpha=0.45,
                   edgecolor=THEME.surface, linewidth=0.9, zorder=3, label=model)
        if fired:
            ax.scatter([r["n_diagnostic_in_segment"] for r in fired], [r[metric] for r in fired],
                       s=86, facecolor=colours[model], edgecolor=THEME.ink, linewidth=1.6,
                       zorder=5, marker="o")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("diagnostic sites inside the contaminated block",
                  color=THEME.ink_secondary, fontsize=9)
    ax.set_ylabel("site AUC", color=THEME.ink_secondary, fontsize=9)
    _style(ax)
    _title(ax, "AUC against diagnostic sites", "ringed = the permutation test fired")
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=7,
                      markerfacecolor=colours[m], markeredgecolor=THEME.surface, label=m)
               for m in models]
    handles.append(Line2D([], [], marker="o", linestyle="none", markersize=9,
                          markerfacecolor=THEME.neutral, markeredgecolor=THEME.ink,
                          markeredgewidth=1.6, label="detected"))
    legend = ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower right")
    for text in legend.get_texts():
        text.set_color(THEME.ink_secondary)

    fig.suptitle(title, color=THEME.ink, fontsize=12, fontweight="bold", x=0.02, ha="left",
                 y=1.07)
    fig.subplots_adjust(wspace=0.28)
    return _save(fig, path)


def plot_label_calibration(
    recovery: Mapping[str, float],
    *,
    path: str | Path = "figures/label_calibration.png",
    title: str = "The apparatus labels are calibrated",
) -> Path:
    """Sequence recovery by apparatus label — are the labels earning their keep?"""
    order = [k for k in ("certain", "probable", "conjectural") if k in recovery]
    fig = _figure(5.4, 3.2)
    ax = fig.add_subplot(111)

    ax.bar(range(len(order)), [recovery[k] for k in order], width=0.55,
           color=[LABEL_COLORS[k] for k in order], zorder=3)
    for index, key in enumerate(order):
        ax.text(index, recovery[key] + 0.03, f"{recovery[key]:.2f}", ha="center",
                fontsize=10, color=THEME.ink, fontweight="bold")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("recovery vs true root", color=THEME.ink_secondary, fontsize=9)
    _style(ax)
    _title(ax, title, "if 'certain' did not recover better than 'conjectural', the labels would be decorative")
    return _save(fig, path)

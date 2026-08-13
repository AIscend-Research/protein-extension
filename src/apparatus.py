"""Critical apparatus for ancestral sequence reconstruction.

Takes ProteinMPNN's per-residue probabilities (the .npz written by
`--save_probs`), computes per-position entropy, joins it against FastML marginal
posterior probabilities for the same ancestral node, and labels every position

    certain | probable | conjectural

in the philological sense: how much of the reconstruction is actually attested
by the evidence, and how much is editorial conjecture.

This module is deliberately independent of the rest of the pipeline — it needs
only an .npz and (optionally) a FastML posterior table, so it can be developed
and tested against the smoke-test output.

    python src/apparatus.py --probs scratch/smoke_test/probs/5L33.npz \\
        --fastml data/interim/node12.marginal_prob.csv \\
        --node N12 --out data/designs/node12_apparatus.csv
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

# ProteinMPNN's output alphabet; the last symbol is the unknown token.
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
CANONICAL = ALPHABET[:20]

CERTAIN = "certain"
PROBABLE = "probable"
CONJECTURAL = "conjectural"


@dataclass(frozen=True)
class Thresholds:
    """Cut-points for the three-way label.

    `entropy` values are normalised to [0, 1] (H / ln 20), so they are
    comparable across alphabets and independent of log base.
    """

    certain_posterior: float = 0.95
    certain_entropy: float = 0.35
    probable_posterior: float = 0.80
    probable_entropy: float = 0.60
    require_agreement_for_certain: bool = True


@dataclass
class PositionCall:
    """One row of the apparatus."""

    position: int  # 1-based, in the coordinate system of the MPNN probabilities
    mpnn_aa: str
    mpnn_prob: float
    mpnn_entropy: float  # nats, over the 20 canonical amino acids
    mpnn_norm_entropy: float  # entropy / ln(20)
    fastml_aa: str | None
    fastml_prob: float | None
    agreement: bool | None
    label: str
    note: str = ""


def load_probs(path_or_array: str | Path | np.ndarray, *, key: str = "probs") -> np.ndarray:
    """Return a (n_res, 21) distribution, averaging over samples if needed."""
    if isinstance(path_or_array, np.ndarray):
        arr = path_or_array
    else:
        with np.load(Path(path_or_array), allow_pickle=True) as data:
            if key not in data.files:
                raise KeyError(f"{key!r} not in {path_or_array} (has {data.files})")
            arr = data[key]
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:  # (n_samples, n_res, 21)
        arr = arr.mean(axis=0)
    if arr.ndim != 2:
        raise ValueError(f"expected (n_res, n_aa) probabilities, got shape {arr.shape}")
    return arr


def canonical_view(probs: np.ndarray) -> np.ndarray:
    """Drop the unknown token and renormalise over the 20 canonical amino acids."""
    if probs.shape[1] == len(ALPHABET):
        probs = probs[:, : len(CANONICAL)]
    elif probs.shape[1] != len(CANONICAL):
        raise ValueError(f"unexpected alphabet size {probs.shape[1]}")
    totals = probs.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return probs / totals


def entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy per position, in nats, over the canonical alphabet."""
    p = canonical_view(probs)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return -terms.sum(axis=1)


def normalised_entropy(probs: np.ndarray) -> np.ndarray:
    """Entropy scaled so 0 = one residue certain, 1 = uniform over 20 residues."""
    return entropy(probs) / math.log(len(CANONICAL))


def load_fastml_posteriors(
    path: str | Path, *, node: str | None = None
) -> dict[int, tuple[str, float, dict[str, float]]]:
    """Read FastML marginal posterior probabilities into {position: (aa, prob, row)}.

    Accepts either of FastML's shapes:

      wide : Ancestral Node, Pos, A, C, D, ...     (one row per position)
      long : Node, Pos, AA, Prob                    (one row per position/residue)

    Column names are matched case-insensitively; `node` filters to a single
    ancestral node when the table holds several.
    """
    rows = list(csv.DictReader(Path(path).open(newline="")))
    if not rows:
        raise ValueError(f"{path} is empty")

    fields = {(k or "").strip().lower(): (k or "") for k in rows[0]}

    def col(*names: str) -> str | None:
        for name in names:
            if name in fields:
                return fields[name]
        return None

    node_col = col("node", "ancestral node", "ancestralnode", "name")
    pos_col = col("pos", "position", "site", "column")
    aa_col = col("aa", "amino acid", "residue", "char", "character")
    prob_col = col("prob", "probability", "posterior", "p")
    if pos_col is None:
        raise ValueError(f"{path}: no position column found (looked for pos/position/site)")

    if node is not None and node_col is None:
        raise ValueError(f"{path}: --node given but no node column present")

    long_format = aa_col is not None and prob_col is not None
    accumulator: dict[int, dict[str, float]] = {}

    for row in rows:
        if node is not None and str(row[node_col]).strip() != node:
            continue
        try:
            pos = int(float(str(row[pos_col]).strip()))
        except (TypeError, ValueError):
            continue
        bucket = accumulator.setdefault(pos, {})
        if long_format:
            aa = str(row[aa_col]).strip().upper()[:1]
            try:
                bucket[aa] = float(row[prob_col])
            except (TypeError, ValueError):
                continue
        else:
            for key, value in row.items():
                name = (key or "").strip().upper()
                if len(name) == 1 and name in CANONICAL:
                    try:
                        bucket[name] = float(value)
                    except (TypeError, ValueError):
                        continue

    if not accumulator:
        raise ValueError(f"{path}: no rows matched (node={node!r})")

    out: dict[int, tuple[str, float, dict[str, float]]] = {}
    for pos, dist in accumulator.items():
        if not dist:
            continue
        best_aa = max(dist, key=dist.get)
        out[pos] = (best_aa, dist[best_aa], dist)
    return out


def label_position(
    mpnn_norm_entropy: float,
    fastml_prob: float | None,
    agreement: bool | None,
    thresholds: Thresholds = Thresholds(),
) -> tuple[str, str]:
    """Return (label, note) for one position.

    With no FastML posterior the call rests on ProteinMPNN alone, which is a
    weaker claim — the note records that so it can be reported honestly.
    """
    t = thresholds
    if fastml_prob is None:
        if mpnn_norm_entropy <= t.certain_entropy:
            return PROBABLE, "structure-only (no phylogenetic posterior)"
        if mpnn_norm_entropy <= t.probable_entropy:
            return CONJECTURAL, "structure-only, moderate entropy"
        return CONJECTURAL, "structure-only, high entropy"

    if agreement is False:
        if fastml_prob >= t.certain_posterior and mpnn_norm_entropy <= t.certain_entropy:
            return PROBABLE, "contested: both sources confident but disagree"
        return CONJECTURAL, "sources disagree"

    if (
        fastml_prob >= t.certain_posterior
        and mpnn_norm_entropy <= t.certain_entropy
        and (agreement is True or not t.require_agreement_for_certain)
    ):
        return CERTAIN, ""
    if fastml_prob >= t.probable_posterior and mpnn_norm_entropy <= t.probable_entropy:
        return PROBABLE, ""
    if fastml_prob < t.probable_posterior and mpnn_norm_entropy > t.probable_entropy:
        return CONJECTURAL, "weak posterior and high entropy"
    return CONJECTURAL, ""


def build_apparatus(
    probs: str | Path | np.ndarray,
    fastml: str | Path | Mapping[int, tuple[str, float, dict[str, float]]] | None = None,
    *,
    node: str | None = None,
    offset: int = 0,
    thresholds: Thresholds = Thresholds(),
) -> list[PositionCall]:
    """Join MPNN probabilities with FastML posteriors and label every position.

    `offset` is added to the 1-based MPNN position before looking it up in the
    FastML table, for when the alignment column numbering differs from the
    structure's residue numbering.
    """
    p = load_probs(probs)
    canon = canonical_view(p)
    h = entropy(p)
    h_norm = h / math.log(len(CANONICAL))
    top_idx = canon.argmax(axis=1)

    if fastml is None:
        posteriors: Mapping[int, tuple[str, float, dict[str, float]]] = {}
    elif isinstance(fastml, (str, Path)):
        posteriors = load_fastml_posteriors(fastml, node=node)
    else:
        posteriors = fastml

    calls: list[PositionCall] = []
    for i in range(canon.shape[0]):
        pos = i + 1
        mpnn_aa = CANONICAL[int(top_idx[i])]
        entry = posteriors.get(pos + offset)
        if entry is None:
            fastml_aa, fastml_prob, agreement = None, None, None
        else:
            fastml_aa, fastml_prob, _ = entry
            agreement = fastml_aa == mpnn_aa
        label, note = label_position(float(h_norm[i]), fastml_prob, agreement, thresholds)
        if fastml is not None and entry is None:
            note = (note + "; " if note else "") + "no FastML row for this position"
        calls.append(
            PositionCall(
                position=pos,
                mpnn_aa=mpnn_aa,
                mpnn_prob=float(canon[i, top_idx[i]]),
                mpnn_entropy=float(h[i]),
                mpnn_norm_entropy=float(h_norm[i]),
                fastml_aa=fastml_aa,
                fastml_prob=None if fastml_prob is None else float(fastml_prob),
                agreement=agreement,
                label=label,
                note=note,
            )
        )
    return calls


def consensus_sequence(calls: Sequence[PositionCall], *, source: str = "fastml") -> str:
    """Reconstructed sequence; conjectural positions are lower-cased, as in an edition."""
    letters = []
    for call in calls:
        aa = call.fastml_aa if source == "fastml" and call.fastml_aa else call.mpnn_aa
        letters.append(aa if call.label != CONJECTURAL else aa.lower())
    return "".join(letters)


def summarize(calls: Sequence[PositionCall]) -> dict[str, object]:
    counts = {CERTAIN: 0, PROBABLE: 0, CONJECTURAL: 0}
    for call in calls:
        counts[call.label] += 1
    n = len(calls) or 1
    scored = [c for c in calls if c.agreement is not None]
    return {
        "n_positions": len(calls),
        "counts": counts,
        "fractions": {k: round(v / n, 4) for k, v in counts.items()},
        "mean_norm_entropy": round(float(np.mean([c.mpnn_norm_entropy for c in calls])), 4),
        "n_with_posterior": len(scored),
        "agreement_rate": (
            round(sum(c.agreement for c in scored) / len(scored), 4) if scored else None
        ),
    }


FIELDS = [
    "position", "mpnn_aa", "mpnn_prob", "mpnn_entropy", "mpnn_norm_entropy",
    "fastml_aa", "fastml_prob", "agreement", "label", "note",
]


def write_apparatus(calls: Iterable[PositionCall], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for call in calls:
            row = asdict(call)
            for key in ("mpnn_prob", "mpnn_entropy", "mpnn_norm_entropy", "fastml_prob"):
                if row[key] is not None:
                    row[key] = round(row[key], 6)
            writer.writerow(row)
    return path


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs", required=True, help="ProteinMPNN .npz from --save_probs")
    ap.add_argument("--fastml", default=None, help="FastML marginal posterior CSV")
    ap.add_argument("--node", default=None, help="Ancestral node to select from the FastML table")
    ap.add_argument("--offset", type=int, default=0, help="Added to MPNN position before FastML lookup")
    ap.add_argument("--out", default=None, help="Output CSV (default: alongside the .npz)")
    ap.add_argument("--certain-posterior", type=float, default=Thresholds.certain_posterior)
    ap.add_argument("--certain-entropy", type=float, default=Thresholds.certain_entropy)
    ap.add_argument("--probable-posterior", type=float, default=Thresholds.probable_posterior)
    ap.add_argument("--probable-entropy", type=float, default=Thresholds.probable_entropy)
    args = ap.parse_args()

    thresholds = Thresholds(
        certain_posterior=args.certain_posterior,
        certain_entropy=args.certain_entropy,
        probable_posterior=args.probable_posterior,
        probable_entropy=args.probable_entropy,
    )
    calls = build_apparatus(
        args.probs, args.fastml, node=args.node, offset=args.offset, thresholds=thresholds
    )
    out = Path(args.out) if args.out else Path(args.probs).with_suffix(".apparatus.csv")
    write_apparatus(calls, out)
    print(f"wrote {out}")
    print(consensus_sequence(calls))
    print(json.dumps(summarize(calls), indent=2))


if __name__ == "__main__":
    main()

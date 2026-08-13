"""Turn phylogenetic evidence into ProteinMPNN conditioning files.

The constructive half of the project: having separated the sub-histories, design
from each coherent sub-ancestor rather than from the mosaic. Every design phase
is "write a different jsonl and call [runner]", and this module writes them.

Two channels, which ProteinMPNN treats differently:

  bias_by_res  A per-position, per-amino-acid value added to the decoder's
               logits. Soft, unnormalised, and the natural home for "this site's
               ancestral posterior was 0.98, lean on it; that one was 0.3, don't".
  pssm         A per-position probability profile, mixed into the distribution
               under `--pssm_multi` with an optional `--pssm_threshold` cutoff.
               Blunter, but it is the channel MPNN's own docs use for profiles.

`posteriors_to_bias` is the link to the phylogenetics: marginal posteriors become
a log-odds bias tempered by `strength`, so confident ancestral states pull hard
and conjectural ones barely pull at all. That is the critical apparatus acting on
the design, not merely reported beside it.

    from stemma import reconstruct
    from conditioning import posteriors_to_bias, write_bias_by_res
    rec = reconstruct(seqs, taxa=clade_a)
    bias = posteriors_to_bias(rec.posteriors, strength=2.0)
    write_bias_by_res("5L33", "A", bias, "data/interim/bias.jsonl")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

# ProteinMPNN's bias/pssm arrays are indexed by its own 21-letter alphabet.
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
CANONICAL = ALPHABET[:20]
EPS = 1e-9


def _pad_to_alphabet(profile: np.ndarray) -> np.ndarray:
    """(L, 20) -> (L, 21), leaving the unknown token at zero."""
    profile = np.asarray(profile, dtype=float)
    if profile.shape[1] == len(ALPHABET):
        return profile
    if profile.shape[1] != len(CANONICAL):
        raise ValueError(f"expected 20 or 21 columns, got {profile.shape[1]}")
    padded = np.zeros((profile.shape[0], len(ALPHABET)))
    padded[:, : len(CANONICAL)] = profile
    return padded


def posteriors_to_bias(
    posteriors: np.ndarray,
    *,
    strength: float = 2.0,
    confidence_weighted: bool = True,
    floor: float = 1e-4,
) -> np.ndarray:
    """Marginal ancestral posteriors -> per-residue logit bias, (L, 21).

    The bias is `strength * log(posterior)`, which is a log-odds nudge toward the
    ancestral state. With `confidence_weighted`, each position is additionally
    scaled by its own maximum posterior, so a site the reconstruction is sure
    about biases the design hard and a conjectural site is left nearly free —
    the certain/probable/conjectural distinction expressed as a force rather
    than a label.
    """
    post = np.asarray(posteriors, dtype=float)
    post = post / np.clip(post.sum(axis=1, keepdims=True), EPS, None)
    bias = strength * np.log(np.clip(post, floor, None))
    bias -= bias.mean(axis=1, keepdims=True)  # only differences matter to a softmax
    if confidence_weighted:
        bias *= post.max(axis=1, keepdims=True)
    return _pad_to_alphabet(bias)


def sequence_to_bias(seq: str, *, strength: float = 2.0) -> np.ndarray:
    """One-hot bias toward a fixed sequence — the blunt "lock to this ancestor"."""
    bias = np.zeros((len(seq), len(ALPHABET)))
    for i, residue in enumerate(seq):
        index = ALPHABET.find(residue.upper())
        if index >= 0:
            bias[i, index] = strength
    return bias


def mask_segment(
    bias: np.ndarray, segment: tuple[int, int], *, inside: bool = True, scale: float = 0.0
) -> np.ndarray:
    """Scale the bias inside (or outside) a segment.

    The intended use is to stop conditioning on a block flagged as contaminated:
    keep the coherent part of the reconstruction driving the design and let the
    suspect region be redesigned from the structure alone.
    """
    out = np.array(bias, dtype=float, copy=True)
    start, stop = segment
    selector = np.zeros(len(out), dtype=bool)
    selector[start:stop] = True
    if not inside:
        selector = ~selector
    out[selector] *= scale
    return out


def write_bias_by_res(
    pdb_name: str,
    chain: str,
    bias: np.ndarray,
    path: str | Path,
    *,
    extra_chains: Mapping[str, np.ndarray] | None = None,
) -> Path:
    """Write ProteinMPNN's `--bias_by_res_jsonl` format.

    Shape: {"pdb_name": {"chain": [[21 floats] per position]}}
    """
    entry: dict[str, list] = {chain: _pad_to_alphabet(bias).tolist()}
    for other, values in (extra_chains or {}).items():
        entry[other] = _pad_to_alphabet(values).tolist()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({pdb_name: entry}) + "\n")
    return path


def write_pssm(
    pdb_name: str,
    chain: str,
    profile: np.ndarray,
    path: str | Path,
    *,
    coefficient: float | Sequence[float] = 1.0,
    log_odds_background: np.ndarray | None = None,
) -> Path:
    """Write ProteinMPNN's `--pssm_jsonl` format.

    Shape: {"pdb_name": {"chain": {"pssm_coef": [L], "pssm_bias": [L][21],
                                   "pssm_log_odds": [L][21]}}}

    `pssm_bias` is the probability profile itself; `pssm_log_odds` is that
    profile against a background (uniform unless one is supplied) and is what
    `--pssm_threshold` filters on.
    """
    prof = _pad_to_alphabet(profile)
    prof = prof / np.clip(prof.sum(axis=1, keepdims=True), EPS, None)

    background = (
        np.full(len(ALPHABET), 1.0 / len(CANONICAL))
        if log_odds_background is None
        else _pad_to_alphabet(np.asarray(log_odds_background)[None, :])[0]
    )
    log_odds = np.log(np.clip(prof, EPS, None) / np.clip(background, EPS, None))

    length = prof.shape[0]
    coef = (
        np.full(length, float(coefficient))
        if np.isscalar(coefficient)
        else np.asarray(coefficient, dtype=float)
    )
    if len(coef) != length:
        raise ValueError(f"coefficient has length {len(coef)}, expected {length}")

    payload = {
        pdb_name: {
            chain: {
                "pssm_coef": coef.tolist(),
                "pssm_bias": prof.tolist(),
                "pssm_log_odds": log_odds.tolist(),
            }
        }
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
    return path


def write_fixed_positions(
    pdb_name: str, chain: str, positions: Sequence[int], path: str | Path
) -> Path:
    """Write `--fixed_positions_jsonl`; positions are 1-based, as MPNN expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({pdb_name: {chain: [int(p) for p in positions]}}) + "\n")
    return path


def confident_positions(posteriors: np.ndarray, *, threshold: float = 0.95) -> list[int]:
    """1-based positions whose ancestral state is called above `threshold`.

    Handing these to `write_fixed_positions` is the strict reading of the
    apparatus: hold everything the reconstruction is certain about and redesign
    only what it is not.
    """
    post = np.asarray(posteriors, dtype=float)
    return [int(i) + 1 for i in np.flatnonzero(post.max(axis=1) >= threshold)]

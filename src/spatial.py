"""Contamination that is contiguous in space rather than in sequence.

Everything measured so far swaps a contiguous *sequence* block, and the scan
that looks for it runs over alignment position. That is a game a string
comparison wins by construction: the answer is written in the sequence and the
search is over the sequence.

Gene conversion and domain-level recombination in a folded protein do not have
to respect sequence order. A converted surface, an exchanged binding face, a
swapped structural element after loop remodelling — these are contiguous on the
*backbone* while scattered across the alignment. That case is invisible to a
one-dimensional scan no matter what score feeds it, because the thing being
searched for does not exist in the coordinate the search runs over.

So this module supplies the two halves of that experiment:

    spatial_patch      pick a set of residues contiguous in 3D, scattered in 1D
    spatial_scan       find the spatial cluster whose score is most extreme,
                       with a permutation null that destroys spatial structure
                       while preserving the marginal distribution of the score

The comparison that matters is a 2x2, not a 1x2. Both the structural score and
the sequence-identity score must be run through both scans:

                  1D scan          3D scan
    MPNN          measured         measured
    identity      measured         measured

If MPNN+3D beats identity+3D, the structural *model* is doing the work. If the
two 3D cells are equal and both beat the 1D cells, the structural *scan* is
doing the work and the model is still not earning its place — a different, and
more likely, conclusion that this design can actually distinguish.
"""

from __future__ import annotations

import numpy as np

from conflict import smooth


def ca_coords(scorer) -> np.ndarray:
    """(L, 3) alpha-carbon coordinates. Backbone.X is (1, L, 4, 3) = N, CA, C, O."""
    return scorer.backbone.X[0, :, 1, :].detach().cpu().numpy()


def neighbour_sets(coords: np.ndarray, k: int = 12) -> np.ndarray:
    """(L, k) indices of each residue's k nearest neighbours, itself included."""
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return np.argsort(d, axis=1)[:, :k]


def spatial_patch(coords: np.ndarray, centre: int, size: int) -> np.ndarray:
    """The `size` residues nearest `centre` in space — a 3D-contiguous patch.

    Returned sorted by position, which is how it becomes visibly *dis*contiguous
    in sequence: a compact ball on the fold is typically drawn from several
    separate stretches of the chain.
    """
    d = np.linalg.norm(coords - coords[centre], axis=1)
    return np.sort(np.argsort(d)[:size])


def sequence_runs(positions: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs in a sorted position array — how broken up a patch is."""
    runs: list[tuple[int, int]] = []
    start = prev = int(positions[0])
    for p in positions[1:]:
        p = int(p)
        if p == prev + 1:
            prev = p
            continue
        runs.append((start, prev + 1))
        start = prev = p
    runs.append((start, prev + 1))
    return runs


def spatial_scan(
    values: np.ndarray, neighbours: np.ndarray, *, min_size: int = 6
) -> tuple[np.ndarray, float]:
    """Find the spatial neighbourhood whose mean score most differs from the rest.

    The structural analogue of `conflict.scan_segment`: same pooled-variance
    two-sample t, so the two scans are compared on equal terms and a difference
    between them is a difference in *where* they search, not in how they score.

    Returns (member indices of the best neighbourhood, statistic).
    """
    x = np.asarray(values, dtype=float)
    L = len(x)
    best: tuple[np.ndarray, float] = (np.arange(L), 0.0)
    for centre in range(L):
        members = neighbours[centre]
        members = members[members < L]
        if len(members) < min_size:
            continue
        mask = np.zeros(L, dtype=bool)
        mask[members] = True
        n_in = int(mask.sum())
        n_out = L - n_in
        if n_out < min_size:
            continue
        m_in, m_out = x[mask].mean(), x[~mask].mean()
        pooled = (n_in * x[mask].var() + n_out * x[~mask].var()) / max(L - 2, 1)
        if pooled <= 1e-12:
            continue
        stat = abs(m_in - m_out) / np.sqrt(pooled * (1 / n_in + 1 / n_out))
        if stat > best[1]:
            best = (np.flatnonzero(mask), float(stat))
    return best


def spatial_permutation_test(
    values: np.ndarray, neighbours: np.ndarray, *, n_perm: int = 200,
    alpha: float = 0.05, seed: int = 0, min_size: int = 6,
) -> tuple[float, float]:
    """Null: the same per-site scores, scattered at random over the backbone.

    Identical in spirit to the 1D permutation test — what is being tested is
    spatial clustering of the score, not its overall magnitude.
    """
    rng = np.random.default_rng(seed)
    observed = spatial_scan(values, neighbours, min_size=min_size)[1]
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = spatial_scan(rng.permutation(values), neighbours, min_size=min_size)[1]
    p = float((1 + np.sum(null >= observed)) / (1 + n_perm))
    return p, float(np.quantile(null, 1 - alpha))


def scan_1d(values: np.ndarray, sites: np.ndarray, *, window: int = 5, min_len: int = 3):
    """The existing 1D scan, restricted to `sites`, for side-by-side comparison."""
    from conflict import scan_segment

    sub = np.asarray(values)[sites]
    (start, stop), stat = scan_segment(smooth(np.sign(sub), window), min_len=min_len)
    return np.array(sites[start:stop]), float(stat)


def permutation_test_1d(
    values: np.ndarray, sites: np.ndarray, *, n_perm: int = 200, seed: int = 0,
    window: int = 5, min_len: int = 3,
) -> float:
    from conflict import permutation_test

    return permutation_test(
        np.sign(np.asarray(values)[sites]), n_perm=n_perm, seed=seed,
        min_len=min_len, window=window
    )[0]


def patch_jaccard(found: np.ndarray, true: np.ndarray) -> float:
    """Set Jaccard — the patch is a set of positions, not an interval."""
    a, b = set(int(x) for x in found), set(int(x) for x in true)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

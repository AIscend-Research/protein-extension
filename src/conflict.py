"""Structural detection of contamination in a reconstructed ancestor.

Marginal ASR infers every site independently, so a reconstruction stitched from
two incompatible histories is invisible to per-site diagnostics: each site is
individually well supported. The pathology is joint.

ProteinMPNN's decoder is a joint-compatibility model — it scores a residue given
the *other residues* and the backbone — so it can see what marginal posteriors
cannot. Two probes, both built on decoding-order control from [mpnn_api]:

**Context swap.** Reconstruct the two candidate sub-histories separately (the two
clades of the stemma) and ask, for each site of the mosaic ancestor,

    delta_i = log p(a_i | context = ancestor_A) - log p(a_i | context = ancestor_B)

A coherent ancestor gives delta ~ 0 with no spatial structure: its residues are
equally at home in either context because the contexts barely differ. A
contaminated ancestor gives a *contiguous run* of sites that prefer one context
and a complementary run that prefers the other. The breakpoint is where the sign
flips.

**Order instability.** Decode the ancestor under many random decoding orders and
measure how much site i's preferred residue depends on which neighbours happened
to be filled in first. A residue that is plausible given lineage-A neighbours and
implausible given lineage-B neighbours flips; a coherent site does not.

Significance comes from a permutation test that destroys spatial contiguity while
preserving the marginal distribution of delta — so the null is "these per-site
scores, arranged at random", and what gets tested is the segmental structure that
contamination produces and ordinary functional coupling does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from mpnn_api import CANONICAL, MPNNScorer, seq_to_idx


# ------------------------------------------------------------------ segments


def scan_segment(
    values: np.ndarray, *, min_len: int = 5, max_frac: float = 0.75
) -> tuple[tuple[int, int], float]:
    """Find the contiguous window whose mean most differs from the rest.

    Returns ((start, stop), statistic). The statistic is a pooled-variance
    two-sample t, so it does not reward tiny windows of extreme sites.
    """
    x = np.asarray(values, dtype=float)
    L = len(x)
    max_len = max(min_len, int(L * max_frac))
    prefix = np.concatenate([[0.0], np.cumsum(x)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(x * x)])
    total, total_sq = prefix[-1], prefix_sq[-1]

    best: tuple[tuple[int, int], float] = ((0, L), 0.0)
    for start in range(0, L - min_len + 1):
        for stop in range(start + min_len, min(start + max_len, L) + 1):
            n_in = stop - start
            n_out = L - n_in
            if n_out < min_len:
                continue
            s_in = prefix[stop] - prefix[start]
            q_in = prefix_sq[stop] - prefix_sq[start]
            s_out, q_out = total - s_in, total_sq - q_in
            m_in, m_out = s_in / n_in, s_out / n_out
            var_in = max(q_in / n_in - m_in ** 2, 0.0)
            var_out = max(q_out / n_out - m_out ** 2, 0.0)
            pooled = (n_in * var_in + n_out * var_out) / max(L - 2, 1)
            if pooled <= 1e-12:
                continue
            stat = abs(m_in - m_out) / np.sqrt(pooled * (1 / n_in + 1 / n_out))
            if stat > best[1]:
                best = ((start, stop), float(stat))
    return best


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average; contamination is contiguous, isolated sites are noise."""
    if window <= 1:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(np.asarray(values, dtype=float), (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def permutation_test(
    values: np.ndarray,
    *,
    n_perm: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
    min_len: int = 5,
    window: int = 1,
) -> tuple[float, float, np.ndarray]:
    """Null: the same per-site scores in random spatial arrangement.

    Smoothing is re-applied *after* each permutation, so the null carries the
    same autocorrelation the smoother induces and the test measures genuine
    contiguity rather than the width of the window.

    Returns (p_value, threshold_at_alpha, null_statistics).
    """
    rng = np.random.default_rng(seed)
    observed = scan_segment(smooth(values, window), min_len=min_len)[1]
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = scan_segment(smooth(rng.permutation(values), window), min_len=min_len)[1]
    p = float((1 + np.sum(null >= observed)) / (1 + n_perm))
    threshold = float(np.quantile(null, 1 - alpha))
    return p, threshold, null


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC; 0.5 is chance."""
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for k in np.flatnonzero(counts > 1):
        mask = inverse == k
        ranks[mask] = ranks[mask].mean()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def jaccard(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float:
    if a is None or b is None:
        return 0.0
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


# -------------------------------------------------------------------- probes


@dataclass
class ConflictResult:
    """Per-site conflict scores plus the segment they imply."""

    delta: np.ndarray
    logp_a: np.ndarray
    logp_b: np.ndarray
    instability: np.ndarray
    segment: tuple[int, int] | None
    statistic: float
    p_value: float
    threshold: float
    detected: bool
    ancestor: str
    context_a: str
    context_b: str
    diff_sites: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    def assignment(self) -> np.ndarray:
        """Per-site sub-history call: +1 leans to context A, -1 to context B."""
        return np.where(self.delta >= 0, 1, -1)

    def summary(self) -> dict[str, object]:
        return {
            "segment": list(self.segment) if self.segment else None,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 5),
            "threshold": round(self.threshold, 4),
            "detected": self.detected,
            "mean_abs_delta": round(float(np.mean(np.abs(self.delta))), 4),
            "n_diff_sites": int(len(self.diff_sites)),
        }


def context_swap(
    scorer: MPNNScorer, ancestor: str, context_a: str, context_b: str
) -> tuple[np.ndarray, np.ndarray]:
    """log p(a_i | rest = A) and log p(a_i | rest = B), for every site."""
    return (
        scorer.site_log_prob(ancestor, context_seq=context_a),
        scorer.site_log_prob(ancestor, context_seq=context_b),
    )


def order_instability(
    scorer: MPNNScorer, seq: str, *, n_orders: int = 64, seed: int = 0
) -> np.ndarray:
    """Per-site sensitivity of the preferred residue to the decoding order.

    Decoding under a random order fills a random subset of neighbours before
    site i. If i is compatible with everything, its argmax is stable; if it is
    plausible only given one lineage's neighbourhood, the argmax flips with the
    order. Score = 1 - (frequency of the modal argmax).
    """
    rng = np.random.default_rng(seed)
    orders = scorer.random_orders(n_orders, rng)
    lp = scorer.log_probs(seq, orders)[:, :, :20]  # (K, L, 20)
    argmax = lp.argmax(axis=2)  # (K, L)
    L = argmax.shape[1]
    out = np.empty(L)
    for i in range(L):
        counts = np.bincount(argmax[:, i], minlength=20)
        out[i] = 1.0 - counts.max() / n_orders
    return out


def detect_contamination(
    scorer: MPNNScorer,
    ancestor: str,
    context_a: str,
    context_b: str,
    *,
    n_perm: int = 200,
    alpha: float = 0.05,
    min_len: int = 3,
    n_orders: int = 64,
    seed: int = 0,
    window: int = 5,
    min_diagnostic: int = 12,
) -> ConflictResult:
    """Run both probes and test whether the conflict is segmentally structured.

    The scan runs in *diagnostic-site space*: only positions where the two
    candidate sub-ancestors actually differ carry information about which
    sub-history a site came from, and positions where they agree contribute pure
    noise. Scoring the sign of delta rather than its magnitude keeps a handful of
    structurally extreme sites from dominating, and a moving average over
    neighbouring diagnostic sites exploits the fact that contamination arrives in
    contiguous blocks.

    Detection needs enough diagnostic sites to be possible at all; below
    `min_diagnostic` the result is reported as undetected rather than guessed.
    """
    logp_a, logp_b = context_swap(scorer, ancestor, context_a, context_b)
    delta = logp_a - logp_b
    instability = order_instability(scorer, ancestor, n_orders=n_orders, seed=seed)

    diff_sites = np.flatnonzero(np.array(list(context_a)) != np.array(list(context_b)))
    if len(diff_sites) < max(min_diagnostic, 2 * min_len):
        return ConflictResult(
            delta=delta, logp_a=logp_a, logp_b=logp_b, instability=instability,
            segment=None, statistic=0.0, p_value=1.0, threshold=float("inf"),
            detected=False, ancestor=ancestor, context_a=context_a,
            context_b=context_b, diff_sites=diff_sites,
        )

    signs = np.sign(delta[diff_sites])
    scanned = smooth(signs, window)
    (start, stop), statistic = scan_segment(scanned, min_len=min_len)
    segment = (int(diff_sites[start]), int(diff_sites[stop - 1]) + 1)
    p_value, threshold, _ = permutation_test(
        signs, n_perm=n_perm, alpha=alpha, seed=seed, min_len=min_len, window=window
    )

    return ConflictResult(
        delta=delta,
        logp_a=logp_a,
        logp_b=logp_b,
        instability=instability,
        segment=segment if p_value <= alpha else None,
        statistic=statistic,
        p_value=p_value,
        threshold=threshold,
        detected=bool(p_value <= alpha),
        ancestor=ancestor,
        context_a=context_a,
        context_b=context_b,
        diff_sites=diff_sites,
    )


def pairwise_conflict(
    scorer: MPNNScorer,
    ancestor: str,
    context_a: str,
    context_b: str,
    *,
    max_sites: int = 48,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Influence matrix: how much does flipping site j's ancestry move site i?

    Returns (sites, M) with M[j, i] = |log p(a_i | ctx_A) - log p(a_i | ctx_A but
    j taken from B)|. Rows cluster into the two sub-histories, which is the
    structural analogue of separating hyparchetypes.
    """
    rng = np.random.default_rng(seed)
    diff = np.flatnonzero(np.array(list(context_a)) != np.array(list(context_b)))
    if len(diff) > max_sites:
        diff = np.sort(rng.choice(diff, size=max_sites, replace=False))

    base = scorer.site_log_prob(ancestor, context_seq=context_a)
    rows = np.empty((len(diff), len(ancestor)))
    for k, j in enumerate(diff):
        ctx = list(context_a)
        ctx[j] = context_b[j]
        rows[k] = np.abs(scorer.site_log_prob(ancestor, context_seq="".join(ctx)) - base)
    return diff, rows


def spectral_split(matrix: np.ndarray) -> np.ndarray:
    """Two-way split of rows by the Fiedler vector of their correlation graph."""
    centred = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    corr = (centred / np.clip(norms, 1e-9, None)) @ (centred / np.clip(norms, 1e-9, None)).T
    affinity = np.clip(corr, 0, None)
    np.fill_diagonal(affinity, 0.0)
    degree = affinity.sum(axis=1)
    laplacian = np.diag(degree) - affinity
    with np.errstate(invalid="ignore"):
        vals, vecs = np.linalg.eigh(laplacian)
    order = np.argsort(vals)
    fiedler = vecs[:, order[1]] if len(vals) > 1 else np.zeros(len(vals))
    return np.where(fiedler >= 0, 1, -1)

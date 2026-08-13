"""Evaluation of designed / reconstructed sequences.

Sequence recovery against a reference, perplexity from ProteinMPNN log-probs,
agreement with the phylogenetic posterior, and per-label breakdowns from
[apparatus] (does recovery differ across certain / probable / conjectural
positions?).

Planned surface:
    recovery(seq, reference) -> float
    perplexity(log_probs, seq) -> float
    recovery_by_label(seq, reference, calls) -> dict[str, float]

For a known-good baseline, run the sequence-recovery benchmark from the separate
reproduction repo (see README) rather than reimplementing it here.
"""

from __future__ import annotations

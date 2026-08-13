"""Homolog family assembly — the "witnesses" to an ancestral sequence.

Collect homologs for a target protein, filter and align them, and record the
provenance of each witness (accession, source database, identity to the target,
coverage) so the apparatus can cite what supports each position.

Planned surface:
    fetch_homologs(query, ...) -> list[Witness]
    filter_family(witnesses, min_identity=..., max_identity=..., min_coverage=...)
    write_alignment(witnesses, path)

Consumed by [stemma] (tree inference / ancestral reconstruction) and, through the
FastML posteriors, by [apparatus].
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Witness:
    """One homologous sequence supporting the reconstruction."""

    accession: str
    seq: str
    source: str = ""
    identity: float | None = None
    coverage: float | None = None
    organism: str = ""

"""Assembling the witnesses: real homolog families from public data.

Everything upstream of the detector so far has been simulated. This module is
the path to empirical families, which is where the method's claim actually has
to hold — the `selection` simulator samples from ProteinMPNN's own joint
distribution and so assumes the epistasis the detector then senses.

Data sources, all public and key-free:

  UniProt REST      sequences, with structure cross-references
  RCSB PDB          experimental backbones
  AlphaFold DB      a predicted model for essentially any UniProt accession

Nothing here calls an alignment program. Families are fetched and filtered, then
handed to MAFFT (or any aligner) externally; `read_alignment` reads the result
back. Keeping the aligner out of process means no extra install for the parts of
the pipeline that only need sequences.

    python src/witnesses.py --query '"three-finger toxin" AND reviewed:true AND database:pdb' \\
        --out data/raw/3ftx.fasta --limit 200
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
ALPHAFOLD_PDB = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.pdb"
RCSB_PDB = "https://files.rcsb.org/download/{pdb_id}.pdb"
USER_AGENT = "protein-extension/0.1 (research; contact via repository)"

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class Witness:
    """One homologous sequence, with enough provenance to cite it."""

    accession: str
    seq: str
    name: str = ""
    organism: str = ""
    length: int = 0
    pdb_ids: list[str] = field(default_factory=list)
    identity: float | None = None
    coverage: float | None = None

    @property
    def has_structure(self) -> bool:
        return bool(self.pdb_ids)

    def to_dict(self) -> dict:
        return {
            "accession": self.accession,
            "name": self.name,
            "organism": self.organism,
            "length": self.length,
            "pdb_ids": self.pdb_ids,
            "identity": self.identity,
            "coverage": self.coverage,
        }


def _get(url: str, *, retries: int = 3, pause: float = 1.0) -> bytes:
    """GET with a couple of retries; public endpoints throttle under bursts."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - surfaced after the final retry
            last = exc
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def fetch_homologs(
    query: str, *, limit: int = 200, page_size: int = 100, require_structure: bool = False
) -> list[Witness]:
    """Search UniProt and return witnesses, following pagination.

    `query` is UniProt query syntax, e.g.
    `'"three-finger toxin" AND reviewed:true AND database:pdb'`.
    """
    fields = "accession,id,organism_name,length,sequence,xref_pdb"
    url = (
        f"{UNIPROT_SEARCH}?query={urllib.parse.quote(query)}"
        f"&fields={fields}&format=json&size={min(page_size, limit)}"
    )
    out: list[Witness] = []
    while url and len(out) < limit:
        raw = _get(url)
        payload = json.loads(raw)
        for entry in payload.get("results", []):
            seq = entry.get("sequence", {}).get("value", "")
            pdb_ids = [
                ref.get("id", "")
                for ref in entry.get("uniProtKBCrossReferences", [])
                if ref.get("database") == "PDB"
            ]
            if require_structure and not pdb_ids:
                continue
            out.append(
                Witness(
                    accession=entry.get("primaryAccession", ""),
                    seq=seq,
                    name=entry.get("uniProtkbId", ""),
                    organism=entry.get("organism", {}).get("scientificName", ""),
                    length=entry.get("sequence", {}).get("length", len(seq)),
                    pdb_ids=sorted(set(pdb_ids)),
                )
            )
            if len(out) >= limit:
                break
        url = None  # UniProt paginates via a Link header, not the body
    return out


def fetch_structure(
    accession_or_pdb: str, out_dir: str | Path, *, source: str = "alphafold"
) -> Path:
    """Download one backbone. `source` is "alphafold" (UniProt accession) or "pdb"."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if source == "alphafold":
        url = ALPHAFOLD_PDB.format(accession=accession_or_pdb)
        path = out_dir / f"AF-{accession_or_pdb}.pdb"
    elif source == "pdb":
        url = RCSB_PDB.format(pdb_id=accession_or_pdb.upper())
        path = out_dir / f"{accession_or_pdb.upper()}.pdb"
    else:
        raise ValueError(f"unknown source {source!r}; use alphafold|pdb")
    path.write_bytes(_get(url))
    return path


def identity(a: str, b: str) -> float:
    """Fraction of aligned, non-gap columns that match."""
    pairs = [(x, y) for x, y in zip(a, b) if x != "-" and y != "-"]
    if not pairs:
        return 0.0
    return float(np.mean([x == y for x, y in pairs]))


def coverage(seq: str, reference: str) -> float:
    """Fraction of the reference covered by non-gap positions of `seq`."""
    if not reference:
        return 0.0
    return float(sum(1 for x, y in zip(seq, reference) if x != "-" and y != "-") / len(reference))


def filter_family(
    witnesses: Sequence[Witness],
    *,
    reference: str | None = None,
    min_identity: float = 0.25,
    max_identity: float = 0.95,
    min_length: int = 40,
    max_length: int = 400,
    require_structure: bool = False,
    drop_nonstandard: bool = True,
) -> list[Witness]:
    """Keep a usable family: diverged enough to be informative, close enough to align.

    The identity band matters for this project specifically. Near-identical
    witnesses contribute no diagnostic sites, so the two sub-histories become
    indistinguishable and there is nothing to detect; wildly diverged ones make
    the alignment itself the dominant source of error.
    """
    kept: list[Witness] = []
    for witness in witnesses:
        if not (min_length <= len(witness.seq) <= max_length):
            continue
        if require_structure and not witness.has_structure:
            continue
        if drop_nonstandard and not set(witness.seq) <= CANONICAL:
            continue
        if reference is not None:
            witness.identity = identity(witness.seq, reference)
            if not (min_identity <= witness.identity <= max_identity):
                continue
        kept.append(witness)
    return kept


def deduplicate(witnesses: Sequence[Witness], *, threshold: float = 0.98) -> list[Witness]:
    """Drop witnesses that are near-copies of one already kept.

    Redundant witnesses inflate apparent support for whichever sub-history they
    belong to, which is the sequence analogue of counting one manuscript twice.
    """
    kept: list[Witness] = []
    for witness in witnesses:
        if all(identity(witness.seq, other.seq) < threshold for other in kept):
            kept.append(witness)
    return kept


def write_fasta(witnesses: Iterable[Witness], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for witness in witnesses:
            header = f"{witness.accession}"
            if witness.organism:
                header += f" {witness.organism.replace(' ', '_')}"
            handle.write(f">{header}\n{witness.seq}\n")
    return path


def write_manifest(witnesses: Iterable[Witness], path: str | Path) -> Path:
    """Provenance for every witness — what the apparatus ultimately cites."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([w.to_dict() for w in witnesses], indent=2))
    return path


def read_alignment(path: str | Path) -> dict[str, str]:
    """Read an aligned FASTA (from MAFFT or similar); all rows must be equal length."""
    seqs: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(chunks)
            name = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if name is not None:
        seqs[name] = "".join(chunks)

    lengths = {len(s) for s in seqs.values()}
    if len(lengths) > 1:
        raise ValueError(f"not an alignment: {len(lengths)} distinct row lengths {sorted(lengths)}")
    return seqs


def ungapped_columns(alignment: dict[str, str], *, max_gap_fraction: float = 0.2) -> list[int]:
    """Columns retained after dropping gap-heavy ones — 0-based indices.

    A reconstruction is only meaningful on columns most witnesses actually
    occupy, and the shared-backbone assumption needs a common core.
    """
    rows = list(alignment.values())
    if not rows:
        return []
    length = len(rows[0])
    keep = []
    for column in range(length):
        gaps = sum(1 for row in rows if row[column] == "-")
        if gaps / len(rows) <= max_gap_fraction:
            keep.append(column)
    return keep


def project(alignment: dict[str, str], columns: Sequence[int]) -> dict[str, str]:
    """Restrict an alignment to the given columns."""
    return {name: "".join(seq[c] for c in columns) for name, seq in alignment.items()}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Fetch and filter a homolog family from UniProt.")
    ap.add_argument("--query", required=True, help="UniProt query syntax")
    ap.add_argument("--out", default="data/raw/family.fasta")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--require-structure", action="store_true")
    ap.add_argument("--min-identity", type=float, default=0.25)
    ap.add_argument("--max-identity", type=float, default=0.95)
    ap.add_argument("--reference", default=None, help="accession to measure identity against")
    ap.add_argument("--structures", default=None, help="directory to download AlphaFold models into")
    args = ap.parse_args()

    found = fetch_homologs(args.query, limit=args.limit, require_structure=args.require_structure)
    print(f"fetched {len(found)} entries")
    if not found:
        return

    reference = None
    if args.reference:
        match = next((w for w in found if w.accession == args.reference), None)
        if match is None:
            raise SystemExit(f"reference {args.reference} not in the result set")
        reference = match.seq
    else:
        reference = max(found, key=lambda w: len(w.pdb_ids)).seq

    kept = deduplicate(
        filter_family(
            found,
            reference=reference,
            min_identity=args.min_identity,
            max_identity=args.max_identity,
            require_structure=args.require_structure,
        )
    )
    print(f"kept {len(kept)} after filtering and deduplication")
    print(f"with structures: {sum(w.has_structure for w in kept)}")

    out = write_fasta(kept, args.out)
    manifest = write_manifest(kept, Path(args.out).with_suffix(".manifest.json"))
    print(f"wrote {out} and {manifest}")
    print("next: align, e.g.  mafft --auto", out, "> alignment.fasta")

    if args.structures:
        for witness in kept[:5]:
            try:
                path = fetch_structure(witness.accession, args.structures)
                print(f"  {witness.accession} -> {path}")
            except RuntimeError as exc:
                print(f"  {witness.accession}: {exc}")


if __name__ == "__main__":
    main()

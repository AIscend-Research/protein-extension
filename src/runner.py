"""Thin wrapper around ProteinMPNN's command-line entry point.

This is the seam between this project and dauparas/ProteinMPNN: nothing else in
`src/` should know that ProteinMPNN is a subprocess, a CLI, or a set of .fa/.npz
files on disk. Downstream phases generate a different bias/pssm jsonl and call
`run_mpnn`.

Usage:
    from runner import run_mpnn
    res = run_mpnn("data/raw/5L33.pdb", num_seq=8, temperature=0.2)
    res.probs.shape        # (n_samples, n_residues, 21)
    res.designs[0].seq
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# ProteinMPNN's fixed output alphabet (21st symbol is the unknown/mask token).
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

REPO_ROOT = Path(__file__).resolve().parents[1]
MPNN_DIR = REPO_ROOT / "proteinmpnn"
STOCK_SCRIPT = MPNN_DIR / "protein_mpnn_run.py"
# Kept outside the submodule so it stays tracked here; it imports
# protein_mpnn_utils, so MPNN_DIR is put on PYTHONPATH when it runs.
MPS_SCRIPT = REPO_ROOT / "patches" / "protein_mpnn_run_mps_patch.py"


class MPNNError(RuntimeError):
    """ProteinMPNN exited non-zero, or produced no parseable output."""


@dataclass
class Design:
    """One sampled sequence plus the metadata ProteinMPNN puts in the FASTA header."""

    seq: str
    header: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def temperature(self) -> float | None:
        return self.meta.get("T")

    @property
    def score(self) -> float | None:
        return self.meta.get("score")

    @property
    def seq_recovery(self) -> float | None:
        return self.meta.get("seq_recovery")


@dataclass
class MPNNResult:
    """Everything one ProteinMPNN invocation produced, already loaded."""

    name: str
    native: Design
    designs: list[Design]
    probs: np.ndarray | None  # (n_samples, n_res, 21)
    log_probs: np.ndarray | None
    S: np.ndarray | None  # (n_samples, n_res) sampled token indices
    mask: np.ndarray | None  # (1, n_res)
    chain_order: np.ndarray | None
    fasta_path: Path
    probs_path: Path | None
    out_folder: Path
    command: list[str]

    @property
    def sequences(self) -> list[str]:
        return [d.seq for d in self.designs]

    @property
    def mean_probs(self) -> np.ndarray | None:
        """Per-position distribution averaged over samples: (n_res, 21)."""
        return None if self.probs is None else self.probs.mean(axis=0)


def _pick_script(device: str) -> Path:
    """Choose the stock runner or the Apple-silicon MPS patch.

    device: "auto" | "mps" | "stock"
    """
    if device == "stock":
        return STOCK_SCRIPT
    if device == "mps":
        if not MPS_SCRIPT.exists():
            raise MPNNError(f"MPS patch not found at {MPS_SCRIPT}")
        return MPS_SCRIPT
    if device != "auto":
        raise ValueError(f"unknown device {device!r}; use auto|mps|stock")

    if MPS_SCRIPT.exists():
        try:
            import torch  # noqa: PLC0415 - optional, only to probe the backend

            if torch.backends.mps.is_available() and not torch.cuda.is_available():
                return MPS_SCRIPT
        except Exception:
            pass
    return STOCK_SCRIPT


def _parse_header(header: str) -> dict[str, Any]:
    """Parse `>name, score=1.23, global_score=..., T=0.1, sample=1` style headers."""
    meta: dict[str, Any] = {}
    for chunk in header.split(","):
        if "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        key, value = key.strip(), value.strip()
        try:
            meta[key] = float(value)
        except ValueError:
            meta[key] = value
    return meta


def parse_fasta(path: Path) -> tuple[Design, list[Design]]:
    """Read a ProteinMPNN .fa. Record 0 is the native sequence; the rest are designs."""
    records: list[Design] = []
    header: str | None = None
    chunks: list[str] = []

    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                seq = "".join(chunks)
                records.append(Design(seq=seq, header=header, meta=_parse_header(header)))
            header = line[1:]
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        records.append(Design(seq="".join(chunks), header=header, meta=_parse_header(header)))

    if not records:
        raise MPNNError(f"no records parsed from {path}")
    return records[0], records[1:]


def run_mpnn(
    pdb_path: str | Path,
    out_folder: str | Path | None = None,
    *,
    num_seq: int = 1,
    temperature: float | Sequence[float] = 0.1,
    bias_by_res_jsonl: str | Path | None = None,
    pssm_jsonl: str | Path | None = None,
    pssm_multi: float | None = None,
    pssm_threshold: float | None = None,
    pssm_log_odds_flag: bool = False,
    pssm_bias_flag: bool = False,
    fixed_positions_jsonl: str | Path | None = None,
    omit_aa_jsonl: str | Path | None = None,
    tied_positions_jsonl: str | Path | None = None,
    chain_id_jsonl: str | Path | None = None,
    model_name: str = "v_48_020",
    path_to_model_weights: str | Path | None = None,
    use_soluble_model: bool = False,
    ca_only: bool = False,
    backbone_noise: float = 0.0,
    batch_size: int = 1,
    seed: int = 0,
    save_probs: bool = True,
    save_score: bool = False,
    unconditional_probs_only: bool = False,
    conditional_probs_only: bool = False,
    device: str = "auto",
    python: str | Path | None = None,
    extra_args: Sequence[str] = (),
    quiet: bool = True,
) -> MPNNResult:
    """Run ProteinMPNN once and return its sequences plus per-residue probabilities.

    `bias_by_res_jsonl` / `pssm_jsonl` are the two conditioning channels the rest
    of this project uses; everything else is passed straight through.
    """
    pdb_path = Path(pdb_path).resolve()
    if not pdb_path.exists():
        raise FileNotFoundError(pdb_path)

    name = pdb_path.stem
    out_folder = Path(out_folder) if out_folder else REPO_ROOT / "scratch" / "runner" / name
    out_folder = out_folder.resolve()
    out_folder.mkdir(parents=True, exist_ok=True)

    temps = [temperature] if isinstance(temperature, (int, float)) else list(temperature)
    script = _pick_script(device)
    interpreter = str(python) if python else sys.executable

    cmd = [
        interpreter,
        str(script),
        "--pdb_path", str(pdb_path),
        "--out_folder", str(out_folder),
        "--num_seq_per_target", str(num_seq),
        "--sampling_temp", " ".join(str(t) for t in temps),
        "--batch_size", str(batch_size),
        "--backbone_noise", str(backbone_noise),
        "--model_name", model_name,
        "--seed", str(seed),
        "--save_probs", "1" if save_probs else "0",
        "--save_score", "1" if save_score else "0",
    ]

    if path_to_model_weights is None and script != STOCK_SCRIPT:
        # The runner scripts resolve weights relative to their own location, and
        # the MPS patch lives outside the submodule — point it back at the weights.
        folder = "ca_model_weights" if ca_only else (
            "soluble_model_weights" if use_soluble_model else "vanilla_model_weights"
        )
        path_to_model_weights = MPNN_DIR / folder

    optional: list[tuple[str, Any]] = [
        ("--path_to_model_weights", path_to_model_weights),
        ("--bias_by_res_jsonl", bias_by_res_jsonl),
        ("--pssm_jsonl", pssm_jsonl),
        ("--pssm_multi", pssm_multi),
        ("--pssm_threshold", pssm_threshold),
        ("--fixed_positions_jsonl", fixed_positions_jsonl),
        ("--omit_AA_jsonl", omit_aa_jsonl),
        ("--tied_positions_jsonl", tied_positions_jsonl),
        ("--chain_id_jsonl", chain_id_jsonl),
    ]
    for flag, value in optional:
        if value is not None:
            cmd += [flag, str(value)]

    if pssm_log_odds_flag:
        cmd += ["--pssm_log_odds_flag", "1"]
    if pssm_bias_flag:
        cmd += ["--pssm_bias_flag", "1"]
    if unconditional_probs_only:
        cmd += ["--unconditional_probs_only", "1"]
    if conditional_probs_only:
        cmd += ["--conditional_probs_only", "1"]
    if use_soluble_model:
        cmd += ["--use_soluble_model"]
    if ca_only:
        cmd += ["--ca_only"]
    cmd += list(extra_args)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(MPNN_DIR), env.get("PYTHONPATH", "")]))
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(MPNN_DIR), env=env)
    if proc.returncode != 0:
        raise MPNNError(
            f"ProteinMPNN failed (exit {proc.returncode})\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    if not quiet:
        print(proc.stdout)

    fasta_path = out_folder / "seqs" / f"{name}.fa"
    if not fasta_path.exists():
        raise MPNNError(f"expected {fasta_path}; ProteinMPNN stdout:\n{proc.stdout[-2000:]}")
    native, designs = parse_fasta(fasta_path)

    probs_path = out_folder / "probs" / f"{name}.npz"
    arrays: dict[str, np.ndarray | None] = {
        "probs": None, "log_probs": None, "S": None, "mask": None, "chain_order": None
    }
    if save_probs:
        if not probs_path.exists():
            raise MPNNError(f"--save_probs was requested but {probs_path} is missing")
        with np.load(probs_path, allow_pickle=True) as data:
            for key in list(arrays):
                if key in data.files:
                    arrays[key] = data[key]
    else:
        probs_path = None

    return MPNNResult(
        name=name,
        native=native,
        designs=designs,
        probs=arrays["probs"],
        log_probs=arrays["log_probs"],
        S=arrays["S"],
        mask=arrays["mask"],
        chain_order=arrays["chain_order"],
        fasta_path=fasta_path,
        probs_path=probs_path,
        out_folder=out_folder,
        command=cmd,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test the ProteinMPNN wrapper.")
    ap.add_argument("pdb", nargs="?", default=str(MPNN_DIR / "inputs/PDB_monomers/pdbs/5L33.pdb"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--num-seq", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.1)
    args = ap.parse_args()

    res = run_mpnn(args.pdb, args.out, num_seq=args.num_seq, temperature=args.temp)
    print(f"{res.name}: {len(res.designs)} designs, probs {None if res.probs is None else res.probs.shape}")
    print(f"native : {res.native.seq}")
    for i, d in enumerate(res.designs):
        print(f"design{i}: {d.seq}  (score={d.score}, recovery={d.seq_recovery})")

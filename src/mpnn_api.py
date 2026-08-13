"""In-process ProteinMPNN with explicit control over the decoding order.

[runner] treats ProteinMPNN as a black-box CLI, which is right for *designing*
sequences. Detecting contamination needs something the CLI does not expose: the
ability to choose which residues are visible when a given site is decoded.

ProteinMPNN's decoder is autoregressive over a random permutation of positions,
and `ProteinMPNN.forward(..., use_input_decoding_order=True, decoding_order=...)`
is teacher-forced: it returns

    log p(s_i | s_j for every j preceding i in the order, backbone)

for all i at once. Fixing the order therefore fixes the conditioning set, which
turns the decoder into a joint-compatibility probe:

    order_last(i)   -> log p(s_i | all other sites)      (full context)
    order_first(i)  -> log p(s_i | backbone alone)       (no sequence context)

and, by swapping the *content* of the context between two candidate ancestries,
the context-swap conflict score in [conflict].

Two details that matter for reproducibility:

  * `augment_eps` (backbone coordinate noise) is forced to 0, otherwise every
    forward pass sees a slightly different structure and the conflict signal is
    swamped by noise.
  * The model is loaded once and reused; scoring is batched over the L decoding
    orders, so a full-context sweep of an L-residue protein is one forward pass.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MPNN_DIR = REPO_ROOT / "proteinmpnn"
if str(MPNN_DIR) not in sys.path:
    sys.path.insert(0, str(MPNN_DIR))

from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize  # noqa: E402

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
CANONICAL = ALPHABET[:20]
AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}

# Hidden sizes are fixed across the released ProteinMPNN checkpoints.
_HIDDEN = 128
_NUM_LAYERS = 3


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seq_to_idx(seq: str) -> np.ndarray:
    return np.array([AA_TO_IDX.get(aa.upper(), 20) for aa in seq], dtype=np.int64)


def idx_to_seq(idx: Iterable[int]) -> str:
    return "".join(ALPHABET[int(i)] for i in idx)


@dataclass
class Backbone:
    """A parsed structure, featurised once and reused for every scoring call."""

    X: torch.Tensor  # (1, L, 4, 3)
    mask: torch.Tensor  # (1, L)
    chain_M: torch.Tensor  # (1, L)
    residue_idx: torch.Tensor  # (1, L)
    chain_encoding: torch.Tensor  # (1, L)
    native_seq: str

    @property
    def L(self) -> int:
        return self.X.shape[1]


class MPNNScorer:
    """ProteinMPNN held in memory, with decoding-order-controlled scoring."""

    def __init__(
        self,
        pdb_path: str | Path,
        *,
        model_name: str = "v_48_020",
        weights_dir: str | Path | None = None,
        device: str = "auto",
        use_soluble_model: bool = False,
        backbone_noise: float = 0.0,
        seed: int | None = 0,
    ) -> None:
        self.device = pick_device(device)
        self.pdb_path = Path(pdb_path)
        if seed is not None:
            torch.manual_seed(seed)

        folder = "soluble_model_weights" if use_soluble_model else "vanilla_model_weights"
        weights = Path(weights_dir) if weights_dir else MPNN_DIR / folder
        checkpoint_path = weights / f"{model_name}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = ProteinMPNN(
            num_letters=21,
            node_features=_HIDDEN,
            edge_features=_HIDDEN,
            hidden_dim=_HIDDEN,
            num_encoder_layers=_NUM_LAYERS,
            num_decoder_layers=_NUM_LAYERS,
            augment_eps=backbone_noise,
            k_neighbors=checkpoint["num_edges"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        # Determinism: no coordinate jitter between calls.
        self.model.features.augment_eps = backbone_noise

        self.backbone = self._load_backbone(self.pdb_path)

    # ------------------------------------------------------------------ setup

    def _load_backbone(self, pdb_path: Path) -> Backbone:
        pdb_dict_list = parse_PDB(str(pdb_path), ca_only=False)
        batch = [pdb_dict_list[0]]
        name = batch[0]["name"]
        chains = [k[-1] for k in batch[0] if k.startswith("seq_chain_")]
        chain_id_dict = {name: (chains, [])}
        (
            X, S, mask, _lengths, chain_M, chain_encoding, _cl, _vl, _ml, _mcl,
            chain_M_pos, _omit, residue_idx, _dm, _tp, _pc, _pb, _plo, _bbr, _tb,
        ) = tied_featurize(
            batch, self.device, chain_id_dict, None, None, None, None, None, ca_only=False
        )
        return Backbone(
            X=X,
            mask=mask,
            chain_M=chain_M * chain_M_pos,
            residue_idx=residue_idx,
            chain_encoding=chain_encoding,
            native_seq=idx_to_seq(S[0].cpu().numpy()),
        )

    @property
    def L(self) -> int:
        return self.backbone.L

    @property
    def native_seq(self) -> str:
        return self.backbone.native_seq

    # --------------------------------------------------------------- scoring

    def _batched(self, n: int) -> tuple[torch.Tensor, ...]:
        bb = self.backbone
        return (
            bb.X.repeat(n, 1, 1, 1),
            bb.mask.repeat(n, 1),
            bb.chain_M.repeat(n, 1),
            bb.residue_idx.repeat(n, 1),
            bb.chain_encoding.repeat(n, 1),
        )

    @torch.no_grad()
    def log_probs(
        self,
        seqs: str | Sequence[str],
        decoding_orders: np.ndarray | torch.Tensor,
        *,
        chunk: int = 64,
    ) -> np.ndarray:
        """Teacher-forced log p(s_i | predecessors in the order) for each sequence.

        seqs             : one sequence, or B sequences
        decoding_orders  : (B, L) permutations of range(L)
        returns          : (B, L, 21) log-probabilities
        """
        if isinstance(seqs, str):
            seqs = [seqs]
        orders = torch.as_tensor(np.asarray(decoding_orders), dtype=torch.long)
        if orders.ndim == 1:
            orders = orders[None, :]
        if len(seqs) == 1 and orders.shape[0] > 1:
            seqs = list(seqs) * orders.shape[0]
        if orders.shape[0] == 1 and len(seqs) > 1:
            orders = orders.repeat(len(seqs), 1)
        if len(seqs) != orders.shape[0]:
            raise ValueError(f"{len(seqs)} sequences vs {orders.shape[0]} decoding orders")
        if orders.shape[1] != self.L:
            raise ValueError(f"decoding orders have length {orders.shape[1]}, expected {self.L}")

        S_all = torch.as_tensor(np.stack([seq_to_idx(s) for s in seqs]), dtype=torch.long)
        out = np.empty((len(seqs), self.L, 21), dtype=np.float32)

        for start in range(0, len(seqs), chunk):
            stop = min(start + chunk, len(seqs))
            n = stop - start
            X, mask, chain_M, residue_idx, chain_enc = self._batched(n)
            S = S_all[start:stop].to(self.device)
            order = orders[start:stop].to(self.device)
            randn = torch.zeros(n, self.L, device=self.device)  # unused when order is given
            lp = self.model(
                X, S, mask, chain_M, residue_idx, chain_enc, randn,
                use_input_decoding_order=True, decoding_order=order,
            )
            out[start:stop] = lp.detach().cpu().numpy()
        return out

    # ------------------------------------------------- canonical order shapes

    def orders_site_last(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """(L, L): order k puts site k last, so it sees every other residue."""
        rng = rng or np.random.default_rng(0)
        base = np.arange(self.L)
        orders = np.empty((self.L, self.L), dtype=np.int64)
        for k in range(self.L):
            rest = base[base != k]
            rest = rng.permutation(rest)
            orders[k, :-1] = rest
            orders[k, -1] = k
        return orders

    def orders_site_first(self) -> np.ndarray:
        """(L, L): order k puts site k first, so it sees no sequence context."""
        base = np.arange(self.L)
        orders = np.empty((self.L, self.L), dtype=np.int64)
        for k in range(self.L):
            orders[k, 0] = k
            orders[k, 1:] = base[base != k]
        return orders

    def random_orders(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        return np.stack([rng.permutation(self.L) for _ in range(n)])

    # ------------------------------------------------------------ conveniences

    def full_context_log_probs(self, seq: str, rng: np.random.Generator | None = None) -> np.ndarray:
        """(L, 21): distribution at each site given *all* other sites of `seq`."""
        orders = self.orders_site_last(rng)
        lp = self.log_probs(seq, orders)
        return np.stack([lp[k, k] for k in range(self.L)])

    def structure_only_log_probs(self, seq: str | None = None) -> np.ndarray:
        """(L, 21): distribution at each site given the backbone alone.

        The sequence argument is irrelevant to the result (nothing precedes the
        site in the order) and defaults to the native sequence.
        """
        orders = self.orders_first = self.orders_site_first()
        lp = self.log_probs(seq or self.native_seq, orders)
        return np.stack([lp[k, k] for k in range(self.L)])

    def site_log_prob(self, seq: str, context_seq: str | None = None) -> np.ndarray:
        """(L,): log p(seq[i] | all other sites taken from `context_seq`).

        With `context_seq` set, site i keeps its own residue from `seq` but every
        other position is filled from the context — the primitive the
        context-swap conflict score is built from.
        """
        orders = self.orders_site_last()
        if context_seq is None:
            chimeras = [seq] * self.L
        else:
            if len(context_seq) != len(seq):
                raise ValueError("seq and context_seq must be the same length")
            ctx = list(context_seq)
            chimeras = []
            for k in range(self.L):
                merged = ctx.copy()
                merged[k] = seq[k]
                chimeras.append("".join(merged))
        lp = self.log_probs(chimeras, orders)
        idx = seq_to_idx(seq)
        return np.array([lp[k, k, idx[k]] for k in range(self.L)], dtype=np.float64)

    def pseudo_log_likelihood(self, seq: str) -> float:
        """Mean over sites of log p(s_i | all other sites) — the stability proxy."""
        return float(np.mean(self.site_log_prob(seq)))

    @torch.no_grad()
    def autoregressive_score(
        self, seq: str, n_orders: int = 16, rng: np.random.Generator | None = None
    ) -> float:
        """ProteinMPNN's own score: mean log-likelihood over random decoding orders."""
        orders = self.random_orders(n_orders, rng)
        lp = self.log_probs(seq, orders)
        idx = seq_to_idx(seq)
        picked = lp[:, np.arange(self.L), idx]
        return float(picked.mean())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Sanity-check the in-process MPNN API.")
    ap.add_argument("pdb", nargs="?", default=str(MPNN_DIR / "inputs/PDB_monomers/pdbs/5L33.pdb"))
    args = ap.parse_args()

    scorer = MPNNScorer(args.pdb)
    print(f"device={scorer.device} L={scorer.L}")
    print(f"native: {scorer.native_seq}")
    struct = scorer.structure_only_log_probs()
    full = scorer.full_context_log_probs(scorer.native_seq)
    print(f"structure-only argmax : {idx_to_seq(struct[:, :20].argmax(1))}")
    print(f"full-context argmax   : {idx_to_seq(full[:, :20].argmax(1))}")
    print(f"pseudo-log-likelihood : {scorer.pseudo_log_likelihood(scorer.native_seq):.4f}")
    print(f"autoregressive score  : {scorer.autoregressive_score(scorer.native_seq):.4f}")
    # determinism check
    a = scorer.pseudo_log_likelihood(scorer.native_seq)
    b = scorer.pseudo_log_likelihood(scorer.native_seq)
    print(f"deterministic         : {a == b}")

"""Pre-simulate and cache clean families so the sweeps become cheap.

Gibbs sampling a `selection` family is the only expensive step in the whole
project (~11 min on CPU) and it does not depend on the contamination, so every
sweep can reuse it. Warming the cache separately means the long, unattended part
is decoupled from the analysis, and it parallelises trivially across seeds where
a single sweep does not.

    python experiments/warm_cache.py --model selection --seeds 3 20 --stride 3 --offset 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from mpnn_api import MPNN_DIR, MPNNScorer  # noqa: E402
from sweeps import clean_family  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="selection", choices=["f81", "selection"])
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 20], metavar=("START", "STOP"))
    ap.add_argument("--stride", type=int, default=1, help="run every Nth seed (for sharding)")
    ap.add_argument("--offset", type=int, default=0, help="shard index, 0 <= offset < stride")
    ap.add_argument("--n-per-clade", type=int, default=6)
    ap.add_argument("--pdb", default=str(MPNN_DIR / "inputs/PDB_monomers/pdbs/5L33.pdb"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    scorer = MPNNScorer(args.pdb, device=args.device)
    todo = [s for s in range(*args.seeds) if s % args.stride == args.offset]
    print(f"shard {args.offset}/{args.stride}: seeds {todo}", flush=True)
    for seed in todo:
        t0 = time.time()
        clean_family(scorer, args.model, seed, n_per_clade=args.n_per_clade)
        print(f"  seed {seed} ready ({time.time() - t0:.0f}s)", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()

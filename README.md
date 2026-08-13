# protein-extension

Ancestral sequence reconstruction read as textual criticism: homolog families are
*witnesses*, the phylogeny is a *stemma*, and the reconstruction ships with a
**critical apparatus** that labels every position `certain` / `probable` /
`conjectural` by combining ProteinMPNN's structural confidence with FastML's
phylogenetic posteriors.

ProteinMPNN is vendored as a git submodule and treated as a black box; the only
code that knows it exists is [src/runner.py](src/runner.py).

## Layout

```
proteinmpnn/     dauparas/ProteinMPNN (submodule) + MPS patch
src/witnesses.py    homolog family assembly     (stub)
src/stemma.py       tree + FastML reconstruction (stub)
src/conditioning.py posteriors -> bias/pssm jsonl (stub)
src/runner.py       ProteinMPNN subprocess wrapper  ✅
src/apparatus.py    entropy + posterior -> apparatus ✅
src/scoring.py      recovery / perplexity        (stub)
configs/ experiments/ data/{raw,interim,designs}/ scratch/
```

## Setup

```bash
git clone --recurse-submodules <this repo>
uv venv --python 3.12 .venv       # torch has no 3.14 wheels yet
VIRTUAL_ENV=.venv uv pip install torch numpy
```

Smoke-test the submodule before anything else:

```bash
.venv/bin/python proteinmpnn/protein_mpnn_run.py \
  --pdb_path proteinmpnn/inputs/PDB_monomers/pdbs/5L33.pdb \
  --out_folder scratch/smoke_test \
  --num_seq_per_target 2 --sampling_temp 0.1 --save_probs 1
```

This should write `scratch/smoke_test/seqs/5L33.fa` and
`scratch/smoke_test/probs/5L33.npz` (`probs` of shape `(n_samples, n_res, 21)`).

### Apple silicon

`proteinmpnn/protein_mpnn_run_mps_patch.py` is a drop-in replacement for
`protein_mpnn_run.py` that selects the `mps` device. It comes from
[AIscend-Research/protein-repro](https://github.com/AIscend-Research/protein-repro)
— credit to AIscend-Research; it is the only file shared between the two
projects. One local change: `bias_AAs_np` is allocated as `float32`, since MPS
cannot convert float64 tensors.

`runner.run_mpnn(..., device="auto")` picks the patch when MPS is available and
CUDA is not; pass `device="stock"` to force the unpatched script.

## Usage

```python
import sys; sys.path.insert(0, "src")
from runner import run_mpnn
from apparatus import build_apparatus, write_apparatus, summarize

res = run_mpnn("proteinmpnn/inputs/PDB_monomers/pdbs/5L33.pdb",
               "scratch/demo", num_seq=8, temperature=0.2)
res.probs.shape          # (8, 106, 21)
res.sequences            # designed sequences

calls = build_apparatus(res.probs_path, "data/interim/node12.marginal_prob.csv", node="N12")
write_apparatus(calls, "data/designs/node12_apparatus.csv")
summarize(calls)         # counts + agreement rate
```

Both modules also run as scripts:

```bash
.venv/bin/python src/runner.py --num-seq 3 --temp 0.2 --out scratch/runner_test
.venv/bin/python src/apparatus.py --probs scratch/smoke_test/probs/5L33.npz \
  --fastml data/interim/node12.marginal_prob.csv --node N12 \
  --out data/designs/node12_apparatus.csv
```

`apparatus.py` accepts FastML's marginal-probability table in either shape: wide
(`Node, Pos, A, C, D, ...`) or long (`Node, Pos, AA, Prob`), with `--offset` for
alignment-vs-structure numbering differences. Note that MPNN entropy depends on
the sampling temperature used to produce the `.npz` — use a neutral temperature
(or `--unconditional_probs_only`) when the entropy is meant to be evidence.

## Benchmarking

If Phase 4 numbers look strange, clone
[AIscend-Research/protein-repro](https://github.com/AIscend-Research/protein-repro)
into a *separate* directory and run its known-good sequence-recovery benchmark to
tell a conditioning problem from a setup problem. Nothing here is built on top of
it.

# Contaminated ancestors

**Detecting recombination in ancestral sequence reconstruction with a structural
joint-compatibility model.**

The hardest problem in stemmatics is not building a family tree of manuscripts —
it is *contamination*. A scribe with two exemplars on the desk copies from both.
The resulting witness has no single parent, and Lachmannian method, which assumes
that every copy descends from exactly one exemplar, does not fail loudly on a
contaminated tradition. It returns a confident, clean-looking stemma that is
wrong, and an archetype that never existed in any scriptorium.

Ancestral sequence reconstruction (ASR) has exactly this failure mode and no
equivalent alarm. Recombination, gene conversion, domain shuffling and horizontal
transfer all violate the tree assumption. When they do, maximum-likelihood ASR
does not error out: it returns a chimeric ancestor with *high per-site posterior
probabilities*, because each site is inferred independently and each site is
individually fine. The pathology is joint, and every standard ASR diagnostic is
marginal. You cannot see a chimera by looking at sites one at a time.

ProteinMPNN's decoder conditions each residue on the backbone *and on the other
residues already decoded*. That is precisely a joint-compatibility model, which
makes it the right instrument rather than a decorative one. Fix the decoding
order and it reports

    log p(residue i | a chosen set of other residues, backbone)

so a site can be scored against lineage-A neighbours and against lineage-B
neighbours and asked whether it cares. A coherent ancestor is context-stable. A
contaminated one shows systematic, *spatially contiguous* context-dependent
disagreement — which is the textual-critical task of separating hyparchetypes,
done with structural epistasis instead of variant readings.

Existing recombination detection (GARD, RDP and relatives) is entirely
sequence-based: it looks for phylogenetic incongruence across alignment windows,
and degrades exactly where ASR is most used and most consequential — deep
divergence, saturated sites, short genes. A structural detector is orthogonal
signal in the regime where the sequence methods run out.

## Method

Three probes, all built on decoding-order control ([src/mpnn_api.py](src/mpnn_api.py)):

**Context swap** — reconstruct the two candidate sub-histories separately (the
two clades of the stemma) and score each site of the mosaic ancestor in each
context:

```
delta_i = log p(a_i | context = ancestor_A) - log p(a_i | context = ancestor_B)
```

The scan runs in *diagnostic-site space* — only positions where the two
sub-ancestors actually differ carry ancestry information — using the sign of
delta rather than its magnitude, smoothed over neighbouring diagnostic sites
because contamination arrives in contiguous blocks. Significance comes from a
permutation test that destroys spatial contiguity while preserving the marginal
distribution of delta, so what is tested is segmental structure, not overall
conflict.

**Order instability** — decode the ancestor under many random decoding orders and
measure how much site *i*'s preferred residue depends on which neighbours were
filled in first.

**Repair** — instead of one chimeric archetype, output the two internally
coherent sub-ancestors and design from each separately.

## Layout

```
src/mpnn_api.py     in-process ProteinMPNN with decoding-order control  ← the instrument
src/evolve.py       simulate families on a fixed backbone; inject contamination
src/stemma.py       NJ + midpoint rooting + F81 marginal ASR (deliberately standard)
src/conflict.py     context-swap / order-instability probes, segment scan, permutation test
src/repair.py       mosaic vs two coherent sub-ancestors, scored
src/pipeline.py     one family, end to end
src/apparatus.py    per-site certain / probable / conjectural labels, + sub-history
src/scoring.py      pseudo-log-likelihood, recovery, perplexity, per-label breakdowns
src/conditioning.py posteriors -> bias_by_res / pssm jsonl, for designing from an ancestor
src/witnesses.py    fetch and filter real homolog families (UniProt / PDB / AlphaFold DB)
src/runner.py       subprocess wrapper around stock ProteinMPNN (for design)
experiments/run_experiments.py
proteinmpnn/        dauparas/ProteinMPNN (submodule, kept pristine)
patches/            Apple-silicon MPS runner
```

The apparatus is the paper's output format, and `annotate_contamination` folds
the detector into it: a site inside a segment judged contaminated is demoted to
`conjectural` no matter how sharp its marginal posterior, because per-site
confidence is precisely the thing that fails to notice a chimera. Each row also
records which sub-history the site leans toward, since a contaminated tradition
has no single archetype to be confident *about*.

The labels are calibrated rather than decorative — on the headline family,
sequence recovery against the known true root runs 1.00 / 0.69 / 0.48 for
certain / probable / conjectural (`scoring.recovery_by_label`).

## Ground truth, cheaply

Every witness is simulated on **one shared backbone** — a family of structural
homologs keeps its fold, which is what makes a single ancestral backbone
meaningful in the first place — so no structure prediction enters the loop and
the evaluation runs on CPU.

Two evolution models, and the contrast is the experiment:

| model | process | epistasis | role |
|---|---|---|---|
| `f81` | site-independent substitution toward per-site equilibrium profiles | none | **control** |
| `selection` | Gibbs sampling from ProteinMPNN's joint distribution over the backbone | yes | test |

The detector is supposed to work by sensing broken joint compatibility. So it
should fire on `selection` data and *not* on `f81` data. If it fires on both it
is reading composition, not structural incoherence; if it fires on neither,
MPNN's receptive field is too local for the signal. Running both is the early
sensitivity check that decides whether the method is viable at all.

## Setup

```bash
git clone --recurse-submodules <this repo>
uv venv --python 3.12 .venv        # torch has no 3.14 wheels yet
VIRTUAL_ENV=.venv uv pip install torch numpy
.venv/bin/python src/mpnn_api.py   # sanity-check the instrument
```

## Running

```bash
# the control: no epistasis, so nothing should be detected
.venv/bin/python experiments/run_experiments.py --exp all --models f81 --seeds 3

# the test
.venv/bin/python experiments/run_experiments.py --exp all --models selection --seeds 3
```

Results land in `experiments/results/` as JSON and CSV. Individual stages also
run standalone:

```bash
.venv/bin/python src/evolve.py --model selection --breakpoint 30,60 --out data/interim/sim
.venv/bin/python src/stemma.py data/interim/sim/witnesses.fasta
```

## Figures

```bash
.venv/bin/python experiments/make_figures.py --model selection --seed 0 --breakpoint 55,85
.venv/bin/python experiments/build_gallery.py     # self-contained HTML plate gallery
```

[src/viz.py](src/viz.py) renders the whole set into `figures/`: the inferred
stemma with contaminated witnesses marked, the per-site conflict profile against
the true breakpoint, the permutation null, the apparatus, two views of the score
painted onto the backbone, and the contact map showing which contacts bridge the
flagged block. `make_figures.py` caches the simulated family, so re-rendering
after a figure tweak skips the slow part.

`viz.write_bfactor_pdb` also writes the conflict score into a PDB's B-factor
column, so the same numbers can be rendered in PyMOL or ChimeraX (`spectrum b`)
without this project depending on either.

Colour follows the data's job rather than taste: the conflict score is polar, so
it is diverging blue-red on a neutral midpoint; apparatus labels are ordinal, so
they are a single neutral ink ramp (deliberately *not* blue — the sub-history
strips are blue and orange, and two encodings sharing a hue is how a reader
conflates "well attested" with "belongs to clade A"). Every palette was checked
with a validator for colour-vision separation and surface contrast.

## Headline figure run

`--model selection --seed 0 --breakpoint 55,85`, 12 witnesses, 106 residues:

| | |
|---|---|
| ASR accuracy vs the true root | 0.60 |
| ASR mean max posterior | **0.95** — confident and wrong, which is the premise |
| Detected | yes, p = 0.030 |
| Segment recovered | (58, 83) against a true (55, 85) — Jaccard 0.83 |
| Site AUC on diagnostic sites | 0.90 |
| Diagnostic sites | 43 |

## Limitations

- **Segment length.** Detection needs a contaminated block long enough to contain
  enough diagnostic sites. A 10-residue swap is not reliably recoverable
  (measured AUC 0.57, versus 0.87 for a 30-residue swap).
- **Divergence.** The two lineages are distinguishable only at sites where they
  differ. A family whose between-clade divergence is no larger than its
  within-clade divergence has almost nothing to detect.
- **Locality.** MPNN conditions on a local structural neighbourhood, so a
  contiguous sequence swap is strained mainly near its structural junctions. The
  whole-sequence penalty for a mosaic is small; the signal lives in *where* the
  conflict sits, not in its magnitude.
- **Circularity.** The `selection` simulator samples from ProteinMPNN's own joint
  distribution, so it asserts the epistasis the detector then senses. The `f81`
  control is what keeps this honest; empirical families are the real test.
- **In silico throughout.** No wet-lab validation, and stability is a
  pseudo-log-likelihood proxy, not a measured ΔΔG.

## Credits

`patches/protein_mpnn_run_mps_patch.py` selects the `mps` device on Apple
silicon. It comes from
[AIscend-Research/protein-repro](https://github.com/AIscend-Research/protein-repro)
— credit to AIscend-Research. Two local changes: `bias_AAs_np` is allocated as
`float32` (MPS cannot convert float64), and it lives outside the submodule so git
tracks it here.

ProteinMPNN is Dauparas et al., *Science* 2022, vendored as a submodule.

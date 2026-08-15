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
divergence, saturated sites, short genes. A structural detector would be
orthogonal signal in the regime where the sequence methods run out.

That is the thesis. The rest of this README describes the apparatus built to test
it. **What the apparatus actually reports is largely negative**, and the honest
summary belongs before the method rather than after it:

> - The premise checks out: contaminated ASR returns ancestors at 0.93 mean max
>   posterior that are 69% correct. Confident and wrong, exactly as claimed.
> - The permutation test **does not discriminate**. Over matched conditions it
>   fires 2 times in 15 on epistatic data and 2 times in 15 on the no-epistasis
>   control. Only *where* it points separates them — mean Jaccard against the true
>   breakpoint is 0.81 versus 0.08.
> - Detection is gated by how many **diagnostic sites** fall inside the
>   contaminated block, not by how long the block is. Nothing below 13 ever fired.
> - **The structural model does not earn its place.** A sequence-identity
>   statistic that uses no structure and no network fires 4/6 against MPNN's 2/6
>   and recovers the window better (mean Jaccard 0.62 vs 0.27), once perfectly
>   (1.00) on a run where MPNN missed entirely. Scrambling the backbone does take
>   MPNN to 0/6, so its signal is genuinely structural — it is simply not better
>   than string comparison.
> - The **repair prediction fails** once witness count is controlled.
> - Starving the sequence evidence (down to 2 witnesses per clade) does **not**
>   let the structural model catch up to identity — no crossing at any point,
>   because marginal ASR bakes the shortcut in regardless of how little data it
>   saw: 95–99% of diagnostic sites hold one sub-ancestor's residue exactly,
>   at every witness count tested.
> - One structural idea does earn its place: contamination contiguous in
>   *space* rather than in *sequence* is found by a scan over structural
>   neighbourhoods and missed by the ordinary sequence scan (7/18 vs 1/18,
>   p = 0.041) — a case no sequence-window method can address by construction.
>   But even there, `identity` matches the scan at least as well as MPNN does;
>   what earns its place is *where the scan looks*, not the structural model.
> - On a real 3FTx family the detector returns nothing, but that family sits near
>   the diagnostic-site floor, so the result is inconclusive rather than negative.
> - A 20-seed, higher-power rerun of the core sweeps is done — see
>   [RESULTS.md](RESULTS.md) §12 for detection rates with 95% confidence
>   intervals.
>
> Full numbers, with the reasoning: **[RESULTS.md](RESULTS.md)**.

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

One subtlety that turned out to matter more than expected. The scan is
sign-symmetric: the intruding block and its complement are both "the window whose
mean differs most from the rest", so delta says which context a site prefers but
not which side is the intrusion. Something label-free has to break that tie, and
the obvious rule — *the intrusion is the rarer sign* — is wrong, because outside
the intruding block the mosaic ancestor is genuinely intermediate between the two
clades, so delta there is ≈ 0 with random sign and the "majority" is decided by
noise. It inverted the site AUC to 0.05 on exactly the run where detection
succeeded. `conflict.oriented_delta` instead orients by the window the scan
itself flagged; see [RESULTS.md](RESULTS.md) §6 for the measurement, including
the fact that this rule inflates the null AUC and so must be read against the
`f81` control rather than against 0.5.

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
src/spatial.py       structural (3D) analogue of the sequence scan and its null

experiments/run_experiments.py  detect / null / repair on simulated families
experiments/sweeps.py           segment length, divergence, orientation, repair sweeps
experiments/ablation.py         identity-only and scrambled-backbone controls
experiments/thin_evidence.py    does MPNN catch up to identity as witnesses thin out?
experiments/spatial_contamination.py  2x2x2: {block,patch} x {mpnn,identity} x {1D,3D}
experiments/real_family.py      UniProt -> MAFFT -> IQ-TREE -> the same detector
experiments/check_fold.py       disulfide topology of a folded ancestor
experiments/warm_cache.py       pre-simulate + cache clean families, shardable by seed
experiments/run_power_sweeps.sh 20-seed rerun of the segment sweep + ablation (done, see RESULTS.md §12)
experiments/summarize.py        every headline number, recomputed from results/
experiments/make_figures.py     the figure set
experiments/build_gallery.py    self-contained HTML plate gallery

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
sequence recovery against the known true root runs 1.00 / 0.80 / 0.45 for
certain / probable / conjectural (`scoring.recovery_by_label`). This ordering
holds independently of whether the contamination detector pans out, which is
worth saying plainly given that §3 of [RESULTS.md](RESULTS.md) reports that it
largely does not.

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
# without uv:  python3.12 -m venv .venv && .venv/bin/pip install torch numpy
# empirical path also needs:  brew install mafft iqtree3
.venv/bin/python src/mpnn_api.py   # sanity-check the instrument
```

## Running

```bash
# the control: no epistasis for a joint model to sense. It should not fire more
# often than alpha — and measured over enough conditions, that is what it does.
.venv/bin/python experiments/run_experiments.py --exp all --models f81 --seeds 3

# the test
.venv/bin/python experiments/run_experiments.py --exp all --models selection --seeds 3
```

Results land in `experiments/results/` as JSON and CSV. Every headline number is
recomputed from those files by

```bash
.venv/bin/python experiments/summarize.py
```

so nothing in [RESULTS.md](RESULTS.md) is transcribed by hand.

The sensitivity sweeps cache each evolved family to `scratch/families/`, because
Gibbs sampling a clean family is the only expensive step and it does not depend
on the contamination — so one simulation serves every contamination condition:

```bash
.venv/bin/python experiments/sweeps.py --sweep segment     --model selection --seeds 3
.venv/bin/python experiments/sweeps.py --sweep divergence  --model selection --seeds 2
.venv/bin/python experiments/sweeps.py --sweep orientation --model selection --seeds 3
.venv/bin/python experiments/sweeps.py --sweep repair      --model selection --seeds 3
```

The ablation asks whether the structural model earns its place, against a
sequence-identity statistic that uses no structure at all and against the same
network with its backbone scrambled:

```bash
.venv/bin/python experiments/ablation.py --model selection --seeds 3
```

Two follow-ups push on where the structural model might still earn its place —
neither rescues it, and the reasoning is in [RESULTS.md](RESULTS.md) §9-10:

```bash
# does MPNN catch up to identity as the sequence evidence thins out? (no)
.venv/bin/python experiments/thin_evidence.py --seeds 3 --width 50

# does a scan over structural neighbourhoods find contamination a sequence
# scan cannot, for contamination that is contiguous in 3D but not in sequence?
# (yes, but identity ties it within that scan — the scan is the contribution)
.venv/bin/python experiments/spatial_contamination.py --seeds 3
```

Everything above ran at 2–3 seeds. `experiments/run_power_sweeps.sh` reran the
segment sweep and the ablation at 20 seeds with blocks sized above the
diagnostic-site floor (50/65/80 residues, since §4 shows 10/20/30 are
unwinnable by construction) — see [RESULTS.md](RESULTS.md) §12 for the result.
`experiments/warm_cache.py` pre-simulates the 20 families it needs and can be
sharded across processes by seed:

```bash
.venv/bin/python experiments/warm_cache.py --model selection --seeds 0 20 --stride 3 --offset 0 &
.venv/bin/python experiments/warm_cache.py --model selection --seeds 0 20 --stride 3 --offset 1 &
.venv/bin/python experiments/warm_cache.py --model selection --seeds 0 20 --stride 3 --offset 2 &
wait
bash experiments/run_power_sweeps.sh
```

The empirical path — fetch, MAFFT, IQ-TREE tree + per-clade ASR, then the same
detector — is one command per rooting convention (needs `mafft` and `iqtree3`):

```bash
.venv/bin/python experiments/real_family.py --split midpoint --backbone <ancestor.pdb>
.venv/bin/python experiments/check_fold.py  <ancestor.pdb> data/raw/pdb/3EBX.pdb
```

`check_fold.py` verifies a folded ancestor's disulfide topology before anything
downstream trusts it, and is validated against 3EBX, where it reproduces the
crystal structure's own `SSBOND` records to 0.01 Å.

Individual stages also run standalone:

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
| Detected | yes, p = 0.020 |
| Segment recovered | (50, 92) against a true (55, 85) — Jaccard 0.71 |
| Site AUC on diagnostic sites | 0.95 |
| Diagnostic sites | 37, of which 13 fall inside the contaminated block |

**Read this as one run, not as the result.** It is the best case, and it is shown
because the figures have to illustrate *something*; across all twelve matched
conditions the detector fires twice, and so does the no-epistasis control — see
[RESULTS.md](RESULTS.md) §3. The 13 diagnostic sites inside the block are the
reason this run works and most do not.

The figure pipeline runs at Gibbs temperature 0.5 to match `run_experiments.py`
and `sweeps.py`. `evolve.make_evolver` defaults to 0.6, so `make_figures.py`
passes the value explicitly — otherwise the headline figure would be drawn from a
different regime than every reported table, which is how a plate and a results
column quietly stop describing the same experiment.

## Limitations

Full results, including two negative ones, are in **[RESULTS.md](RESULTS.md)**.
The short version:

- **The permutation test alone does not discriminate.** On matched conditions the
  detector fires 2/12 on epistatic data and 2/12 on the no-epistasis control. What
  separates them is *where* it points: mean Jaccard against the true breakpoint is
  0.81 on `selection` and 0.08 on `f81`. A yes/no verdict from this test carries
  no information; the flagged window does.
- **Diagnostic sites, not segment length, set the floor.** Only positions where
  the two sub-ancestors differ carry ancestry information. The runs that fired had
  13 and 22 diagnostic sites inside the contaminated block; a 50-residue block
  with 16 of them failed. Some blocks contain zero, and are undetectable in
  principle.
- **Divergence and witness count locate that floor but do not chart a curve.**
  Nothing fires below ~20 diagnostic sites in total, and both detections in the
  grid came from six witnesses per clade rather than three — at matched
  diagnostic-site count, which makes it a reconstruction-quality effect, not a
  site-count effect. But the grid is 2 / 12 overall with both hits in one seed, so
  seed variance exceeds the effect of either variable. The floor is established;
  the dose-response is not.
- **The epistatic regime starves the detector.** `f81` families carry 67–78
  diagnostic sites; `selection` families carry 23–41. Structural constraint keeps
  the two lineages similar, so the regime the method is designed for is also the
  one that gives it least to work with.
- **The oriented site AUC has an inflated null.** Resolving the score's sign
  ambiguity by the detector's own flagged window pushes null AUC to 0.50–0.68.
  Read it against the `f81` series, never against 0.5.
- **A sequence-identity control beats the structural probe.** Fires 4/6 vs 2/6,
  mean Jaccard 0.62 vs 0.27. Scrambling the backbone takes MPNN to 0/6, so the
  signal is structural — but structural is not the same as useful.
- **Thinning the sequence evidence doesn't change that.** Down to two witnesses
  per clade, identity still matches or beats MPNN. 95–99% of diagnostic sites hold
  one sub-ancestor's residue exactly regardless of witness count, because that is
  how marginal ASR resolves ties, not a function of how much data went in — so
  there is no regime along this axis where starving the input rescues the model.
- **Spatially contiguous contamination is where structure earns something —
  but it's the scan, not the model.** A patch compact on the folded backbone
  breaks into several separate runs in sequence, so a scan over alignment
  position cannot find it whatever score feeds it. A scan over structural
  neighbourhoods does (7/18 vs 1/18, p = 0.041) — a real capability no
  sequence-window method has. Within that scan, though, `identity` still edges
  `mpnn` (4/9 vs 3/9), so the contribution is *where the scan looks*, not the
  structural model computing the score.
- **The repair prediction is not supported.** Controlled for the number of
  witnesses, the coherent-versus-incoherent gap is +0.007 ± 0.059 with the
  predicted sign in 4/12 runs. The apparent effect is a sample-size artifact — it
  is just as large at zero contamination.
- **The empirical test was inconclusive, not negative.** On 56 three-finger
  toxins the detector returns p = 0.42 / 0.32 under two rootings, but the family
  has only 32–38 diagnostic sites in total, near the floor where the method has no
  power regardless of the truth.
- **Circularity.** The `selection` simulator samples from ProteinMPNN's own joint
  distribution, so it asserts the epistasis the detector then senses. The `f81`
  control is what keeps this honest; empirical families are the real test.
- **Locality.** MPNN conditions on a local structural neighbourhood, so a
  contiguous sequence swap is strained mainly near its structural junctions. The
  whole-sequence penalty for a mosaic is small; the signal lives in *where* the
  conflict sits, not in its magnitude.
- **One backbone, in silico throughout.** Every simulated witness is evolved on
  5L33. No wet-lab validation, and stability is a pseudo-log-likelihood proxy, not
  a measured ΔΔG.
- **Underpowered.** Three seeds per configuration cannot put an interval on a
  detection rate.

## A reproducibility hazard worth knowing

On Apple silicon, PyTorch's MPS backend has been observed failing a Metal command
buffer partway through a long run, printing `Error: command buffer exited with
error status` to stderr, and then **continuing with corrupted tensors without
raising**. The run completes and writes plausible-looking output: in one case the
same seed and settings that gave p = 0.030 returned p = 1.000 with every score
degraded to noise.

`MPNNScorer.log_probs` now checks that the model's output is finite and raises if
it is not, so this fails loudly instead of quietly. The figure pipeline defaults
to `--device cpu` for the same reason — it is a few minutes slower and does not
invent results. Anything intended for the paper should be produced on CPU or
CUDA, not MPS.

## Credits

`patches/protein_mpnn_run_mps_patch.py` selects the `mps` device on Apple
silicon. It comes from
[AIscend-Research/protein-repro](https://github.com/AIscend-Research/protein-repro)
— credit to AIscend-Research. Two local changes: `bias_AAs_np` is allocated as
`float32` (MPS cannot convert float64), and it lives outside the submodule so git
tracks it here.

ProteinMPNN is Dauparas et al., *Science* 2022, vendored as a submodule.

# Results

All numbers here are produced by `experiments/summarize.py`, which recomputes
them from the files in `experiments/results/`. Nothing is transcribed by hand.

Everything below is CPU-only, on one backbone (5L33, 106 residues), 12 witnesses
per family unless stated. The earlier contents of `experiments/results/` were
deleted before this run: they predated the conflict-score orientation fix and
carried inverted AUCs.

---

## 1. The premise holds

Marginal ASR on contaminated data does exactly what the project claims it does.
Across the `selection` detect conditions the reconstruction has a **mean maximum
posterior of 0.93** while being **69% correct** against the known true root. It
is confident and wrong, and no per-site number in the reconstruction says so.

That is the failure this method exists to catch. Whether it catches it is the
rest of this document.

## 2. The control, and the headline correction

`f81` — site-independent evolution, no epistasis for a joint model to sense — is
the experiment that decides whether the detector reads structural incoherence or
merely composition.

In the main experiment it looks immaculate: **0 / 6** detections, **0 / 3** false
positives, site AUC 0.498 ± 0.058, mean p = 0.73. On that evidence alone the
obvious claim is "the detector never fires on the control".

**That claim is wrong, and the wider sweep is what shows it.** Across the
12-condition segment sweep the `f81` control fires **2 / 12** times, at p = 0.015
twice. That is close to the nominal α = 0.05 — which is what a correctly
calibrated permutation test is *supposed* to do. The main experiment's 0 / 6 was
a small-sample artifact, not a property of the method.

So the honest statement is not "the control is silent" but "the control fires at
about its nominal rate".

## 3. Detection rate does not separate test from control — window accuracy does

Run on identical conditions, the two models fire equally often. Pooling the main
experiment with the segment sweep and de-duplicating the three shared conditions,
`selection` fires 2 times in 15 and `f81` fires 2 times in 15. Within the
12-condition sweep alone:

| model | detections | mean Jaccard **on the runs that fired** |
|---|---|---|
| `selection` | 2 / 12 | **0.81** |
| `f81` control | 2 / 12 | **0.08** |

The firing rate carries no information at all. The separation is entirely in
whether the flagged window is the right one:

| model | run | p | flagged window | true block | Jaccard |
|---|---|---|---|---|---|
| `selection` | seed 0, w30 | 0.020 | 50–92 | 55–85 | **0.71** |
| `selection` | seed 1, w50 | 0.005 | 53–102 | 55–105 | **0.90** |
| `f81` | seed 2, w10 | 0.015 | 39–103 | 55–65 | 0.16 |
| `f81` | seed 2, w20 | 0.015 | 9–39 | 55–75 | **0.00** |

When `selection` fires it lands on the breakpoint — Jaccard 0.71 and 0.90, with
site AUC 0.95 and 0.90. When `f81` fires it lands somewhere arbitrary, once with
zero overlap at all. Both `f81` firings come from the same seed, so they are not
even independent events.

This is the central result, and it reframes the method. The permutation test on
its own is not a contamination detector: its firing rate is the same on epistatic
and non-epistatic data. What the joint model contributes is *where* it points
once it fires. Any usable version of this method has to report the window and its
support, not a yes/no verdict — and needs a second, independent criterion to
separate a real firing from a calibrated false one, which this work does not yet
supply.

Two firings in twelve is also simply underpowered. Three seeds cannot put an
interval on a detection rate.

## 4. What actually governs detection: diagnostic sites, not block length

The sweep was designed to find where detection dies as a function of block
length. It found something more useful — block length is the wrong variable.

| block length | `selection` detections | diagnostic sites inside the block (per seed) |
|---|---|---|
| 10 | 0 / 3 | 5, 1, 3 |
| 20 | 0 / 3 | 8, 0, 4 |
| 30 | **1 / 3** | **13**, 0, 2 |
| 50 | **1 / 3** | 22, **22**, 16 |

Sorting the same twelve runs by diagnostic-site count rather than by block
length makes the constraint obvious: **no run with fewer than 13 diagnostic sites
in the block ever fired (0 of 8)**, and 2 of the 4 runs with 13 or more did. It
is a necessary condition, not a sufficient one — a 50-residue block holding 16 of
them still failed at p = 0.33, and a 30-residue block holding 13 succeeded at
p = 0.02.

Only positions where the two sub-ancestors differ carry ancestry information, so
a long block that happens to span conserved structure contains almost none. Two
of the twelve blocks contained **zero**, and were undetectable in principle.

The `f81` firings do not follow this pattern at all — they occur at 2 and 10
diagnostic sites, where `selection` never fires. That mismatch is a second,
independent sign that the control's firings are spurious rather than weak
detections.

This also explains the `nan` entries in the orientation table: on some seeds the
contaminated block contained **zero** diagnostic sites, so there was nothing to
detect there even in principle.

A second-order observation worth recording: `f81` families carry **67–78**
diagnostic sites where `selection` families carry **23–41**. The epistatic model
produces more similar clades, because structural constraint limits where the two
lineages can drift apart. The regime the detector is designed for is also the
regime that starves it of the sites it needs.

## 5. Divergence and witness count: mapping the floor

Between-clade divergence (`stem`, the branch separating the two lineages) and
witnesses per clade were varied together, 2 seeds each. Divergence is the knob
that manufactures diagnostic sites in the first place; witness count is what
determines whether the two sub-ancestors can be reconstructed well enough to
tell them apart.

| stem | n/clade | detected | mean diagnostic sites | site AUC |
|---|---|---|---|---|
| 0.5 | 3 | 0 / 2 | 16.0 | 0.591 |
| 0.5 | 6 | 0 / 2 | 17.5 | 0.540 |
| 2.0 | 3 | 0 / 2 | 31.0 | 0.556 |
| 2.0 | 6 | **1 / 2** | 30.0 | **0.952** |
| 4.0 | 3 | 0 / 2 | 33.0 | 0.698 |
| 4.0 | 6 | **1 / 2** | 39.5 | **0.951** |

Divergence buys diagnostic sites, and it saturates: stem 0.5 → 16.8 on average,
stem 2.0 → 30.5, stem 4.0 → 36.2. Doubling divergence again past 2.0 adds little,
because the two lineages are already as different as the shared backbone will let
them be.

Two things are consistent across the grid and worth stating:

- **Nothing fires below ~20 diagnostic sites.** Both `stem = 0.5` rows are 0 / 4
  at 16–18 sites, which agrees with §4's finding that detection needs roughly 13
  *inside the block* — a family with 17 in total cannot supply that.
- **Both detections are at n = 6, never at n = 3** (0 / 6 across every divergence
  level). Note that `stem = 2.0, n = 3` and `stem = 2.0, n = 6` have the *same*
  mean diagnostic-site count (31.0 vs 30.0) and only the larger clade detects, so
  this is not just the site count again. Reconstructing a sub-ancestor from three
  witnesses is too noisy for the comparison to mean anything.

**What this table does not support** is a dose-response claim. The overall rate is
2 / 12, and **both detections are in seed 0 — seed 1 is 0 / 6 throughout**. Seed
variance is larger than the effect of either variable, which is what two seeds per
cell buys you. The two bullets above are the floor (a necessary condition, cheaply
established); the shape of the curve above the floor is not measured here. The
site AUC column shows the same split — 0.95 in both detecting cells, 0.54–0.70
elsewhere — but it is 2 runs against 10, not a trend.

## 6. The orientation rule was a real bug, now fixed

The scan is sign-symmetric: the intruding block and its complement are both "the
window whose mean differs most from the rest". Something label-free has to decide
which side is the intrusion, and the rule that shipped — *the intrusion is the
rarer sign among diagnostic sites* — is wrong.

| condition | detected | AUC, `minority` rule | AUC, `segment` rule |
|---|---|---|---|
| seed 0, 55–85 | **yes** (p = 0.02) | **0.048** | **0.952** |
| seed 1, 25–55 | no | 0.828 | 0.172 |
| seed 2, 55–85 | no | 0.239 | 0.239 |

On the one run where detection succeeded, the shipped rule reported AUC 0.048 —
the correct signal, perfectly inverted. The rule fails because outside the
intruding block the mosaic ancestor is genuinely intermediate between the two
clades: delta there is ≈ 0 with essentially random sign, so the "majority" is
decided by noise rather than by ancestry.

The replacement (`conflict.oriented_delta`) orients by the window the scan itself
flagged. It uses no ground truth, and it is not circular — the window is the
detector's *estimate*, and when the estimate is wrong the AUC lands below 0.5
rather than being rescued, which is what keeps the number falsifiable.

**Caveat, and it matters:** this rule inflates AUC on null data. The `f81`
control at matched block lengths returns 0.50–0.68 rather than 0.50. So 0.5 is
*not* the right baseline for the oriented AUC — the `f81` column is, which is why
the sensitivity figure plots both series on the same axes.

## 7. The repair prediction is not supported

The prediction: the mosaic archetype should score worse under ProteinMPNN's joint
model than a coherent sub-ancestor, and the gap should widen with contamination.

As originally measured the comparison is confounded. The mosaic is reconstructed
from all 12 witnesses and each sub-ancestor from 6, so "coherent" and
"reconstructed from fewer sequences" vary together — and a reconstruction built
from more data sits closer to the family consensus, which is exactly where a
joint model assigns high likelihood. That pull runs opposite to the effect being
tested.

The tell is that the confounded gap is **just as positive at zero contamination**
(+0.036), where there is nothing to repair.

Holding witness count fixed — one coherent clade of 6 against a mixed 6 drawn
across the root — removes it:

| comparison | mean gap | predicted sign |
|---|---|---|
| vs mosaic, n=12 vs n=6 (confounded) | +0.025 ± 0.033 | 7 / 12 |
| vs mixed, n=6 vs n=6 (**controlled**) | **+0.007 ± 0.059** | **4 / 12** |

Controlled, the effect is indistinguishable from zero, shows no trend with
contamination, and carries the predicted sign in fewer than half the runs. On the
real family the gap is **negative** (−0.23): the mosaic scores *better* than
either sub-ancestor.

This is a negative result and is reported as one. No new compute was spent
chasing a regime where it might hold.

## 8. Ablating the instrument: the structural model does not earn its place

The project's own pre-registered test was: *swap MPNN for a sequence-only model;
if the conflict signal survives, the structural claim is wrong.* Here is that
test, plus a second one in the opposite direction.

The sequence-only stand-in is as cheap as it gets — at each site, ask which
sub-ancestor the mosaic matches:

```
identity_i = [a_i == subA_i] - [a_i == subB_i]
```

No structure, no backbone, no network. It is fed to the *same* segment scan and
the *same* permutation test, so any difference in outcome is a difference in
signal rather than in procedure. The third arm keeps ProteinMPNN but permutes the
backbone coordinates across residues, destroying the structural neighbourhood
while preserving the model, the alphabet and the composition.

| seed | block | diagnostic sites in block | `mpnn` p / J | `identity` p / J | `scrambled` p / J |
|---|---|---|---|---|---|
| 0 | 30 | 13 | 0.020 / 0.71 | **0.005 / 0.97** | 0.582 / 0.00 |
| 0 | 50 | 22 | 0.060 / 0.00 | **0.005 / 1.00** | 0.463 / 0.00 |
| 1 | 30 | 0 | 0.915 / 0.00 | 0.294 / 0.00 | 0.612 / 0.00 |
| 1 | 50 | 22 | 0.005 / 0.90 | **0.005 / 0.94** | 0.542 / 0.00 |
| 2 | 30 | 2 | 0.318 / 0.00 | 0.154 / 0.00 | 0.950 / 0.00 |
| 2 | 50 | 16 | 0.328 / 0.00 | **0.010 / 0.82** | 0.871 / 0.00 |

| arm | fired | mean Jaccard | mean site AUC |
|---|---|---|---|
| `mpnn` | 2 / 6 | 0.27 | 0.72 |
| **`identity`** | **4 / 6** | **0.62** | **0.74** |
| `scrambled` | 0 / 6 | 0.00 | 0.55 |

Both results are real and they point in opposite directions.

**The MPNN signal is genuinely structural.** Scrambling the backbone takes it to
0 / 6 firings and site AUC 0.55. That is not a trivial breakage — the same
scramble drops the native sequence's pseudo-log-likelihood from −1.49 to −3.06,
which is the expected magnitude for destroying a fold, not a NaN cascade. So when
MPNN reports conflict, it is reporting something about structure.

**But it does not beat string comparison.** Sequence identity fires twice as
often, recovers the window more than twice as well, and edges it on site AUC. On
seed 0 / width 50 it recovers the breakpoint *perfectly* (Jaccard 1.00) on a run
where MPNN failed to reach significance at all. On seed 2 / width 50, MPNN was
silent (p = 0.33) while identity found the block at p = 0.010, J = 0.82.

By the project's own criterion, **the structural claim is not supported**. The
conflict signal survives the sequence-only ablation, and thrives.

What this does *not* say is that the framing was wrong. The README's premise —
that per-site posteriors within one reconstruction cannot see a chimera — still
holds. What the ablation isolates is *which step* does the work: it is not the
joint structural model, it is **reconstructing the two sub-histories separately
and comparing the mosaic to each**. That comparison is already the informative
operation, and once you have done it, string equality reads it off exactly while
ProteinMPNN estimates it noisily through a structural proxy.

The constructive reading is that the *repair* framing — separate the
hyparchetypes, then compare — is the contribution, and the instrument chosen to
perform the comparison is the part that should be cheap.

One caveat on scope: `identity` is not GARD or RDP. It consumes the two
sub-ancestors this pipeline reconstructs, so it is a control on the instrument,
not a head-to-head against published sequence-based recombination detectors.
That comparison remains unrun.

## 9. Starving the sequence evidence: does the structural model ever win?

The README's claim is specific: a structural detector is orthogonal signal *"in
the regime where the sequence methods run out — deep divergence, saturated
sites, short genes."* Nothing measured so far tests that regime — every
condition had six well-behaved witnesses per clade, which is where sequence
evidence is abundant and the `identity` control (§8) wins outright.

`identity` depends entirely on how well the two sub-ancestor reconstructions
turned out; MPNN additionally conditions on the backbone, which does not
degrade. So thinning the witnesses per clade should hurt `identity` and leave
MPNN comparatively less damaged — if the structural prior is worth anything,
the two curves should cross as evidence gets scarce. (Below four witnesses per
clade, NJ + midpoint rooting stops recovering the true clade split at all — it
returned 5 | 1 on a 3 | 3 family — so the known split is supplied directly
rather than inferred, isolating reconstruction quality from tree-inference
failure.)

| witnesses/clade | verbatim | mpnn Jaccard | mpnn AUC | mpnn fired | identity Jaccard | identity AUC | identity fired |
|---|---|---|---|---|---|---|---|
| 6 | 97% | 0.30 | 0.68 | 1/3 | **0.61** | **0.81** | **2/3** |
| 4 | 99% | 0.27 | 0.73 | 1/3 | **0.94** | **0.94** | **3/3** |
| 3 | 98% | 0.00 | 0.44 | 0/3 | 0.00 | 0.57 | 0/3 |
| 2 | 95% | 0.25 | 0.59 | 1/3 | 0.27 | 0.44 | 2/3 |

No crossing at any witness count. `identity` matches or beats `mpnn` all the
way down to two witnesses per clade.

The `verbatim` column says why the rescue was never possible: it stays
**95–99% even at two witnesses per clade**. Marginal ML ASR takes an argmax
over 20 residues, and in a two-clade family that argmax is essentially always
one of the two clade consensus residues — a property of *how marginal ASR
resolves ties*, not of how much data went in. Thinning the evidence does not
touch it, so this route cannot rescue the structural claim.

## 10. Contamination contiguous in space rather than in sequence

Every condition up to this point swapped a contiguous *sequence* block and
searched for it with a scan over *sequence* position — a game a string
comparison wins by construction. Gene conversion of a folded structural element
does not have to respect sequence order: a 30-residue patch compact on the
backbone typically breaks into 2–4 separate runs spanning up to 90 positions
when read off the chain. A scan over alignment position cannot represent that
target at all, whatever score feeds it.

A 2×2×2 keeps the attribution honest — both scores go through both scans:

| contamination | score | scan | fired | mean Jaccard |
|---|---|---|---|---|
| block (sequence-contiguous) | mpnn | 1D | 1/9 | 0.10 |
| block | mpnn | 3D | 0/9 | 0.06 |
| block | identity | **1D** | **3/9** | 0.14 |
| block | identity | 3D | 1/9 | 0.06 |
| patch (structure-contiguous) | mpnn | 1D | 1/9 | 0.12 |
| patch | mpnn | **3D** | **3/9** | 0.09 |
| patch | identity | 1D | 0/9 | 0.09 |
| patch | identity | **3D** | **4/9** | **0.15** |

Pooling both scores, patch contamination is found far more often by the 3D scan
than the 1D scan (7/18 vs 1/18, Fisher p = 0.041), and matching the scan to the
contamination's geometry beats mismatching it overall (11/36 vs 2/36, Fisher
p = 0.012). **This is a real, significant interaction**, and it is a capability
no sequence-window method — GARD, RDP, or `identity`+1D — has by construction:
none of them can search a coordinate they never look at.

Two things temper it. Localisation is weak — best-cell mean Jaccard is 0.149
against roughly 0.08 expected from a random found-set of that size, so this
detects that something is off rather than saying precisely where. And within
the 3D scan on patches, `identity` still edges `mpnn` (4/9, J 0.149 vs 3/9, J
0.089) — so even here, what earns its place is the **structural scan**, not
the **structural model**. The useful idea this experiment surfaces is *where
you search*, not *what you search with*.

**Statistical power here is thin (n = 9 per cell) and this has not been
re-run at higher power** — see the note on the pending 20-seed rerun in
Limitations.

## 11. The empirical family: nothing fires

67 structure-backed three-finger toxins from UniProt, aligned with MAFFT, trimmed
to 58 core columns (≤ 20% gaps), 56 witnesses retained after alignment-space
filtering and deduplication. Tree and marginal ASR with IQ-TREE, run
independently per clade so neither sub-ancestor has been told about the other.

The reconstruction was folded before being trusted: **pLDDT 87.4**, and the
disulfide connectivity is **canonical 3FTx** — (1,3), (2,4), (5,6), (7,8) in
bonded-cysteine rank, identical to erabutoxin b (3EBX), plus one supernumerary
free cysteine. The checker was validated against 3EBX, where it reproduces the
crystal structure's own `SSBOND` records to 0.01 Å.

| split | clades | diagnostic sites | detected | p |
|---|---|---|---|---|
| midpoint | 53 \| 3 | 38 | no | 0.42 |
| most-even | 33 \| 23 | 32 | no | 0.32 |

No detection under either rooting. Two things this does *not* mean: it is not
evidence that the family is recombination-free, and it is not a clean test of
the method. Given section 4 — that detection needs on the order of 13+ diagnostic
sites *inside* a contaminated block, out of 32–38 total here — this family is
close to the floor where the detector has no power regardless of what is true.

Midpoint rooting is also pulled onto the long branch and returns a 53 | 3 split,
which is not two comparable sub-histories at all. The most-even split is reported
beside it because which split you take is a free parameter that changes the
answer, and that dependence belongs in the open.

---

## Limitations

- **The 20-seed rerun is done — see §12.** Sections 4, 8, 9 and 10 above are
  still anecdote-scale (2–3 seeds); §12 repeats the segment sweep and the
  ablation at 20 seeds with blocks above the diagnostic-site floor, with Wilson
  95% confidence intervals on every detection rate.
- **In silico throughout.** No wet-lab validation. Stability is a
  pseudo-log-likelihood proxy, not a measured ΔΔG.
- **One backbone.** Every simulated witness is evolved on 5L33. Nothing here
  shows the result transfers to another fold.
- **The simulator asserts what the detector senses.** The `selection` model Gibbs-
  samples from ProteinMPNN's own joint distribution, so it builds in the epistasis
  the probe then reports. The `f81` control is what keeps this honest, and the
  empirical family is the real test — which returned nothing.
- **The divergence grid is underpowered, even after tripling it.** §5's
  original 2-seed grid found a floor, not a dose-response curve. §14 reruns
  it at 6 seeds and finds a real-looking step at stem = 4.0 (67% vs 33%) —
  but the 95% CIs still overlap, and detection does not track divergence
  monotonically (stem 0.5 and 2.0 are tied). Separating this with confidence
  would need the same 20-seed investment §12 used elsewhere.
- **Underpowered.** Pooling the main experiment and the segment sweep and removing
  the three conditions they share: **2 detections across 15 distinct contaminated
  conditions — and the `f81` control also fires 2 times in its matching 15.**
  Three seeds per configuration is too few to put an interval on a detection rate.
- **A sequence-identity control beats the structural probe.** Section 8. Fires
  4/6 against MPNN's 2/6, mean Jaccard 0.62 against 0.27. Scrambling the backbone
  takes MPNN to 0/6, so the signal is structural — but structural is not the same
  as useful, and the ablation is the result that most constrains the thesis.
- **The GARD head-to-head (§15) is measured, not run at full strength.** On
  `selection` data GARD fires 0/6 against `mpnn`'s 2/6 and `identity`'s
  4/6 — real evidence for orthogonality — but GARD was run in amino-acid
  mode with no codon layer to feed it, a weaker deployment than its normal
  use, and `selection` is independently the hardest case by diagnostic-site
  count (§4). GARD cannot run at all on the real 3FTx family (needs 235
  sites for 56 taxa, has 58) — the sharpest single data point for the
  README's "where sequence methods run out" claim, but still one family.
  RDP remains untested.
- **The oriented AUC has an inflated null.** See section 6. Read it against the
  `f81` series, never against 0.5.
- **The empirical test was inconclusive, not negative — and §16 extends it to
  three more families, same conclusion.** 0/4 real families detected (3FTx,
  ribonuclease A, lysozyme C, cytochrome c), p-values 0.36-0.89 with no
  clustering near significance. Consistent, but still four families chosen
  for tractability, not sampled from the literature, and none has a known
  contaminated block to test power against.
- **The clade split is a free parameter.** On real data, midpoint and most-even
  rooting give different sub-histories and different answers.
- **The method only covers a single breakpoint.** Section 13. Splitting the
  same 50-residue contaminated budget into 2+ separate runs (multiple
  crossovers, multi-tract gene conversion) collapses detection to near zero
  — `mpnn` 1/9 and `identity` 0/9 once fragmented, against 2/3 and 3/3 at a
  single contiguous block. `scan_segment` was built to find one window; every
  other detection rate in this document is a single-breakpoint number and
  does not generalize past it.
- **Locality.** MPNN conditions on a local structural neighbourhood, so a
  contiguous sequence swap is strained mainly near its structural junctions. The
  whole-sequence penalty for a mosaic is small; the signal lives in *where* the
  conflict sits, not in its magnitude.

## Reproducibility hazard

On Apple silicon, PyTorch's MPS backend has been observed failing a Metal command
buffer partway through a long run and then continuing with corrupted tensors
without raising. In a previously recorded incident (not a run in this document)
the same seed and settings that had given p = 0.030 returned p = 1.000 with every
score degraded to noise. `MPNNScorer.log_probs` now checks
that the model output is finite and raises if it is not. Everything in this
document was produced with `--device cpu`.

## 12. Firming up the statistics: 20-seed rerun, blocks above the floor

The sections above ran at 2–3 seeds, with 10/20/30-residue blocks that §4
showed were mostly below the diagnostic-site floor. This reruns the segment
sweep and the ablation at 20 seeds with blocks of 50/65/80 residues, so the
detection-rate numbers below carry a real confidence interval instead of
standing on 2 or 3 observations. Produced by
`experiments/run_power_sweeps.sh`; raw output in `*_power20.json`.

### Detection rate by width, selection vs f81

| model | width | fired | rate | 95% CI | mean Jaccard (fired) |
|---|---|---|---|---|---|
| selection | 50 | 0/20 | 0% | [0%, 16%] | n/a |
| selection | 65 | 7/20 | 35% | [18%, 57%] | 0.754 |
| selection | 80 | 6/20 | 30% | [15%, 52%] | 0.727 |
| f81 | 50 | 1/20 | 5% | [1%, 24%] | 0.602 |
| f81 | 65 | 0/20 | 0% | [0%, 16%] | n/a |
| f81 | 80 | 2/20 | 10% | [3%, 30%] | 0.245 |

Pooled across widths: selection 13/60 (22%, 95% CI [13%, 34%]), f81 3/60 (5%, 95% CI [2%, 14%]). The two intervals **overlap**.

### Ablation at 20 seeds

| arm | width | fired | rate | 95% CI | mean Jaccard (fired) |
|---|---|---|---|---|---|
| mpnn | 50 | 6/20 | 30% | [15%, 52%] | 0.678 |
| mpnn | 80 | 5/20 | 25% | [11%, 47%] | 0.396 |
| identity | 50 | 18/20 | 90% | [70%, 97%] | 0.683 |
| identity | 80 | 18/20 | 90% | [70%, 97%] | 0.160 |
| scrambled | 50 | 1/20 | 5% | [1%, 24%] | 0.000 |
| scrambled | 80 | 1/20 | 5% | [1%, 24%] | 0.662 |

Pooled: `mpnn` 11/40 (95% CI [16%, 43%]), `identity` 36/40 (95% CI [77%, 96%]).

One nuance the per-width table above hides: **mean Jaccard among firings only,
split by width**, is `mpnn` 0.678 vs `identity` 0.683 at width 50 — essentially
tied — but `mpnn` 0.396 vs `identity` 0.160 at width 80, where 15 of
`identity`'s 18 width-80 firings land at Jaccard < 0.1 (`mpnn`'s width-80
firings are noisy too — 3 of 5 near-zero — just less so). So this is not "mpnn
localizes better in general"; it is "at width 80, most of `identity`'s firings
look like the permutation test tripping on a window that happens to pass
significance rather than a real localization." It does not change the headline
of this section: `identity` still detects far more often overall (90% vs 28%,
non-overlapping CIs), which is the metric that determines whether a real
contamination event gets flagged at all.

## 13. Multiple breakpoints: the single-window scan does not survive fragmentation

Every condition above — main experiment, segment sweep, divergence grid,
ablation, spatial patches — copies exactly one contiguous run from one donor.
`conflict.scan_segment` looks for exactly one contiguous window, so this has
never been a fair test of what happens when it isn't given one. Real
recombination is rarely a single crossover: multiple breakpoints between the
same two parents, or multi-tract gene conversion, split the contaminated
sequence into several separate runs rather than one.

The total contaminated length is held at *exactly* 50 residues — the width
already established in §12 as sitting above the diagnostic-site floor for a
single block — and only the number of separate runs it is split into varies
(1, 2, 3, 4), evenly spaced across the backbone. No new detection code: this
reuses `evolve.contaminate_positions` (already generalized to an arbitrary
position *set*, not just an interval — built for §10) and scores the
shipped 1D scan against the true position *set* with `spatial.patch_jaccard`,
exactly as §10 does for a spatially- rather than sequence-discontiguous
target. 3 seeds, both `mpnn` and `identity` scores, both through the same
scan.

| n_blocks | score | fired | mean Jaccard | mean sites found |
|---|---|---|---|---|
| 1 | mpnn | 2/3 | 0.251 | 15.0 |
| 1 | identity | 3/3 | 0.378 | 20.7 |
| 2 | mpnn | 0/3 | 0.164 | 13.0 |
| 2 | identity | 0/3 | 0.060 | 12.0 |
| 3 | mpnn | 1/3 | 0.057 | 8.3 |
| 3 | identity | 0/3 | 0.066 | 4.7 |
| 4 | mpnn | 0/3 | 0.056 | 7.0 |
| 4 | identity | 0/3 | 0.019 | 4.7 |

One clean, one confounded, and both point the same way.

**The clean comparison is within seed 0.** Its four conditions carry 22, 15,
15 and 18 diagnostic sites *inside the true positions* — all above the
13-site floor §4 established for reliable detection, so site count is not
the variable changing. `mpnn` still goes from p = 0.010 (n_blocks=1) to
p = 0.139 / 0.413 / 0.856 (n_blocks=2/3/4): fragmenting the same budget of
diagnostic sites into separate runs kills detection even when there are
enough of them.

**Seeds 1 and 2 confound it.** Diagnostic sites inside the true positions
drop with n_blocks in both (15→16→4→5 and 21→18→3→2) — an artifact of
where this experiment's fixed, evenly-spaced block layout happens to land
relative to each family's own diagnostic sites, not a property of
fragmentation itself. For those two seeds, the collapse in detection is at
least partly the already-known floor effect from §4 wearing a new hat, not
new evidence about fragmentation specifically. This was not deconfounded by
re-running with placement held fixed and only spacing varied — the seed-0
result already answers the qualitative question without it, and a cleaner
dose-response design is future work rather than new compute spent here.

**Both scores collapse together, and pooled detection is symmetric:**
`mpnn` fires 3/12 overall (2 at n_blocks=1, 1 at n_blocks=3), `identity`
fires 3/12 (all at n_blocks=1). The one fragmented firing — `mpnn`,
n_blocks=3, seed 2, p = 0.040 — lands at Jaccard 0.03, ten sites found
against three true ones. That is the same pattern §2-3 established for the
`f81` control: a permutation test can clear α on a window that has nothing
to do with the true positions, so a bare "detected" count is not evidence of
localization. Reading Jaccard rather than the fired count, both scores are
indistinguishable from noise once split past one run — mean Jaccard
0.02–0.16 against the ~0.08 chance level §10 estimated for a random
found-set of comparable size.

This is a negative result for the method as shipped, not a new failure mode
invented to be negative: `scan_segment` was built to find one window, and a
target that is not one window is not found. It also means every detection
rate reported elsewhere in this document — the main experiment, the power
rerun, the ablation — describes the single-breakpoint case specifically, and
does not extend to gene conversion or multi-crossover recombination without
a scan that can represent a discontiguous target, which is exactly what §10
already built for the spatial case and this section did not need to
duplicate.

Produced by `experiments/multi_breakpoint.py --seeds 3`; raw output in
`multi_breakpoint.json`, recomputed by `summarize.py`. `multi_block_positions`
originally floor-divided the residue budget (`total_size // n_blocks`) and
dropped the remainder, so n_blocks=3/4 actually swapped 48 residues against
50 for n_blocks=1/2 — a real, if small, confound on the "same total budget"
claim this section depends on, caught in a later review pass and fixed to
distribute the remainder across the first few blocks instead. The numbers
above are from the corrected code; the conclusion did not change.

## 14. Dose-response: donor distance at real power

Section 5's divergence grid was 2 seeds per cell and the honest conclusion
was "the floor is established; the dose-response is not." This reruns it at
3x the seeds (6, not 2), holding everything else fixed at the settings §12
already validated as informative: n = 6 witnesses/clade (§5 found n = 3
never detects at any divergence), a 50-residue contaminated block (above the
diagnostic-site floor), 3 stem levels spanning near-sibling donor (0.5) to
deep outgroup (4.0). `sweep_divergence` gained the same per-cell
checkpointing `sweep_segment` already had — a run this long (several fresh
Gibbs simulations at ~12–28 minutes each, scaling with `stem` itself, since
more branch length means more Gibbs sweeps) needed it.

| stem | fired | rate | 95% CI | mean diagnostic sites |
|---|---|---|---|---|
| 0.5 (near sibling) | 2/6 | 33% | [10%, 70%] | 21.5 |
| 2.0 | 2/6 | 33% | [10%, 70%] | 36.2 |
| 4.0 (deep outgroup) | 4/6 | 67% | [30%, 90%] | 44.7 |

**Not a clean monotonic curve.** If detection tracked divergence smoothly,
stem = 0.5 should trail stem = 2.0. It does not — they are tied at 2/6.
Only stem = 4.0 shows a step up. Diagnostic-site count does climb
monotonically and saturates (21.5 → 36.2 → 44.7), the same qualitative
shape §5 found (16.8 → 30.5 → 36.2, different seeds) — divergence still
buys diagnostic sites reliably. What does not follow smoothly is detection
built on top of that count, which is the thing that actually matters.

**The gap at stem = 4.0 is suggestive, not established.** The 95% Wilson
intervals — [10%, 70%] at stem = 0.5/2.0 against [30%, 90%] at stem = 4.0
— overlap substantially. Six seeds per cell triples §5's sample but is
still short of separating a 33% rate from a 67% one with confidence; the
same 20-seed-per-cell investment that separated `selection` from `f81` in
§12 would be needed to make this a real dose-response curve rather than a
step that six seeds happened to land on.

Produced by `experiments/sweeps.py --sweep divergence --model selection
--seeds 6 --stems 0.5 2.0 4.0 --clade-sizes 6 --width 50 --tag power6`; raw
output in `sweep_divergence_selection_power6.json`, recomputed by
`summarize.py`.

## 15. Head-to-head against a published detector: GARD

`identity` (§8) is not GARD or RDP — it consumes the two sub-ancestors this
pipeline reconstructs, so it is a control on this project's own instrument,
not the orthogonality test the README's thesis actually asks for: *"a
structural detector would be orthogonal signal in the regime where the
sequence methods run out."* That claim has been asserted since the README
was first written and never measured. This measures it, against GARD
(Kosakovsky Pond et al. 2006) — a genetic algorithm over alignment
partitions that fits a substitution model to each candidate partition and
picks the partition count by c-AIC. HyPhy 2.5 ships GARD for amino-acid
alignments directly (`hyphy gard --type amino-acid --model JTT`), so it runs
on exactly the protein data this project already has — no nucleotide or
codon layer exists here to hand it, which is a weaker use of GARD than its
usual deployment on coding sequence, where synonymous substitution adds
power a pure amino-acid fit does not have. That asymmetry is real and is not
hidden below.

Run on the same six conditions already reported for `mpnn` / `identity` /
`scrambled` in §8 — no new simulation, the new column drops straight into
that table:

| seed | width | diagnostic sites in block | mpnn | identity | gard |
|---|---|---|---|---|---|
| 0 | 30 | 13 | fired | fired | — |
| 0 | 50 | 22 | — | fired | — |
| 1 | 30 | 0 | — | — | — |
| 1 | 50 | 22 | fired | fired | — |
| 2 | 30 | 2 | — | — | — |
| 2 | 50 | 16 | — | fired | — |

**GARD fires 0/6 on the `selection` families — the regime this project is
actually about.** Both `mpnn` (2/6) and `identity` (4/6) find something GARD
finds nothing on, on identical data. This is the cleanest evidence so far for
the orthogonality claim, with the caveat that it is evidence for a narrower
version of it: GARD is being run at a disadvantage (amino-acid only, 12
witnesses, 106 sites) relative to its normal use, and §4 already established
that `selection` families are the *hardest* case by diagnostic-site count
(23-41 sites, against `f81`'s 67-78) — so part of what GARD is failing on
here is the same site-count floor that constrains this project's own
detector, not necessarily something GARD would still miss with codon data
and more taxa.

**On the `f81` control — more diagnostic sites, the easier case — GARD does
fire, 3/6, but only ever finds *one* breakpoint, never the two that would
bound the true block.** `n_breakpoints` was 1 in all three firings. A single
breakpoint splits the alignment into two partitions with no bounded interval
to score, so no Jaccard comparison is possible for these — GARD is flagging
that the alignment is not tree-like without recovering where the
non-tree-like region actually is. That is a real capability gap from this
project's own detector, which supplies a segment and a Jaccard score even
when it is wrong (§3); it is also a real capability GARD has and `mpnn`
does not on this same control — 3/6 beats `mpnn`'s and `identity`'s
combined presence in the `f81` main experiment (§2: 0/6 for both at the
default 30-residue width). Different widths, so not an apples-to-apples
number, but the qualitative point holds: GARD is not blind to `f81`-style
composition-only contamination, it is blind to `selection`-style
structurally-coherent contamination, and localizing it is a separate
capability it does not have here regardless of which regime it is run on.

**GARD cannot run on the real 3FTx family at all.** 56 sequences, 58
aligned sites. HyPhy's own assertion: *"The alignment is too short to
permit c-AIC based model comparison. Need at least 235 sites for 56
sequences to fit a two-partition model."* This project's own detector runs
on exactly this alignment (§11) and returns an answer — an inconclusive
one, but a computable one. This is the sharpest evidence in this document
for the README's specific claim about *where* a structural detector would
be orthogonal signal: not generically better, but applicable in a regime —
short genes, deep divergence pushed onto few informative sites — where a
partition-fitting sequence method's own parameter-count requirements rule
it out before comparison is even possible.

Produced by `experiments/gard_baseline.py --model selection --seeds 3` and
`--model f81 --seeds 3` (both `--widths 30 50`, matching §8), plus a
one-off run of `run_gard` against `data/interim/3ftx/core.fasta`; raw output
in `gard_selection.json`, `gard_f81.json`, `gard_3ftx.json`, recomputed by
`summarize.py`. `hyphy` via `brew install hyphy` (2.5.101); not vendored,
not a project dependency for anything else here.

## 16. The Bedier audit: the detector across several published families

Section 11 runs the detector on one real family and is explicit that a
single non-detection near the diagnostic-site floor is inconclusive, not
evidence either way. The obvious next question is what the conflict-score
distribution looks like across more than one — not a claim that any of
these families is actually contaminated (there is no ground truth for any
of them, 3FTx included), but a report of where the method lands on real,
independently published reconstructions rather than one hand-picked case.

Three more families, chosen only for being small enough to fold on CPU in
minutes and having enough structure-backed reviewed UniProt entries to form
a family at all — not for any expected outcome: ribonuclease A, lysozyme C,
cytochrome c, all classic ASR/phylogenetics subjects with deep structural
literature. Each goes through [real_family]'s identical staged pipeline
(fetch, MAFFT, trim, IQ-TREE tree + per-clade marginal ASR — independently
per clade, same as §11), with one addition automated by
`experiments/bedier_audit.py`: the mosaic ancestor is folded with ColabFold
so the detector has a backbone that corresponds to it residue-for-residue,
the same two-phase workflow §11 used for 3FTx, just run across a family
list instead of by hand.

| family | witnesses | clades | diagnostic sites | detected | p | mean pLDDT |
|---|---|---|---|---|---|---|
| 3ftx (§11) | 56 | 53 \| 3 | 38 | no | 0.418 | n/a* |
| rnase_a | 22 | 10 \| 12 | 75 | no | 0.891 | 89.2 |
| lysozyme_c | 22 | 15 \| 7 | 73 | no | 0.657 | 97.0 |
| cytochrome_c | 29 | 21 \| 8 | 24 | no | 0.363 | 86.6 |

\* 3FTx's fold check (§11) reports pLDDT 87.4 by a different route — folded
before this audit existed, using two ColabFold recycles rather than three —
so it is not re-quoted here as if produced by the same run.

**0/4 detected.** This is not evidence that any of these four families is
recombination-free — §4 and §5 already established that detection needs
roughly 13+ diagnostic sites *inside* whatever block is being searched for,
and none of these families has a known block to search for in the first
place, so a non-detection here carries exactly the same limited weight §11
gives the 3FTx result on its own. What the audit adds is that the pattern is
consistent rather than a one-off: four different real families, four
non-detections, p-values spread from 0.36 to 0.89 with no sign of
clustering near significance the way the `f81` control's spurious firings
did in §2 (p = 0.015, close to α). If this pipeline were prone to firing on
ordinary real proteins for reasons unrelated to contamination, four
independent families would be a reasonable chance to see it, and none did.

**cytochrome c's alignment is a real audit finding on its own.** The query
pulled 79 structure-backed reviewed entries — evolutionarily broader than
intended — and after gap-filtering only 35 of 326 aligned columns survived
at ≤20% gaps, the shortest core alignment of any family tried, real or
simulated. 24 diagnostic sites is close to the ~20-site floor §5 found
below which nothing fires regardless of truth. This is exactly the failure
mode the README's thesis names — a short, divergence-saturated alignment —
occurring by accident from an ordinary UniProt query, not constructed to
demonstrate the point.

**Fold confidence does not track detection.** pLDDT ranges from 86.6 to
97.0, all comfortably in a range that would be called "confident" for an
AlphaFold2 model, and detection is uniformly absent regardless. This rules
out one confound worth naming: it is not that some ancestors are folding
badly and are undetectable for that reason.

**Scope, stated plainly.** Three families, chosen for tractability rather
than sampled from the literature at random, plus 3FTx — four is a real
number but not a large one, and "audited many published reconstructions"
would overstate what four data points support. The family-specific
disulfide-topology check §11 used for 3FTx (validated against a crystal
structure) is not repeated here; only mean pLDDT gates fold quality, since
building a per-family reference topology check for each new family is out
of scope for this audit. Every clade split here is midpoint rooting only —
§11's own finding that midpoint and most-even rooting can disagree is not
re-tested across these three.

Produced by `experiments/bedier_audit.py --families rnase_a lysozyme_c
cytochrome_c` (`colabfold_batch` via `.venv/bin/pip install colabfold`; no
GPU, ColabFold's public MSA server for the alignment step); raw output in
`bedier_audit.json`, recomputed by `summarize.py` together with the
existing `real_3ftx_midpoint.json`. Building this surfaced a real,
independent bug in `real_family.py`: its results filename hardcoded
`real_3ftx_{tag}.json` regardless of which family `--work` pointed at, so
the first run against `rnase_a` silently overwrote `real_3ftx_midpoint.json`
with the ribonuclease result. Caught immediately because the printed clade
sizes didn't match section 11's; the original file was restored from git
and the filename now derives from `--work`'s directory name. No results in
this document were affected — the overwrite and its correction both
happened after §11 was written and were resolved before anything was
computed from the corrupted file.

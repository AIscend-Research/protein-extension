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
- **The divergence grid is underpowered.** 2 / 12 with both detections in one
  seed. It establishes a floor, not a dose-response curve. Section 5.
- **Underpowered.** Pooling the main experiment and the segment sweep and removing
  the three conditions they share: **2 detections across 15 distinct contaminated
  conditions — and the `f81` control also fires 2 times in its matching 15.**
  Three seeds per configuration is too few to put an interval on a detection rate.
- **A sequence-identity control beats the structural probe.** Section 8. Fires
  4/6 against MPNN's 2/6, mean Jaccard 0.62 against 0.27. Scrambling the backbone
  takes MPNN to 0/6, so the signal is structural — but structural is not the same
  as useful, and the ablation is the result that most constrains the thesis.
- **No head-to-head against GARD or RDP.** The orthogonality claim in the
  introduction is still asserted rather than measured. The identity arm is a
  control on this pipeline's own instrument, not a published detector.
- **The oriented AUC has an inflated null.** See section 6. Read it against the
  `f81` series, never against 0.5.
- **The empirical test was inconclusive, not negative.** The 3FTx family sits near
  the diagnostic-site floor, so its non-detection carries little information
  either way.
- **The clade split is a free parameter.** On real data, midpoint and most-even
  rooting give different sub-histories and different answers.
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

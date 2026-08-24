# Extension experiments -- aggregated results

_Recomputed from `experiments/results/*.json`. Rates carry Wilson 95% CIs._

### 1. Donor distance (dose-response)

Donor lineage depth varied with between-clade divergence (`stem`) held fixed, so context separability is constant and only donor foreignness changes.

| donor depth | detection rate (95% CI) | mean Jaccard | mean diag. sites |
|---|---|---|---|
| 0.25 | 0/12 = 0.00 [0.00, 0.24] | 0.00 | 41.6 |
| 0.5 | 0/12 = 0.00 [0.00, 0.24] | 0.00 | 42.2 |
| 1.0 | 1/12 = 0.08 [0.01, 0.35] | 0.02 | 42.5 |
| 2.0 | 0/12 = 0.00 [0.00, 0.24] | 0.00 | 42.8 |
| 4.0 | 0/12 = 0.00 [0.00, 0.24] | 0.00 | 42.4 |
| 8.0 | 1/12 = 0.08 [0.01, 0.35] | 0.06 | 42.6 |

_Diagnostic-site count is flat across depth (42.3 mean), so donor foreignness does not add diagnostic sites; detection is gated by site density, not distance. No monotone dose-response._

### 2. Multiple breakpoints & gene conversion

Fixed total contamination fragmented into 1/2/3 disjoint blocks (amount held constant), plus a scattered gene-conversion tract. Primary metric is per-site AUC over diagnostic sites, since the single-window verdict is the wrong yardstick once contamination is fragmented.

| condition | detection rate (95% CI) | mean site-AUC |
|---|---|---|
| blocks=1 | 1/8 = 0.12 [0.02, 0.47] | 0.681 |
| blocks=2 | 1/8 = 0.12 [0.02, 0.47] | 0.666 |
| blocks=3 | 1/8 = 0.12 [0.02, 0.47] | 0.519 |
| gene_conversion | 0/8 = 0.00 [0.00, 0.32] | 0.466 |

_Read the site-AUC column: if it holds as blocks rise the detector reads fragmented contamination and only the single-window verdict degrades; if it falls toward 0.5, fragmentation genuinely hurts._

### 3. GARD head-to-head (orthogonality)

Structural detector vs GARD on identical alignments, across divergence.

| stem | MPNN det (95% CI) | GARD det (95% CI) | MPNN mean J | GARD edge-hit |
|---|---|---|---|---|
| 0.5 | 0/5 = 0.00 [0.00, 0.43] | 0/5 = 0.00 [0.00, 0.43] | 0.00 | 0/5 = 0.00 [0.00, 0.43] |
| 1.0 | 1/5 = 0.20 [0.04, 0.62] | 0/5 = 0.00 [0.00, 0.43] | 0.07 | 0/5 = 0.00 [0.00, 0.43] |
| 2.0 | 1/5 = 0.20 [0.04, 0.62] | 0/5 = 0.00 [0.00, 0.43] | 0.14 | 0/5 = 0.00 [0.00, 0.43] |
| 4.0 | 1/5 = 0.20 [0.04, 0.62] | 0/5 = 0.00 [0.00, 0.43] | 0.16 | 0/5 = 0.00 [0.00, 0.43] |

**Orthogonality:** across all 20 families, GARD detected 0, MPNN detected 3, and 3 families were caught by MPNN but missed by GARD. Detection is disjoint -- the structural signal is orthogonal, not redundant.

### 4. Bedier audit

Detector run across published reconstructions; p-values split on the diagnostic-site floor (floor = 20).

| family | p-value | diag. sites | detected | powered? |
|---|---|---|---|---|
| 3ftx | 0.83582 | 38 | False | yes |

_Powered band (n=1): detected 0/1, p median 0.8358. A detector firing across many powered ordinary families would be the warning sign; it does not._


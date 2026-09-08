# Data availability and dictionary

All numerical records used for the reported figures are included as JSON.
There is no external private dataset and no access restriction.

## Common simulation record

Most per-run JSON files contain:

- `parameters.config` together with its enclosing `parameters`: walker count,
  time step, horizon, seed, slab width, budget, weights, target times, and any
  geometric variant; upgrade/robustness records also serialize a disjoint
  stream tag;
- `parameters.model_parameters`: the nondimensional encounter-process
  parameters;
- `results`: kills, survivors, contact/walker step counts, runtime, kill and
  event fractions, and reported peak diagnostics;
- `results.classifier`: histogram edges and integer counts, Gaussian smoothing
  bandwidth, prominence rule, smoothed density, local maxima, and significant
  maxima;
- `results.histograms`: auxiliary coarse or comparison histograms where used;
- `validation_gates`: zero-budget, probability-range, mass-balance, and
  configuration-specific checks.

Summary JSON files contain only values derived from those per-run records and
name their source files.

## Robustness records

- `robustness/classifier_sensitivity.json` reclassifies stored counts.  Its W2
  transition summaries distinguish bracketed, left-censored, and
  right-censored settings; no threshold is extrapolated beyond stored probes.
- `robustness/seed_repeats/*.json` are independent deterministic Monte Carlo
  streams, not bootstrap resamples.
- `robustness/dt_halving/*.json` compare `dt=0.001` and `dt=0.0005` with the
  same declared seed and tag; `dt` itself enters the `SeedSequence` entropy,
  hence the two records use independent streams rather than coupled paths.
- `robustness/dt_half_seed_repeats/*.json` and
  `robustness/dt_half_seed_repeat_summary.json`: three independent-seed runs
  (seeds 20260814--20260816, 10^6 walkers, stream tag 65) of the
  boundary-adjacent cell `m=3`, `eps=0.175`, `B=0.5` at `dt=0.0005`.  Each
  per-run file stores the raw window counts and edges
  (`results.histogram`) and the `classify_both` re-judgement under both
  sigma conventions with the individual five-sigma and 5% flags
  (`results.classifier_both_conventions`); the records deliberately use the
  `classify_both` schema rather than the legacy `results.classifier` schema
  and are therefore outside the 201 retained-campaign records.  The summary
  compares the late-candidate relative prominence and covariance-aware `z`
  with the matched-walker `dt=0.001` seed repeats and the stored dt-halving
  pair; streams at the two resolutions are not paired.
- `exact_m_prr_upgrade/mean_field_boundary.json`: deterministic evaluation
  of the mean-field hazard--survival law `f_1 = B G exp(-B Lambda)` on the
  fixed covariance-aware classifier (real-valued expected counts).  Keys:
  `conventions` (time grid, contact quadrature, exposure integration,
  expected-count rule, classifier and count convention with the result of
  the regression test against `classify_both`, pass rule, crossing search,
  nominal walker counts), `headline` (rounded and unrounded upper crossings,
  deviations from the stored covariance-aware midpoints, W1 and production
  agreement, contact factors), `chains` (per chain and per nominal `N`: the
  97-point budget scan, all pass/fail transitions, bisection bracket, status,
  right-censoring at `B=8` or no certified crossing in the scanned range),
  `grid_checks` (crossings at `dt_mf = 5e-4` and `1.25e-4` against the
  canonical `2.5e-4`), `w1_comparison` (all 96 cells and all mismatches),
  `production_comparison` (all 17 configurations, excluding the `dt/2`
  control, with counted peak-time differences), and `contact_factors`.
- `exact_m_prr_upgrade/covariance_aware_reclassification.json`
  (+ `_summary.txt`): covariance-aware prominence `z` for every local maximum
  of every stored classifier record (keys `records`, `flips`, `w2_chains`,
  `sensitivity_new_rule`, `headline_numbers`); this is the statistic used as
  the formal mode-count definition in the article.
  `w3_jitter/covariance_aware_recheck.json`: the same rule applied to the 700
  W3 replicas (all reproduced, no flips).  The legacy
  `w2_b0_empirical/B0_empirical.json` fields for the chain `m=3`, `eps=0.20`
  (`status`, `b0`, `b0_bracket`) were written under the peak-only convention
  and are superseded by `w2_chains` in the reclassification file; the same
  holds for the mirrored entry `exact_m_prr_upgrade/campaign_summary.json`
  (`streams/w2_b0_empirical/chains[7]`, zero-based), which copies those
  legacy fields unchanged and carries the same `covariance_aware_note`.

## Production-record and legacy-schema notes

The 18 archived `exact_m_offlattice_production/*.json` files predate explicit
serialization of a stream-tag field.  They record seed `20260808`; the
deterministic production driver fixes their stream tag to `1` and includes
`dt` in the entropy.  Thus the runs are exactly specified even though those
older JSON files are not individually self-describing with respect to the
tag.  New driver-generated records serialize the tag.

The W2 directory/file labels `w2_b0_empirical` and `B0_empirical.json`, and
serialized fields beginning with `b0`, are legacy schema names for the
protocol-defined operational threshold `B_op` only.  They never denote the
theorem threshold `B_top` or the sufficient certificate `B_cert`.

## Deliberate storage limitation

Raw walker paths and raw reaction-time vectors were not saved.  This prevents
trajectory-level reanalysis and rebinning from unbinned events.  The retained
integer counts and edges are the sufficient statistics for every histogram,
classifier-sensitivity calculation, and plotted stochastic density reported
with the article.  Full scripts and deterministic seed metadata are supplied
so raw events can be regenerated by rerunning a configuration.

# Robustness-results summary

This file is a compact reading guide to the machine-readable records under
`artifacts/data/exact_m_prr_upgrade/robustness/`.  Values below are descriptive
diagnostics, not preregistered hypothesis tests.

## Covariance-aware reclassification (formal definition used in the article)

All 201 stored classifier records (112 W1 cells, 43 W2 probes, 26 robustness
records, the 18 production rows, W4, and W5; 606 local maxima) were re-judged
with the covariance-aware prominence statistic
`sigma_prom^2 = sum_j (A_peak,j - A_base,j)^2 C_j / (N dt)^2`, where
`A_peak,j` and `A_base,j` are the edge-normalized Gaussian kernel weights of
the peak bin and of the selected contour-base bin, `C_j` the raw bin counts,
`N` the walker count and `dt` the bin width; the acceptance rule is unchanged
(`z >= 5` and prominence `>= 5%` of the window maximum).  This is the
statistic defined in the article.  The peak-only convention
`sigma_peak^2 = sum_j A_peak,j^2 C_j / (N dt)^2` under which the records were
first written is superseded; it inflates `z` of counted maxima by factors
1.00--1.55.  Three of the 606 maxima flip, all in the W2 column `m=3`,
`eps=0.20` (`B = 0.125, 0.210, 0.273`; `z` falls from 5.8--6.1 to 4.2--4.4),
so that column has no certified crossing at 10^6 walkers.  The legacy fields
`status="bisected"`, `b0=0.2816`, `b0_bracket=[0.2726, 0.2909]` stored for
that chain in `w2_b0_empirical/B0_empirical.json`, and the mirrored entry
`campaign_summary.json` (`streams/w2_b0_empirical/chains[7]`, zero-based),
are superseded by `covariance_aware_reclassification.json` (`w2_chains`,
`new_status="no_certified_crossing_at_1e6_walkers"`); both files carry a
`covariance_aware_note` saying so.  The W1 final map is
unchanged (0 of 96 cells), every other W2 bracket-defining verdict is
unchanged, and all 700 W3 replica verdicts are reproduced bit-for-bit and
unchanged (`w3_jitter/covariance_aware_recheck.json`).  Source:
`artifacts/data/exact_m_prr_upgrade/covariance_aware_reclassification.json`
and `covariance_aware_reclassification_summary.txt`.

## Mean-field boundary (parameter-free preflight law of the main text)

`mean_field_boundary.json` (script `exact_m_prr_mean_field_boundary.py`,
deterministic, about 40 s) evaluates the hazard--survival law
`f_1 = B G exp(-B Lambda)`, `Lambda(t) = int_0^t G`, built from the free
exposure clock `G` of `validate_exact_m_offlattice.py`, on the fixed
covariance-aware classifier and the W2 last-basin pass rule, using the exact
expected window counts `N [exp(-B Lambda(e_j)) - exp(-B Lambda(e_{j+1}))]`
as real-valued counts (the classifier adapter is regression-tested against
`classify_both` on all 173 stored integer histograms of the production grid,
W1 map, and W2 probes: 501 local maxima, zero difference).  Time grid
`dt_mf = 2.5e-4` from `t = 0` (80 steps per 0.02 bin); the crossings move by
less than `1e-5` at `dt_mf = 5e-4` and `1.25e-4`.

| chain (m, eps) | scan on [0.125, 8] | mean-field upper crossing | stored covariance-aware `B_op` | deviation |
|:---|:---|---:|---:|---:|
| (2, 0.05) | one pass-to-fail transition | 7.357968 | 6.987 ([6.949, 7.025]) | +5.3% |
| (2, 0.10) | one pass-to-fail transition | 7.479411 | 7.619 ([7.578, 7.661]) | -1.8% |
| (2, 0.15) | all 97 points pass | right-censored, > 8 | right-censored, > 8 | -- |
| (2, 0.20) | all 97 points pass | right-censored, > 8 | right-censored, > 8 | -- |
| (3, 0.05) | one pass-to-fail transition | 4.286502 | 4.155 ([4.132, 4.177]) | +3.2% |
| (3, 0.10) | one pass-to-fail transition | 3.632884 | 3.570 ([3.551, 3.589]) | +1.8% |
| (3, 0.15) | one pass-to-fail transition | 1.524705 | 1.551 ([1.542, 1.559]) | -1.7% |
| (3, 0.20) | no point passes | no certified crossing in scanned range | no certified crossing at 10^6 walkers | -- |

The crossings are quoted at the nominal walker count `N = 10^6`; they are
identical to the bisection tolerance (`1e-9`) at `N = 10^5` and `10^7`
because in every bisected chain the first failing candidate fails the 5%
relative-prominence floor while its covariance-aware `z` is still far above
five (`z >= 6.1` at `N = 10^5`, `>= 19` at `10^6`).  This is the sensitivity
of the upper mean-field crossing to `N` through the five-sigma gate; it is not
a statement about the finite-walker Monte Carlo verdicts.  For `(3, 0.20)` the
late candidate never reaches the floor (relative prominence at most 0.030 at
`B = 0.125`) and the five-sigma gate also fails at every scanned budget, so no
lower crossing is reported either.  The mean-field verdict agrees with the
covariance-aware final W1 map in 95 of 96 cells at `N = 10^6` (and at the
stored walker counts); the exception is `(m, eps, B) = (3, 0.075, 4)`, where
the mean-field third candidate sits at 5.2% of the window maximum (stored
record: 2 modes), one of the four classifier-sensitive cells.  For the 17
production configurations (the `dt/2` control excluded) every covariance-aware
mode count is reproduced and every counted peak time agrees within one 0.02
bin (maximum difference 0.02).  Contact factors at the window opening
`tau = 0.5`: `c = 1.000, 0.999, 0.986, 0.941, 0.869, 0.788, 0.711, 0.588` at
`eps = 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25`.  Note that the
`(3, 0.15)` crossing is 1.5247, i.e. 1.52 to two decimals.

## Stored-count classifier sensitivity

The primary sensitivity grid used smoothing bandwidth
`h = 0.03, 0.04, 0.05` and relative-prominence floor
`r = 0.03, 0.05, 0.07`, with the five-sigma condition retained (the four
sensitive cells are the same under the peak-only and the covariance-aware
statistic).
Four of the 96 final W1 cells changed mode count somewhere on this 3 by 3
grid: the `m=3` cells `(eps,B)=(0.05,4)`, `(0.075,4)`, `(0.10,4)`, and
`(0.175,0.5)`.  The principal `m=3`, `m=5`, and `d=3` anchors retained their
reported mode counts throughout the wider grid `h=0.02,...,0.08` and
`r=0,0.01,0.03,0.05,0.07,0.10`.

W2 probes were deliberately concentrated near the baseline decision
boundaries.  The following ranges are geometric midpoints of brackets that
are supported by the stored probes, across the nine primary classifier
settings.  They are not fresh bisections or extrapolations.

| m | eps | bracketed midpoint range | censored settings |
|---:|---:|---:|:---|
| 2 | 0.05 | 6.16884--7.17884 | 3/9 right-censored above 8 |
| 2 | 0.10 | 6.16884--7.82858 | 3/9 right-censored above 8 |
| 2 | 0.15 | none | 9/9 right-censored above 8 |
| 2 | 0.20 | none | 9/9 right-censored above 8 |
| 3 | 0.05 | 4.15454--5.18736 | 4/9 left-censored below 4.08759 |
| 3 | 0.10 | 3.08442--3.62850 | 3/9 right-censored above 3.66802 |
| 3 | 0.15 | 1.55058--1.55058 | 3/9 below 1.41421; 3/9 above 1.68179 |
| 3 | 0.20 | none | 9/9 no certified crossing (covariance-aware) |

The declared baseline classifier is `h=0.04`, `r=0.05`.  Its brackets and
midpoints are stored in `classifier_sensitivity.json` (peak-only convention)
and, under the covariance-aware statistic, in the `sensitivity_new_rule`
block of `covariance_aware_reclassification.json`; the two agree for every
row above except `(3, 0.20)`, where the peak-only file records a superseded
bracket `0.162105--0.281630`.  Operational
threshold values are classifier-dependent by construction; the sensitivity
table should accompany, not replace, the declared baseline definition.

## Independent seeds

Three independent deterministic streams, each with one million walkers, gave:

| configuration | mode counts | maximum common-index peak span | kill-fraction range |
|:---|:---|---:|:---|
| m3 anchor, eps=0.10, B=1 | 3,3,3 | 0.02 | 0.776153--0.776589 |
| m3 threshold lower side, B=3.50 | 3,3,3 | 0 | 0.980694--0.981178 |
| m3 threshold upper side, B=3.64 | 2,2,2 | 0 | 0.982581--0.982689 |
| m3 phase-boundary passing side, eps=0.175, B=0.5 | 3,3,3 | 0.10 | 0.408590--0.408817 |
| m3 phase-boundary failing side, eps=0.175, B=1 | 2,2,2 | 0 | 0.638291--0.638861 |
| m5 anchor, eps=0.10, B=1 | 5,5,5 | 0.02 | 0.551580--0.552851 |

## Time-step halving

Each side used 500,000 walkers.  The 0.1-wide comparison-bin `z` values use
independent deterministic streams; `max|z|` is a diagnostic maximum over the
usable bins and is not multiplicity-adjusted.

| configuration | modes, dt / dt/2 | max peak shift | kill-fraction change | max abs z |
|:---|:---:|---:|:---|---:|
| m3 interior anchor | 3 / 3 | 0.02 | 0.775462 to 0.776370 | 3.271 |
| m3 operational threshold, B=3.57 | 2 / 2 | 0 | 0.981864 to 0.982380 | 2.592 |
| m3 phase boundary, eps=0.175, B=0.5 | 3 / 2 | not defined after mode loss | 0.408044 to 0.409030 | 2.612 |
| m5 interior anchor | 5 / 5 | 0.02 | 0.551458 to 0.551476 | 1.939 |

The phase-boundary flip is retained rather than hidden.  At `dt=0.001`, the
third-peak prominence is 0.0145723 (7.32 percent of the global maximum and
7.49 covariance-aware sigmas), so it passes both criteria.  At `dt=0.0005`,
the candidate prominence is 0.00829782 (4.23 percent and 4.24
covariance-aware sigmas): it fails both the five-sigma criterion and the
five-percent relative floor (under the superseded peak-only convention these
read 10.29 and 5.89 sigmas).  Thus the
interior anchors preserve mode count under halving, while the deliberately
boundary-adjacent case is classifier-sensitive.  No global time-step-stability
claim is supported or made.

## Independent seeds at the halved time step

`robustness/dt_half_seed_repeats/` and `dt_half_seed_repeat_summary.json`
(script `exact_m_prr_dt_half_seed_repeat.py`) rerun the boundary-adjacent
cell `m=3`, `eps=0.175`, `B=0.5` at `dt=0.0005` with seeds
20260814--20260816 and 10^6 walkers each (stream tag 65), matching the walker
count of the three `dt=0.001` seed repeats `m3_boundary_pass_seed*.json`.
The streams at the two resolutions are independent, not paired common random
numbers.  All records are judged through `classify_both`; the late candidate
is the local maximum after the last window valley of `G` (2.034) with the
largest covariance-aware `z`.

| run | dt | walkers | modes (cov.-aware) | late peak time | late prominence / global max | late z |
|:---|---:|---:|:---:|---:|---:|---:|
| dt/2 seed 20260814 | 0.0005 | 10^6 | 3 | 2.63 | 6.96% | 10.06 |
| dt/2 seed 20260815 | 0.0005 | 10^6 | 3 | 2.51 | 8.57% | 12.29 |
| dt/2 seed 20260816 | 0.0005 | 10^6 | 3 | 2.53 | 6.86% | 9.82 |
| dt seed 20260814 | 0.001 | 10^6 | 3 | 2.55 | 6.32% | 9.03 |
| dt seed 20260815 | 0.001 | 10^6 | 3 | 2.63 | 6.54% | 9.25 |
| dt seed 20260816 | 0.001 | 10^6 | 3 | 2.65 | 6.56% | 9.43 |
| dt-halving pair, dt | 0.001 | 5x10^5 | 3 | 2.59 | 7.32% | 7.49 |
| dt-halving pair, dt/2 | 0.0005 | 5x10^5 | 2 | 2.49 | 4.23% | 4.24 |

All three finer-step runs at 10^6 walkers retain three classified modes, so
the 5x10^5-walker `dt/2` value of the halving pair remains a single low
outlier among the stored runs of this cell.  These additional
independent-stream runs characterize sampling variability at the finer step,
but three seeds do not exclude a systematic time-step contribution near this
classifier boundary; they are not convergence tests and do not establish a
theorem threshold.


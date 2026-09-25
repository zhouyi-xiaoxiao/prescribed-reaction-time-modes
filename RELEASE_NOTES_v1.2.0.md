# Release notes: v1.2.0

Release `v1.2.0` accompanies the final submitted version of **"Programming the
number and weights of reaction-time peaks with a fixed budget of static
reactivity"** (CNSNS).  It supersedes `v1.1.1` and keeps all of its code and
records unchanged.  No manuscript PDF is included.

## What is new

The code and records behind the results added in the final version: the exact
programming of peak times and weights (Theorem 4 and Box 1 of the article), the
timing horizon, the total-error protocol of every demonstrated design, the
certified bifurcation set of the fixed-width fold, and the programmed plug-flow
channel of Fig. 1(b).  Paths below are relative to
`artifacts/data/exact_m_fixed_budget/`.

| topic (article) | records | script(s) in `code/` |
|:---|:---|:---|
| Differentiable exact-law design estimator: pathwise derivatives on one fixed ensemble of unkilled paths, finite-difference self-test, bitwise reproduction of a stored exact-law ensemble, check against direct kill (SM-B, design estimator) | `V2_NU_A/` (`selftest.json`, `crosscheck_fk.json`, `validate_dk.json`, `dkfd_n11_*.json`) | `fb_v2_design_estimator.py` |
| Exact programming of peak times and weights by sample-average Newton; four- and five-peak designs, out-of-sample replicates, sample-size study (Sec. 4.2, Fig. design programming; SM-B) | `V2_design/` (`demo_eps{0.05,0.025}.json`, `showcase_m5_eps0.05_B8_equal.json`, `samplesize_*.json`, `failed_cells_nocap.json`, `oos_r2_failed_cells_nocap_tmax6.json`, `designs/`, run logs `logs/demo_eps*.log`) | `fb_v2_saa_newton.py`, `fb_v2_oos_replicate.py` |
| Timing horizon by continuation and its fit (Sec. 4.2, Fig. design programming (c)) | `V2_design/horizon_exact_eps*.json`, `horizon_mean_field.json`, `horizon_fit.json`, `logs/horizon_eps0.01_failed_nan_prefix.log` | `fb_v2_timing_horizon.py` |
| Total-error protocol: direct kill at `dt`, `dt/2`, `dt/4` (coupled and independent steps) for all 35 designs, the ten max--min cells of the allocation study and the shift-compensated designs; protocol-P visibility (Sec. 4.2; SM-B) | `V2_TEP/` (`nue_summary.json`, `n1_tep_summary.json`, `protocol_P_design_law.json`, per-design `*/tep_*.json` and `level*.json`, input designs `_designs/`) | `fb_v2_tep.py`, `fb_v2_tep_ladder.py`, `fb_v2_tep_n1.py`, `fb_v2_nue_summary.py`, `fb_v2_protocol_p_designs.py` |
| Certified fold curve of two stripes for widths in `[0.15, 0.5715]`, fold curves along allocation families, the codimension-two switch at three stripes, theorem-level validation (Prop. 4, Fig. bifurcation set; SM-A Sections A7.3--A7.4) | `V2_bifurcation/` (`fold_curve_m2.json`, `fold_curve_m2_boxes.json`, `codim2_switch.json`, `codim2_switch_boxes.json`, `theorem_validation*.json`, `maxmin_m3_float_scan.json`) | `fb_v2_fold_curve_certificate.py`, `fb_v2_codim2_switch.py`, `fb_v2_cert_validator.py`, `fb_v2_maxmin_m3_scan.py`, `fb_v2_fold_curve_figure.py`, `fb_v2_bifurcation_figure.py` |
| Programmed plug-flow channel with the Poiseuille contrast (Sec. 2.4, Fig. 1(b); SM-B) | `V2_hero/` (`design_eps{0.03,0.01}.json`, `designs/`, `tep_eps*.json`, `oos_*.json`, `oos_study_*.json`, `pois_contrast.json`, `selftest_*.json`, `summary.json`) | `fb_v2_hero_channel.py`; `crop_hero_panel_a.py` (Fig. 1(b) panel, needs PyMuPDF) |
| Graphical abstract of the submission (deterministic, no simulation) | `V2_hero/design_eps0.03.json`, `V2_hero/designs/` | `fb_v2_graphical_abstract.py` |

New figures in `artifacts/figures/`: `fb_v2_design_programming`,
`fb_v2_tep_validation`, `fb_v2_bifurcation_set`, `fb_v2_fold_curve_m2`,
`fb_v2_hero_channel` and the single panel `fb_v2_hero_channel_a.pdf` used in
Fig. 1(b), which `crop_hero_panel_a.py` writes from `fb_v2_hero_channel.pdf` by
removing the clipped-away panels (the kept region is unchanged).

## Seeds and determinism

The stochastic V2 drivers use NumPy Philox streams from `SeedSequence` entropy
built from the base seed `20260923`, a stream tag and the replicate, chunk and
level indices; every record stores its full entropy tuple.  Stream tags: 201--206
(estimator self-tests, Newton designs, out-of-sample ensembles, horizon,
channel), 211--215 (direct-kill total-error protocol, coupled ladder), 217 (the
ten max--min cells).  The interval certificates (`V2_bifurcation/`) draw no random
numbers (mpmath `iv`, 100-bit outward rounding, mpmath 1.4.1).

## Not archived

As in earlier releases only JSON records (and three cited run logs) are
archived: the per-level count arrays `V2_TEP/*/level*.npz` (about 60 MB) are
intermediate, their SHA-256 hashes and every derived number are in the JSON
records, and `fb_v2_tep.py` / `fb_v2_tep_ladder.py` regenerate them from the
recorded seeds.  Smoke tests and superseded pilot runs (`V2_TEP/smoke_*`,
`V2_hero/_smoke/`, `V2_hero/_superseded_T12345/`,
`V2_NU_A/dkfd_2e5_superseded/`) and the other driver logs are not archived.

## Sanitisation

Besides the path rewrites of earlier releases, the ssh host alias of the
cluster login used by the `submit`/`fetch` commands of `fb_v2_tep.py` is
replaced by the neutral alias `isambard3`; set it in your own ssh
configuration to use those commands.

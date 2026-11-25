# Programmable multimodality and saddle-node loss of reaction-time peaks — reproduction package

**Question.** Reacting molecules drift through a region and can react only
where catalyst has been placed. With a fixed total amount of catalyst, can the
placement be chosen so that the distribution of reaction times has a
prescribed number of peaks, each at a chosen time and with a chosen weight?

**Result.** Yes, within limits that the work maps out (for example, a latest
reachable peak time). The placement can be designed to set the number, times
and weights of the peaks; designs with up to five peaks are demonstrated. For a
given placement, once the catalyst budget passes a threshold a peak is lost
through a saddle-node (fold) bifurcation.

**Status.** Manuscript in preparation.

**Start here.** [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) maps every result
to its record and script.

Labels such as `N3`, `I4`, `W2` or `TH8` name numerical campaigns and checks;
they are listed under [Labels used in this archive](#labels-used-in-this-archive).

**Release.** `v1.3.0` accompanies the manuscript **“Programmable multimodality
and saddle-node loss of reaction-time peaks under a fixed catalyst budget”** by
Xiaoxiao Zhouyi (University of Bristol). An earlier version was titled
**“Programming the number and weights of reaction-time peaks with a fixed budget
of static reactivity”**.

The archive contains:

- the simulation, analysis and plotting code;
- the machine-readable numerical records behind the numbers and figures of the
  manuscript and its Supplementary Material, with the figure-generation scripts;
- the interval-arithmetic certificates and enclosures (scripts and outputs) that
  are part of three proofs of the Supplementary Material;
- the Lean 4 sources;
- the drivers, job scripts and records of the large-scale checks run on the
  Isambard 3 supercomputer.

**Manuscript.** The manuscript, its Supplementary Material and the graphical
abstract are not part of this release; the manuscript will be linked here once
it is published.  Theorem, proposition and section numbers quoted in these
documents refer to the current manuscript draft and may change.

This package is tag `v1.3.0` of
https://github.com/zhouyi-xiaoxiao/prescribed-reaction-time-modes.  The
default branch also carries documentation and comment edits made after the
release (listed in `PROVENANCE.md`); no number in any record changed.  Verify
the tree from the archive root with:

```bash
shasum -a 256 -c MANIFEST.sha256
```

If a persistent identifier is later minted for this release, cite the
identifier shown by the repository record together with the tag.

## Contents

Prefixes such as `prr`, `fb` and `exact_m` in file and folder names are historical
working names.

- `code/`: all scripts.
  - Fixed-budget campaign (2026-09):
    - `exact_m_prr_fk_exact_law.py`: exact-law Feynman–Kac estimator.
    - `exact_m_prr_fk_n0_validation.py`: estimator validation.
    - `fb_allocation_law.py`: allocation law.
    - `fb_n1_allocation_design.py`: N1 allocation design.
    - `fb_n2_meanfield_residual.py`: N2 mean-field residual.
    - `fb_n3_contact_factorial.py`: N3 contact factorial.
    - `fb_n4_preparation.py`: N4 preparation sensitivity.
    - `fb_n5_dt_ladder.py`: N5 time-step ladder.
    - `fb_n6_visibility_thresholds.py`: N6 visibility thresholds.
    - `fb_n8_robustness.py`: N8 robustness.
    - `fb_n9a_matching_census.py`: N9a matching census.
    - `fb_n9b_fold.py`: N9b finite-width fold of the exact law.
    - `fb_theory_core_checks.py`: theory checks.
    - `fb_th1_passage_sign.py`: passage-sign check.
    - `fb_local_profile_check.py`: local-profile check.
    - `fb_meanfield_levelset_check.py`: mean-field level-set check.
    - `exact_m_fb_th6_tilted_bridge_checks.py`: tilted-bridge check.
    - `fb_th6_contact_constants.py`: contact constants.
    - `fb_frozen_gate_check.py`: frozen-gate check.
    - `fb_finite_width_det.py`: finite-width stripes, deterministic fold budget.
    - `fb_preparation_predictions.py`: preparation predictions.
    - `fb_bcert_window_check.py`: nominal evaluation of the sufficient budget `B_cert`.
    - `fb_referee_checks.py`: additional robustness checks.
    - Stream-integration drivers:
      `exact_m_prr_upgrade_w6_single_particle.py` and
      `exact_m_prr_w6_width_diagnostic.py` (single-particle control, I1),
      `exact_m_prr_w2_seed_repeat.py` (seed repeats),
      `exact_m_prr_release_time_smearing.py` (release-time smearing, I3),
      `exact_m_prr_mean_field_topology.py` (mean-field topology, I4),
      `exact_m_prr_classifier_null_calibration.py` (classifier null
      calibration, I5).
    - `fb_make_paper_figures.py`: publication figures.
  - Closing campaign (2026-09):
    - `fb_th8_interval_certificate.py`: interval certificate of the finite-width fold.
    - `fb_c1_preparation_count.py`: preparation-count certificate.
    - `fb_th8b_fold_normal_form.py`: saddle-node normal form of the exact law.
    - `fb_p5_gate_switching.py`: frozen-gate profile and gate-switching correction.
    - `fb_p5_step6_tail.py`: tail bound in step 6 of the gate-switching proof.
    - `fb_n14_universality.py`: universality demos.
    - `fb_n14_general_mc.py`: general-transport Monte Carlo check.
    - `fb_n14_unimodality.py`: unimodality check.
    - `fb_n14_fix_checks.py`: deterministic checks of the universality remarks.
    - `fb_n10_largeB_validation.py`: large-budget validation.
    - `fb_n11_shift_compensation.py`: shift-compensated timing design.
    - `fb_n12_census_resolution.py`: census resolution.
    - `fb_n13_third_order.py`: third-order mean-field residual.
    - `fb_derived_numbers.py`: derived numbers.
    - `fb_graphical_abstract.py`: graphical abstract.
  - General-transport counterexamples (new in `v1.1.1`):
    `fb_gx_counterexample_checks.py`. It computes 60-digit outward-rounded
    interval enclosures (mpmath `iv`, rational inputs) of the arrival actions
    and their derivatives in the variance-driven example, the constants,
    crossing and endpoint checks of the spiral example, closed-form counts of
    the free envelopes on dense grids, and a seeded finite-noise illustration
    of the killed density (not a proof).
  - Frozen-gate exact count under a contact floor (new in `v1.1.1`):
    `fb_gb2_frozen_gate_checks.py`. It runs 40-digit checks of the bridge,
    slope and score identities, the free-interval margin at the anchors, the
    contact probability of the boundary-tangent gate by exact quadrature, the
    minimum-image and orthant checks, and a Monte Carlo comparison of two
    transverse directions (consistency checks, not a proof).
  - Exact programming, total-error protocol, bifurcation set and programmed
    channel (new in `v1.2.0`): `fb_v2_*.py` and `crop_hero_panel_a.py`; see
    `RELEASE_NOTES_v1.2.0.md`.
  - Large-scale checks on Isambard 3 (2026-09; Supplementary Material,
    section "Large-scale checks on Isambard 3"):
    - `fb_hpc_n5full.py`: time-step ladder with exact Ornstein–Uhlenbeck transitions and Brownian-bridge refinement.
    - `fb_hpc_n3full.py`: contact factorial with exact transitions.
    - `fb_hpc_headline.py`: direct-kill confirmation of the design cells.
    - `fb_hpc_census.py`: census of the three gated cells with up to 2×10⁹ exact-law paths.
    - `fb_hpc_common.py`: shared helpers.
    - `fb_hpc_n5n3_report.py`: deterministic assessment and figures.
    - `fb_hpc_snippets.py`: result-snippet generator.
    - `code/hpc_jobs/`: the Slurm job scripts, the prologue `_common.sh` they
      source and the tiny end-to-end validation `hpc_validate_tiny.py`.
  - Release `v1.0.0` (legacy records):
    - `validate_exact_m_offlattice.py`: deterministic off-lattice simulator.
    - `exact_m_offlattice_production.py`: production driver.
    - `exact_m_prr_upgrade_w1.py` through `exact_m_prr_upgrade_w5.py`: W1–W5 drivers.
    - `exact_m_prr_robustness.py`: robustness driver.
    - `reclassify_covariance_aware.py`: covariance-aware reclassification.
    - `remake_b0_figure_covariance.py`: `B_op(eps)` figure under the covariance-aware classifier.
    - `remake_jitter_figure_covariance.py`: W3 jitter figure under the covariance-aware classifier.
    - `exact_m_prr_mean_field_boundary.py`: mean-field boundary evaluation.
    - `exact_m_prr_dt_half_seed_repeat.py`: halved-time-step seed repeats.
    - `b0_dyson_numerics.py`: Dyson-bound numerics.
    - `b0_dyson_chaincheck.py`: Dyson-bound chain check.
- `artifacts/data/exact_m_fixed_budget/`: records of the fixed-budget
  campaign and its closing campaign (see `DATA_AVAILABILITY.md` and
  `RESULTS_SUMMARY.md`), including the ensemble index records
  `fk_ensembles/*.json`, the certificate outputs `TH8_certificate/` and
  `C1_preparation_count/`, the counterexample checks `GX_counterexamples/`
  and the frozen-gate checks `GB2_frozen_gate/` (both new in `v1.1.1`), the
  records `V2_NU_A/`, `V2_design/`, `V2_TEP/`, `V2_bifurcation/`, `V2_hero/`
  (new in `v1.2.0`), and the records of the Isambard 3 checks
  (`HPC_N5full/`, `HPC_N3full/`, `HPC_headline/`, `HPC_census/`, and
  `HPC_jobs/`: the job scripts and accounting record of the cluster runs).
- `artifacts/data/exact_m_offlattice_production/`,
  `artifacts/data/exact_m_prr_upgrade/`: the records of release `v1.0.0`,
  plus the single-particle control `w6_single_particle/`, the seed repeats
  `robustness/w2_seed_repeat_summary.json` and the release-time smearing
  `robustness/release_time_smearing/` used by the Supplementary Material.
- `artifacts/figures/`: PDF (and, where available, PNG) versions of every
  numerical figure included by the manuscript and the Supplementary Material,
  the six figures of release `v1.0.0`, the provenance/check record
  `fb_paper_figures_manifest.json` written by `fb_make_paper_figures.py`, and
  the TikZ source `fig1_schematic.tex` of the schematic Fig. 1.  The builder
  checks that each figure included by the manuscript or the Supplementary
  Material is byte-identical to the file shipped here (some are included under a shorter name; the map is
  `copied_as` in the figure manifest).
- `notes/gap_diagnosis_20260923/`: the theory-scout records that the manuscript
  cites (`theory_scout/analysis_m2.json`, `analysis_m3.json`,
  `analysis_h3.json`), the scout scripts that produce them and the path
  caches they read (`fk_paths.py`, `fk_bop.py`, `fk_var.py`,
  `analyze_fk.py`, `analyze_h3.py`), and the contact-escape check read by
  `fb_frozen_gate_check.py` (`theory_checks/contact_escape_check.{py,json}`).
- `lean/formal_lean_prr/`: the Lean 4 package of the Supplementary Material
  (see below).
- `environment/`: package versions of the two reference environments and the
  platform provenance; `reference_platform.json` records the source commit
  the archive was built from (`release_commit`, written by the builder).
- `.zenodo.json` and `CITATION.cff`: deposit and citation metadata (no DOI
  has been minted yet).
- `RESULTS_SUMMARY.md`: a map from the results of the manuscript to their records
  and scripts, followed by the reading guide to the `v1.0.0` robustness
  records.
- `RELEASE_NOTES_v1.2.0.md`: what release `v1.2.0` adds.
- `RELEASE_NOTES_v1.3.0.md`: what release `v1.3.0` adds.
- `PROVENANCE.md`, `DATA_AVAILABILITY.md`: seeds, platforms, data dictionary,
  what is not archived, and how workstation paths were neutralised.
- `MANIFEST.sha256`: SHA-256 checksums, generated only after the release tree
  is frozen.

All paths inside scripts and records are relative to the archive root, which
mirrors the layout of the source repository: run every command below from the
archive root.

## Labels used in this archive

The definitions follow the campaign records and the data dictionary
(`RESULTS_SUMMARY.md`, `DATA_AVAILABILITY.md`).

- `N1` allocation design; `N2` mean-field residual and sandwich tests; `N3`
  contact factorial and frozen gate; `N4` preparation study; `N5` time-step
  ladder; `N6` visibility thresholds.
- `N8` robustness; `N9a` matching census; `N9b` finite-width fold; `N10`
  large-budget validation; `N11` shift-compensated timing design; `N12`
  census resolution; `N13` third-order residual; `N14` universality.
- `I1` single-particle control; `I3` release-time smearing; `I4` mean-field
  topology; `I5` classifier null calibration and power.
- `W1` phase diagram; `W2` operational budget threshold by bisection; `W3`
  jitter robustness; `W4` five-peak demonstration; `W5` three-dimensional
  spot check.
- `TH8` interval certificate and fold checks; `P5` frozen-gate passage profile
  and gate-switching correction; `C1` small-budget preparation count; `GX`
  general-transport counterexamples; `GB2` frozen-gate exact count under a
  contact floor; `SM-B` part B of the Supplementary Material, the numerical
  record (`SM-A`: proofs of the main results).

## Environment

Two reference environments were used (both arm64 macOS, Apple M4 Pro):

- the 2026-09 fixed-budget campaign: CPython 3.9.6 with
  `environment/requirements_fixed_budget.txt` (NumPy, SciPy, Matplotlib);
- the 2026-08 campaigns of release `v1.0.0`: CPython 3.12.13 with
  `environment/requirements.txt` (NumPy, Matplotlib, mpmath).

The large-scale checks ran on the Isambard 3 supercomputer with CPython
3.11.15 and NumPy 2.0.2, as recorded in the `env` field of each `HPC_*`
record (`environment/reference_platform.json`, key `hpc_campaign`).

The interval certificates of release `v1.2.0` (`V2_bifurcation/`) record
CPython 3.14.6 and mpmath 1.4.1 in their `provenance` field; the other
`v1.2.0` drivers need only NumPy (and Matplotlib for the figures) and ran on
the same workstation.  The Arb recheck of release `v1.3.0`
(`fb_v3_arb_recheck.py`) needs python-flint (recorded: CPython 3.14.6,
python-flint 0.9.0, FLINT 3.6.0).

Create an isolated environment from the archive root, for example

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements_fixed_budget.txt mpmath==1.4.1
```

NumPy drives every simulation; SciPy is needed by `fb_n9a_matching_census.py`
(PDE census) and the universality drivers, Matplotlib for figures, and mpmath
(1.4.1 was used) for the Dyson-bound checks and for all interval-arithmetic
certificates (`mpmath.iv`) and high-precision quadratures of the closing
campaign.
All random streams are NumPy `SeedSequence`-based; each record stores its
seed (and, where applicable, stream tag or full entropy), and the ensemble
index records name the bit generator.  Floating-point results of a rerun on a
different platform or NumPy version can differ in the last digits.

## Fast reproduction from stored records

These deterministic checks need no path ensemble; each rewrites its own JSON
record under `artifacts/data/exact_m_fixed_budget/` (run them in a copy of the
archive if you want `MANIFEST.sha256` to keep verifying):

```bash
python code/fb_allocation_law.py            # allocation law (Prop. 1), allocation_law.json
python code/fb_th1_passage_sign.py          # TH_core/th1_passage_sign.json
python code/fb_finite_width_det.py          # TH8/finite_width_det.json
python code/fb_preparation_predictions.py   # TH10/preparation_predictions.json
python code/fb_bcert_window_check.py        # TH11/bcert_nominal_windows.json
python code/fb_th6_contact_constants.py     # TH6/th6_contact_constants_anchor.json
python code/fb_derived_numbers.py           # derived_numbers.json
```

The interval-arithmetic certificates and the other deterministic checks of
the closing campaign also run from stored records (mpmath required; no random
numbers):

```bash
python code/fb_th8_interval_certificate.py  # TH8_certificate/certificate.json (Prop. 4, (D1)-(D4), B_top^det)
python code/fb_c1_preparation_count.py interval   # C1_preparation_count/c1_interval_certificates.json
python code/fb_c1_preparation_count.py floatG     # C1_preparation_count/c1_floatG_census.json
python code/fb_th8b_fold_normal_form.py post      # TH8_certificate/fold_normal_form.json (post-processing only)
python code/fb_p5_gate_switching.py predict       # P5_gate_switching/p5_predict.json (from stored N3 records)
python code/fb_p5_step6_tail.py                   # P5_gate_switching/p5_step6_tail.json
python code/fb_n14_unimodality.py                 # N14_universality/n14_unimodality.json
python code/fb_n14_fix_checks.py                  # N14_universality/n14_fix_checks.json
python code/fb_n10_largeB_validation.py compare   # N10_largeB_validation/n10_summary.json (stored DK records + FK arrays)
python code/fb_graphical_abstract.py              # writes artifacts/figures/graphical_abstract.{pdf,png} (not shipped)
python code/fb_hpc_n5n3_report.py derive          # HPC_N5full/hpc_n5full_assessment.json, HPC_N3full/hpc_n3full_assessment.json
python code/fb_hpc_n5n3_report.py figures         # artifacts/figures/fb_hpc_n5full.pdf, fb_hpc_n3full.pdf
```

`fb_th8_interval_certificate.py` aborts instead of writing a certificate if
a bisection cell becomes narrower than its `MIN_WIDTH` without being resolved.

The general-transport counterexample checks (new in `v1.1.1`; mpmath and
NumPy; about 30 s on the reference workstation) are one command:

```bash
python code/fb_gx_counterexample_checks.py        # GX_counterexamples/checks.json
python code/fb_gx_counterexample_checks.py --skip-mc   # without the seeded illustration: checks_quick.json
```

Parts 1–4 and the free-envelope grid counts are deterministic.  The
finite-noise illustration of part 5 draws bridges with the recorded
`SeedSequence([20260924, 61, tag])` streams (one process) and is labelled an
illustration, not a proof.  Enclosures are written with outward rounding
(lower endpoint rounded down, upper endpoint up, to the stated number of
significant digits); the certified comparisons themselves are made on the
exact rational endpoints inside the script.  The two development sources whose
claims the script checks are not archived; their SHA-256 values are recorded
in the `sources` field of the record (a rerun from this archive records them
as absent).

The frozen-gate checks (new in `v1.1.1`; mpmath and NumPy; about 45 s) are
one command:

```bash
python code/fb_gb2_frozen_gate_checks.py         # GB2_frozen_gate/checks.json
```

Part 1 uses Python's `random.seed(20260924)`, parts 4a–4b NumPy's
`default_rng(20260924)`; the other parts draw no random numbers.

The publication figures are redrawn by

```bash
python code/fb_make_paper_figures.py --sm --copy-to <directory>
```

Most panels are drawn from the archived JSON summaries.  Panels that need
per-path accumulators (the whole-axis late-basin masses of the contact
figure, and the Supplementary redraws of the estimator validation and of the
residual decomposition) read the path-ensemble store described next.

The stored-count computations of release `v1.0.0` run as before:

```bash
python code/exact_m_prr_robustness.py --phase sensitivity
python code/reclassify_covariance_aware.py
python code/exact_m_prr_mean_field_boundary.py
python code/exact_m_prr_upgrade_w1.py --phase figure
python code/remake_b0_figure_covariance.py
python code/remake_jitter_figure_covariance.py
python code/exact_m_prr_upgrade_w4.py --replot
python code/exact_m_prr_upgrade_w5.py --replot
python code/b0_dyson_numerics.py
python code/b0_dyson_chaincheck.py
```

`reclassify_covariance_aware.py` and `exact_m_prr_mean_field_boundary.py`
rewrite their records in place; see `RESULTS_SUMMARY.md` for what they
contain.

## Full stochastic reproduction

**Fixed-budget campaign.**  Every exact-law estimate is an average over an
ensemble of unkilled Euler–Maruyama paths (the discrete Feynman–Kac
identity documented at the top of `code/exact_m_prr_fk_exact_law.py`).  The
ensembles themselves (about 16 GB of compressed chunk files) are not archived.
Each ensemble has an index record `artifacts/data/exact_m_fixed_budget/fk_ensembles/<name>.json`
holding its full model specification (`spec`), storage mode, variants, path
and chunk counts, seed, stream tag, replicate index, the complete
`SeedSequence` entropy, the time grid and, per chunk, the SHA-256 and size of
the original chunk file.  An ensemble is regenerated into the directory named
by the environment variable `PRR_FK_ENSEMBLE_ROOT` (default
`~/.local-build/prr_fk_ensembles`) with

```bash
export PRR_FK_ENSEMBLE_ROOT=/path/with/space   # about 0.1-0.2 GB per 1e5 paths
python code/exact_m_prr_fk_exact_law.py simulate --name <name> \
  --m <spec.m> --eps <spec.eps> --paths <n_paths> --tag <tag> \
  --replicate <replicate> --mode <mode> [further --options from spec] --workers 3
python code/exact_m_prr_fk_exact_law.py selftest
```

(`python code/exact_m_prr_fk_exact_law.py simulate --help` lists every
option; each corresponds to a field of `spec`).  The item drivers
`fb_n1_*.py` ... `fb_n9b_*.py`, `exact_m_prr_fk_n0_validation.py`,
`fb_referee_checks.py`, `fb_local_profile_check.py`,
`fb_meanfield_levelset_check.py` and `fb_theory_core_checks.py` then rebuild
their records from the ensembles; `python code/<script> --help` shows their
options where they take any.  The closing-campaign drivers follow the same
pattern with sub-commands (`--help` lists them): `fb_n10_largeB_validation.py
{dk,fk,compare,figure}`, `fb_n11_shift_compensation.py
{design,fk,fkcorr,dk,analyze,figure}`, `fb_n12_census_resolution.py`,
`fb_n13_third_order.py {simulate,analyze}` (re-generates the N2 paths bit for
bit from their stored seed entropy), `fb_n14_universality.py
{simulate,chunk,law,analyze,figure}`, `fb_n14_general_mc.py
{run,analyze,figure}`, `fb_p5_gate_switching.py {jfun,predict,oracle,profile,...}`
and `fb_c1_preparation_count.py fk`; their scratch caches are given with
`--cache` where the script takes it.  Several theory checks and the N0 validation also
read the small scout path caches (`~/.local-build/prr_gap_fk_cache/`, not
archived), which the scout scripts regenerate:

```bash
cd notes/gap_diagnosis_20260923/theory_scout
python fk_paths.py 2 0.1 200000 11 fk_m2_e0.1.npz      # also: (3,0.1,12) (2,0.05,13) (3,0.05,14) (2,0.035,15) (3,0.035,16)
FK_FULL_ONLY=1 FK_BUDGETS=5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,11 python fk_bop.py 2 0.05 200000 21 fkb_m2_e0.05.npz
FK_RPAR0=0.0 FK_RPERP0=0.4 FK_BUDGETS=0.5,1,2,4,8,20,50,200 python fk_var.py 2 0.05 200000 31 fkh3_m2_e0.05.npz
python analyze_fk.py fk_m2_e0.1.npz ...                 # JSON on standard output
```

Arguments are `m eps N seed output.npz`.  The remaining scout runs are
`fk_bop.py 2 0.035 200000 22` (same environment as above),
`fk_bop.py 3 0.05 200000 23` and `fk_bop.py 3 0.035 200000 24` (with
`FK_FULL_ONLY=1 FK_BUDGETS=3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,5.5,6,7`), and
`fk_var.py 2 0.025 200000 32` (same environment as the `fk_var.py` line
above); the output names follow the pattern shown.  The Monte Carlo drivers
listed above take their own `--help` options and record their seeds and stream
tags.

**Large-scale checks (Isambard 3).**  The four drivers run as
`simulate` then `analyze` sub-commands (`--help` lists the options).  They
keep per-chunk accumulators in `$PRR_HPC_WORK` (default `~/prr_hpc_work`,
outside the archive; not archived) and write only the reduced summaries
`artifacts/data/exact_m_fixed_budget/HPC_<item>/hpc_<item>.json`.  The job
scripts in `code/hpc_jobs/` record the cluster invocations; in the
archived copies the cluster locations of the repository clone
and its interpreter are replaced by the variables `PRR_ARCHIVE_ROOT` (the
archive root, required) and `PRR_PYTHON` (default `python3`):

```bash
export PRR_ARCHIVE_ROOT=$PWD PRR_PYTHON=python3
sbatch code/hpc_jobs/prr_n5full.sbatch      # time-step ladder, tag 95
sbatch code/hpc_jobs/prr_n3full.sbatch      # contact factorial, tag 96
sbatch code/hpc_jobs/prr_headline.sbatch    # design-cell direct kill, tag 97
sbatch code/hpc_jobs/prr_census_m2.sbatch   # census, 1e9 paths, tag 98
sbatch code/hpc_jobs/prr_census_m3.sbatch   # census, 2e9 paths, tag 98
sbatch --dependency=afterany:<m2 job>:<m3 job> code/hpc_jobs/prr_census_analyze.sbatch
```

Without Slurm, run the commands of the job scripts from `code/` (for example
`python3 fb_hpc_headline.py simulate --workers 8` then `python3
fb_hpc_headline.py analyze`); the random streams are indexed by chunk through
the recorded entropies and do not depend on the number of workers (the order
of floating-point accumulation can).  The drivers stop cleanly at
`--deadline-min`; the census driver resumes or extends a run from its
checkpoint.  The analyses of the
ladder and factorial compare with the workstation records `N5/n5_dt_ladder.json`
and `N3/n3_summary.json`, which are archived.

**Release `v1.0.0` campaigns.**  The production matrix and the W1–W5 and
robustness campaigns are rerun exactly as documented for `v1.0.0`:

```bash
python code/exact_m_offlattice_production.py --phase audit
python code/exact_m_offlattice_production.py --phase smoke --workers 3
python code/exact_m_offlattice_production.py --phase full --workers 3 \
  --output-subdir exact_m_offlattice_production_rerun
python code/exact_m_prr_upgrade_preflight.py
python code/exact_m_prr_upgrade_w1.py --phase all --workers 5
python code/exact_m_prr_upgrade_w2.py --workers 5
python code/exact_m_prr_upgrade_w3.py --workers 5 --replicas 50
python code/exact_m_prr_upgrade_w4.py --walkers 5000000
python code/exact_m_prr_upgrade_w5.py --walkers 5000000
python code/exact_m_prr_robustness.py --phase seeds --workers 3 --seed-walkers 1000000
python code/exact_m_prr_robustness.py --phase dt --workers 3 --dt-walkers 500000
python code/exact_m_prr_dt_half_seed_repeat.py --workers 3
python code/exact_m_prr_upgrade_campaign_summary.py
python code/w3_jitter_covariance_recheck.py --workers 5
```

The preflight, campaign-summary and Dyson-check scripts take no options: any
invocation, including `--help`, executes the full script and rewrites its
record in place.

## Lean 4 package

`lean/formal_lean_prr/` is the package `formal_prr` version 0.2.0 (toolchain
`leanprover/lean4:v4.32.0-rc1`, mathlib4 pinned by `lake-manifest.json`).
Every file of the package is listed with its SHA-256 in the package's own
`MANIFEST.sha256` (`cd lean/formal_lean_prr && shasum -a 256 -c MANIFEST.sha256`),
and the builder refuses to archive sources whose hashes differ from that
manifest or from the anchors quoted in the Supplementary Material table.
Build and audit it with the commands of its `README.md`:

```bash
cd lean/formal_lean_prr
lake exe cache get
lake build                                  # default target: library FormalPRR
lake env lean audit_paper_20260923.lean     # axiom audit
```

The reference outputs `lean_build_20260923_output.txt`,
`audit_paper_20260923_output.txt` and `BUILD_RECEIPT.txt` ship with it.  The
separate library `FormalPRRCompanion` (kernels of a different manuscript) is
included because the historical axiom records name its declarations, but it
is not a default build target, no `FormalPRR` module imports it, and it is not
counted anywhere in the manuscript (`lake build FormalPRRCompanion` builds it on
request; reference output `audit_companion_20260923_output.txt`).  The files
listed under "Historical files" in the package README are the audit trail of
the `v1.0.0` package and do not describe the current one.

## What is and is not retained

Individual walker trajectories, unbinned reaction times, the chunk files of
the path ensembles (about 16 GB) and the per-chunk accumulators of the
Isambard 3 runs are not retained.  The direct-kill records store the
declared histogram counts and bin edges; the fixed-budget records store the
exact-law estimates, their standard errors and every derived quantity the
manuscript quotes, together with the specification and seeds needed to
regenerate the ensembles.  See `DATA_AVAILABILITY.md` for the data dictionary
and these limitations in full.

## Integrity and licensing

After the release tree is frozen, verify it from the archive root with:

```bash
shasum -a 256 -c MANIFEST.sha256
```

Source code is licensed under the MIT License (`LICENSE-CODE`).  Stored data,
figures and documentation are licensed under Creative Commons Attribution 4.0
International (`LICENSE-DATA`).

## Release history

**Release `v1.3.0`.** Release `v1.3.0` supersedes `v1.2.0` and keeps all of
its code and records. It adds the code and records of Sections B16 and B17 of
the numerical record (SM-B): the approach of the exact law to its weak-noise
limits across the noise level (`code/fb_v3_bridge.py`,
`artifacts/data/exact_m_fixed_budget/V3_bridge/`, figure `fb_v3_bridge`) and
the independent recheck of the interval certificates with FLINT/Arb
(`code/fb_v3_arb_recheck.py`,
`artifacts/data/exact_m_fixed_budget/V3_arb_recheck/`). See
`RELEASE_NOTES_v1.3.0.md`.

**Release `v1.2.0`.** Release `v1.2.0` supersedes `v1.1.1` and keeps all of
its code and records. It adds the code and records of the results added in
that version of the manuscript: the exact programming of peak times and
weights (Theorem 4, Box 1), the timing horizon, the total-error protocol of
every demonstrated design, the certified bifurcation set of the fixed-width
fold and the programmed plug-flow channel of Fig. 1(b)
(`code/fb_v2_*.py`, `code/crop_hero_panel_a.py`,
`artifacts/data/exact_m_fixed_budget/V2_*/`, five new figures). See
`RELEASE_NOTES_v1.2.0.md` for the file list, seeds and what is not archived.

**Release `v1.1.1`.** Release `v1.1.1` superseded `v1.1.0`. It adds the
checks of the general-transport counterexamples of the Supplementary Material
(section "Exact count beyond the harmonic trap: counterexamples, plug flow and
a sufficient criterion"):
`code/fb_gx_counterexample_checks.py` and its record
`artifacts/data/exact_m_fixed_budget/GX_counterexamples/checks.json`, whose
60-digit interval enclosures of arrival actions are part of the proof of the
variance-driven counterexample. It also adds the checks of the frozen-gate
exact count under a uniform contact floor (Supplementary Material, section
"Random frozen gates: the exact count under a uniform contact floor"):
`code/fb_gb2_frozen_gate_checks.py` and its record
`artifacts/data/exact_m_fixed_budget/GB2_frozen_gate/checks.json`
(consistency checks of the algebra and constants, not part of a proof). All
other code and records are those of `v1.1.0`; only these documents and the
metadata files were updated.

**Release `v1.1.0`.** Release `v1.1.0` kept all code and numerical records of
`v1.0.0` as legacy records and added the fixed-budget campaign and its closing
campaign.

**Release `v1.0.0`.** Tag `v1.0.0` (2026-09-08) archives the code and
records of an earlier version of this work, under a different title.  The
full proof of the sufficient budget `B_cert` and the earlier direct-kill
campaigns are documented there; release `v1.1.0` kept all code and numerical
records of `v1.0.0` as legacy records.

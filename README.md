# Reproduction package: finite-window reaction-time modes

This archive accompanies **“Prescribing finite-window reaction-time modes
with a static fixed-budget Doi reactivity field”** by Xiaoxiao Zhouyi.  It
contains the exact scripts, stored Monte Carlo sufficient statistics,
robustness checks, and publication figures used for the numerical claims.

This package is released as tag `v1.0.0` of
https://github.com/zhouyi-xiaoxiao/prescribed-reaction-time-modes; verify it
from the archive root with `shasum -a 256 -c MANIFEST.sha256`.  If a
persistent identifier is later minted for this release, cite the identifier
shown by the repository record together with the tag.

## Contents

- `code/`: deterministic off-lattice simulator, W1–W5 drivers, campaign
  summary, robustness driver, and the covariance-aware reclassification and
  figure scripts (`reclassify_covariance_aware.py`,
  `w3_jitter_covariance_recheck.py`, `remake_b0_figure_covariance.py`,
  `remake_jitter_figure_covariance.py`), the mean-field evaluation of the
  retention boundary (`exact_m_prr_mean_field_boundary.py`), and the
  independent-seed repeats at the halved time step
  (`exact_m_prr_dt_half_seed_repeat.py`).
- `artifacts/data/exact_m_offlattice_production/`: high-statistics anchor
  simulations.
- `artifacts/data/exact_m_prr_upgrade/`: W1 phase-diagram cells, W2 empirical
  operational-threshold probes, W3 jitter tests, W4 five-mode example, W5
  three-dimensional spot check, the added classifier/seed/time-step
  robustness results, and the covariance-aware re-judgement of every stored
  classifier record (`covariance_aware_reclassification.json`, its
  `_summary.txt`, and `w3_jitter/covariance_aware_recheck.json`), the
  mean-field hazard--survival evaluation quoted in the main text
  (`mean_field_boundary.json`), and the three independent-seed runs at the
  halved time step of the boundary-adjacent cell
  (`robustness/dt_half_seed_repeats/` and
  `robustness/dt_half_seed_repeat_summary.json`).
- `artifacts/figures/`: PDF and PNG versions of the six numerical figures.
- `manuscript/`: final compiled article and Supplemental Material, and their
  LaTeX sources under `manuscript/source/` (the six submitted figure PDFs
  are duplicated under `manuscript/source/figures/`, so the sources compile
  in place).
- `lean/formal_lean_prr/`: the Lean 4 package of the Supplemental Material
  (toolchain `leanprover/lean4:v4.32.0-rc1`, mathlib4 pinned by
  `lake-manifest.json`); build with `lake exe cache get && lake build` from
  that directory (see its `README.md`).  `consolidated_axioms.txt` is the
  authoritative axiom record: the 138 `#print axioms` lines of the audited
  theorems, each on `[propext, Classical.choice, Quot.sound]` and none on
  `sorryAx` (`axioms_report_alpha.txt` and `axioms_report_beta.txt` are
  earlier per-file extracts kept for history; no separate build log ships).
  The two adversarial audit reports (`codex_lean_audit.txt`,
  `codex_lean_recheck.txt`) and the per-finding resolution record
  (`codex_lean_recheck_resolution.txt`) ship alongside the sources; the
  SHA-256 anchors quoted in the Supplemental Material are
  `shasum -a 256 FormalPRR/<Module>.lean | cut -c1-16`, and the builder
  refuses to archive sources whose anchors differ from those quoted.
  `FORMALIZATION_TARGETS.md` is the original task brief and refers to the
  author's working paths and to a `tex_anchors/` mirror that is not shipped.
- `environment/`: the Python package versions and platform provenance of the
  reference run, and the source commit the archive was built from
  (`reference_platform.json`, key `release_commit`, written by the builder).
- `.zenodo.json` and `CITATION.cff`: deposit and citation metadata with no
  fabricated DOI.
- `RESULTS_SUMMARY.md`: a compact, explicitly qualified guide to the
  covariance-aware reclassification and to the added classifier, seed, and
  time-step robustness records.
- `MANIFEST.sha256`: SHA-256 checksums, generated only after the release tree
  is frozen.

## Environment

The reference computations used CPython 3.12.13 on arm64 macOS.  Create an
isolated environment from the archive root:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.txt
```

Only NumPy is needed for simulation.  Matplotlib is needed to regenerate the
figures, and mpmath is needed for the independent Dyson-bound calculations.

## Fast reproduction from stored counts

These commands do not rerun Monte Carlo walkers:

```bash
python code/exact_m_prr_robustness.py --phase sensitivity
python code/reclassify_covariance_aware.py
python code/exact_m_prr_mean_field_boundary.py
python code/exact_m_prr_upgrade_w1.py --phase figure
python code/remake_b0_figure_covariance.py
python code/remake_jitter_figure_covariance.py
python code/exact_m_prr_upgrade_w4.py --replot
python code/exact_m_prr_upgrade_w5.py --replot
```

The first command reclassifies the stored 0.02-wide histogram counts over the
declared bandwidth and relative-prominence grids under the peak-only
convention (legacy `robustness/classifier_sensitivity.json`).  The second
re-judges all 201 stored classifier records with the covariance-aware
statistic that the article uses as its formal definition and rewrites
`covariance_aware_reclassification.json` and its `_summary.txt` in place.
The third evaluates the parameter-free mean-field law
`f_1 = B G exp(-B Lambda)` of the main text on the fixed covariance-aware
classifier: it first regression-tests its real-valued-count classifier
against `classify_both` on all 173 stored integer histograms of the
production grid, W1 map, and W2 probes (501 local maxima; this adapter test
is distinct from the covariance-aware reclassification of all 201 stored
classifier records), then scans and bisects the eight W2 chains, compares
the 96 W1 cells and the 17 production configurations with their
covariance-aware verdicts, and rewrites
`mean_field_boundary.json` (about 40 s; all numerical conventions are stored
under its `conventions` key).
The remaining commands recreate the six published figures in
`artifacts/figures/`: `remake_b0_figure_covariance.py` draws the
operational-threshold figure from the stored W2 chains and the
covariance-aware verdicts (the legacy `exact_m_prr_upgrade_w2.py
--figure-only` reproduces the superseded peak-only version), and
`remake_jitter_figure_covariance.py` redraws the W3 figure from the stored
`w3_jitter/covariance_aware_recheck.json`.  `reclassify_covariance_aware.py`
and the two `remake_*` scripts take no options and run on any invocation,
including `--help`.

The independent high-precision analytic checks can be rerun with:

```bash
python code/b0_dyson_numerics.py
python code/b0_dyson_chaincheck.py
```

## Full stochastic reproduction

First audit that the archive contains the exact 18-row production matrix
underlying Supplemental Table S-I:

```bash
python code/exact_m_offlattice_production.py --phase audit
```

A quick executable test traverses the same complete matrix at reduced walker
count in a separate, explicitly non-production directory:

```bash
python code/exact_m_offlattice_production.py --phase smoke --workers 3
```

The 18 production rows (17 at `dt=0.001`, plus the reported `dt=0.0005`
twin) can be rerun without overwriting the archived records with:

```bash
python code/exact_m_offlattice_production.py --phase full --workers 3 \
  --output-subdir exact_m_offlattice_production_rerun
```

This uses five million walkers per row, chunk size 250,000, seed 20260808,
`tmax=4`, and classifier bandwidth 0.04.  The subsequent W1--W5 upgrade and
robustness campaigns can be rerun with:

```bash
python code/exact_m_prr_upgrade_preflight.py
python code/exact_m_prr_upgrade_w1.py --phase all --workers 5
python code/exact_m_prr_upgrade_w2.py --workers 5
python code/exact_m_prr_upgrade_w3.py --workers 5 --replicas 50
python code/exact_m_prr_upgrade_w4.py --walkers 5000000
python code/exact_m_prr_upgrade_w5.py --walkers 5000000
python code/exact_m_prr_robustness.py --phase seeds --workers 3 \
  --seed-walkers 1000000
python code/exact_m_prr_robustness.py --phase dt --workers 3 \
  --dt-walkers 500000
python code/exact_m_prr_dt_half_seed_repeat.py --workers 3
python code/exact_m_prr_upgrade_campaign_summary.py
```

The preflight, campaign-summary and Dyson-check scripts take no options: any
invocation, including `--help`, executes the full script.
`exact_m_prr_upgrade_preflight.py` and
`exact_m_prr_upgrade_campaign_summary.py` rewrite
`artifacts/data/exact_m_prr_upgrade/preflight_theory.json` and
`campaign_summary.json` in place (deterministically, byte-identical to the
archived files); run them in a copy of the archive if you wish to keep
`MANIFEST.sha256` verification intact.  `b0_dyson_numerics.py` and
`b0_dyson_chaincheck.py` only print to standard output.

The covariance-aware W3 recheck is a deterministic Monte Carlo rerun, not a
stored-count computation:

```bash
python code/w3_jitter_covariance_recheck.py --workers 5
```

This re-simulates all 700 W3 replicas (200,000 walkers each)
deterministically, checks that every stored peak-only verdict is reproduced
bit-for-bit, and rewrites `w3_jitter/covariance_aware_recheck.json`.

The high-statistics campaign is computationally substantial.  Each JSON
records its own walker count, seed, step size, model parameters, measured
runtime, histogram counts, classifier output, and validation gates.  Upgrade
and robustness records also serialize their stream tags.  The 18 older
production records serialize the seed but not the tag; the deterministic
production driver fixes their tag to `1`, audits their complete parameter
matrix, and writes the tag explicitly in new reruns.  The robustness driver
is resumable: it reuses completed per-run JSON files unless `--force` is
given; so is `exact_m_prr_dt_half_seed_repeat.py`, which reruns the
boundary-adjacent cell `(m, eps, B) = (3, 0.175, 0.5)` at `dt = 0.0005` with
seeds 20260814--20260816 and 10^6 walkers each (stream tag 65) and
classifies the stored counts through `classify_both`.

## Legacy schema names

The directory/file names `w2_b0_empirical`, `B0_empirical.json`, and
`exact_m_b0_empirical_prr`, together with serialized W2 keys such as `b0`,
`b0_bracket`, and `b0_lower_bound`, are frozen legacy schema identifiers.
They mean the protocol-defined operational transition
`B_op(eps)` under the stated finite-walker histogram classifier only.  They
must not be read as the theorem threshold `B_top` or the sufficient bound
`B_cert`.  Filenames such as `..._B0.125...` instead encode the ordinary
budget value `B=0.125` and do not use a `B0` symbol.

## What is and is not retained

Individual walker trajectories and unbinned reaction times are not retained.
They are generated in memory and reduced to declared histogram counts to keep
the archive compact.  The stored counts and bin edges are sufficient to
reproduce the reported classifier, sensitivity analysis, peak locations, and
figures, but not analyses that require trajectories, unbinned event times, or
a different base histogram bin width.  See `DATA_AVAILABILITY.md` for the data
dictionary and this limitation in full.

## Integrity and licensing

After the release tree is frozen, verify it from the archive root with:

```bash
shasum -a 256 -c MANIFEST.sha256
```

Source code is licensed under the MIT License (`LICENSE-CODE`).  Stored data,
figures, documentation, and compiled manuscript files are licensed under
Creative Commons Attribution 4.0 International (`LICENSE-DATA`).

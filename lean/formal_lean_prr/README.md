# FormalPRR: Lean 4 kernels for the fixed-budget reaction-time-mode paper

This package holds the machine-checked kernels of the paper "Prescribing
finite-window reaction-time modes with a static fixed-budget Doi reactivity
field" (X. Zhouyi; CNSNS package `manuscript/cnsns_submission/`). It was rebuilt
on 2026-09-23 from the 2026-08 PRR package, following gap-closure items L1–L3 of
`notes/gap_diagnosis_20260923/GAP_CLOSURE_PLAN.md`.

Toolchain: `leanprover/lean4:v4.32.0-rc1`. Install it with elan; `lean-toolchain`
pins the version. `lake-manifest.json` pins the dependencies (mathlib4
`v4.32.0-rc1`, rev `360da6fa66c1273b76b6b2d8c5666fd5ac2e3b56`).

## Reproduce

```
lake exe cache get                         # download the mathlib oleans
lake build                                 # the paper library FormalPRR (default target)
lake env lean audit_paper_20260923.lean    # axiom audit: 181 named lines + full sweep
lake build FormalPRRCompanion              # optional: companion library, see below
```

What you should see (reference outputs are included):
- `lake build` ends with `Build completed successfully` and exits 0, with no
  warnings. It prints `#print axioms` lines from the three `AxiomsReport*`
  modules. The reference log is `lean_build_20260923_output.txt` (3612 jobs).
- `audit_paper_20260923.lean` exits 0. Reference output:
  `audit_paper_20260923_output.txt`.
  - It prints 181 lines of the form
    `'<declaration>' depends on axioms: [propext, Classical.choice, Quot.sound]`.
    101 cover the declarations carried over from the PRR package. These lines
    are identical to the matching lines of the historical record
    `audit_all_138_20260903_output.txt`. The other 80 cover every
    theorem/lemma in the five modules added on 2026-09-23.
  - It then prints one `SWEEP` line per module and a final `SWEEP PASS`. The
    sweep is a meta-level check of every constant in a `FormalPRR.*` module:
    376 theorems (233 of them non-auxiliary) and 36 other constants. Every
    axiom footprint is a subset of the three standard axioms, and `sorryAx`
    occurs 0 times.
- There is no `sorry`, `admit` or `axiom` in any source file.

## Layout

The paper library `FormalPRR` is the default target. It has 18 modules plus
the root file `FormalPRR.lean`; the 18 include three audit printers and a smoke
test. The library has 215 `theorem`/`lemma` declarations.

Modules carried over from the PRR package, with their source files unchanged
(the SHA-256 anchors are identical to the 2026-09-03 receipt):

| module | content |
|---|---|
| ExpPolyZeros | ≤ 2m−1 zero bound for affine-exponential sums, distinct and with multiplicity |
| ZeroBound | independent distinct-zeros encoding |
| MixtureIdentities, GaussianMixture | log-derivative identities of the Gaussian mixture (two encodings); crossover-point iff |
| CrossoverBounds | adjacent-odds and crossover ratios; nonadjacent smallness **on the crossover windows**, explicit constants |
| BudgetThreshold, BZeroThreshold | budget-threshold inversion E_c(B) < M̂ ⇔ B < B_cert (two encodings) |
| B0ChainKernel | scalar steps of the B_cert lemma chain; margin equivalence |
| WindowSignature | exhaustiveness half of the pure-mixture window signature |

Modules added on 2026-09-23 (items L2 and L3):

| module | plan item | content |
|---|---|---|
| MeanFieldLevelSet | L2(a), TH-5 | f₁ = B G e^{−BΛ}, Λ' = G ⇒ f₁' = B e^{−BΛ}(G' − BG²); sign f₁' = sign(G' − BG²); level set {G'/G² > B}; the rising flank is downward-closed in B; antitone above the threshold, strictly increasing below it, local rise at any level-set point |
| StickBreaking | L2(b), TH-2/TH-4 | M_j = e^{−Σ_{i<j}λ_i}(1 − e^{−λ_j}); Σ M_j = 1 − e^{−Σλ} (< 1); inverse design λ_j = ln(S_j/S_{j+1}) in both directions; budget identity Σ c_j λ_j = B for λ_j = B w_j/c_j and its converse; componentwise monotonicity of the design cost; max–min ⇒ equal masses, with a unique and explicit optimal allocation w_j = c_j ln[(1−jp*)/(1−(j+1)p*)]/B; existence (IVT), uniqueness and strict monotonicity of p*(B) |
| SurvivalSandwich | L2(c), TH-1 | scalar core e^{−y} − 1 + y ≤ (y²/2)e^{K} for y ≥ −K; tangent-line and quadratic majorants; for any nonnegative integrable X on a probability space, e^{−B E X} ≤ E e^{−BX} ≤ e^{−B E X} + (B²/2) Var X (also stated with mathlib's `variance` for X ∈ L²); explicit event-mass floor |
| SignatureStability | L3 | perturbation-stable signature lemma: with disjoint tubes, \|G'\| ≥ μ₁ off the tubes, a sign change of G' across each tube, \|G''\| ≥ μ₂ on the tubes, \|F'−G'\| < μ₁ and \|F''−G''\| < μ₂, F' has exactly one zero per tube (interior, F'' ≠ 0 with the sign of G''), none elsewhere (endpoints included), and the zero set has exactly n points. No continuity of F'' is assumed (Darboux). Corollaries: chained with `margins_iff` (jet bounds E/r₀, 2E/r₀²), and with `Ec_lt_iff` (B < B_cert) |
| PassageProfile | L2(d), TH-3 / A8 | h_β = u + βφ has exactly one zero u₀ < 0, so p_β' = −p_β h_β has one sign change; identity (A8) at a critical point; F''_β(y) < 0 at every critical point; a continuous function whose zeros all have a negative derivative has at most one zero |

Audit printers: `AxiomsReportAlpha`, `AxiomsReportBeta` (the companion sections
were removed on 2026-09-23) and `AxiomsReportFixedBudget` (the 80 new
declarations).

**Companion library `FormalPRRCompanion`** (folder `FormalPRRCompanion/`, not a
default target). It holds SeedConditioning, NewtonKernel, NewtonContraction and
SigmaBound: 39 theorem/lemma declarations, 37 of them audited. These encode the
fold-transfer arithmetic of a separate manuscript and no display of this paper.
Before the move we checked the import graph: no `FormalPRR` module imports a
companion module. The four files were moved byte-for-byte, and their namespaces
still read `FormalPRR.*`, so the historical axiom record names them verbatim.
They are not counted anywhere in this paper. Reference output:
`audit_companion_20260923_output.txt` (37 lines, all standard).

## What is and is not machine-checked (plain-language statement for the SM)

The package checks the finite-dimensional and one-variable-calculus core of the
arguments. It does not check the stochastic analysis.

*New results.* The step decomposition below is the Lean agent's own; re-map it
if the final theorem numbering differs.
- **Allocation law (TH-4): checked in full, given the limiting mass formula.**
  This covers the inverse design, the budget identity, the max–min theorem (the
  optimum has equal masses, and the optimal weights are unique and explicit),
  and the existence, uniqueness and monotonicity of p*(B).
- **Survival sandwich (TH-1).** Checked in full: the lower bound, the upper
  bound and the event-mass floor, for an arbitrary nonnegative square-integrable
  exposure X on a probability space.
  Not checked: identifying X with the Doi exposure, the two-time formula for
  Var X_t, and the covariance expansion of f − f₁.
- **Mean-field level set (TH-5).** Checked in full: the sign law and its
  level-set/flank consequences, for given G, Λ with Λ' = G.
  Not checked: the asymptotic expansion of ln B_top^mf.
- **Fixed-budget passage theorem (TH-2/TH-3).** Six steps: none in full, 3 in
  part, 3 by hand.
  - Partial: identifying the limit masses (the sandwich and the stick-breaking
    algebra are checked; the ε → 0 bookkeeping is by hand); uniqueness and
    nondegeneracy of the frozen-profile peak (differentiation under the integral
    sign and existence of the maximum are hypotheses); and the transfer from C²
    closeness to one nondegenerate maximum per passage (the signature lemma, with
    C² closeness as a hypothesis).
  - By hand only: the mean-exposure limit, the O(ε) variance bound, and the
    C²_loc convergence of the rescaled density.
- **Across these four results: 18 steps.** The sandwich has 5 steps (3 full, 2
  by hand), the allocation law 4 (all full), the mean-field level set 3 (2 full,
  1 by hand) and the passage theorem 6 (3 partial, 3 by hand). In total 9 are
  checked in full, 3 in part and 6 by hand only. The sandwich is counted once,
  under TH-1.

*Theorem 1 (exact topology at small B, twelve proof steps).*
- **Two are formalized as stated.** These are the mixture log-derivative
  identities and the 2m−1 zero bound. The bound is proved with multiplicity for
  general affine-exponential sums; for H′ only distinct zeros are counted, which
  is all the proof uses.
- **Three are formalized in part:**
  - the exact adjacent-odds and crossover identities, together with the
    exhaustiveness half of the pure-mixture mode count;
  - nonadjacent smallness, with explicit constants, on the crossover windows
    only;
  - the per-tube sign-preservation step of the compactness transfer, via the
    stable signature lemma, with the jet bounds and tubes as hypotheses.
- **Seven are proved by hand only:**
  - the analytic killed semigroup;
  - the Dyson/Cauchy jet bound;
  - the three Gaussian process-calculus steps (stationary variance, contact
    tail, free-exposure factorization);
  - the posterior-sector certificate;
  - the slow-factor theorem.

  Parts of the partially formalized steps are also by hand only: the existence
  and location of the modes, and the implicit-function/compactness construction
  of the root tubes.

The quoted value of B_cert is a floating-point evaluation, not an interval
enclosure.

Suggested SM sentence:
> The Lean package (library FormalPRR, 215 declarations, 0 sorry, only Lean's
> three standard axioms, verified for every one of its 376 theorems including
> auxiliary ones) checks the finite-dimensional and calculus core of our
> arguments. It does not check their stochastic parts. It verifies the
> allocation law completely, given the limiting mass formula. It also verifies
> the survival sandwich and event-mass floor for an arbitrary nonnegative
> exposure, and the mean-field sign law.
>
> For the fixed-budget passage theorem it verifies the peak-uniqueness argument
> and the stable-signature transfer, taking the analytic inputs as hypotheses.
> Of the 18 proof steps behind the new results, 9 are checked in full, 3 in
> part and 6 by hand only. Of the 12 steps behind Theorem 1, 2 are checked in
> full, 3 in part and 7 by hand only. The vanishing-noise limits, the variance
> and C² convergence estimates and all Feynman–Kac and semigroup statements are
> proved by hand.

## Historical files (PRR v1.0.0, 2026-08/09)

These files are kept unchanged as the audit trail of the PRR-era package. That
package counted 138 audited declarations (174 theorem/lemma declarations),
including the 37 (39) companion ones.
- Axiom records: `audit_all_138_20260903.lean`,
  `audit_all_138_20260903_output.txt`, `consolidated_axioms.txt`,
  `consolidated_axioms_rebuild_20260903.txt`, `axioms_report_alpha.txt` and
  `axioms_report_beta.txt`.
- Audit trail: `codex_lean_audit.txt`, `codex_lean_recheck.txt`,
  `codex_lean_recheck_resolution.txt` and `SOURCE_HASHES_pre_recheck.txt`.

These do not describe the current package. The current records are
`BUILD_RECEIPT.txt`, `MANIFEST.sha256`, `lean_build_20260923_output.txt`,
`audit_paper_20260923_output.txt` and `audit_companion_20260923_output.txt`.

The docstrings of the carried-over modules cite labels of the PRR-era proof
sources (`exact_m_theorem_spine.tex`, `exact_m_theorem_full_proof.tex`,
`prr_assets/b0_quantitative_bound.tex`). The new modules cite the items of
`GAP_CLOSURE_PLAN.md` (TH-1…TH-5, L2, L3).

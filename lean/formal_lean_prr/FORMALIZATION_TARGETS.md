# Formalization targets for "Prescribed finite-window reaction-time modality" (PRR v2)

Source of truth (present submission): /Users/ae23069/Library/CloudStorage/OneDrive-UniversityofBristol/Desktop/valley-k-small/research/reports/encounter_multimodal_prr/manuscript/prr_submission/ — exact_m_theorem_spine.tex (spine), exact_m_theorem_full_proof.tex (full proof), prr_assets/b0_quantitative_bound.tex (B0 propositions); mirrored for anchoring under tex_anchors/. The JCP-companion targets (A4/A5 below) anchor instead to the mirrored companion supplement tex_anchors/COMPANION_jcp_supplement.tex (fold-transfer theorem proof, "Step 0 (seed conditioning)" ≈ lines 1934–1943 and "Step 2 (simplified Newton)" ≈ lines 1952–1983 of the mirror). Every Lean theorem must carry a docstring naming the source equation/lemma it encodes (or an explicit "internal helper, no direct paper display" note). NO sorry, NO added axioms (verify with #print axioms), NO strawman weakening without an explicit `-- SCOPE NOTE:` comment stating the gap vs the paper.

## Tier A — present-submission targets (must deliver)

A1. EXPONENTIAL-POLYNOMIAL ZERO BOUND (spine Eq. exact-m-exp-polynomial; full proof Lemma on the 2m-1 count).
Statement: for real λ₁ < ... < λ_m and coefficients a_j b_j (not all zero), the function
  f(x) = Σ_{j=1..m} (a_j + b_j x) exp(λ_j x)
has at most 2m−1 real zeros. Deliver at minimum the DISTINCT-zeros version via generalized Rolle induction (divide by exp(λ₁x), differentiate twice to kill the affine term, induct on m; mathlib: exists_deriv_eq_zero / Rolle, analyticity or explicit differentiability). STRETCH: zeros counted with multiplicity (multiplicity via iterated-derivative vanishing). If only the distinct version lands, add a SCOPE NOTE that the paper uses the with-multiplicity count.
Status: DELIVERED — ExpPolyZeros (distinct + with-multiplicity) and ZeroBound (independent distinct-zeros encoding; ncard forms carry Set.Finite explicitly).

A2. MIXTURE LOG-DERIVATIVE IDENTITIES (spine Eq. exact-m-log-slope).
For H(x) = Σ_j w_j exp(−(x−c_j)²/(2σ²)) with w_j > 0, σ > 0, define π_j(x) = q_j(x)/H(x), c̄(x) = Σ π_j c_j, Var(x) = Σ π_j (c_j − c̄)². Prove:
  (log H)'(x) = (c̄(x) − x)/σ²  and  (log H)''(x) = Var(x)/σ⁴ − 1/σ².
Pure finite-sum calculus; deliver fully.
Status: DELIVERED — MixtureIdentities and GaussianMixture (independent encodings; the latter adds the crossover-point iff, T4).

A3. CROSSOVER RATIO BOUNDS (spine Eq. exact-m-crossover and surrounding claims).
With s_j = (c_j+c_{j+1})/2 + σ²/(c_{j+1}−c_j) · ln(w_j/w_{j+1}) prove: q_{j+1}(s_j)/q_j(s_j) = 1, and at x = s_j ∓ σ² ln 9/(c_{j+1}−c_j) the ratio is 1/9 resp. 9. Plus nonadjacent smallness: under spacing c_{k+1}−c_k ≥ Δ > 0 for all k, for x in the crossover window [s_j − σ²ln9/Δ, s_j + σ²ln9/Δ] and any i ∉ {j, j+1}: q_i(x)/max(q_j(x), q_{j+1}(x)) ≤ C·exp(−Δ²/(cσ²)) with explicit C, c (derive the exact constants; they need not match the prose's implicit ones but must be stated and proved).
Status: DELIVERED — CrossoverBounds.

A5 (present submission). B0 LEMMA-CHAIN ARITHMETIC KERNELS (prr_assets/b0_quantitative_bound.tex: lemmas lem:exactmfull-b0-domination, lem:exactmfull-b0-legs, lem:exactmfull-b0-absorption; proposition prop:exactmfull-b0 Step 2; the closed-form radius remark; Eq. eq:exactmfull-b0-vinf).
Selected scalar, Gaussian-integral, and normalization kernels of the sharpened B_cert route: the Young cross-term step, the legs(a) Z-block real-part bound, the legs(c) sine bound, the legs(d) Z-block penalty budget and budget sum, the legs(e) secant/tangent bounds with the heterogeneous block-product and the two Young factors reaching the κ̂ display shape, the absorption(a)/(c) complete-square and initial-law Gaussian identities, the absorption(d) normalization chain, the Step-2 margin equivalence, the closed-form radius identity plus its uniform O(1) bound under (R1), and the m = 2 slab-mixture supremum bound.
Status: DELIVERED (as scoped) — B0ChainKernel; the file-level SCOPE NOTE enumerates exactly which chain steps are NOT formalized. This module is the present-submission A5 deliverable (the original contraction-constant A5 below was re-scoped to the JCP companion).

A6. B0 ASSEMBLY INVERSION (b0_quantitative_bound.tex, weighted-route final display and sharpened-route final display).
For constants v∞, T, v*, κ, Q, M > 0: E(B) := v*κQ(exp((3/2)v∞TB) − 1) is strictly increasing with E(0)=0, and B₀ := (2/(3v∞T))·ln(1 + M/(v*κQ)) satisfies E(B₀) = M and E(B) < M for 0 ≤ B < B₀. Same skeleton for the sharpened form B₀ = ln(1 + m̂/(k̂v∞e^Π))/(k̂(1+δ)v∞(T+r₀)) with its own monotone chain (read the file for the exact final inequality it needs). Pure real analysis; deliver fully, including continuity of the assembly maps (Step 1 of prop:exactmfull-b0).
Status: DELIVERED — BudgetThreshold (both routes, two-sided iff, continuity via continuous_E/continuous_Ewt/continuous_Ec) and BZeroThreshold (independent IVT-route encoding with the closed-form identification explicit_is_threshold).

## JCP-companion targets (valid mathematics, NOT displays of the present submission)

These two targets were originally drafted as present-paper A4/A5 but anchor to the fold-transfer theory of the RELATED JCP manuscript's supplement (mirror: tex_anchors/COMPANION_jcp_supplement.tex, fold-transfer proof ≈ lines 1934–1983). The delivering modules carry COMPANION scope notes and must not be cited as encoding displays of the present PRR submission.

A4 (companion). SEED-CONDITIONING 2×2 BOUND (COMPANION_jcp_supplement.tex, fold-transfer proof, "Step 0 (seed conditioning)" ≈ lines 1934–1943).
For a real 2×2 matrix J = [[p, q], [r, s]] with p = 0 (F_tt(q_*)=0), |r| ≥ μ₃, |q| ≥ μ_θ, |r| ≤ Λ, |s| ≤ Λ: prove σ_min(J) ≥ μ₃μ_θ/√(μ_θ² + 2Λ²), via σ_min = |det J|/σ_max, σ_max ≤ ‖J‖_F, and monotonicity of x ↦ x/√(x²+2Λ²). mathlib: Matrix, singular values may need hand-rolled defs for 2×2 (define σ_min/σ_max concretely via det and Frobenius norm; that is faithful for 2×2).
Status: DELIVERED (companion) — SeedConditioning (hand-rolled 2×2 singular values + faithfulness certificate) and SigmaBound (independent |det|/‖J‖_F-proxy encoding).

A5 (companion). CONTRACTION-CONSTANT ARITHMETIC (COMPANION_jcp_supplement.tex, fold-transfer proof, "Step 2 (simplified Newton)" ≈ lines 1952–1983).
Abstract lemma over ℝ: given σ > 0, ε̄ ≥ 0, ω ≥ 0 with 2ε̄ ≤ σ/16 and 2ω ≤ σ/8:
  (i) prove ‖DH_h(q_*)⁻¹‖ ≤ (8/7)(1/σ) from the Neumann bound (formalize as: if η ≤ σ/16 then 1/(σ−η) ≤ (16/15)/σ ≤ (8/7)/σ);
  (ii) (8/7)(1/σ)(2ω + 4ε̄) ≤ (8/7)(1/σ)(σ/4) = 2/7;
  (iii) fixed-point bound: α := (8√2/7)Kε and contraction factor 2/7 give α/(1−2/7) = (7/5)α = (8√2/5)Kε ≤ 2√2·Kε.
Then STRETCH: instantiate mathlib's Banach fixed point (ContractingWith) on a closed ball to conclude existence+uniqueness of the zero for an abstract map satisfying the bounds (state hypotheses abstractly: Φ maps closed ball to itself, Lipschitz 2/7).
Status: DELIVERED (companion) — NewtonKernel (incl. the Banach STRETCH) and NewtonContraction (independent scalar-chain encoding).

## Tier B (stretch, only after Tier A compiles sorry-free)

B1. EXACT-m WINDOW SIGNATURE OF THE PURE MIXTURE: for m centers with spacing ≥ Δ and σ ≤ σ₀(Δ, m) explicit, H has exactly m local maxima and m−1 local minima on an interval containing all centers with margin. Build from A1+A2+A3.
Status: PARTIAL — WindowSignature delivers the exhaustiveness (upper-bound) half only; see its SCOPE NOTE.

B2. Multiplicity version of A1.
Status: DELIVERED — ExpPolyZeros.expPoly_zeros_with_multiplicity_le.

## Project layout

~/.local-build/formal_lean_prr/ — package formal_prr, lib FormalPRR.
Root import file: FormalPRR.lean (16 modules, matching this list):
* FormalPRR/Smoke.lean — mathlib link smoke test.
* FormalPRR/ExpPolyZeros.lean — A1 + B2.
* FormalPRR/ZeroBound.lean — A1, independent distinct-zeros encoding.
* FormalPRR/MixtureIdentities.lean — A2.
* FormalPRR/GaussianMixture.lean — A2 independent encoding + crossover-point iff.
* FormalPRR/CrossoverBounds.lean — A3.
* FormalPRR/BudgetThreshold.lean — A6, both routes + continuity.
* FormalPRR/BZeroThreshold.lean — A6 independent encoding (IVT route).
* FormalPRR/B0ChainKernel.lean — A5 (present submission): b0 lemma-chain arithmetic kernels.
* FormalPRR/WindowSignature.lean — B1 partial (exhaustiveness half).
* FormalPRR/SeedConditioning.lean — A4 (JCP companion kernel).
* FormalPRR/SigmaBound.lean — A4 independent encoding (JCP companion kernel).
* FormalPRR/NewtonKernel.lean — A5 (JCP companion kernel, incl. Banach STRETCH).
* FormalPRR/NewtonContraction.lean — A5 independent encoding (JCP companion kernel).
* FormalPRR/AxiomsReportAlpha.lean — axiom audit, A1/A2/B1/B2 half.
* FormalPRR/AxiomsReportBeta.lean — axiom audit, A3/A4/A5/A6 half.
(ZeroBound, GaussianMixture, BZeroThreshold, NewtonContraction and SigmaBound additionally carry in-file `#print axioms` blocks.)

Rules: every file header comments the tex source; `lake build` must pass from clean; run `#print axioms <each main theorem>` into AxiomsReport.lean + a text log. Toolchain pinned v4.32.0-rc1 with the local mathlib cache (never `lake update`).

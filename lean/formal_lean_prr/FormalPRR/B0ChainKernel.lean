/-
FormalPRR/B0ChainKernel.lean  (target A5, replacement: the real-arithmetic
lemma chain of the sharpened B_cert route of THE PRESENT PAPER)

Tex source (docstring anchors):
  prr_submission/prr_assets/b0_quantitative_bound.tex —
  lemma `lem:exactmfull-b0-domination` (complex Gaussian domination),
  lemma `lem:exactmfull-b0-legs` (complex-leg kernel data, parts (a)-(e),
    Eqs. eq:exactmfull-b0-budget, eq:exactmfull-b0-kappahat),
  lemma `lem:exactmfull-b0-absorption` (penalty absorption,
    Eqs. eq:exactmfull-b0-square, eq:exactmfull-b0-initlaw and the
    normalization chain in the proof of part (d)),
  proposition `prop:exactmfull-b0` (Step 2 margin equivalence,
    Eq. eq:exactmfull-b0-conditions), the closed-form remark after it
    (Omega_Z yhat^2 = (1 + t_theta^2/lambda) tau/(4(tau - r_0)) at the
    radius eq:exactmfull-b0-r0), and Eq. eq:exactmfull-b0-vinf (m = 2
    slab-mixture supremum).

-- SCOPE NOTE (file-level): this file formalizes SELECTED scalar,
Gaussian-integral, and normalization kernels mined from the chain above —
NOT every self-contained real-arithmetic step.  Each docstring below states
exactly which display or proof sentence it encodes.  Real-arithmetic steps
of the chain that are explicitly NOT formalized here:
  * the parallel and transverse real-part lower bounds of legs(a)
    (Re v_par >= 4 D_0 h u e^{-A} and Re v_perp = 4 D_0 h u; only the
    Z-block bound `re_vZ_lower` is proved);
  * the legs(b) tangent/argument bound |tan arg v(zeta)| <= |Im z|/u
    (all three blocks);
  * the parallel penalty inequality of legs(d)
    (omega_par(h) <= Omega_par h with its extra factor 1/4; only the
    Z-block case `penalty_exponent_le` is proved);
  * the weighted-mixture supremum bound S_w of absorption(a)
    (Eq. eq:exactmfull-b0-Sw and the inflation slack delta_v of
    Eq. eq:exactmfull-b0-deltav);
  * absorption(b) (the contact-node bound e^{sigma' y_par^2} <= e^{sigma' a^2});
  * the full e^Pi aggregation of absorption(d)
    (Eq. eq:exactmfull-b0-Pi: the assembled product-of-all-factors bound;
    only its scalar normalization steps `log_inv_one_sub_le`,
    `div_one_sub_le`, `inv_sqrt_one_sub_le`, `prod_inv_sqrt_le` and the
    budget step `budget_sum` are proved).
The heterogeneous block-product assembly of legs(e) with the two
(1-lambda)^{-1/2} Young factors IS formalized at the level of abstract
angle data (`kappa_block_prod_pow`, `kappa_hat_display_bound`); the
identification of those angles with arg v(zeta) is not.  The
complex-analytic and operator-theoretic scaffolding (analyticity of R_B,
the Dyson expansion, Cauchy estimates on disks, torus image sums, the
Tonelli/Morera/identity-theorem bridge of prop:exactmfull-b0-chain) is NOT
formalized.  The final inversion Step 3 of prop:exactmfull-b0 is formalized
in FormalPRR/BudgetThreshold.lean (`Ec_lt_iff`).
-/
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds

namespace FormalPRR
namespace B0ChainKernel

noncomputable section
open Real MeasureTheory Finset

/-! ## Lemma `lem:exactmfull-b0-domination`: the Young step -/

/-- Young step in the proof of lemma `lem:exactmfull-b0-domination`:
"the cross term ... is bounded in absolute value by Young's inequality,
`|2 b m_I xi| <= lambda a xi^2 + b^2 m_I^2/(lambda a)`", for `lambda a > 0`.
The identification of `a`, `b`, `m_I` with `Re`/`Im` data of the complex
variance is the (unformalized) complex-arithmetic step. -/
theorem young_cross_term {lam a : ℝ} (hlam : 0 < lam) (ha : 0 < a) (b mI ξ : ℝ) :
    |2 * b * mI * ξ| ≤ lam * a * ξ ^ 2 + b ^ 2 * mI ^ 2 / (lam * a) := by
  have hla : 0 < lam * a := mul_pos hlam ha
  have h2 : |2 * b * mI * ξ| = 2 * (|b * mI| * |ξ|) := by
    rw [show 2 * b * mI * ξ = 2 * ((b * mI) * ξ) by ring, abs_mul, abs_mul, abs_two]
  have hD : b ^ 2 * mI ^ 2 = b ^ 2 * mI ^ 2 / (lam * a) * (lam * a) :=
    (div_mul_cancel₀ _ hla.ne').symm
  have key : 2 * (|b * mI| * |ξ|) * (lam * a) ≤ (lam * a) ^ 2 * ξ ^ 2 + b ^ 2 * mI ^ 2 := by
    nlinarith [sq_nonneg (lam * a * |ξ| - |b * mI|), sq_abs (b * mI), sq_abs ξ,
      abs_nonneg (b * mI), abs_nonneg ξ]
  rw [h2]
  nlinarith [key, hD, hla]

/-! ## Lemma `lem:exactmfull-b0-legs` (a): real-part lower bounds -/

/-- legs(a), second inequality: `1 - e^{-A} >= A e^{-A}` (the tex uses it for
`A = 2 gamma h u >= 0`; the inequality holds for every real `A`, equivalent to
`1 + A <= e^A`). -/
theorem one_sub_exp_neg_ge {A : ℝ} (_hA : 0 ≤ A) :
    A * Real.exp (-A) ≤ 1 - Real.exp (-A) := by
  have h := Real.add_one_le_exp A
  have hpos : (0 : ℝ) < Real.exp (-A) := Real.exp_pos _
  have h2 : (A + 1) * Real.exp (-A) ≤ 1 := by
    have h3 := mul_le_mul_of_nonneg_right h hpos.le
    rwa [← Real.exp_add, add_neg_cancel, Real.exp_zero] at h3
  nlinarith

/-- legs(a), first inequality: `Re(1 - e^{-2 gamma zeta}) = 1 - e^{-A} cos(...)
>= 1 - e^{-A}` (the cosine only helps). -/
theorem one_sub_exp_neg_cos_ge (A c : ℝ) :
    1 - Real.exp (-A) ≤ 1 - Real.exp (-A) * Real.cos c := by
  have h1 : Real.cos c ≤ 1 := Real.cos_le_one c
  have h2 : (0 : ℝ) < Real.exp (-A) := Real.exp_pos _
  nlinarith

/-- legs(a) assembled for the midpoint block: for the leg data `h >= 0`,
`u >= 0` (the tex has `h ∈ (0,1]`, `u = Re z > 0`) and `A = 2 gamma h u`,
`(D_0/(2 gamma)) (1 - e^{-A} cos(2 gamma h Im z)) >= D_0 h u e^{-A}` — the
display "Re v_Z(zeta) >= D_0 h u e^{-A}" of legs(a), up to the
(unformalized) identification `Re(1 - e^{-2 gamma zeta}) =
1 - e^{-A} cos(2 gamma h Im z)` of the complex real part. -/
theorem re_vZ_lower {D0 gamma h u : ℝ} (hD : 0 ≤ D0) (hg : 0 < gamma)
    (hh : 0 ≤ h) (hu : 0 ≤ u) (c : ℝ) :
    D0 * h * u * Real.exp (-(2 * gamma * h * u)) ≤
      D0 / (2 * gamma) * (1 - Real.exp (-(2 * gamma * h * u)) * Real.cos c) := by
  have hA : (0 : ℝ) ≤ 2 * gamma * h * u := by positivity
  have h1 := one_sub_exp_neg_ge hA
  have h2 := one_sub_exp_neg_cos_ge (2 * gamma * h * u) c
  have hcoef : (0 : ℝ) ≤ D0 / (2 * gamma) := by positivity
  have h3 : D0 / (2 * gamma) * ((2 * gamma * h * u) * Real.exp (-(2 * gamma * h * u))) ≤
      D0 / (2 * gamma) * (1 - Real.exp (-(2 * gamma * h * u)) * Real.cos c) :=
    mul_le_mul_of_nonneg_left (le_trans h1 h2) hcoef
  calc D0 * h * u * Real.exp (-(2 * gamma * h * u))
      = D0 / (2 * gamma) * ((2 * gamma * h * u) * Real.exp (-(2 * gamma * h * u))) := by
        field_simp
    _ ≤ _ := h3

/-! ## Lemma `lem:exactmfull-b0-legs` (b), (c): sine bounds -/

/-- legs(b)/(c) scalar core: `sin^2 x <= x^2` (mathlib), in the exact shape of
legs(c): `e^{-A} sin^2(gamma h w) <= e^{-A} gamma^2 h^2 w^2` with
`A = 2 gamma h u`; the identification of the left side with
`(Im e^{-gamma zeta})^2` is the (unformalized) complex-arithmetic step. -/
theorem mean_multiplier_im_sq_le (gamma h u w : ℝ) :
    Real.exp (-(2 * gamma * h * u)) * Real.sin (gamma * h * w) ^ 2 ≤
      Real.exp (-(2 * gamma * h * u)) * (gamma ^ 2 * h ^ 2 * w ^ 2) := by
  have hs : Real.sin (gamma * h * w) ^ 2 ≤ (gamma * h * w) ^ 2 := Real.sin_sq_le_sq
  have h2 : (gamma * h * w) ^ 2 = gamma ^ 2 * h ^ 2 * w ^ 2 := by ring
  exact mul_le_mul_of_nonneg_left (by nlinarith) (Real.exp_pos _).le

/-! ## Lemma `lem:exactmfull-b0-legs` (d): penalty budget -/

/-- `Omega_Z` of Eq. (eq:exactmfull-b0-budget):
`Omega_Z = (1 + t_theta^2/lambda) gamma^2 r_0^2 / (2 eps^2 D_0 (tau - r_0))`. -/
def OmegaZ (lam ttheta gamma r0 eps D0 tau : ℝ) : ℝ :=
  (1 + ttheta ^ 2 / lam) * (gamma ^ 2 * r0 ^ 2) / (2 * eps ^ 2 * D0 * (tau - r0))

/-- `Omega_parallel = Omega_Z / 4` (Eq. eq:exactmfull-b0-budget). -/
def OmegaPar (lam ttheta gamma r0 eps D0 tau : ℝ) : ℝ :=
  OmegaZ lam ttheta gamma r0 eps D0 tau / 4

/-- legs(d), exact cancellation sentence "the factors e^{-A} cancel exactly":
`(1 + t^2/lam) (gamma^2 h^2 w^2 e^{-A}) / (2 eps^2 D_0 h u e^{-A})
 = (1 + t^2/lam) gamma^2 h w^2 / (2 eps^2 D_0 u)`. -/
theorem penalty_exponent_cancel {h eps D0 u : ℝ} (hh : h ≠ 0) (he : eps ≠ 0)
    (hD : D0 ≠ 0) (hu : u ≠ 0) (lam ttheta gamma w A : ℝ) :
    (1 + ttheta ^ 2 / lam) * (gamma ^ 2 * h ^ 2 * w ^ 2 * Real.exp (-A)) /
        (2 * eps ^ 2 * D0 * h * u * Real.exp (-A)) =
      (1 + ttheta ^ 2 / lam) * (gamma ^ 2 * h * w ^ 2) / (2 * eps ^ 2 * D0 * u) := by
  have hexp : Real.exp (-A) ≠ 0 := Real.exp_ne_zero _
  field_simp

/-- Pure ring form of `Omega_Z * h`. -/
theorem OmegaZ_mul_h (lam ttheta gamma r0 eps D0 tau h : ℝ) :
    OmegaZ lam ttheta gamma r0 eps D0 tau * h =
      (1 + ttheta ^ 2 / lam) * (gamma ^ 2 * h * r0 ^ 2) /
        (2 * eps ^ 2 * D0 * (tau - r0)) := by
  unfold OmegaZ
  ring

/-- legs(d), final bound of the display: under `(Im z)^2 <= r_0^2` and
`u >= tau - r_0 > 0`,
`(1 + t^2/lam) gamma^2 h w^2 / (2 eps^2 D_0 u) <= Omega_Z h`
(the tex multiplies both sides by `(y_Z - z-bar)^2 >= 0`; multiply the
conclusion by any nonnegative factor to recover that form). -/
theorem penalty_exponent_le {lam ttheta gamma h u eps D0 w r0 tau : ℝ}
    (hlam : 0 < lam) (hh : 0 ≤ h) (he : 0 < eps) (hD : 0 < D0)
    (hw : w ^ 2 ≤ r0 ^ 2) (hu : tau - r0 ≤ u) (htr : 0 < tau - r0) :
    (1 + ttheta ^ 2 / lam) * (gamma ^ 2 * h * w ^ 2) / (2 * eps ^ 2 * D0 * u) ≤
      OmegaZ lam ttheta gamma r0 eps D0 tau * h := by
  rw [OmegaZ_mul_h]
  have hu' : 0 < u := lt_of_lt_of_le htr hu
  have hd1 : (0 : ℝ) < 2 * eps ^ 2 * D0 * u := by positivity
  have hd2 : (0 : ℝ) < 2 * eps ^ 2 * D0 * (tau - r0) := by positivity
  rw [div_le_div_iff₀ hd1 hd2]
  have hC : (0 : ℝ) ≤ 1 + ttheta ^ 2 / lam := by positivity
  have hw0 : (0 : ℝ) ≤ w ^ 2 := sq_nonneg w
  have hcore : w ^ 2 * (tau - r0) ≤ r0 ^ 2 * u := by
    calc w ^ 2 * (tau - r0) ≤ r0 ^ 2 * (tau - r0) :=
          mul_le_mul_of_nonneg_right hw htr.le
      _ ≤ r0 ^ 2 * u := mul_le_mul_of_nonneg_left hu (sq_nonneg r0)
  nlinarith [mul_le_mul_of_nonneg_left hcore
    (mul_nonneg (mul_nonneg hC (sq_nonneg gamma)) hh),
    mul_pos (mul_pos (mul_pos two_pos (pow_pos he 2)) hD) hu']

/-- legs(d), closing sentence "Linearity in h and `sum_k h_k = 1` give the
budget statement": if every leg penalty satisfies `f k <= Omega * h k` and the
leg fractions sum to one, the total penalty is at most `Omega`, independently
of the chain order and the leg configuration. -/
theorem budget_sum {ι : Type*} (S : Finset ι) (f h : ι → ℝ) (Omega : ℝ)
    (hf : ∀ k ∈ S, f k ≤ Omega * h k) (hsum : ∑ k ∈ S, h k = 1) :
    ∑ k ∈ S, f k ≤ Omega := by
  calc ∑ k ∈ S, f k ≤ ∑ k ∈ S, Omega * h k := Finset.sum_le_sum hf
    _ = Omega * ∑ k ∈ S, h k := (Finset.mul_sum _ _ _).symm
    _ = Omega := by rw [hsum, mul_one]

/-! ## Lemma `lem:exactmfull-b0-legs` (e): the domination constant -/

/-- legs(e) core: for `cos theta > 0` and `|tan theta| <= t`,
`1/cos theta <= sqrt(1 + t^2)` — the sentence
"`cos theta >= (1 + t_theta^2)^{-1/2}`". -/
theorem sec_le_sqrt_one_add_tan_sq {theta t : ℝ} (hc : 0 < Real.cos theta)
    (ht : |Real.tan theta| ≤ t) :
    1 / Real.cos theta ≤ Real.sqrt (1 + t ^ 2) := by
  have hsec : (1 / Real.cos theta) ^ 2 = 1 + Real.tan theta ^ 2 := by
    rw [Real.tan_eq_sin_div_cos]
    have h := Real.sin_sq_add_cos_sq theta
    field_simp
    linarith
  have h1 : (1 / Real.cos theta) ^ 2 ≤ 1 + t ^ 2 := by
    rw [hsec]
    nlinarith [sq_abs (Real.tan theta), abs_nonneg (Real.tan theta)]
  have h2 : 0 ≤ 1 / Real.cos theta := by positivity
  calc 1 / Real.cos theta = Real.sqrt ((1 / Real.cos theta) ^ 2) :=
        (Real.sqrt_sq h2).symm
    _ ≤ Real.sqrt (1 + t ^ 2) := Real.sqrt_le_sqrt h1

/-- Scalar secant-square bound behind legs(e): for `cos theta > 0` and
`|tan theta| <= t`, `(1/cos theta)^2 <= 1 + t^2` (squared form of
`sec_le_sqrt_one_add_tan_sq`; internal helper shared by the powered and
product forms below). -/
theorem sec_sq_le_one_add_tan_sq {theta t : ℝ} (hc : 0 < Real.cos theta)
    (ht : |Real.tan theta| ≤ t) :
    (1 / Real.cos theta) ^ 2 ≤ 1 + t ^ 2 := by
  have hsec : (1 / Real.cos theta) ^ 2 = 1 + Real.tan theta ^ 2 := by
    rw [Real.tan_eq_sin_div_cos]
    have h := Real.sin_sq_add_cos_sq theta
    field_simp
    linarith
  rw [hsec]
  nlinarith [sq_abs (Real.tan theta), abs_nonneg (Real.tan theta)]

/-- legs(e), ONE-ANGLE powered form: `(1/cos theta)^(2n) <= (1 + t^2)^n` for a
single repeated angle `theta`.
-- SCOPE NOTE: this theorem alone is only one scalar ingredient of
Eq. (eq:exactmfull-b0-kappahat) — it carries neither the heterogeneous
per-block angles nor the two `(1-lambda)^{-1/2}` Young factors of the
display.  The heterogeneous product form is `kappa_block_prod_pow`; the
exact display shape `hat-kappa = (1-lambda)^{-1} (1 + t_theta^2)^{(d+1)/4}`
(as a natural-power expression in fourth roots, certified by
`kappa_hat_pow_four`) is reached in `kappa_hat_display_bound`. -/
theorem kappa_block_pow {theta t : ℝ} (hc : 0 < Real.cos theta)
    (ht : |Real.tan theta| ≤ t) (n : ℕ) :
    ((1 / Real.cos theta) ^ 2) ^ n ≤ (1 + t ^ 2) ^ n :=
  pow_le_pow_left₀ (by positivity) (sec_sq_le_one_add_tan_sq hc ht) n

/-- legs(e), HETEROGENEOUS product form: for a finite family of blocks with
per-block angles `theta k`, per-block tangent bounds `|tan (theta k)| <= tt k`
and per-block multiplicities `n k`,
`prod_k ((1/cos theta_k)^2)^(n_k) <= prod_k (1 + tt_k^2)^(n_k)` —
the product over potentially different block angles assembled in the proof
of legs(e) ("with |tan theta| <= t_theta throughout by (b)"). -/
theorem kappa_block_prod_pow {ι : Type*} (S : Finset ι) (theta tt : ι → ℝ)
    (n : ι → ℕ) (hc : ∀ k ∈ S, 0 < Real.cos (theta k))
    (ht : ∀ k ∈ S, |Real.tan (theta k)| ≤ tt k) :
    (∏ k ∈ S, ((1 / Real.cos (theta k)) ^ 2) ^ n k) ≤
      ∏ k ∈ S, (1 + tt k ^ 2) ^ n k := by
  apply Finset.prod_le_prod
  · intro k _
    positivity
  · intro k hk
    exact pow_le_pow_left₀ (sq_nonneg _)
      (sec_sq_le_one_add_tan_sq (hc k hk) (ht k hk)) (n k)

/-- Fourth-root certificate: `(sqrt (sqrt x))^4 = x` for `x >= 0`, so
`sqrt (sqrt x)` is the real fourth root `x^{1/4}`. -/
theorem sqrt_sqrt_pow_four {x : ℝ} (hx : 0 ≤ x) :
    Real.sqrt (Real.sqrt x) ^ 4 = x := by
  have h1 : Real.sqrt (Real.sqrt x) ^ 2 = Real.sqrt x :=
    Real.sq_sqrt (Real.sqrt_nonneg x)
  calc Real.sqrt (Real.sqrt x) ^ 4 = (Real.sqrt (Real.sqrt x) ^ 2) ^ 2 := by ring
    _ = Real.sqrt x ^ 2 := by rw [h1]
    _ = x := Real.sq_sqrt hx

/-- The natural-power encoding of the display exponent: the fourth power of
`(sqrt (sqrt (1 + t^2)))^(d+1)` is `(1 + t^2)^(d+1)`, i.e.
`(sqrt (sqrt (1 + t^2)))^(d+1)` IS the display's `(1 + t_theta^2)^{(d+1)/4}`
of Eq. (eq:exactmfull-b0-kappahat). -/
theorem kappa_hat_pow_four (t : ℝ) (d : ℕ) :
    (Real.sqrt (Real.sqrt (1 + t ^ 2)) ^ (d + 1)) ^ 4 = (1 + t ^ 2) ^ (d + 1) := by
  rw [← pow_mul, mul_comm (d + 1) 4, pow_mul,
    sqrt_sqrt_pow_four (by positivity : (0:ℝ) ≤ 1 + t ^ 2)]

/-- legs(e) / Eq. (eq:exactmfull-b0-kappahat), EXACT DISPLAY SHAPE: the
one-leg domination constant.  The `Z` and `parallel` blocks contribute
`[(1-lambda) cos theta]^{-1/2}` each (the two Young factors), the `d - 1`
transverse blocks `(cos theta)^{-1/2}` each; under the per-block hypotheses
`cos theta > 0` and `|tan theta| <= t` (legs(b)), the product is at most
`hat-kappa = (1 - lambda)^{-1} (1 + t^2)^{(d+1)/4}`, where the fourth root
is encoded as `sqrt (sqrt (1 + t^2))` (certified by `kappa_hat_pow_four`).
-- SCOPE NOTE: the block angles are abstract data here; their
identification with `arg v(zeta)` of the complex block variances is the
(unformalized) complex-arithmetic step. -/
theorem kappa_hat_display_bound {lam t : ℝ} (hlam : lam < 1) {d : ℕ}
    (hd : 1 ≤ d) (thZ thPar : ℝ) (thPerp : Fin (d - 1) → ℝ)
    (hcZ : 0 < Real.cos thZ) (htZ : |Real.tan thZ| ≤ t)
    (hcP : 0 < Real.cos thPar) (htP : |Real.tan thPar| ≤ t)
    (hcp : ∀ i, 0 < Real.cos (thPerp i))
    (htp : ∀ i, |Real.tan (thPerp i)| ≤ t) :
    Real.sqrt (1 / ((1 - lam) * Real.cos thZ)) *
        Real.sqrt (1 / ((1 - lam) * Real.cos thPar)) *
        ∏ i, Real.sqrt (1 / Real.cos (thPerp i)) ≤
      (1 - lam)⁻¹ * Real.sqrt (Real.sqrt (1 + t ^ 2)) ^ (d + 1) := by
  have h1l : (0 : ℝ) < 1 - lam := by linarith
  set F := Real.sqrt (Real.sqrt (1 + t ^ 2)) with hF
  have hFnn : 0 ≤ F := Real.sqrt_nonneg _
  -- per-block bound: sqrt (1/cos theta) <= F
  have hblock : ∀ {th : ℝ}, 0 < Real.cos th → |Real.tan th| ≤ t →
      Real.sqrt (1 / Real.cos th) ≤ F := by
    intro th hcth htth
    exact Real.sqrt_le_sqrt (sec_le_sqrt_one_add_tan_sq hcth htth)
  -- split the two Young factors
  have hsplit : ∀ {th : ℝ}, 0 < Real.cos th →
      Real.sqrt (1 / ((1 - lam) * Real.cos th)) =
        Real.sqrt (1 / (1 - lam)) * Real.sqrt (1 / Real.cos th) := by
    intro th hcth
    rw [← Real.sqrt_mul (by positivity)]
    congr 1
    field_simp
  -- transverse product bound
  have hperp : (∏ i, Real.sqrt (1 / Real.cos (thPerp i))) ≤ F ^ (d - 1) := by
    calc (∏ i, Real.sqrt (1 / Real.cos (thPerp i)))
        ≤ ∏ _i : Fin (d - 1), F :=
          Finset.prod_le_prod (fun i _ => Real.sqrt_nonneg _)
            (fun i _ => hblock (hcp i) (htp i))
      _ = F ^ (d - 1) := by
          rw [Finset.prod_const, Finset.card_univ, Fintype.card_fin]
  have hperp_nn : (0 : ℝ) ≤ ∏ i, Real.sqrt (1 / Real.cos (thPerp i)) :=
    Finset.prod_nonneg fun i _ => Real.sqrt_nonneg _
  have hZP : Real.sqrt (1 / Real.cos thZ) * Real.sqrt (1 / Real.cos thPar) ≤
      F * F :=
    mul_le_mul (hblock hcZ htZ) (hblock hcP htP) (Real.sqrt_nonneg _) hFnn
  have hpow : F * F * F ^ (d - 1) = F ^ (d + 1) := by
    rw [show d + 1 = 2 + (d - 1) by omega, pow_add, sq]
  calc Real.sqrt (1 / ((1 - lam) * Real.cos thZ)) *
        Real.sqrt (1 / ((1 - lam) * Real.cos thPar)) *
        ∏ i, Real.sqrt (1 / Real.cos (thPerp i))
      = (Real.sqrt (1 / (1 - lam)) * Real.sqrt (1 / (1 - lam))) *
          (Real.sqrt (1 / Real.cos thZ) * Real.sqrt (1 / Real.cos thPar) *
            ∏ i, Real.sqrt (1 / Real.cos (thPerp i))) := by
        rw [hsplit hcZ, hsplit hcP]
        ring
    _ = (1 - lam)⁻¹ *
          (Real.sqrt (1 / Real.cos thZ) * Real.sqrt (1 / Real.cos thPar) *
            ∏ i, Real.sqrt (1 / Real.cos (thPerp i))) := by
        rw [Real.mul_self_sqrt (by positivity), one_div]
    _ ≤ (1 - lam)⁻¹ * (F * F * F ^ (d - 1)) := by
        apply mul_le_mul_of_nonneg_left _ (inv_nonneg.mpr h1l.le)
        exact mul_le_mul hZP hperp hperp_nn (by positivity)
    _ = (1 - lam)⁻¹ * F ^ (d + 1) := by rw [hpow]

/-! ## Lemma `lem:exactmfull-b0-absorption` (a): the complete square -/

/-- The unit-normalized Gaussian kernel `N(u; v) = e^{-u^2/(2v)}/sqrt(2 pi v)`
of the "Kernels" paragraph (real duration/variance case). -/
def gaussN (u v : ℝ) : ℝ := Real.exp (-u ^ 2 / (2 * v)) / Real.sqrt (2 * π * v)

/-- Internal helper: nonnegativity of `gaussN`; no direct paper display. -/
theorem gaussN_nonneg (u : ℝ) {v : ℝ} (_hv : 0 < v) : 0 ≤ gaussN u v := by
  unfold gaussN
  positivity

/-- Eq. (eq:exactmfull-b0-square) of lemma `lem:exactmfull-b0-absorption`(a),
verbatim: for `x = 2 v sigma < 1` (tex: `x = 2 eps^2 rho^2 sigma < 1` with
`v = eps^2 rho^2`),
`N(y - c; v) e^{sigma (y - z-bar)^2}
  = (1-x)^{-1/2} exp[sigma (c - z-bar)^2/(1-x)] N(y - c(x); v/(1-x))`,
`c(x) = (c - x z-bar)/(1-x)`.  Exact identity; `sigma` may be any real with
`2 v sigma < 1` (the tex assumes `sigma >= 0`, which is not needed). -/
theorem complete_square_identity {v sig : ℝ} (hv : 0 < v) (hx : 2 * v * sig < 1)
    (c zbar y : ℝ) :
    gaussN (y - c) v * Real.exp (sig * (y - zbar) ^ 2) =
      (Real.sqrt (1 - 2 * v * sig))⁻¹ *
        Real.exp (sig * (c - zbar) ^ 2 / (1 - 2 * v * sig)) *
        gaussN (y - (c - 2 * v * sig * zbar) / (1 - 2 * v * sig))
          (v / (1 - 2 * v * sig)) := by
  have h1x : (0 : ℝ) < 1 - 2 * v * sig := by linarith
  have hS : (0 : ℝ) < Real.sqrt (2 * π * v) := Real.sqrt_pos.mpr (by positivity)
  have hs1x : (0 : ℝ) < Real.sqrt (1 - 2 * v * sig) := Real.sqrt_pos.mpr h1x
  unfold gaussN
  have hsq : Real.sqrt (2 * π * (v / (1 - 2 * v * sig))) =
      Real.sqrt (2 * π * v) / Real.sqrt (1 - 2 * v * sig) := by
    rw [show 2 * π * (v / (1 - 2 * v * sig)) = 2 * π * v / (1 - 2 * v * sig) by ring,
      Real.sqrt_div (by positivity) _]
  rw [hsq]
  have hexp : -(y - c) ^ 2 / (2 * v) + sig * (y - zbar) ^ 2 =
      sig * (c - zbar) ^ 2 / (1 - 2 * v * sig) +
        -(y - (c - 2 * v * sig * zbar) / (1 - 2 * v * sig)) ^ 2 /
          (2 * (v / (1 - 2 * v * sig))) := by
    field_simp
    ring
  rw [div_mul_eq_mul_div, ← Real.exp_add, hexp, Real.exp_add]
  field_simp
  rw [div_self (by
    rw [show (1 - sig * 2 * v : ℝ) = 1 - 2 * v * sig by ring]
    exact hs1x.ne' : Real.sqrt (1 - sig * 2 * v) ≠ 0)]

/-! ## Lemma `lem:exactmfull-b0-absorption` (c): initial-law identities -/

/-- The kernel `gaussN` has unit mass: `∫ N(y - m; v) dy = 1` for `v > 0`
(the "unit-mass real Gaussian density" property used throughout the chain;
via mathlib's `integral_gaussianPDFReal_eq_one`). -/
theorem integral_gaussN_eq_one {v : ℝ} (hv : 0 < v) (m : ℝ) :
    ∫ y, gaussN (y - m) v = 1 := by
  have hnn : Real.toNNReal v ≠ 0 := (Real.toNNReal_pos.mpr hv).ne'
  have heq : ∀ y, gaussN (y - m) v =
      ProbabilityTheory.gaussianPDFReal m (Real.toNNReal v) y := by
    intro y
    unfold gaussN ProbabilityTheory.gaussianPDFReal
    rw [Real.coe_toNNReal v hv.le, div_eq_inv_mul]
  simp_rw [heq]
  exact ProbabilityTheory.integral_gaussianPDFReal_eq_one m hnn

/-- Eq. (eq:exactmfull-b0-initlaw), first display (midpoint block), verbatim
with `v = eps^2 D_0/(2 gamma)` and `x_Z = eps^2 D_0 sigma/gamma = 2 v sigma`:
`∫ N(y - z_0; v) e^{sigma (y - z-bar)^2} dy
  = (1 - x_Z)^{-1/2} exp[sigma (z_0 - z-bar)^2/(1 - x_Z)]` —
"exact Gaussian identities" of lemma `lem:exactmfull-b0-absorption`(c). -/
theorem initial_law_integral {v sig : ℝ} (hv : 0 < v) (hx : 2 * v * sig < 1)
    (z0 zbar : ℝ) :
    ∫ y, gaussN (y - z0) v * Real.exp (sig * (y - zbar) ^ 2) =
      (Real.sqrt (1 - 2 * v * sig))⁻¹ *
        Real.exp (sig * (z0 - zbar) ^ 2 / (1 - 2 * v * sig)) := by
  have h1x : (0 : ℝ) < 1 - 2 * v * sig := by linarith
  have hv' : (0 : ℝ) < v / (1 - 2 * v * sig) := by positivity
  have hfun : (fun y => gaussN (y - z0) v * Real.exp (sig * (y - zbar) ^ 2)) =
      fun y => ((Real.sqrt (1 - 2 * v * sig))⁻¹ *
          Real.exp (sig * (z0 - zbar) ^ 2 / (1 - 2 * v * sig))) *
        gaussN (y - (z0 - 2 * v * sig * zbar) / (1 - 2 * v * sig))
          (v / (1 - 2 * v * sig)) := by
    funext y
    rw [complete_square_identity hv hx z0 zbar y]
  rw [hfun, MeasureTheory.integral_const_mul,
    integral_gaussN_eq_one hv' ((z0 - 2 * v * sig * zbar) / (1 - 2 * v * sig)),
    mul_one]

/-- Eq. (eq:exactmfull-b0-initlaw), second display (longitudinal block):
`∫ N(y - r_par0; v) e^{sigma' y^2} dy
  = (1 - x_par)^{-1/2} exp[sigma' r_par0^2/(1 - x_par)]` with
`v = eps^2 u_0^2`, `x_par = 2 v sigma'` — "the first with ... penalty centre 0"
(proof of lemma `lem:exactmfull-b0-absorption`(c)). -/
theorem initial_law_integral_parallel {v sig : ℝ} (hv : 0 < v)
    (hx : 2 * v * sig < 1) (rpar0 : ℝ) :
    ∫ y, gaussN (y - rpar0) v * Real.exp (sig * y ^ 2) =
      (Real.sqrt (1 - 2 * v * sig))⁻¹ *
        Real.exp (sig * rpar0 ^ 2 / (1 - 2 * v * sig)) := by
  have h := initial_law_integral hv hx rpar0 0
  simpa using h

/-! ## Lemma `lem:exactmfull-b0-absorption` (d): normalization bookkeeping -/

/-- Proof of absorption(d), first normalization step:
`log[1/(1-x)] <= x/(1-x)` for `0 <= x < 1`. -/
theorem log_inv_one_sub_le {x : ℝ} (_hx0 : 0 ≤ x) (hx1 : x < 1) :
    Real.log (1 / (1 - x)) ≤ x / (1 - x) := by
  have h1 : (0 : ℝ) < 1 - x := by linarith
  calc Real.log (1 / (1 - x)) ≤ 1 / (1 - x) - 1 :=
        Real.log_le_sub_one_of_pos (by positivity)
    _ = x / (1 - x) := by
        field_simp
        ring

/-- Proof of absorption(d), second normalization step:
`x/(1-x) <= x/(1-x_max)` for `0 <= x <= x_max < 1`. -/
theorem div_one_sub_le {x xmax : ℝ} (hx0 : 0 ≤ x) (hxm : x ≤ xmax)
    (hm1 : xmax < 1) :
    x / (1 - x) ≤ x / (1 - xmax) := by
  have h1 : (0 : ℝ) < 1 - xmax := by linarith
  have h2 : (0 : ℝ) < 1 - x := by linarith
  rw [div_le_div_iff₀ h2 h1]
  nlinarith

/-- Per-factor normalization bound of absorption(d):
`(1-x)^{-1/2} <= exp(x/(2(1-x_max)))` for `0 <= x <= x_max < 1`. -/
theorem inv_sqrt_one_sub_le {x xmax : ℝ} (hx0 : 0 ≤ x) (hxm : x ≤ xmax)
    (hm1 : xmax < 1) :
    (Real.sqrt (1 - x))⁻¹ ≤ Real.exp (x / (2 * (1 - xmax))) := by
  have h1 : (0 : ℝ) < 1 - xmax := by linarith
  have h2 : (0 : ℝ) < 1 - x := by linarith
  have hchain : (1 - x)⁻¹ ≤ Real.exp (x / (1 - xmax)) := by
    have hlog : Real.log ((1 - x)⁻¹) ≤ x / (1 - xmax) := by
      rw [← one_div]
      exact le_trans (log_inv_one_sub_le hx0 (by linarith))
        (div_one_sub_le hx0 hxm hm1)
    calc (1 - x)⁻¹ = Real.exp (Real.log ((1 - x)⁻¹)) :=
          (Real.exp_log (by positivity)).symm
      _ ≤ Real.exp (x / (1 - xmax)) := Real.exp_le_exp.mpr hlog
  calc (Real.sqrt (1 - x))⁻¹ = Real.sqrt ((1 - x)⁻¹) := (Real.sqrt_inv _).symm
    _ ≤ Real.sqrt (Real.exp (x / (1 - xmax))) := Real.sqrt_le_sqrt hchain
    _ = Real.exp (x / (1 - xmax) / 2) := (Real.exp_half _).symm
    _ = Real.exp (x / (2 * (1 - xmax))) := by rw [div_div]; ring_nf

/-- Chain-product normalization bound of absorption(d):
"`prod_k (1-x_k)^{-1/2} <= exp[sum_k x_k/(2(1-x_max))]`", uniformly in the
chain order and the leg configuration. -/
theorem prod_inv_sqrt_le {ι : Type*} (S : Finset ι) (x : ι → ℝ) (xmax : ℝ)
    (hm1 : xmax < 1) (hx : ∀ k ∈ S, 0 ≤ x k ∧ x k ≤ xmax) :
    (∏ k ∈ S, (Real.sqrt (1 - x k))⁻¹) ≤
      Real.exp (∑ k ∈ S, x k / (2 * (1 - xmax))) := by
  rw [Real.exp_sum]
  apply Finset.prod_le_prod
  · intro k _
    positivity
  · intro k hk
    exact inv_sqrt_one_sub_le (hx k hk).1 (hx k hk).2 hm1

/-! ## Proposition `prop:exactmfull-b0`, Step 2: the margin equivalence -/

/-- Step 2/Step 3 pivot of the proof of `prop:exactmfull-b0`, verbatim:
the two jet conditions of Eq. (eq:exactmfull-b0-conditions),
`r_0^{-1} E < mu_1` and `2 r_0^{-2} E < mu_2`, hold "if and only if
`E < M-hat`" with `M-hat = min(r_0 mu_1, r_0^2 mu_2/2)`
(Eq. eq:exactmfull-b0-threshold). -/
theorem margins_iff {r0 : ℝ} (hr : 0 < r0) (E mu1 mu2 : ℝ) :
    (E / r0 < mu1 ∧ 2 * E / r0 ^ 2 < mu2) ↔
      E < min (r0 * mu1) (r0 ^ 2 * mu2 / 2) := by
  have hr2 : (0 : ℝ) < r0 ^ 2 := by positivity
  rw [lt_min_iff]
  constructor
  · rintro ⟨h1, h2⟩
    have h1' := (div_lt_iff₀ hr).mp h1
    have h2' := (div_lt_iff₀ hr2).mp h2
    constructor
    · nlinarith
    · nlinarith
  · rintro ⟨h1, h2⟩
    constructor
    · rw [div_lt_iff₀ hr]
      nlinarith
    · rw [div_lt_iff₀ hr2]
      nlinarith

/-! ## The closed-form radius remark after `prop:exactmfull-b0` -/

/-- The closed-form sentence following Eq. (eq:exactmfull-b0-radius):
at the radius `r_0 = (eps/(gamma yhat)) sqrt(D_0 tau/2)`
[Eq. eq:exactmfull-b0-r0], the leading penalty term is
`Omega_Z yhat^2 = (1 + t_theta^2/lambda) tau / (4 (tau - r_0))`.
This theorem is the pointwise ALGEBRAIC IDENTITY only; the paper's
"manifestly O(1) uniformly in eps" conclusion is the separate quantitative
bound `radius_penalty_uniform_O1` below, proved under the paper's radius
condition (R1) `r_0 <= tau/2` [Eq. eq:exactmfull-b0-radius] and fixed
`lambda > 0`. -/
theorem radius_penalty_closed_form {eps gamma yhat D0 tau : ℝ}
    (he : eps ≠ 0) (hg : gamma ≠ 0) (hy : yhat ≠ 0) (hD : 0 < D0)
    (htau : 0 < tau)
    (_hr : tau - eps / (gamma * yhat) * Real.sqrt (D0 * tau / 2) ≠ 0)
    (lam ttheta : ℝ) :
    OmegaZ lam ttheta gamma (eps / (gamma * yhat) * Real.sqrt (D0 * tau / 2))
        eps D0 tau * yhat ^ 2 =
      (1 + ttheta ^ 2 / lam) * tau /
        (4 * (tau - eps / (gamma * yhat) * Real.sqrt (D0 * tau / 2))) := by
  have hs : Real.sqrt (D0 * tau / 2) ^ 2 = D0 * tau / 2 :=
    Real.sq_sqrt (by positivity)
  unfold OmegaZ
  rw [mul_pow, div_pow, hs]
  field_simp
  ring

/-- Scalar core of the uniform O(1) bound: under the paper's radius
condition (R1) `r_0 <= tau/2` [Eq. eq:exactmfull-b0-radius] and fixed
`lambda > 0`, with the paper's `t_theta = r_0/(tau - r_0)`
[Eq. eq:exactmfull-b0-budget],
`(1 + t_theta^2/lambda) tau / (4 (tau - r_0)) <= (1 + 1/lambda)/2` —
a bound independent of `r_0` (hence of `eps`). -/
theorem penalty_O1_of_R1 {tau r0 lam : ℝ} (htau : 0 < tau) (hr0 : 0 ≤ r0)
    (hR1 : r0 ≤ tau / 2) (hlam : 0 < lam) :
    (1 + (r0 / (tau - r0)) ^ 2 / lam) * tau / (4 * (tau - r0)) ≤
      (1 + 1 / lam) / 2 := by
  have hd : (0 : ℝ) < tau - r0 := by linarith
  have ht0 : (0 : ℝ) ≤ r0 / (tau - r0) := div_nonneg hr0 hd.le
  have htle : r0 / (tau - r0) ≤ 1 := by
    rw [div_le_one hd]
    linarith
  have ht2 : (r0 / (tau - r0)) ^ 2 ≤ 1 := by nlinarith
  have hfac1 : 1 + (r0 / (tau - r0)) ^ 2 / lam ≤ 1 + 1 / lam := by
    have h := div_le_div_of_nonneg_right ht2 hlam.le
    linarith
  have hfac2 : tau / (4 * (tau - r0)) ≤ 1 / 2 := by
    rw [div_le_div_iff₀ (by positivity) two_pos]
    linarith
  have hc' : (0 : ℝ) ≤ tau / (4 * (tau - r0)) :=
    div_nonneg htau.le (by linarith)
  calc (1 + (r0 / (tau - r0)) ^ 2 / lam) * tau / (4 * (tau - r0))
      = (1 + (r0 / (tau - r0)) ^ 2 / lam) * (tau / (4 * (tau - r0))) := by
        ring
    _ ≤ (1 + 1 / lam) * (1 / 2) :=
        mul_le_mul hfac1 hfac2 hc' (by positivity)
    _ = (1 + 1 / lam) / 2 := by ring

/-- The paper's "manifestly O(1) uniformly in eps" conclusion after
Eq. (eq:exactmfull-b0-radius), made quantitative: at the closed-form radius
`r_0 = (eps/(gamma yhat)) sqrt(D_0 tau/2)` [Eq. eq:exactmfull-b0-r0], with
the paper's `t_theta = r_0/(tau - r_0)` and under the radius condition (R1)
`r_0 <= tau/2` and fixed `lambda > 0`,
`Omega_Z yhat^2 <= (1 + 1/lambda)/2` — a bound uniform in `eps` (which
enters only through `r_0`). -/
theorem radius_penalty_uniform_O1 {eps gamma yhat D0 tau lam : ℝ}
    (he : eps ≠ 0) (hg : gamma ≠ 0) (hy : yhat ≠ 0) (hD : 0 < D0)
    (htau : 0 < tau) (hlam : 0 < lam)
    (hr0 : 0 ≤ eps / (gamma * yhat) * Real.sqrt (D0 * tau / 2))
    (hR1 : eps / (gamma * yhat) * Real.sqrt (D0 * tau / 2) ≤ tau / 2) :
    OmegaZ lam
        (eps / (gamma * yhat) * Real.sqrt (D0 * tau / 2) /
          (tau - eps / (gamma * yhat) * Real.sqrt (D0 * tau / 2)))
        gamma (eps / (gamma * yhat) * Real.sqrt (D0 * tau / 2)) eps D0 tau *
        yhat ^ 2 ≤
      (1 + 1 / lam) / 2 := by
  have hd : (0 : ℝ) <
      tau - eps / (gamma * yhat) * Real.sqrt (D0 * tau / 2) := by linarith
  rw [radius_penalty_closed_form he hg hy hD htau hd.ne']
  exact penalty_O1_of_R1 htau hr0 hR1 hlam

/-! ## Eq. (eq:exactmfull-b0-vinf): the m = 2 slab-mixture supremum -/

/-- The sentence before Eq. (eq:exactmfull-b0-vinf): "every z ∈ ℝ lies at
distance at least Delta_phys/2 from one of the two slab centres". -/
theorem far_from_one_center (z c1 c2 : ℝ) :
    |c1 - c2| / 2 ≤ |z - c1| ∨ |c1 - c2| / 2 ≤ |z - c2| := by
  rcases le_total (|c1 - c2| / 2) (|z - c1|) with h | h
  · exact Or.inl h
  · right
    have htri : |c1 - c2| ≤ |z - c1| + |z - c2| := by
      calc |c1 - c2| = |(c1 - z) + (z - c2)| := by ring_nf
        _ ≤ |c1 - z| + |z - c2| := abs_add_le _ _
        _ = |z - c1| + |z - c2| := by rw [abs_sub_comm c1 z]
    linarith

/-- Gaussian tail monotonicity: if `delta <= |u|` then
`N(u; v) <= e^{-delta^2/(2v)}/sqrt(2 pi v)`. -/
theorem gaussN_le_of_far {v : ℝ} (hv : 0 < v) {u delta : ℝ}
    (hd : 0 ≤ delta) (h : delta ≤ |u|) :
    gaussN u v ≤ Real.exp (-delta ^ 2 / (2 * v)) / Real.sqrt (2 * π * v) := by
  have hS : (0 : ℝ) < Real.sqrt (2 * π * v) := Real.sqrt_pos.mpr (by positivity)
  unfold gaussN
  rw [div_le_div_iff_of_pos_right hS]
  apply Real.exp_le_exp.mpr
  have h2 : delta ^ 2 ≤ u ^ 2 := by nlinarith [sq_abs u, abs_nonneg u]
  rw [div_le_div_iff_of_pos_right (by positivity : (0:ℝ) < 2 * v)]
  linarith

/-- Flat sup bound for the kernel: `N(u; v) <= 1/sqrt(2 pi v)`. -/
theorem gaussN_le_max {v : ℝ} (hv : 0 < v) (u : ℝ) :
    gaussN u v ≤ 1 / Real.sqrt (2 * π * v) := by
  have hS : (0 : ℝ) < Real.sqrt (2 * π * v) := Real.sqrt_pos.mpr (by positivity)
  unfold gaussN
  rw [div_le_div_iff_of_pos_right hS]
  have h1 : -u ^ 2 / (2 * v) ≤ 0 := by
    apply div_nonpos_of_nonpos_of_nonneg <;> [nlinarith [sq_nonneg u]; positivity]
  calc Real.exp (-u ^ 2 / (2 * v)) ≤ Real.exp 0 := Real.exp_le_exp.mpr h1
    _ = 1 := Real.exp_zero

/-- Eq. (eq:exactmfull-b0-vinf) for m = 2, pointwise in z (hence a bound for
the supremum `v_infinity`): with `v = eps^2 rho^2` and
`Delta_phys = |c_1 - c_2|`,
`w_1 N(z - c_1; v) + w_2 N(z - c_2; v)
  <= (max w_1 w_2)(1 + e^{-Delta_phys^2/(8v)})/sqrt(2 pi v)`.
-- SCOPE NOTE: the tex display divides both sides by the transverse
normalization `W^{d-1}` (a positive constant carried by every kernel of the
slab family); the display is recovered by multiplying this inequality by
`W^{-(d-1)} > 0`.  Only the longitudinal factor is formalized. -/
theorem two_slab_sup_bound {v : ℝ} (hv : 0 < v) {w1 w2 : ℝ}
    (hw1 : 0 ≤ w1) (_hw2 : 0 ≤ w2) (c1 c2 z : ℝ) :
    w1 * gaussN (z - c1) v + w2 * gaussN (z - c2) v ≤
      max w1 w2 * ((1 + Real.exp (-(c1 - c2) ^ 2 / (8 * v))) /
        Real.sqrt (2 * π * v)) := by
  have hS : (0 : ℝ) < Real.sqrt (2 * π * v) := Real.sqrt_pos.mpr (by positivity)
  have hexp8 : ∀ c c' : ℝ, -(|c - c'| / 2) ^ 2 / (2 * v) = -(c - c') ^ 2 / (8 * v) := by
    intro c c'
    rw [div_pow, sq_abs]
    ring
  have hmax1 : w1 ≤ max w1 w2 := le_max_left _ _
  have hmax2 : w2 ≤ max w1 w2 := le_max_right _ _
  have hmaxnn : 0 ≤ max w1 w2 := le_trans hw1 hmax1
  rcases far_from_one_center z c1 c2 with hfar | hfar
  · -- z is far from c1
    have h1 : gaussN (z - c1) v ≤
        Real.exp (-(c1 - c2) ^ 2 / (8 * v)) / Real.sqrt (2 * π * v) := by
      have := gaussN_le_of_far hv (by positivity : (0:ℝ) ≤ |c1 - c2| / 2) hfar
      rwa [hexp8 c1 c2] at this
    have h2 : gaussN (z - c2) v ≤ 1 / Real.sqrt (2 * π * v) := gaussN_le_max hv _
    have g1nn : 0 ≤ gaussN (z - c1) v := gaussN_nonneg _ hv
    have g2nn : 0 ≤ gaussN (z - c2) v := gaussN_nonneg _ hv
    calc w1 * gaussN (z - c1) v + w2 * gaussN (z - c2) v
        ≤ max w1 w2 * (Real.exp (-(c1 - c2) ^ 2 / (8 * v)) / Real.sqrt (2 * π * v)) +
            max w1 w2 * (1 / Real.sqrt (2 * π * v)) := by
          apply add_le_add
          · exact mul_le_mul hmax1 h1 g1nn hmaxnn
          · exact mul_le_mul hmax2 h2 g2nn hmaxnn
      _ = max w1 w2 * ((1 + Real.exp (-(c1 - c2) ^ 2 / (8 * v))) /
            Real.sqrt (2 * π * v)) := by ring
  · -- z is far from c2
    have h1 : gaussN (z - c2) v ≤
        Real.exp (-(c1 - c2) ^ 2 / (8 * v)) / Real.sqrt (2 * π * v) := by
      have := gaussN_le_of_far hv (by positivity : (0:ℝ) ≤ |c1 - c2| / 2) hfar
      rw [hexp8 c1 c2] at this
      exact this
    have h2 : gaussN (z - c1) v ≤ 1 / Real.sqrt (2 * π * v) := gaussN_le_max hv _
    have g1nn : 0 ≤ gaussN (z - c1) v := gaussN_nonneg _ hv
    have g2nn : 0 ≤ gaussN (z - c2) v := gaussN_nonneg _ hv
    calc w1 * gaussN (z - c1) v + w2 * gaussN (z - c2) v
        ≤ max w1 w2 * (1 / Real.sqrt (2 * π * v)) +
            max w1 w2 * (Real.exp (-(c1 - c2) ^ 2 / (8 * v)) / Real.sqrt (2 * π * v)) := by
          apply add_le_add
          · exact mul_le_mul hmax1 h2 g1nn hmaxnn
          · exact mul_le_mul hmax2 h1 g2nn hmaxnn
      _ = max w1 w2 * ((1 + Real.exp (-(c1 - c2) ^ 2 / (8 * v))) /
            Real.sqrt (2 * π * v)) := by ring

end
end B0ChainKernel
end FormalPRR


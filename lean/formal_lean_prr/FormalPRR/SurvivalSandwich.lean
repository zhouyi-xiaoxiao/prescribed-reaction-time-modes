/-
FormalPRR/SurvivalSandwich.lean — gap-closure item L2(c) (plan TH-1).

Source (paper, CNSNS rebuild): the mean-field survival sandwich.  With the
nonnegative cumulative exposure X_t = ∫₀ᵗ V(Z_s, R_s) ds, its mean Λ(t) = E X_t
and the survival S_B(t) = E e^{−B X_t}, for every B ≥ 0:
    e^{−BΛ} ≤ S_B ≤ e^{−BΛ} + (B²/2) Var X_t,
and for the basin mass between two times s < s' (exposures X ≤ X'),
    S_B(s) − S_B(s') ≥ e^{−BΛ(s)} − e^{−BΛ(s')} − (B²/2) Var X_{s'}.

Proof route (plan TH-1, "synth check"): lower bound = tangent line of exp at
BΛ (Jensen); upper bound = the scalar inequality, for y = B(x − Λ) ≥ −BΛ,
    e^{−y} − 1 + y ≤ (y²/2) e^{BΛ},
multiplied by e^{−BΛ}, then integrated.  The expectation versions are proved
here directly by integral monotonicity against the tangent-line and quadratic
majorants (no appeal to an abstract Jensen theorem is needed).

-- SCOPE NOTE: X is an arbitrary nonnegative integrable random variable on a
-- probability space.  Its identification with the model's time-integrated Doi
-- exposure, the identity S_B(s) − S_B(s') = P(s < T_R ≤ s'), the two-time
-- Gaussian (Mehler) formula for Var X_t, and the density-level covariance
-- expansion f − f₁ = −B² Cov(V_t, X_t) + O(B³) are NOT formalized here.
-/
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Probability.Moments.Variance

namespace FormalPRR
namespace Sandwich

open MeasureTheory ProbabilityTheory Set

/-! ### Scalar core -/

/-- For `y ≥ 0`: `e^{−y} − 1 + y ≤ y²/2` (internal helper). -/
lemma exp_neg_sub_le_sq {y : ℝ} (hy : 0 ≤ y) : Real.exp (-y) - 1 + y ≤ y ^ 2 / 2 := by
  have hQ := Real.quadratic_le_exp_of_nonneg hy
  have hQpos : 0 < 1 + y + y ^ 2 / 2 := by positivity
  have h1 : Real.exp (-y) ≤ 1 / (1 + y + y ^ 2 / 2) := by
    rw [Real.exp_neg, ← one_div]
    exact one_div_le_one_div_of_le hQpos hQ
  have h2 : 1 / (1 + y + y ^ 2 / 2) ≤ 1 - y + y ^ 2 / 2 := by
    rw [div_le_iff₀ hQpos]
    nlinarith [sq_nonneg (y ^ 2)]
  linarith

/-- For `a ≥ 0`: `e^{a} − 1 − a ≤ (a²/2) e^{a}` (internal helper; monotonicity of
`a²/2 − 1 + e^{−a}(1 + a)`, whose derivative is `a(1 − e^{−a}) ≥ 0`). -/
lemma exp_sub_one_sub_le {a : ℝ} (ha : 0 ≤ a) :
    Real.exp a - 1 - a ≤ a ^ 2 / 2 * Real.exp a := by
  have hd : ∀ x : ℝ, HasDerivAt (fun x : ℝ => x ^ 2 / 2 - 1 + Real.exp (-x) * (1 + x))
      (x - x * Real.exp (-x)) x := by
    intro x
    have h1 : HasDerivAt (fun x : ℝ => x ^ 2 / 2) x x :=
      ((hasDerivAt_pow 2 x).div_const 2).congr_deriv (by norm_num)
    have h2 : HasDerivAt (fun x : ℝ => Real.exp (-x)) (Real.exp (-x) * (-1)) x :=
      (hasDerivAt_neg x).exp
    have h3 : HasDerivAt (fun x : ℝ => 1 + x) 1 x := (hasDerivAt_id x).const_add 1
    exact ((h1.sub_const 1).add (h2.mul h3)).congr_deriv (by ring)
  have hmono : MonotoneOn (fun x : ℝ => x ^ 2 / 2 - 1 + Real.exp (-x) * (1 + x)) (Ici 0) := by
    apply monotoneOn_of_hasDerivWithinAt_nonneg (convex_Ici 0)
    · exact fun x _ => (hd x).continuousAt.continuousWithinAt
    · exact fun x _ => (hd x).hasDerivWithinAt
    · intro x hx
      rw [interior_Ici] at hx
      have : Real.exp (-x) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith [mem_Ioi.1 hx])
      nlinarith [mem_Ioi.1 hx]
  have hv := hmono self_mem_Ici ha ha
  simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_div, zero_sub,
    neg_zero, Real.exp_zero, add_zero, mul_one] at hv
  have he : Real.exp a * Real.exp (-a) = 1 := by rw [← Real.exp_add]; simp
  have hpos := Real.exp_pos a
  nlinarith [mul_le_mul_of_nonneg_left hv hpos.le]

/-- **Scalar core of the upper bound** (plan TH-1): if `K ≥ 0` and `y ≥ −K`, then
`e^{−y} − 1 + y ≤ (y²/2) e^{K}`. -/
theorem key_quadratic {y K : ℝ} (hK : 0 ≤ K) (hy : -K ≤ y) :
    Real.exp (-y) - 1 + y ≤ y ^ 2 / 2 * Real.exp K := by
  rcases le_or_gt 0 y with h | h
  · calc Real.exp (-y) - 1 + y ≤ y ^ 2 / 2 := exp_neg_sub_le_sq h
      _ ≤ y ^ 2 / 2 * Real.exp K :=
          le_mul_of_one_le_right (by positivity) (Real.one_le_exp hK)
  · have ha : 0 ≤ -y := by linarith
    have h1 := exp_sub_one_sub_le ha
    have h2 : Real.exp (-y) ≤ Real.exp K := Real.exp_le_exp.mpr (by linarith)
    have h3 : (-y) ^ 2 / 2 * Real.exp (-y) ≤ (-y) ^ 2 / 2 * Real.exp K :=
      mul_le_mul_of_nonneg_left h2 (by positivity)
    have h4 : (-y) ^ 2 = y ^ 2 := by ring
    rw [h4] at h3 h1
    linarith

/-- **Tangent-line (Jensen) inequality** (plan TH-1, lower bound, scalar form):
`e^{−BΛ}(1 − B(x − Λ)) ≤ e^{−Bx}` for all real `B, x, Λ`. -/
theorem tangent_lower (B x L : ℝ) :
    Real.exp (-(B * L)) * (1 - B * (x - L)) ≤ Real.exp (-(B * x)) := by
  have he : Real.exp (-(B * x)) = Real.exp (-(B * L)) * Real.exp (-(B * (x - L))) := by
    rw [← Real.exp_add]; congr 1; ring
  rw [he]
  apply mul_le_mul_of_nonneg_left _ (Real.exp_pos _).le
  linarith [Real.add_one_le_exp (-(B * (x - L)))]

/-- **Quadratic majorant** (plan TH-1, upper bound, scalar form): for `B, x, Λ ≥ 0`,
`e^{−Bx} ≤ e^{−BΛ}(1 − B(x − Λ)) + (B²/2)(x − Λ)²`. -/
theorem quad_upper {B x L : ℝ} (hB : 0 ≤ B) (hx : 0 ≤ x) (hL : 0 ≤ L) :
    Real.exp (-(B * x)) ≤ Real.exp (-(B * L)) * (1 - B * (x - L)) + B ^ 2 / 2 * (x - L) ^ 2 := by
  have hk := key_quadratic (y := B * (x - L)) (K := B * L) (mul_nonneg hB hL)
    (by nlinarith [mul_nonneg hB hx])
  have hmul := mul_le_mul_of_nonneg_left hk (Real.exp_pos (-(B * L))).le
  have e1 : Real.exp (-(B * L)) * Real.exp (-(B * (x - L))) = Real.exp (-(B * x)) := by
    rw [← Real.exp_add]; congr 1; ring
  have e2 : Real.exp (-(B * L)) * ((B * (x - L)) ^ 2 / 2 * Real.exp (B * L))
      = B ^ 2 / 2 * (x - L) ^ 2 := by
    have : Real.exp (-(B * L)) * Real.exp (B * L) = 1 := by rw [← Real.exp_add]; simp
    calc Real.exp (-(B * L)) * ((B * (x - L)) ^ 2 / 2 * Real.exp (B * L))
        = B ^ 2 / 2 * (x - L) ^ 2 * (Real.exp (-(B * L)) * Real.exp (B * L)) := by ring
      _ = B ^ 2 / 2 * (x - L) ^ 2 := by rw [this, mul_one]
  rw [e2] at hmul
  rw [← e1]
  nlinarith [hmul]

/-! ### Expectation form -/

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- `e^{−BX}` is integrable for `B ≥ 0`, `X ≥ 0` a.e. (it is bounded by one). -/
lemma integrable_exp_neg_mul {X : Ω → ℝ} (hXm : AEStronglyMeasurable X μ) {B : ℝ}
    (hB : 0 ≤ B) (hX0 : ∀ᵐ ω ∂μ, 0 ≤ X ω) :
    Integrable (fun ω => Real.exp (-(B * X ω))) μ := by
  refine Integrable.of_bound
    (Real.continuous_exp.comp_aestronglyMeasurable ((hXm.const_mul B).neg)) 1 ?_
  filter_upwards [hX0] with ω hω
  rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
  exact Real.exp_le_one_iff.mpr (by nlinarith)

/-- The tangent-line majorant integrates to `e^{−BΛ}` when `Λ = E X`. -/
lemma integral_tangent {X : Ω → ℝ} (hXi : Integrable X μ) (B : ℝ) :
    ∫ ω, Real.exp (-(B * ∫ ω', X ω' ∂μ)) * (1 - B * (X ω - ∫ ω', X ω' ∂μ)) ∂μ
      = Real.exp (-(B * ∫ ω', X ω' ∂μ)) := by
  set L := ∫ ω', X ω' ∂μ with hL
  have hc : Integrable (fun ω => B * (X ω - L)) μ := (hXi.sub (integrable_const L)).const_mul B
  rw [integral_const_mul, integral_sub (integrable_const 1) hc, integral_const_mul,
    integral_sub hXi (integrable_const L)]
  simp [← hL]

lemma integrable_tangent {X : Ω → ℝ} (hXi : Integrable X μ) (B L : ℝ) :
    Integrable (fun ω => Real.exp (-(B * L)) * (1 - B * (X ω - L))) μ :=
  ((integrable_const 1).sub ((hXi.sub (integrable_const L)).const_mul B)).const_mul _

/-- **Mean-field lower bound** (plan TH-1): `e^{−B E X} ≤ E e^{−BX}` for `B ≥ 0`
and `X ≥ 0` integrable. -/
theorem sandwich_lower {X : Ω → ℝ} (hXi : Integrable X μ) (hX0 : ∀ᵐ ω ∂μ, 0 ≤ X ω)
    {B : ℝ} (hB : 0 ≤ B) :
    Real.exp (-(B * ∫ ω, X ω ∂μ)) ≤ ∫ ω, Real.exp (-(B * X ω)) ∂μ := by
  rw [← integral_tangent hXi B]
  exact integral_mono (integrable_tangent hXi B _)
    (integrable_exp_neg_mul hXi.aestronglyMeasurable hB hX0)
    (fun ω => tangent_lower B (X ω) _)

/-- **Variance upper bound** (plan TH-1):
`E e^{−BX} ≤ e^{−B E X} + (B²/2) E (X − E X)²` for `B ≥ 0`, `X ≥ 0` integrable
with `(X − E X)²` integrable. -/
theorem sandwich_upper {X : Ω → ℝ} (hXi : Integrable X μ)
    (hX2 : Integrable (fun ω => (X ω - ∫ ω', X ω' ∂μ) ^ 2) μ)
    (hX0 : ∀ᵐ ω ∂μ, 0 ≤ X ω) {B : ℝ} (hB : 0 ≤ B) :
    ∫ ω, Real.exp (-(B * X ω)) ∂μ ≤
      Real.exp (-(B * ∫ ω, X ω ∂μ)) + B ^ 2 / 2 * ∫ ω, (X ω - ∫ ω', X ω' ∂μ) ^ 2 ∂μ := by
  set L := ∫ ω', X ω' ∂μ with hL
  have hL0 : 0 ≤ L := integral_nonneg_of_ae hX0
  have hmaj : Integrable
      (fun ω => Real.exp (-(B * L)) * (1 - B * (X ω - L)) + B ^ 2 / 2 * (X ω - L) ^ 2) μ :=
    (integrable_tangent hXi B L).add (hX2.const_mul _)
  have hle := integral_mono_ae (integrable_exp_neg_mul hXi.aestronglyMeasurable hB hX0) hmaj
    (by filter_upwards [hX0] with ω hω using quad_upper hB hω hL0)
  rw [integral_add (integrable_tangent hXi B L) (hX2.const_mul _), integral_tangent hXi B,
    integral_const_mul] at hle
  exact hle

/-- **Sandwich with `Var`** (plan TH-1): for `X ∈ L²`, `X ≥ 0`, `B ≥ 0`,
`e^{−B E X} ≤ E e^{−BX} ≤ e^{−B E X} + (B²/2) Var X`. -/
theorem sandwich_variance {X : Ω → ℝ} (hX : MemLp X 2 μ) (hX0 : ∀ᵐ ω ∂μ, 0 ≤ X ω)
    {B : ℝ} (hB : 0 ≤ B) :
    Real.exp (-(B * ∫ ω, X ω ∂μ)) ≤ ∫ ω, Real.exp (-(B * X ω)) ∂μ ∧
    ∫ ω, Real.exp (-(B * X ω)) ∂μ ≤ Real.exp (-(B * ∫ ω, X ω ∂μ)) + B ^ 2 / 2 * Var[X; μ] := by
  have hXi : Integrable X μ := hX.integrable one_le_two
  have hX2 : Integrable (fun ω => (X ω - ∫ ω', X ω' ∂μ) ^ 2) μ :=
    (hX.sub (memLp_const (∫ ω', X ω' ∂μ))).integrable_sq
  refine ⟨sandwich_lower hXi hX0 hB, ?_⟩
  rw [variance_eq_integral hX.aestronglyMeasurable.aemeasurable]
  exact sandwich_upper hXi hX2 hX0 hB

/-- **Explicit finite-budget event-mass floor** (plan TH-1): for exposures `X₁`
(up to the earlier time) and `X₂` (up to the later time), both nonnegative and
integrable with `(X₂ − E X₂)²` integrable,
`E e^{−BX₁} − E e^{−BX₂} ≥ e^{−B E X₁} − e^{−B E X₂} − (B²/2) E (X₂ − E X₂)²`.
With `S_B(s) = E e^{−B X_s}` the left side is the basin (or window) mass. -/
theorem event_mass_floor {X₁ X₂ : Ω → ℝ} (h1 : Integrable X₁ μ) (h10 : ∀ᵐ ω ∂μ, 0 ≤ X₁ ω)
    (h2 : Integrable X₂ μ) (h22 : Integrable (fun ω => (X₂ ω - ∫ ω', X₂ ω' ∂μ) ^ 2) μ)
    (h20 : ∀ᵐ ω ∂μ, 0 ≤ X₂ ω) {B : ℝ} (hB : 0 ≤ B) :
    Real.exp (-(B * ∫ ω, X₁ ω ∂μ)) - Real.exp (-(B * ∫ ω, X₂ ω ∂μ))
        - B ^ 2 / 2 * ∫ ω, (X₂ ω - ∫ ω', X₂ ω' ∂μ) ^ 2 ∂μ
      ≤ ∫ ω, Real.exp (-(B * X₁ ω)) ∂μ - ∫ ω, Real.exp (-(B * X₂ ω)) ∂μ := by
  have hl := sandwich_lower h1 h10 hB
  have hu := sandwich_upper h2 h22 h20 hB
  linarith

end Sandwich
end FormalPRR

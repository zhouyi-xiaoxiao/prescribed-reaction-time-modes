/-
  FormalPRR/NewtonContraction.lean  (COMPANION KERNEL - not a display of the
  present paper)

  Simplified-Newton / fold-transfer constants chain.  Anchor: the
  fold-transfer proof of the mirrored companion JCP supplement,
  tex_anchors/COMPANION_jcp_supplement.tex, "Step 2 (simplified Newton)"
  (approx. lines 1952-1983 of the mirror), where every constant below
  appears verbatim (16/15 and 8/7 Neumann bounds, contraction factor 2/7,
  alpha = (8 sqrt 2/7)||J^{-1}||eps, radius (7/5) alpha, displacement
  <= 2 sqrt 2 ||J^{-1}|| eps).  This module was originally drafted against
  the A5 target list; the A5 deliverable anchored to the present PRR
  submission is FormalPRR/B0ChainKernel.lean.

  Everything here is the real-arithmetic kernel of a quantitative implicit-function
  argument, abstracted away from the Banach-space operators.  We prove, sorry-free:

  * `neumann_inverse_bound`  (N1)  the scalar Neumann step  (1-x)⁻¹ ≤ 16/15 for x ≤ 1/16,
                                    together with 16/15 ≤ 8/7.
  * `contraction_factor`     (N2)  the Lipschitz constant of the Newton map is ≤ 2/7.
  * `fixed_point_displacement`(N3)  the geometric fixed-point displacement bound
                                    (7/5)α ≤ 2√2·Jinv·ε.
  * `self_map`               (N4)  the self-mapping identity of the closed ball.
-/
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic

-- SCOPE: companion kernel. These constants formalize the simplified-Newton
--  fold-transfer theorem of the companion JCP manuscript's supplement
--  (mirrored at tex_anchors/COMPANION_jcp_supplement.tex, fold-transfer
--  proof "Step 2 (simplified Newton)", approx. lines 1952-1983), NOT a
--  display in the exact-m modality submission this project accompanies.
--  Valid mathematics; not claimed as formalizing any equation of the
--  present paper.

namespace FormalPRR.Newton

/-! ### N1 : scalar Neumann bound -/

/-- **N1 (Neumann step).**  For `0 ≤ x ≤ 1/16` the scalar reciprocal `(1-x)⁻¹`
is at most `16/15`.  This is the scalar shadow of the Neumann series bound
`‖(I-P)⁻¹‖ ≤ 1/(1-‖P‖)` used to control the perturbed inverse. -/
theorem neumann_inverse_bound (x : ℝ) (_hx0 : 0 ≤ x) (hx : x ≤ 1 / 16) :
    (1 - x)⁻¹ ≤ 16 / 15 := by
  have h1 : (0 : ℝ) < 1 - x := by linarith
  rw [inv_eq_one_div, div_le_iff₀ h1]
  nlinarith

/-- `16/15 ≤ 8/7`, so the Neumann bound also gives `(1-x)⁻¹ ≤ 8/7`. -/
theorem sixteen_fifteenths_le_eight_sevenths : (16 : ℝ) / 15 ≤ 8 / 7 := by norm_num

/-- Combined form: `(1-x)⁻¹ ≤ 8/7` for `0 ≤ x ≤ 1/16`, i.e. the perturbed inverse
norm is ≤ `(8/7)·Jinv`. -/
theorem neumann_inverse_bound' (x : ℝ) (hx0 : 0 ≤ x) (hx : x ≤ 1 / 16) :
    (1 - x)⁻¹ ≤ 8 / 7 :=
  le_trans (neumann_inverse_bound x hx0 hx) sixteen_fifteenths_le_eight_sevenths

/-! ### N2 : contraction factor of the Newton map -/

/-- **N2 (contraction factor).**  With `σ > 0`, and the perturbation budgets
`ωhat ≤ σ/16` and `εbar ≤ σ/32`, the Lipschitz constant of the Newton map,
`(8/7)·(1/σ)·(2·ωhat + 4·εbar)`, is at most `2/7`.

The intermediate arithmetic is `2·ωhat + 4·εbar ≤ σ/4`, then
`(8/7)(1/σ)(σ/4) = 2/7`. -/
theorem contraction_factor (σ ωhat εbar : ℝ) (hσ : 0 < σ)
    (hω : ωhat ≤ σ / 16) (hε : εbar ≤ σ / 32) :
    (8 / 7) * (1 / σ) * (2 * ωhat + 4 * εbar) ≤ 2 / 7 := by
  have hsum : 2 * ωhat + 4 * εbar ≤ σ / 4 := by linarith
  have hfac : (0 : ℝ) ≤ (8 / 7) * (1 / σ) := by positivity
  have hstep : (8 / 7) * (1 / σ) * (2 * ωhat + 4 * εbar)
      ≤ (8 / 7) * (1 / σ) * (σ / 4) := by
    exact mul_le_mul_of_nonneg_left hsum hfac
  have heval : (8 / 7) * (1 / σ) * (σ / 4) = 2 / 7 := by
    field_simp; ring
  linarith [hstep, heval.le, heval.ge]

/-- At the extreme budgets the factor equals exactly `2/7`. -/
theorem contraction_factor_extreme (σ : ℝ) (hσ : 0 < σ) :
    (8 / 7) * (1 / σ) * (2 * (σ / 16) + 4 * (σ / 32)) = 2 / 7 := by
  field_simp
  ring

/-! ### N3 : fixed-point displacement bound -/

/-- `8√2/5 ≤ 2√2`.  Pure comparison of the geometric-series prefactors. -/
theorem sqrt2_prefactor_le : (8 * Real.sqrt 2 / 5) ≤ 2 * Real.sqrt 2 := by
  nlinarith [Real.sqrt_nonneg 2]

/-- **N3 (fixed-point displacement).**  With contraction factor `2/7` the geometric
sum multiplier is `1/(1-2/7) = 7/5`.  Starting from `α = (8√2/7)·Jinv·ε`, the total
displacement `‖q_h − q*‖ ≤ (7/5)α = (8√2/5)·Jinv·ε ≤ 2√2·Jinv·ε`. -/
theorem fixed_point_displacement (Jinv ε : ℝ) (hJ : 0 ≤ Jinv) (hε : 0 ≤ ε) :
    (7 / 5) * ((8 * Real.sqrt 2 / 7) * Jinv * ε) ≤ 2 * Real.sqrt 2 * Jinv * ε := by
  have hs : (0 : ℝ) ≤ Real.sqrt 2 := Real.sqrt_nonneg 2
  nlinarith [mul_nonneg (mul_nonneg hs hJ) hε, mul_nonneg hJ hε, hs]

/-- The geometric-series multiplier identity `1/(1 - 2/7) = 7/5`. -/
theorem geometric_multiplier : (1 : ℝ) / (1 - 2 / 7) = 7 / 5 := by norm_num

/-! ### N4 : self-mapping identity of the closed ball -/

/-- **N4 (self map).**  If a map is a `2/7`-contraction and the initial displacement
is `α`, then the ball radius `(7/5)α` is preserved:
`α + (2/7)·(7/5)·α = (7/5)·α`.  Pure identity, valid for every `α`. -/
theorem self_map (α : ℝ) : α + (2 / 7) * ((7 / 5) * α) = (7 / 5) * α := by ring

end FormalPRR.Newton

-- Axiom audit for the delivered theorems.
#print axioms FormalPRR.Newton.neumann_inverse_bound
#print axioms FormalPRR.Newton.neumann_inverse_bound'
#print axioms FormalPRR.Newton.contraction_factor
#print axioms FormalPRR.Newton.contraction_factor_extreme
#print axioms FormalPRR.Newton.fixed_point_displacement
#print axioms FormalPRR.Newton.sqrt2_prefactor_le
#print axioms FormalPRR.Newton.self_map

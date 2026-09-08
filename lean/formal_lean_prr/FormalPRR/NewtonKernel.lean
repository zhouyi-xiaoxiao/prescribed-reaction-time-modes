/-
FormalPRR/NewtonKernel.lean  (COMPANION KERNEL - not a display of the
present paper)

-- SCOPE NOTE: generic contraction-constant arithmetic and Banach
fixed-point kernel.  The numeric constants formalized here (Neumann margin
eta <= sigma/16, inverse bound (8/7)/sigma, contraction factor 2/7, residual
alpha = (8 sqrt 2/7) K eps, radius (7/5) alpha) belong to the fold-transfer
Newton argument of the RELATED JCP manuscript's supplement (companion
methodology), NOT to any display of the present PRR submission
(prr_submission/encounter_multimodal_prr_v2_supplement.tex,
exact_m_theorem_full_proof.tex, prr_assets/b0_quantitative_bound.tex).  The
true anchor is the fold-transfer proof of the mirrored companion
supplement, tex_anchors/COMPANION_jcp_supplement.tex, "Step 2 (simplified
Newton)" (approx. lines 1952-1983 of the mirror), where these constants
appear verbatim: the Neumann series gives
||DH_h(q_*)^{-1}|| <= (16/15)||J(q_*)^{-1}|| <= (8/7)||J(q_*)^{-1}||,
||D Phi(q)|| <= (8/7)(1/sigma)(2 omega-hat(r_0) + 4 eps-bar) <= 2/7,
alpha = (8 sqrt 2/7)||J(q_*)^{-1}|| eps_h, Phi maps B(q_*, (7/5) alpha)
into itself, and ||q_h - q_*|| <= (7/5) alpha
<= 2 sqrt 2 ||J(q_*)^{-1}|| eps_h.  This file is retained as a supporting
kernel only and must not be cited as encoding a display of the present
paper.  The A5 deliverable anchored to the present paper is
FormalPRR/B0ChainKernel.lean.

Contents:
  (i)   `neumann_inverse_bound`: eta <= sigma/16 gives
        1/(sigma - eta) <= (16/15)/sigma <= (8/7)/sigma (the Neumann-series
        bound behind ||DH_h(q_*)^{-1}|| <= (8/7)(1/sigma)).
  (ii)  `contraction_factor_le`: (8/7)(1/sigma)(2 omega + 4 eps-bar)
        <= (8/7)(1/sigma)(sigma/4) = 2/7.
  (iii) `alpha_div_arith`: alpha/(1 - 2/7) = (7/5) alpha = (8 sqrt 2/5) K eps
        <= 2 sqrt 2 K eps.
  STRETCH: `contraction_maps_closedBall` (self-mapping of the (7/5) alpha
  ball) and `newton_kernel_fixed_point` / `newton_kernel_root_bound`
  (existence, uniqueness and displacement bound of the zero, via mathlib's
  Banach fixed-point theorem `ContractingWith.exists_fixedPoint'`).
-/
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Topology.MetricSpace.Contracting

namespace FormalPRR
namespace NewtonKernel

noncomputable section
open Real Metric Set Function NNReal

/-! ## (i) Neumann inverse bound -/

/-- Companion-kernel Neumann bound ("Step 2 (simplified Newton)" of the
fold-transfer proof, tex_anchors/COMPANION_jcp_supplement.tex approx. lines
1952-1959; see file-header SCOPE NOTE): if the derivative perturbation
`eta` satisfies `eta <= sigma/16`, then
`1/(sigma - eta) <= (16/15)(1/sigma) <= (8/7)(1/sigma)`.
This is the scalar content of `||DH_h(q_*)^{-1}|| <= (8/7)(1/sigma)` via the
Neumann series `(sigma - eta)^{-1} = sigma^{-1} sum_k (eta/sigma)^k`. -/
theorem neumann_inverse_bound {sigma eta : ℝ} (hs : 0 < sigma) (_heta : 0 ≤ eta)
    (h : eta ≤ sigma / 16) :
    1 / (sigma - eta) ≤ (16 / 15) * (1 / sigma) ∧
      (16 / 15) * (1 / sigma) ≤ (8 / 7) * (1 / sigma) := by
  have hpos : 0 < sigma - eta := by linarith
  constructor
  · rw [show (16 / 15 : ℝ) * (1 / sigma) = (16 / 15) / sigma by ring,
      div_le_div_iff₀ hpos hs]
    linarith
  · have h1 : 0 ≤ 1 / sigma := by positivity
    nlinarith

/-! ## (ii) Contraction-factor arithmetic -/

/-- The exact evaluation `(8/7)(1/sigma)(sigma/4) = 2/7` (Step 2 display). -/
theorem inverse_times_quarter {sigma : ℝ} (hs : sigma ≠ 0) :
    (8 / 7) * (1 / sigma) * (sigma / 4) = 2 / 7 := by
  field_simp
  norm_num

/-- Companion-kernel contraction factor (the `||D Phi(q)|| <= 2/7` display
of "Step 2 (simplified Newton)", tex_anchors/COMPANION_jcp_supplement.tex
approx. lines 1960-1969; see file-header SCOPE NOTE): under the margin
hypotheses `2 eps-bar <= sigma/16` and `2 omega <= sigma/8`,
`(8/7)(1/sigma)(2 omega + 4 eps-bar) <= 2/7`. -/
theorem contraction_factor_le {sigma epsBar omega : ℝ} (hs : 0 < sigma)
    (_he : 0 ≤ epsBar) (_ho : 0 ≤ omega)
    (h1 : 2 * epsBar ≤ sigma / 16) (h2 : 2 * omega ≤ sigma / 8) :
    (8 / 7) * (1 / sigma) * (2 * omega + 4 * epsBar) ≤ 2 / 7 := by
  have key : 2 * omega + 4 * epsBar ≤ sigma / 4 := by linarith
  have h3 : (8 / 7) * (1 / sigma) * (2 * omega + 4 * epsBar) ≤
      (8 / 7) * (1 / sigma) * (sigma / 4) :=
    mul_le_mul_of_nonneg_left key (by positivity)
  have h4 := inverse_times_quarter hs.ne'
  linarith

/-! ## (iii) Fixed-point radius arithmetic -/

/-- Residual scale `alpha = (8 sqrt 2 / 7) K eps` (the
`alpha := (8 sqrt 2/7) ||J(q_*)^{-1}|| eps_h` of "Step 2 (simplified
Newton)", tex_anchors/COMPANION_jcp_supplement.tex approx. lines
1970-1972). -/
def alpha (K eps : ℝ) : ℝ := (8 * Real.sqrt 2 / 7) * K * eps

/-- Companion-kernel radius arithmetic (the displacement display of "Step 2
(simplified Newton)", tex_anchors/COMPANION_jcp_supplement.tex approx.
lines 1979-1983; see file-header SCOPE NOTE): with contraction factor `2/7`,
`alpha/(1 - 2/7) = (7/5) alpha = (8 sqrt 2/5) K eps <= 2 sqrt 2 K eps`. -/
theorem alpha_div_arith {K eps : ℝ} (hK : 0 ≤ K) (he : 0 ≤ eps) :
    alpha K eps / (1 - 2 / 7) = (7 / 5) * alpha K eps ∧
      (7 / 5) * alpha K eps = (8 * Real.sqrt 2 / 5) * K * eps ∧
      (8 * Real.sqrt 2 / 5) * K * eps ≤ 2 * Real.sqrt 2 * (K * eps) := by
  have h2 : 0 ≤ Real.sqrt 2 := Real.sqrt_nonneg 2
  refine ⟨by unfold alpha; ring, by unfold alpha; ring, ?_⟩
  nlinarith [mul_nonneg (mul_nonneg h2 hK) he]

/-! ## STRETCH: Banach fixed point on the (7/5) alpha ball -/

/-- Self-mapping of the Newton ball: if `Phi` is `2/7`-Lipschitz on the closed
ball of radius `(7/5) a` around the seed `x_0` and the first Newton residual
satisfies `dist (Phi x_0) x_0 <= a`, then `Phi` maps that ball into itself
(the "(7/5) alpha radius argument": `(2/7)(7/5) a + a = (7/5) a`). -/
theorem contraction_maps_closedBall {E : Type*} [MetricSpace E]
    (Phi : E → E) (x₀ : E) {a : ℝ} (ha : 0 ≤ a)
    (hlip : LipschitzOnWith (2 / 7 : ℝ≥0) Phi (closedBall x₀ ((7 / 5) * a)))
    (hres : dist (Phi x₀) x₀ ≤ a) :
    MapsTo Phi (closedBall x₀ ((7 / 5) * a)) (closedBall x₀ ((7 / 5) * a)) := by
  intro x hx
  have hR : (0 : ℝ) ≤ (7 / 5) * a := by linarith
  have hx' : dist x x₀ ≤ (7 / 5) * a := mem_closedBall.mp hx
  have hx₀ : x₀ ∈ closedBall x₀ ((7 / 5) * a) := mem_closedBall_self hR
  have hcoe : ((2 / 7 : ℝ≥0) : ℝ) = 2 / 7 := by norm_num
  have h1 : dist (Phi x) (Phi x₀) ≤ (2 / 7) * dist x x₀ := by
    have := hlip.dist_le_mul x hx x₀ hx₀
    rwa [hcoe] at this
  have : dist (Phi x) x₀ ≤ (7 / 5) * a := by
    calc dist (Phi x) x₀ ≤ dist (Phi x) (Phi x₀) + dist (Phi x₀) x₀ :=
          dist_triangle _ _ _
      _ ≤ (2 / 7) * ((7 / 5) * a) + a := by
          have h2 : (2 / 7) * dist x x₀ ≤ (2 / 7) * ((7 / 5) * a) := by linarith
          linarith
      _ = (7 / 5) * a := by ring
  exact mem_closedBall.mpr this

/-- STRETCH main theorem (companion kernel; the Banach fixed-point step of
"Step 2 (simplified Newton)" in the fold-transfer proof of
tex_anchors/COMPANION_jcp_supplement.tex, approx. lines 1952-1983 of the
mirror; see the file-header SCOPE NOTE).
If `Phi` is `2/7`-Lipschitz on the closed ball of radius `(7/5) a` around the
seed `x_0` of a complete metric space and `dist (Phi x_0) x_0 <= a`, then
`Phi` has a fixed point `x` in that ball, unique among fixed points in the
ball, with the displacement bound
`dist x x_0 <= dist (Phi x_0) x_0 / (1 - 2/7) = (7/5) dist (Phi x_0) x_0`. -/
theorem newton_kernel_fixed_point {E : Type*} [MetricSpace E] [CompleteSpace E]
    (Phi : E → E) (x₀ : E) {a : ℝ} (ha : 0 ≤ a)
    (hlip : LipschitzOnWith (2 / 7 : ℝ≥0) Phi (closedBall x₀ ((7 / 5) * a)))
    (hres : dist (Phi x₀) x₀ ≤ a) :
    ∃ x ∈ closedBall x₀ ((7 / 5) * a), IsFixedPt Phi x ∧
      dist x x₀ ≤ (7 / 5) * dist (Phi x₀) x₀ ∧
      ∀ y ∈ closedBall x₀ ((7 / 5) * a), IsFixedPt Phi y → y = x := by
  have hR : (0 : ℝ) ≤ (7 / 5) * a := by linarith
  set s : Set E := closedBall x₀ ((7 / 5) * a) with hs
  have hsc : IsComplete s := Metric.isClosed_closedBall.isComplete
  have hmaps : MapsTo Phi s s := contraction_maps_closedBall Phi x₀ ha hlip hres
  have hK1 : (2 / 7 : ℝ≥0) < 1 := by
    rw [← NNReal.coe_lt_coe]
    norm_num
  have hcontr : ContractingWith (2 / 7 : ℝ≥0) (hmaps.restrict Phi s s) :=
    ⟨hK1, hlip.mapsToRestrict hmaps⟩
  have hx₀s : x₀ ∈ s := mem_closedBall_self hR
  obtain ⟨x, hxs, hfix, -, -⟩ :=
    hcontr.exists_fixedPoint' hsc hmaps hx₀s (edist_ne_top x₀ (Phi x₀))
  have hcoe : ((2 / 7 : ℝ≥0) : ℝ) = 2 / 7 := by norm_num
  -- Lipschitz step between two points of the ball, in real distance
  have hlip' : ∀ u ∈ s, ∀ v ∈ s, dist (Phi u) (Phi v) ≤ (2 / 7) * dist u v := by
    intro u hu v hv
    have := hlip.dist_le_mul u hu v hv
    rwa [hcoe] at this
  refine ⟨x, hxs, hfix, ?_, ?_⟩
  · -- displacement: dist x x₀ = dist (Phi x) x₀ ≤ (2/7) dist x x₀ + dist (Phi x₀) x₀
    have h1 : dist (Phi x) (Phi x₀) ≤ (2 / 7) * dist x x₀ := hlip' x hxs x₀ hx₀s
    have h2 : dist x x₀ ≤ dist (Phi x) (Phi x₀) + dist (Phi x₀) x₀ := by
      calc dist x x₀ = dist (Phi x) x₀ := by rw [hfix.eq]
        _ ≤ dist (Phi x) (Phi x₀) + dist (Phi x₀) x₀ := dist_triangle _ _ _
    linarith
  · -- uniqueness among fixed points of the ball
    intro y hys hyfix
    have h1 : dist y x = dist (Phi y) (Phi x) := by rw [hyfix.eq, hfix.eq]
    have h2 : dist (Phi y) (Phi x) ≤ (2 / 7) * dist y x := hlip' y hys x hxs
    have h3 : dist y x ≤ 0 := by linarith
    exact dist_le_zero.mp h3

/-- Composite root bound: seeding with residual `alpha = (8 sqrt 2/7) K eps`
on the ball of radius `(7/5) alpha` yields a unique fixed point of the simplified-Newton map at displacement at
most `(7/5) alpha = (8 sqrt 2/5) K eps <= 2 sqrt 2 K eps` from the seed
(the assembled Banach conclusion of "Step 2 (simplified Newton)" in the
fold-transfer proof, tex_anchors/COMPANION_jcp_supplement.tex approx. lines
1970-1983; see the file-header SCOPE NOTE). -/
theorem newton_kernel_root_bound {E : Type*} [MetricSpace E] [CompleteSpace E]
    (Phi : E → E) (x₀ : E) {K eps : ℝ} (hK : 0 ≤ K) (he : 0 ≤ eps)
    (hlip : LipschitzOnWith (2 / 7 : ℝ≥0) Phi
      (closedBall x₀ ((7 / 5) * alpha K eps)))
    (hres : dist (Phi x₀) x₀ ≤ alpha K eps) :
    ∃ x ∈ closedBall x₀ ((7 / 5) * alpha K eps), IsFixedPt Phi x ∧
      dist x x₀ ≤ 2 * Real.sqrt 2 * (K * eps) ∧
      ∀ y ∈ closedBall x₀ ((7 / 5) * alpha K eps), IsFixedPt Phi y → y = x := by
  have hα : 0 ≤ alpha K eps := by
    unfold alpha
    have := Real.sqrt_nonneg 2
    positivity
  obtain ⟨x, hxs, hfix, hdist, huniq⟩ :=
    newton_kernel_fixed_point Phi x₀ hα hlip hres
  refine ⟨x, hxs, hfix, ?_, huniq⟩
  obtain ⟨-, harith2, harith3⟩ := alpha_div_arith hK he
  calc dist x x₀ ≤ (7 / 5) * dist (Phi x₀) x₀ := hdist
    _ ≤ (7 / 5) * alpha K eps := by linarith
    _ = (8 * Real.sqrt 2 / 5) * K * eps := harith2
    _ ≤ 2 * Real.sqrt 2 * (K * eps) := harith3

end
end NewtonKernel
end FormalPRR

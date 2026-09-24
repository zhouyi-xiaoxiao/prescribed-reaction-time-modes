/-
FormalPRR/PassageProfile.lean — gap-closure item L2(d) (optional; plan TH-3, GPT-6 A8).

Source (paper, CNSNS rebuild; theory scout theory_gpt6.md step 5): uniqueness and
nondegeneracy of the limiting local passage peak.  With the standard Gaussian
density φ, an antiderivative Φ (Φ' = φ) and β > 0, the frozen passage profile is
    p_β(u) = β φ(u) e^{−β Φ(u)},   p_β'(u) = −p_β(u) h_β(u),   h_β(u) = u + β φ(u),
and F_β = p_β ∗ φ_λ.  The file proves:
  (1) h_β has exactly one zero u₀, u₀ < 0, with h_β < 0 left of u₀ and > 0 right
      of it; hence p_β' has exactly one sign change (+ → −) at u₀, and
      (u − u₀) p_β'(u) < 0 for every u ≠ u₀;
  (2) identity (A8): at a critical point y of F (∫ p_β'(u) k(u) du = 0 with
      k(u) = φ_λ(y − u) > 0),
        ∫ p_β'(u) (u − y)/λ² k(u) du = λ^{−2} ∫ (u − u₀) p_β'(u) k(u) du < 0;
  (3) a continuous function whose zeros all have negative derivative has at most
      one zero; applied to F' this gives "at most one critical point of F, and it
      is a nondegenerate maximum".

-- SCOPE NOTE: the differentiation of F = p_β ∗ φ_λ under the integral sign is NOT
-- formalized: the formulas F'(y) = ∫ p_β'(u) φ_λ(y−u) du and
-- F''(y) = ∫ p_β'(u) (u−y)/λ² φ_λ(y−u) du enter as hypotheses, as does
-- integrability.  The kernel k is an arbitrary positive function (its Gaussian
-- form is not used).  Existence of a maximum of F_β (tail decay) and the C²_loc
-- convergence of the rescaled reaction-time density to the frozen profile
-- (plan TH-3) are NOT formalized.
-/
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Calculus.DerivativeTest
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic

namespace FormalPRR
namespace PassageProfile

open Set MeasureTheory Filter Topology

/-! ### (1) One sign change of p_β' -/

/-- Standard Gaussian density `φ(u) = (2π)^{−1/2} e^{−u²/2}`. -/
noncomputable def phi (u : ℝ) : ℝ := (Real.sqrt (2 * Real.pi))⁻¹ * Real.exp (-(u ^ 2) / 2)

lemma sqrt_two_pi_pos : 0 < Real.sqrt (2 * Real.pi) :=
  Real.sqrt_pos.2 (by positivity)

lemma phi_pos (u : ℝ) : 0 < phi u :=
  mul_pos (inv_pos.2 sqrt_two_pi_pos) (Real.exp_pos _)

lemma phi_le (u : ℝ) : phi u ≤ (Real.sqrt (2 * Real.pi))⁻¹ := by
  have : Real.exp (-(u ^ 2) / 2) ≤ 1 :=
    Real.exp_le_one_iff.2 (by nlinarith [sq_nonneg u])
  calc phi u ≤ (Real.sqrt (2 * Real.pi))⁻¹ * 1 :=
        mul_le_mul_of_nonneg_left this (inv_pos.2 sqrt_two_pi_pos).le
    _ = _ := mul_one _

lemma hasDerivAt_phi (u : ℝ) : HasDerivAt phi (-u * phi u) u := by
  have h1 : HasDerivAt (fun u : ℝ => -(u ^ 2) / 2) (-u) u :=
    (((hasDerivAt_pow 2 u).neg).div_const 2).congr_deriv (by norm_num; ring)
  have h2 := (h1.exp).const_mul (Real.sqrt (2 * Real.pi))⁻¹
  have hfun : phi = fun u => (Real.sqrt (2 * Real.pi))⁻¹ * Real.exp (-(u ^ 2) / 2) := rfl
  rw [hfun]
  exact h2.congr_deriv (by beta_reduce; ring)

/-- `h_β(u) = u + β φ(u)`, so that `p_β' = −p_β h_β`. -/
noncomputable def hfun (β u : ℝ) : ℝ := u + β * phi u

lemma hasDerivAt_hfun (β u : ℝ) : HasDerivAt (hfun β) (1 - β * u * phi u) u := by
  have := (hasDerivAt_id u).add ((hasDerivAt_phi u).const_mul β)
  exact this.congr_deriv (by simp; ring)

/-- **One sign change of `h_β`** (theory_gpt6.md step 5): for `β > 0` there is
`u₀ < 0` with `h_β(u₀) = 0`, `h_β < 0` on `(−∞, u₀)` and `h_β > 0` on `(u₀, ∞)`. -/
theorem hfun_one_sign_change {β : ℝ} (hβ : 0 < β) :
    ∃ u₀ < 0, hfun β u₀ = 0 ∧ (∀ u < u₀, hfun β u < 0) ∧ (∀ u, u₀ < u → 0 < hfun β u) := by
  set c := (Real.sqrt (2 * Real.pi))⁻¹ with hc
  have hc0 : 0 < c := inv_pos.2 sqrt_two_pi_pos
  have hcont : Continuous (hfun β) :=
    continuous_iff_continuousAt.2 fun u => (hasDerivAt_hfun β u).continuousAt
  -- strictly increasing on (−∞, 0]
  have hmono : StrictMonoOn (hfun β) (Iic 0) := by
    apply strictMonoOn_of_hasDerivWithinAt_pos (convex_Iic 0) hcont.continuousOn
    · exact fun u _ => (hasDerivAt_hfun β u).hasDerivWithinAt
    · intro u hu
      rw [interior_Iic] at hu
      have hu' : u < 0 := hu
      have := mul_pos (mul_pos hβ (neg_pos.2 hu')) (phi_pos u)
      nlinarith
  -- positive on [0, ∞)
  have hpos : ∀ u, 0 ≤ u → 0 < hfun β u := fun u hu => by
    have := mul_pos hβ (phi_pos u)
    simp only [hfun]; linarith
  -- a negative value at u₁ = −(β c + 1)
  set u₁ := -(β * c + 1) with hu₁
  have hu₁neg : u₁ < 0 := by rw [hu₁]; nlinarith
  have hneg : hfun β u₁ < 0 := by
    have : β * phi u₁ ≤ β * c := mul_le_mul_of_nonneg_left (phi_le u₁) hβ.le
    simp only [hfun]; rw [hu₁] at this ⊢; linarith
  obtain ⟨u₀, hu₀mem, hu₀⟩ :=
    intermediate_value_Icc hu₁neg.le hcont.continuousOn ⟨hneg.le, (hpos 0 le_rfl).le⟩
  have hu₀neg : u₀ < 0 := by
    rcases eq_or_lt_of_le hu₀mem.2 with h | h
    · exact absurd hu₀ (by rw [h]; exact (hpos 0 le_rfl).ne')
    · exact h
  refine ⟨u₀, hu₀neg, hu₀, fun u hu => ?_, fun u hu => ?_⟩
  · have := hmono (mem_Iic.2 (by linarith)) (mem_Iic.2 hu₀neg.le) hu
    rwa [hu₀] at this
  · rcases le_or_gt u 0 with h | h
    · have := hmono (mem_Iic.2 hu₀neg.le) (mem_Iic.2 h) hu
      rwa [hu₀] at this
    · exact hpos u h.le

/-- Frozen passage profile `p_β(u) = β φ(u) e^{−β Φ(u)}`. -/
noncomputable def pbeta (β : ℝ) (Φ : ℝ → ℝ) (u : ℝ) : ℝ := β * phi u * Real.exp (-(β * Φ u))

lemma pbeta_pos {β : ℝ} (hβ : 0 < β) (Φ : ℝ → ℝ) (u : ℝ) : 0 < pbeta β Φ u :=
  mul_pos (mul_pos hβ (phi_pos u)) (Real.exp_pos _)

/-- **Derivative of the frozen profile**: `p_β' = −p_β h_β` (theory_gpt6.md step 5). -/
theorem hasDerivAt_pbeta (β : ℝ) {Φ : ℝ → ℝ} {u : ℝ} (hΦ : HasDerivAt Φ (phi u) u) :
    HasDerivAt (pbeta β Φ) (-(pbeta β Φ u) * hfun β u) u := by
  have h1 := ((hasDerivAt_phi u).const_mul β).mul ((hΦ.const_mul β).neg.exp)
  have hfun' : pbeta β Φ = fun u => β * phi u * Real.exp (-(β * Φ u)) := rfl
  rw [hfun']
  exact h1.congr_deriv (by simp only [hfun, Pi.neg_apply]; ring)

/-- **Pointwise sign of the curvature integrand** (A8): with `u₀` the zero of `h_β`,
`(u − u₀) p_β'(u) < 0` for every `u ≠ u₀`. -/
theorem curvature_integrand_neg {β : ℝ} (hβ : 0 < β) (Φ : ℝ → ℝ) {u₀ : ℝ}
    (hl : ∀ u < u₀, hfun β u < 0) (hr : ∀ u, u₀ < u → 0 < hfun β u) {u : ℝ} (hu : u ≠ u₀) :
    (u - u₀) * (-(pbeta β Φ u) * hfun β u) < 0 := by
  have hp := pbeta_pos hβ Φ u
  rcases lt_or_gt_of_ne hu with h | h
  · have h1 := hl u h
    have : 0 < (u - u₀) * hfun β u := mul_pos_of_neg_of_neg (by linarith) h1
    nlinarith
  · have h1 := hr u h
    have : 0 < (u - u₀) * hfun β u := mul_pos (by linarith) h1
    nlinarith

/-! ### (2) Identity (A8) and negative curvature at every critical point -/

/-- **Identity (A8)**: at a critical point (`∫ dp·k = 0`),
`∫ dp(u) (u − y)/λ² k(u) du = λ^{−2} ∫ (u − u₀) dp(u) k(u) du` for any `u₀`. -/
theorem curvature_identity {dp k : ℝ → ℝ} (u₀ y lam : ℝ)
    (hI1 : Integrable (fun u => dp u * k u)) (hI2 : Integrable (fun u => u * (dp u * k u)))
    (hcrit : ∫ u, dp u * k u = 0) :
    ∫ u, dp u * ((u - y) / lam ^ 2 * k u) = (lam ^ 2)⁻¹ * ∫ u, (u - u₀) * dp u * k u := by
  have hI3 : Integrable (fun u => (u - u₀) * dp u * k u) := by
    refine (hI2.sub (hI1.const_mul u₀)).congr (Eventually.of_forall fun u => ?_)
    simp only [Pi.sub_apply]
    ring
  calc ∫ u, dp u * ((u - y) / lam ^ 2 * k u)
      = ∫ u, ((lam ^ 2)⁻¹ * ((u - u₀) * dp u * k u)
          + ((lam ^ 2)⁻¹ * (u₀ - y)) * (dp u * k u)) := by
        refine integral_congr_ae (Eventually.of_forall fun u => ?_)
        simp only
        ring
    _ = (lam ^ 2)⁻¹ * (∫ u, (u - u₀) * dp u * k u)
          + ((lam ^ 2)⁻¹ * (u₀ - y)) * (∫ u, dp u * k u) := by
        rw [integral_add (hI3.const_mul _) (hI1.const_mul _), integral_const_mul,
          integral_const_mul]
    _ = (lam ^ 2)⁻¹ * ∫ u, (u - u₀) * dp u * k u := by rw [hcrit, mul_zero, add_zero]

/-- A strictly negative integrand off one point has a strictly negative integral
(Lebesgue measure; internal helper for A8). -/
theorem integral_neg_of_neg_off_point {g : ℝ → ℝ} {u₀ : ℝ} (hg : Integrable g)
    (hneg : ∀ u, u ≠ u₀ → g u < 0) (hle : ∀ u, g u ≤ 0) : ∫ u, g u < 0 := by
  have hnn : 0 ≤ fun u => -g u := fun u => by simp only [Pi.zero_apply]; linarith [hle u]
  have hpos : 0 < ∫ u, -g u := by
    rw [integral_pos_iff_support_of_nonneg hnn hg.neg]
    have hsub : Ioi u₀ ⊆ Function.support (fun u => -g u) := by
      intro u hu
      rw [Function.mem_support]
      have := hneg u (ne_of_gt hu)
      linarith
    calc (0 : ENNReal) < volume (Ioi u₀) := by simp [Real.volume_Ioi]
      _ ≤ _ := measure_mono hsub
  rw [integral_neg] at hpos
  linarith

/-- **Negative curvature at every critical point (A8)**: for `β > 0` and any positive
kernel `k` (standing for `φ_λ(y − ·)`), if `F'(y) = ∫ p_β' k = 0` and
`F''(y) = ∫ p_β'(u) (u − y)/λ² k(u) du`, then `F''(y) < 0`. -/
theorem A8_curvature_neg {β lam y : ℝ} (hβ : 0 < β) (hlam : lam ≠ 0) {Φ : ℝ → ℝ}
    (hΦ : ∀ u, HasDerivAt Φ (phi u) u) {k : ℝ → ℝ} (hk : ∀ u, 0 < k u)
    (hI1 : Integrable (fun u => deriv (pbeta β Φ) u * k u))
    (hI2 : Integrable (fun u => u * (deriv (pbeta β Φ) u * k u)))
    {F1 F2 : ℝ} (hF1 : F1 = ∫ u, deriv (pbeta β Φ) u * k u) (hcrit : F1 = 0)
    (hF2 : F2 = ∫ u, deriv (pbeta β Φ) u * ((u - y) / lam ^ 2 * k u)) :
    F2 < 0 := by
  obtain ⟨u₀, -, -, hl, hr⟩ := hfun_one_sign_change hβ
  have hdp : ∀ u, deriv (pbeta β Φ) u = -(pbeta β Φ u) * hfun β u :=
    fun u => (hasDerivAt_pbeta β (hΦ u)).deriv
  have hid := curvature_identity u₀ y lam hI1 hI2 (hcrit ▸ hF1).symm
  have hI3 : Integrable (fun u => (u - u₀) * deriv (pbeta β Φ) u * k u) := by
    refine (hI2.sub (hI1.const_mul u₀)).congr (Eventually.of_forall fun u => ?_)
    simp only [Pi.sub_apply]
    ring
  have hneg : ∫ u, (u - u₀) * deriv (pbeta β Φ) u * k u < 0 := by
    refine integral_neg_of_neg_off_point (u₀ := u₀) hI3 (fun u hu => ?_) (fun u => ?_)
    · rw [hdp u]
      exact mul_neg_of_neg_of_pos (curvature_integrand_neg hβ Φ hl hr hu) (hk u)
    · rcases eq_or_ne u u₀ with h | h
      · simp [h]
      · rw [hdp u]
        exact (mul_neg_of_neg_of_pos (curvature_integrand_neg hβ Φ hl hr h) (hk u)).le
  rw [hF2, hid]
  exact mul_neg_of_pos_of_neg (inv_pos.2 (pow_pos (lt_of_le_of_ne (sq_nonneg lam)
    (Ne.symm (pow_ne_zero 2 hlam))) 1 |>.trans_eq (pow_one _))) hneg

/-! ### (3) Downcrossing zeros are unique -/

/-- **At most one zero** (plan TH-3, "there can be only one"): if `g` is continuous
and every zero `z` of `g` has `g'(z) < 0`, then `g` has at most one zero.  Applied to
`g = F'` (every critical point has `F'' < 0`) this says `F` has at most one
critical point, necessarily a nondegenerate local maximum. -/
theorem zero_unique_of_downcrossing {g : ℝ → ℝ} (hc : Continuous g)
    (hdown : ∀ z, g z = 0 → deriv g z < 0) {z₁ z₂ : ℝ} (h1 : g z₁ = 0) (h2 : g z₂ = 0) :
    z₁ = z₂ := by
  -- reduce to z₁ < z₂ impossible
  suffices key : ∀ a b, g a = 0 → g b = 0 → a < b → False by
    rcases lt_trichotomy z₁ z₂ with h | h | h
    · exact (key _ _ h1 h2 h).elim
    · exact h
    · exact (key _ _ h2 h1 h).elim
  intro a b ha hb hab
  -- g < 0 just right of a
  have hsa := eventually_nhdsWithin_sign_eq_of_deriv_neg (hdown a ha) ha
  obtain ⟨δ₁, hδ₁, hball₁⟩ := Metric.eventually_nhds_iff.1 hsa
  set s := a + min δ₁ (b - a) / 2 with hs
  have hmin_pos : 0 < min δ₁ (b - a) := lt_min hδ₁ (by linarith)
  have has : a < s := by rw [hs]; linarith
  have hsb : s < b := by rw [hs]; linarith [min_le_right δ₁ (b - a)]
  have hgs : g s < 0 := by
    have hd : dist s a < δ₁ := by
      rw [Real.dist_eq, abs_of_pos (by linarith)]
      rw [hs]; linarith [min_le_left δ₁ (b - a)]
    have := hball₁ hd
    rw [sign_neg (by linarith : a - s < 0)] at this
    exact sign_eq_neg_one_iff.1 this
  -- first zero after s
  set Z := Icc s b ∩ g ⁻¹' {0} with hZ
  have hZc : IsClosed Z := isClosed_Icc.inter (isClosed_singleton.preimage hc)
  have hZne : Z.Nonempty := ⟨b, ⟨hsb.le, le_rfl⟩, hb⟩
  have hZbdd : BddBelow Z := ⟨s, fun t ht => ht.1.1⟩
  set z := sInf Z with hz
  have hzZ : z ∈ Z := hZc.csInf_mem hZne hZbdd
  have hgz : g z = 0 := hzZ.2
  have hsz : s < z := by
    rcases eq_or_lt_of_le hzZ.1.1 with h | h
    · rw [← h] at hgz; linarith
    · exact h
  -- g < 0 on [s, z)
  have hbelow : ∀ t, s ≤ t → t < z → g t < 0 := by
    intro t hst htz
    by_contra hge
    have hge : 0 ≤ g t := not_lt.1 hge
    obtain ⟨r, hr, hgr⟩ := intermediate_value_Icc hst hc.continuousOn ⟨hgs.le, hge⟩
    have hrZ : r ∈ Z := ⟨⟨hr.1, le_trans hr.2 (le_trans htz.le hzZ.1.2)⟩, hgr⟩
    have := csInf_le hZbdd hrZ
    linarith [hr.2]
  -- but g > 0 just left of z
  have hsz' := eventually_nhdsWithin_sign_eq_of_deriv_neg (hdown z hgz) hgz
  obtain ⟨δ₂, hδ₂, hball₂⟩ := Metric.eventually_nhds_iff.1 hsz'
  set x := z - min δ₂ (z - s) / 2 with hx
  have hmin2 : 0 < min δ₂ (z - s) := lt_min hδ₂ (by linarith)
  have hxz : x < z := by rw [hx]; linarith
  have hsx : s ≤ x := by rw [hx]; linarith [min_le_right δ₂ (z - s)]
  have hd : dist x z < δ₂ := by
    rw [Real.dist_eq, abs_of_neg (by linarith)]
    rw [hx]; linarith [min_le_left δ₂ (z - s)]
  have := hball₂ hd
  rw [sign_pos (by linarith : 0 < z - x)] at this
  have hpos := sign_eq_one_iff.1 this
  linarith [hbelow x hsx hxz]

end PassageProfile
end FormalPRR

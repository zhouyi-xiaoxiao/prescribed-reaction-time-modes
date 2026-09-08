/-
  FormalPRR/SigmaBound.lean  (COMPANION KERNEL - not a display of the
  present paper)

  Seed-conditioning 2×2 smallest-singular-value lower bound.  Anchor: the
  fold-transfer proof of the mirrored companion JCP supplement,
  tex_anchors/COMPANION_jcp_supplement.tex, "Step 0 (seed conditioning)"
  (approx. lines 1934-1943 of the mirror):
  sigma = sigma_min(J(q_*)) = |det J(q_*)|/sigma_max(J(q_*))
  >= mu_3 mu_theta / sqrt(mu_theta^2 + 2 Lambda^2), "using
  sigma_max <= ||J||_F ... and monotonicity of
  x -> x/sqrt(x^2 + 2 Lambda^2)".  This module was originally drafted
  against the A4 target list; A4 is a JCP-companion target (see
  FORMALIZATION_TARGETS.md).

  For the 2×2 Jacobian at the fold, `J = !![Ftt, Ftθ; Fttt, Fttθ]`, with the fold
  conditions abstracted:  `Ftt = 0` (fold), `|Ftθ| ≥ μθ > 0`, `|Fttt| ≥ μ3 > 0`, and the
  off-fold entries bounded by `Λ`, we prove a lower bound on the smallest singular value.

  Faithful 2×2 singular-value facts we use (both are classical identities, cited here and
  substantiated by the algebra below):
    σ_max(J) ≤ ‖J‖_F            (spectral norm ≤ Frobenius norm),
    σ_min(J) · σ_max(J) = |det J|   (product of singular values = |det|).
  Hence  σ_min(J) = |det J| / σ_max(J) ≥ |det J| / ‖J‖_F, and we bound the last (fully
  computable) quantity from below.  `spectral_le_frobenius_sq` proves the pointwise
  inequality `‖Jv‖² ≤ ‖J‖_F²‖v‖²` that underlies `σ_max ≤ ‖J‖_F`.

  Delivered sorry-free:
  * `spectral_le_frobenius_sq` (S1 core)  ‖Jv‖² ≤ ‖J‖_F² · ‖v‖².
  * `frob_fold_bound`         (S1)        ‖J‖_F ≤ √(Ftθ² + 2Λ²) at the fold.
  * `ratio_mono`              (key lemma)  x ↦ x/√(x²+c) is monotone on x ≥ 0 (c > 0).
  * `sigma_min_lower`         (S2)        |det J|/‖J‖_F ≥ μ3·μθ/√(μθ² + 2Λ²).
-/
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic

-- SCOPE: companion kernel. This bound formalizes "Step 0 (seed
--  conditioning)" of the fold-transfer theorem in the companion JCP
--  manuscript's supplement (mirrored at
--  tex_anchors/COMPANION_jcp_supplement.tex, approx. lines 1934-1943), NOT
--  a display in the exact-m modality submission this project accompanies.
--  Valid mathematics; not claimed as formalizing any equation of the
--  present paper.

namespace FormalPRR.Sigma

/-! ### S1 core : spectral norm ≤ Frobenius norm (pointwise) -/

/-- **S1 core.**  For `J = !![Ftt, Ftθ; Fttt, Fttθ]` and any vector `v = (v1, v2)`,
`‖J v‖² ≤ ‖J‖_F² · ‖v‖²`.  This is the Cauchy–Schwarz sum-of-squares identity
`‖J‖_F²‖v‖² − ‖Jv‖² = (Ftt·v2 − Ftθ·v1)² + (Fttt·v2 − Fttθ·v1)² ≥ 0`, and it is exactly
what yields `σ_max(J) ≤ ‖J‖_F`. -/
theorem spectral_le_frobenius_sq (Ftt Ftθ Fttt Fttθ v1 v2 : ℝ) :
    (Ftt * v1 + Ftθ * v2) ^ 2 + (Fttt * v1 + Fttθ * v2) ^ 2
      ≤ (Ftt ^ 2 + Ftθ ^ 2 + Fttt ^ 2 + Fttθ ^ 2) * (v1 ^ 2 + v2 ^ 2) := by
  nlinarith [sq_nonneg (Ftt * v2 - Ftθ * v1), sq_nonneg (Fttt * v2 - Fttθ * v1)]

/-! ### S1 : Frobenius norm at the fold -/

/-- **S1.**  At the fold (`Ftt = 0`), with off-fold entries bounded, the Frobenius norm
`‖J‖_F = √(Ftt² + Ftθ² + Fttt² + Fttθ²)` is at most `√(Ftθ² + 2Λ²)`. -/
theorem frob_fold_bound (Ftt Ftθ Fttt Fttθ Λ : ℝ)
    (hfold : Ftt = 0) (h3hi : |Fttt| ≤ Λ) (htθhi : |Fttθ| ≤ Λ) :
    Real.sqrt (Ftt ^ 2 + Ftθ ^ 2 + Fttt ^ 2 + Fttθ ^ 2) ≤ Real.sqrt (Ftθ ^ 2 + 2 * Λ ^ 2) := by
  apply Real.sqrt_le_sqrt
  have h3 := abs_le.mp h3hi
  have htθ := abs_le.mp htθhi
  have hFttt2 : Fttt ^ 2 ≤ Λ ^ 2 := sq_le_sq' h3.1 h3.2
  have hFttθ2 : Fttθ ^ 2 ≤ Λ ^ 2 := sq_le_sq' htθ.1 htθ.2
  rw [hfold]; nlinarith [hFttt2, hFttθ2]

/-! ### Key monotonicity lemma -/

/-- **Key lemma.**  For `c > 0`, the map `x ↦ x/√(x² + c)` is monotone increasing on
`x ≥ 0`.  Proof: cross-multiply (both denominators positive), rewrite each side as a
single square root, and compare radicands: `x²(y²+c) ≤ y²(x²+c) ⇔ c·x² ≤ c·y²`. -/
theorem ratio_mono {c x y : ℝ} (hc : 0 < c) (hx : 0 ≤ x) (hxy : x ≤ y) :
    x / Real.sqrt (x ^ 2 + c) ≤ y / Real.sqrt (y ^ 2 + c) := by
  have hy : 0 ≤ y := le_trans hx hxy
  have hdx : 0 < Real.sqrt (x ^ 2 + c) := Real.sqrt_pos.mpr (by positivity)
  have hdy : 0 < Real.sqrt (y ^ 2 + c) := Real.sqrt_pos.mpr (by positivity)
  rw [div_le_div_iff₀ hdx hdy]
  have e1 : x * Real.sqrt (y ^ 2 + c) = Real.sqrt (x ^ 2 * (y ^ 2 + c)) := by
    rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hx]
  have e2 : y * Real.sqrt (x ^ 2 + c) = Real.sqrt (y ^ 2 * (x ^ 2 + c)) := by
    rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hy]
  rw [e1, e2]
  apply Real.sqrt_le_sqrt
  have hxy2 : x ^ 2 ≤ y ^ 2 := by nlinarith [hx, hy, hxy]
  nlinarith [mul_le_mul_of_nonneg_left hxy2 hc.le]

/-! ### S2 : smallest-singular-value lower bound -/

/-- **S2 (smallest-singular-value lower bound).**  At the fold, the fully computable
proxy `|det J| / ‖J‖_F` — which is a rigorous lower bound for `σ_min(J)` since
`σ_min = |det|/σ_max ≥ |det|/‖J‖_F` — satisfies
`|det J| / ‖J‖_F ≥ μ3·μθ / √(μθ² + 2Λ²)`.

Chain: `det J = −Ftθ·Fttt` at the fold, so `|det J| = |Ftθ|·|Fttt| ≥ |Ftθ|·μ3`; the
denominator is bounded by `√(|Ftθ|² + 2Λ²)`; the resulting `|Ftθ|·μ3/√(|Ftθ|²+2Λ²)`
is increasing in `|Ftθ| ≥ μθ` by `ratio_mono`. -/
theorem sigma_min_lower (Ftt Ftθ Fttt Fttθ μθ μ3 Λ : ℝ)
    (hfold : Ftt = 0)
    (hμθ : 0 < μθ) (hμ3 : 0 < μ3)
    (hθlo : μθ ≤ |Ftθ|) (h3lo : μ3 ≤ |Fttt|)
    (h3hi : |Fttt| ≤ Λ) (htθhi : |Fttθ| ≤ Λ) :
    μ3 * μθ / Real.sqrt (μθ ^ 2 + 2 * Λ ^ 2)
      ≤ |Ftt * Fttθ - Ftθ * Fttt| / Real.sqrt (Ftt ^ 2 + Ftθ ^ 2 + Fttt ^ 2 + Fttθ ^ 2) := by
  -- Basic positivity facts.
  have hΛ : 0 < Λ := lt_of_lt_of_le hμ3 (le_trans h3lo h3hi)
  have hΛ2 : (0 : ℝ) < Λ ^ 2 := pow_pos hΛ 2
  have hc : (0 : ℝ) < 2 * Λ ^ 2 := by linarith
  set X := |Ftθ| with hXdef
  have hXpos : 0 < X := lt_of_lt_of_le hμθ hθlo
  have hX2 : X ^ 2 = Ftθ ^ 2 := sq_abs Ftθ
  -- Rewrite the goal's numerator and denominator at the fold.
  have hnum : |Ftt * Fttθ - Ftθ * Fttt| = X * |Fttt| := by
    rw [hfold, hXdef]; rw [zero_mul, zero_sub, abs_neg, abs_mul]
  have hden : Ftt ^ 2 + Ftθ ^ 2 + Fttt ^ 2 + Fttθ ^ 2 = X ^ 2 + Fttt ^ 2 + Fttθ ^ 2 := by
    rw [hfold, hX2]; ring
  rw [hnum, hden]
  -- Positivity of the two denominators appearing after the rewrite.
  have hQin : 0 < X ^ 2 + Fttt ^ 2 + Fttθ ^ 2 := by
    have := pow_pos hXpos 2; nlinarith [sq_nonneg Fttt, sq_nonneg Fttθ]
  have hQpos : 0 < Real.sqrt (X ^ 2 + Fttt ^ 2 + Fttθ ^ 2) := Real.sqrt_pos.mpr hQin
  -- Frobenius bound: √(X²+Fttt²+Fttθ²) ≤ √(X²+2Λ²).
  have h3 := abs_le.mp h3hi
  have htθ := abs_le.mp htθhi
  have hFttt2 : Fttt ^ 2 ≤ Λ ^ 2 := sq_le_sq' h3.1 h3.2
  have hFttθ2 : Fttθ ^ 2 ≤ Λ ^ 2 := sq_le_sq' htθ.1 htθ.2
  have hfrob : Real.sqrt (X ^ 2 + Fttt ^ 2 + Fttθ ^ 2) ≤ Real.sqrt (X ^ 2 + 2 * Λ ^ 2) :=
    Real.sqrt_le_sqrt (by nlinarith [hFttt2, hFttθ2])
  -- The main chain.
  calc
    μ3 * μθ / Real.sqrt (μθ ^ 2 + 2 * Λ ^ 2)
        = μ3 * (μθ / Real.sqrt (μθ ^ 2 + 2 * Λ ^ 2)) := by ring
    _ ≤ μ3 * (X / Real.sqrt (X ^ 2 + 2 * Λ ^ 2)) :=
          mul_le_mul_of_nonneg_left (ratio_mono hc hμθ.le hθlo) hμ3.le
    _ = (X * μ3) / Real.sqrt (X ^ 2 + 2 * Λ ^ 2) := by ring
    _ ≤ (X * |Fttt|) / Real.sqrt (X ^ 2 + Fttt ^ 2 + Fttθ ^ 2) :=
          div_le_div₀ (mul_nonneg hXpos.le (abs_nonneg _))
            (mul_le_mul_of_nonneg_left h3lo hXpos.le) hQpos hfrob

#print axioms FormalPRR.Sigma.spectral_le_frobenius_sq
#print axioms FormalPRR.Sigma.frob_fold_bound
#print axioms FormalPRR.Sigma.ratio_mono
#print axioms FormalPRR.Sigma.sigma_min_lower

end FormalPRR.Sigma

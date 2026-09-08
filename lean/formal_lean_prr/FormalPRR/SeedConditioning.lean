/-
FormalPRR/SeedConditioning.lean  (target A4: seed-conditioning 2x2 bound)

-- SCOPE NOTE (COMPANION KERNEL - not a display of the present paper):
  generic 2x2 singular-value kernel.  The seed-conditioning bound
  sigma_min(J) >= mu_3 mu_theta / sqrt(mu_theta^2 + 2 Lam^2) formalized below
  is companion methodology (the fold-transfer theory of the RELATED JCP
  manuscript's supplement, where the fold-slice Jacobian J(q_*) has vanishing
  (1,1) entry F_tt(q_*) = 0); it is NOT a display of the present PRR
  submission (prr_submission/encounter_multimodal_prr_v2_supplement.tex,
  exact_m_theorem_full_proof.tex, prr_assets/b0_quantitative_bound.tex).
  The true anchor is the fold-transfer proof of the mirrored companion
  supplement, tex_anchors/COMPANION_jcp_supplement.tex, "Step 0 (seed
  conditioning)" (approx. lines 1934-1943 of the mirror): at q_*, F_tt = 0
  and sigma = sigma_min(J(q_*)) = |det J(q_*)|/sigma_max(J(q_*))
  >= mu_3 mu_theta / sqrt(mu_theta^2 + 2 Lambda^2), "using
  sigma_max <= ||J||_F ... and monotonicity of
  x -> x/sqrt(x^2 + 2 Lambda^2)" - exactly the chain proved below.  This
  file is retained as a supporting kernel only (the hand-rolled 2x2
  singular values and `singularValues_unique` are used as a self-contained
  faithfulness certificate) and must not be cited as encoding a display of
  the present paper.

  Statement formalized (companion): for a real 2x2 matrix
  J = [[p, q], [r, s]] with p = 0, |r| >= mu_3, |q| >= mu_theta, |r| <= Lam,
  |s| <= Lam one has  sigma_min(J) >= mu_3 mu_theta / sqrt(mu_theta^2 + 2 Lam^2),
  via sigma_min = |det J| / sigma_max, sigma_max <= ||J||_F, and monotonicity
  of x -> x / sqrt(x^2 + 2 Lam^2).

Singular values of a 2x2 real matrix are hand-rolled through the closed form
in the Frobenius norm and determinant (as FORMALIZATION_TARGETS.md prescribes:
"define sigma_min/sigma_max concretely via det and Frobenius norm; that is
faithful for 2x2").  Faithfulness is certified by `singularValues_unique`:
any pair 0 <= a <= b with a^2 + b^2 = ||J||_F^2 and a b = |det J| - the two
defining symmetric functions of the 2x2 singular-value pair - coincides with
(sigmaMin, sigmaMax) defined here.
-/
import Mathlib.Analysis.Real.Sqrt

namespace FormalPRR
namespace SeedConditioning

noncomputable section
open Real

/-! ## Hand-rolled 2x2 singular values

A 2x2 real matrix is carried as its four entries `J = [[p, q], [r, s]]`. -/

/-- Squared Frobenius norm `||J||_F^2 = p^2 + q^2 + r^2 + s^2`. -/
def frobSq (p q r s : ℝ) : ℝ := p ^ 2 + q ^ 2 + r ^ 2 + s ^ 2

/-- Determinant `det J = p s - q r`. -/
def det2 (p q r s : ℝ) : ℝ := p * s - q * r

/-- Discriminant `||J||_F^4 - 4 (det J)^2 = (sigma_max^2 - sigma_min^2)^2`. -/
def discr (p q r s : ℝ) : ℝ := frobSq p q r s ^ 2 - 4 * det2 p q r s ^ 2

theorem frobSq_nonneg (p q r s : ℝ) : 0 ≤ frobSq p q r s := by
  unfold frobSq; positivity

/-- The discriminant is a product of two sums of squares (exact 2x2 identity
`||J||_F^4 - 4 det^2 = ((p-s)^2 + (q+r)^2)((p+s)^2 + (q-r)^2)`). -/
theorem discr_eq (p q r s : ℝ) :
    discr p q r s = ((p - s) ^ 2 + (q + r) ^ 2) * ((p + s) ^ 2 + (q - r) ^ 2) := by
  unfold discr frobSq det2; ring

theorem discr_nonneg (p q r s : ℝ) : 0 ≤ discr p q r s := by
  rw [discr_eq]; positivity

/-- Squared larger singular value: `(||J||_F^2 + sqrt(discr)) / 2`. -/
def sigmaMaxSq (p q r s : ℝ) : ℝ :=
  (frobSq p q r s + Real.sqrt (discr p q r s)) / 2

/-- Squared smaller singular value: `(||J||_F^2 - sqrt(discr)) / 2`. -/
def sigmaMinSq (p q r s : ℝ) : ℝ :=
  (frobSq p q r s - Real.sqrt (discr p q r s)) / 2

/-- Larger singular value `sigma_max(J)` of the 2x2 matrix `[[p,q],[r,s]]`. -/
def sigmaMax (p q r s : ℝ) : ℝ := Real.sqrt (sigmaMaxSq p q r s)

/-- Smaller singular value `sigma_min(J)` of the 2x2 matrix `[[p,q],[r,s]]`. -/
def sigmaMin (p q r s : ℝ) : ℝ := Real.sqrt (sigmaMinSq p q r s)

theorem sqrt_discr_le_frobSq (p q r s : ℝ) :
    Real.sqrt (discr p q r s) ≤ frobSq p q r s := by
  have h1 : discr p q r s ≤ frobSq p q r s ^ 2 := by
    unfold discr
    nlinarith [sq_nonneg (det2 p q r s)]
  calc Real.sqrt (discr p q r s) ≤ Real.sqrt (frobSq p q r s ^ 2) :=
        Real.sqrt_le_sqrt h1
    _ = frobSq p q r s := Real.sqrt_sq (frobSq_nonneg p q r s)

theorem sigmaMinSq_nonneg (p q r s : ℝ) : 0 ≤ sigmaMinSq p q r s := by
  unfold sigmaMinSq
  have := sqrt_discr_le_frobSq p q r s
  linarith

theorem sigmaMaxSq_nonneg (p q r s : ℝ) : 0 ≤ sigmaMaxSq p q r s := by
  unfold sigmaMaxSq
  have h1 := frobSq_nonneg p q r s
  have h2 := Real.sqrt_nonneg (discr p q r s)
  linarith

theorem sigmaMin_nonneg (p q r s : ℝ) : 0 ≤ sigmaMin p q r s :=
  Real.sqrt_nonneg _

theorem sigmaMax_nonneg (p q r s : ℝ) : 0 ≤ sigmaMax p q r s :=
  Real.sqrt_nonneg _

theorem sigmaMin_sq (p q r s : ℝ) : sigmaMin p q r s ^ 2 = sigmaMinSq p q r s :=
  Real.sq_sqrt (sigmaMinSq_nonneg p q r s)

theorem sigmaMax_sq (p q r s : ℝ) : sigmaMax p q r s ^ 2 = sigmaMaxSq p q r s :=
  Real.sq_sqrt (sigmaMaxSq_nonneg p q r s)

/-- First symmetric function: `sigma_min^2 + sigma_max^2 = ||J||_F^2`. -/
theorem sq_add_sq (p q r s : ℝ) :
    sigmaMin p q r s ^ 2 + sigmaMax p q r s ^ 2 = frobSq p q r s := by
  rw [sigmaMin_sq, sigmaMax_sq]
  unfold sigmaMinSq sigmaMaxSq
  ring

theorem sigmaMinSq_mul_sigmaMaxSq (p q r s : ℝ) :
    sigmaMinSq p q r s * sigmaMaxSq p q r s = det2 p q r s ^ 2 := by
  have hD : Real.sqrt (discr p q r s) ^ 2 = discr p q r s :=
    Real.sq_sqrt (discr_nonneg p q r s)
  calc sigmaMinSq p q r s * sigmaMaxSq p q r s
      = (frobSq p q r s ^ 2 - Real.sqrt (discr p q r s) ^ 2) / 4 := by
        unfold sigmaMinSq sigmaMaxSq; ring
    _ = (frobSq p q r s ^ 2 - discr p q r s) / 4 := by rw [hD]
    _ = det2 p q r s ^ 2 := by unfold discr; ring

/-- Second symmetric function: `sigma_min sigma_max = |det J|`
(the exact 2x2 route "sigma_min = |det J| / sigma_max" of the target). -/
theorem sigmaMin_mul_sigmaMax (p q r s : ℝ) :
    sigmaMin p q r s * sigmaMax p q r s = |det2 p q r s| := by
  unfold sigmaMin sigmaMax
  rw [← Real.sqrt_mul (sigmaMinSq_nonneg p q r s), sigmaMinSq_mul_sigmaMaxSq,
    Real.sqrt_sq_eq_abs]

theorem sigmaMin_le_sigmaMax (p q r s : ℝ) :
    sigmaMin p q r s ≤ sigmaMax p q r s := by
  apply Real.sqrt_le_sqrt
  unfold sigmaMinSq sigmaMaxSq
  have := Real.sqrt_nonneg (discr p q r s)
  linarith

/-- Division form of the determinant route: for `sigma_max ≠ 0`,
`sigma_min = |det J| / sigma_max`. -/
theorem sigmaMin_eq_abs_det_div (p q r s : ℝ) (h : sigmaMax p q r s ≠ 0) :
    sigmaMin p q r s = |det2 p q r s| / sigmaMax p q r s := by
  rw [eq_div_iff h, sigmaMin_mul_sigmaMax]

/-- Operator-norm domination by the Frobenius norm, 2x2 closed form:
`sigma_max(J) <= ||J||_F` (the step "sigma_max <= ||J||_F" of the target). -/
theorem sigmaMax_le_frobNorm (p q r s : ℝ) :
    sigmaMax p q r s ≤ Real.sqrt (frobSq p q r s) := by
  apply Real.sqrt_le_sqrt
  unfold sigmaMaxSq
  have := sqrt_discr_le_frobSq p q r s
  linarith

/-- Faithfulness certificate for the hand-rolled definitions: any pair
`0 <= a <= b` realizing the two symmetric functions of the 2x2
singular-value pair (`a^2 + b^2 = ||J||_F^2` and `a b = |det J|`) equals
`(sigmaMin, sigmaMax)`.  Since the singular values of a 2x2 real matrix
satisfy exactly these two identities, every legitimate definition of them
agrees with the closed form used in this file. -/
theorem singularValues_unique {p q r s a b : ℝ} (ha : 0 ≤ a) (hab : a ≤ b)
    (hsum : a ^ 2 + b ^ 2 = frobSq p q r s)
    (hprod : a * b = |det2 p q r s|) :
    a = sigmaMin p q r s ∧ b = sigmaMax p q r s := by
  have hb : 0 ≤ b := ha.trans hab
  have habs : (a * b) ^ 2 = det2 p q r s ^ 2 := by rw [hprod]; exact sq_abs _
  have h1 : (b ^ 2 - a ^ 2) ^ 2 = discr p q r s := by
    unfold discr
    rw [← hsum, ← habs]
    ring
  have h2 : b ^ 2 - a ^ 2 = Real.sqrt (discr p q r s) := by
    rw [← h1]
    exact (Real.sqrt_sq (by nlinarith)).symm
  have hasq : a ^ 2 = sigmaMinSq p q r s := by
    unfold sigmaMinSq
    rw [← h2, ← hsum]
    ring
  have hbsq : b ^ 2 = sigmaMaxSq p q r s := by
    unfold sigmaMaxSq
    rw [← h2, ← hsum]
    ring
  constructor
  · rw [sigmaMin, ← hasq, Real.sqrt_sq ha]
  · rw [sigmaMax, ← hbsq, Real.sqrt_sq hb]

/-! ## Monotonicity helper and the Step-0 lower bound -/

/-- Monotonicity of `x ↦ x / sqrt(x^2 + 2 Lam^2)` on `x >= 0`
(the helper named by FORMALIZATION_TARGETS.md A4). -/
theorem div_sqrt_mono {Lam x y : ℝ} (hx : 0 ≤ x) (hxy : x ≤ y) :
    x / Real.sqrt (x ^ 2 + 2 * Lam ^ 2) ≤ y / Real.sqrt (y ^ 2 + 2 * Lam ^ 2) := by
  rcases eq_or_lt_of_le hx with h0 | hxpos
  · rw [← h0, zero_div]
    exact div_nonneg (by linarith) (Real.sqrt_nonneg _)
  · have hy : 0 < y := lt_of_lt_of_le hxpos hxy
    have hsx : 0 < Real.sqrt (x ^ 2 + 2 * Lam ^ 2) :=
      Real.sqrt_pos.mpr (by positivity)
    have hsy : 0 < Real.sqrt (y ^ 2 + 2 * Lam ^ 2) :=
      Real.sqrt_pos.mpr (by positivity)
    rw [div_le_div_iff₀ hsx hsy]
    have key : (x * Real.sqrt (y ^ 2 + 2 * Lam ^ 2)) ^ 2 ≤
        (y * Real.sqrt (x ^ 2 + 2 * Lam ^ 2)) ^ 2 := by
      rw [mul_pow, mul_pow, Real.sq_sqrt (by positivity : (0:ℝ) ≤ y ^ 2 + 2 * Lam ^ 2),
        Real.sq_sqrt (by positivity : (0:ℝ) ≤ x ^ 2 + 2 * Lam ^ 2)]
      nlinarith [mul_self_le_mul_self hx hxy, sq_nonneg Lam]
    calc x * Real.sqrt (y ^ 2 + 2 * Lam ^ 2)
        = Real.sqrt ((x * Real.sqrt (y ^ 2 + 2 * Lam ^ 2)) ^ 2) :=
          (Real.sqrt_sq (by positivity)).symm
      _ ≤ Real.sqrt ((y * Real.sqrt (x ^ 2 + 2 * Lam ^ 2)) ^ 2) :=
          Real.sqrt_le_sqrt key
      _ = y * Real.sqrt (x ^ 2 + 2 * Lam ^ 2) := Real.sqrt_sq (by positivity)

/-- A4 main theorem (seed-conditioning 2x2 bound; COMPANION KERNEL, see the
file-header SCOPE NOTE - this encodes "Step 0 (seed conditioning)" of the
fold-transfer proof in tex_anchors/COMPANION_jcp_supplement.tex
(approx. lines 1934-1943), not a display of the present PRR submission).
For the fold-slice Jacobian `J = [[p, q], [r, s]]` with `p = 0`
(`F_tt(q_*) = 0` at the fold seed), entry margins `mu_3 <= |r|`,
`mu_theta <= |q|` and caps `|r| <= Lam`, `|s| <= Lam`:

  `sigma_min(J) >= mu_3 mu_theta / sqrt(mu_theta^2 + 2 Lam^2)`.

Proved via `sigma_min sigma_max = |det J|`, `sigma_max <= ||J||_F`, and
`div_sqrt_mono`.  The margins are only assumed nonnegative (weaker than the
paper's strictly positive `mu_3, mu_theta`), and no positivity of the entries
themselves is smuggled in. -/
theorem sigmaMin_lower_bound {q r s Lam mu3 muTheta : ℝ}
    (hmu3 : 0 ≤ mu3) (hmuTheta : 0 ≤ muTheta)
    (hr_lo : mu3 ≤ |r|) (hq_lo : muTheta ≤ |q|)
    (hr_hi : |r| ≤ Lam) (hs_hi : |s| ≤ Lam) :
    mu3 * muTheta / Real.sqrt (muTheta ^ 2 + 2 * Lam ^ 2) ≤ sigmaMin 0 q r s := by
  -- |det J| = |q| |r| when p = 0
  have hdet : |det2 0 q r s| = |q| * |r| := by
    unfold det2
    rw [show (0 : ℝ) * s - q * r = -(q * r) by ring, abs_neg, abs_mul]
  -- sigma_max <= sqrt (q^2 + 2 Lam^2), via the Frobenius bound and entry caps
  have hfrob_le : frobSq 0 q r s ≤ q ^ 2 + 2 * Lam ^ 2 := by
    have hr2 : r ^ 2 ≤ Lam ^ 2 := by
      have := abs_nonneg r
      calc r ^ 2 = |r| ^ 2 := (sq_abs r).symm
        _ ≤ Lam ^ 2 := by nlinarith
    have hs2 : s ^ 2 ≤ Lam ^ 2 := by
      have := abs_nonneg s
      calc s ^ 2 = |s| ^ 2 := (sq_abs s).symm
        _ ≤ Lam ^ 2 := by nlinarith
    unfold frobSq
    nlinarith
  have hmax_le : sigmaMax 0 q r s ≤ Real.sqrt (q ^ 2 + 2 * Lam ^ 2) :=
    le_trans (sigmaMax_le_frobNorm 0 q r s) (Real.sqrt_le_sqrt hfrob_le)
  by_cases hq2 : 0 < q ^ 2 + 2 * Lam ^ 2
  · -- main case: the denominator sqrt(q^2 + 2 Lam^2) is positive
    have hD : 0 < Real.sqrt (q ^ 2 + 2 * Lam ^ 2) := Real.sqrt_pos.mpr hq2
    -- |q| |r| = sigma_min sigma_max <= sigma_min sqrt(q^2 + 2 Lam^2)
    have hkey : |q| * |r| ≤ sigmaMin 0 q r s * Real.sqrt (q ^ 2 + 2 * Lam ^ 2) := by
      calc |q| * |r| = |det2 0 q r s| := hdet.symm
        _ = sigmaMin 0 q r s * sigmaMax 0 q r s := (sigmaMin_mul_sigmaMax 0 q r s).symm
        _ ≤ sigmaMin 0 q r s * Real.sqrt (q ^ 2 + 2 * Lam ^ 2) :=
            mul_le_mul_of_nonneg_left hmax_le (sigmaMin_nonneg 0 q r s)
    -- |r| |q| / sqrt(q^2 + 2 Lam^2) <= sigma_min
    have hstep3 : |r| * (|q| / Real.sqrt (q ^ 2 + 2 * Lam ^ 2)) ≤ sigmaMin 0 q r s := by
      rw [← mul_div_assoc, div_le_iff₀ hD]
      calc |r| * |q| = |q| * |r| := mul_comm _ _
        _ ≤ sigmaMin 0 q r s * Real.sqrt (q ^ 2 + 2 * Lam ^ 2) := hkey
    -- chain through the monotone helper
    calc mu3 * muTheta / Real.sqrt (muTheta ^ 2 + 2 * Lam ^ 2)
        = mu3 * (muTheta / Real.sqrt (muTheta ^ 2 + 2 * Lam ^ 2)) := by
          rw [mul_div_assoc]
      _ ≤ mu3 * (|q| / Real.sqrt (|q| ^ 2 + 2 * Lam ^ 2)) :=
          mul_le_mul_of_nonneg_left (div_sqrt_mono hmuTheta hq_lo) hmu3
      _ = mu3 * (|q| / Real.sqrt (q ^ 2 + 2 * Lam ^ 2)) := by rw [sq_abs]
      _ ≤ |r| * (|q| / Real.sqrt (q ^ 2 + 2 * Lam ^ 2)) :=
          mul_le_mul_of_nonneg_right hr_lo
            (div_nonneg (abs_nonneg q) (Real.sqrt_nonneg _))
      _ ≤ sigmaMin 0 q r s := hstep3
  · -- degenerate case q = 0 and Lam = 0: then mu_theta = 0 and the bound is 0
    have hq2' : q ^ 2 + 2 * Lam ^ 2 ≤ 0 := not_lt.mp hq2
    have hq0 : q = 0 := by nlinarith [sq_nonneg q, sq_nonneg Lam]
    have hmuT0 : muTheta = 0 := le_antisymm (by rw [hq0] at hq_lo; simpa using hq_lo) hmuTheta
    rw [hmuT0, mul_zero, zero_div]
    exact sigmaMin_nonneg 0 q r s

end
end SeedConditioning
end FormalPRR

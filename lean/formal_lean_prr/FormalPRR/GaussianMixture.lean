/-
  FormalPRR/GaussianMixture.lean

  Log-derivative identities of a Gaussian mixture density
  (paper: spine Eq. exact-m-log-slope and exact-m-crossover, "MIXTURE LOG-DERIVATIVE
  IDENTITIES" / A2 and "CROSSOVER RATIO BOUNDS" / A3).

  For weights `w j ≥ 0` (not all zero), centres `c j`, and width `σ > 0`, set
    q w c σ j x = w j · exp(−(x − c j)² / (2σ²)),
    H w c σ x   = ∑ j, q w c σ j x,
    π w c σ x j = q w c σ j x / H w c σ x                     (softmax weights)
    c̄ w c σ x  = (∑ j, q j x · c j) / H  = ∑ j, π j · c j     (π-weighted mean)
    Var w c σ x = (∑ j, q j x · (c j)²)/H − c̄²  = ∑ π_j c_j² − c̄².
  We prove sorry-free:

  * `H_pos`            (T1)  H > 0 on all of ℝ.
  * `logDeriv_H`       (T2)  (log H)'(x) = (c̄(x) − x)/σ².
  * `second_logDeriv_H`(T3)  (log H)''(x) = Var(x)/σ⁴ − 1/σ².
  * `crossover_point`  (T4)  q_j x = q_k x  ⇔  x = (c_j+c_k)/2 + σ²/(c_k−c_j)·log(w_j/w_k).

  Plus the π-form bridges `sum_pi`, `cbar_eq_pi`, `var_eq_pi`.
-/
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Tactic

-- SCOPE: in-scope — exact-m modality submission (log-derivative identities of the
--  Gaussian mixture, spine Eq. exact-m-log-slope / exact-m-crossover, A2 & A3).

namespace FormalPRR.MixtureIndep

open Finset

variable {m : ℕ}

/-- A single Gaussian component `q_j(x) = w_j · exp(−(x−c_j)²/(2σ²))`
(the components of the mixture of spine Eq. eq:exact-m-mixture /
Eq. eq:exactmfull-posterior of exact_m_theorem_full_proof.tex). -/
noncomputable def q (w c : Fin m → ℝ) (σ : ℝ) (j : Fin m) (x : ℝ) : ℝ :=
  w j * Real.exp (-(x - c j) ^ 2 / (2 * σ ^ 2))

/-- The mixture density `H(x) = ∑_j q_j(x)`
(spine Eq. eq:exact-m-mixture). -/
noncomputable def H (w c : Fin m → ℝ) (σ : ℝ) (x : ℝ) : ℝ := ∑ j, q w c σ j x

/-- The softmax weight `π_j(x) = q_j(x)/H(x)`
(Eq. eq:exactmfull-posterior of exact_m_theorem_full_proof.tex). -/
noncomputable def piw (w c : Fin m → ℝ) (σ : ℝ) (x : ℝ) (j : Fin m) : ℝ :=
  q w c σ j x / H w c σ x

/-- The π-weighted mean of centres, `c̄(x) = (∑_j q_j(x) c_j)/H(x)`
(the `c̄` of spine Eq. eq:exact-m-log-slope). -/
noncomputable def cbar (w c : Fin m → ℝ) (σ : ℝ) (x : ℝ) : ℝ :=
  (∑ j, q w c σ j x * c j) / H w c σ x

/-- The π-weighted variance of centres, `Var(x) = (∑_j q_j(x) c_j²)/H(x) − c̄(x)²`
(the `Var` of spine Eq. eq:exact-m-log-slope). -/
noncomputable def varc (w c : Fin m → ℝ) (σ : ℝ) (x : ℝ) : ℝ :=
  (∑ j, q w c σ j x * (c j) ^ 2) / H w c σ x - (cbar w c σ x) ^ 2

/-! ### T1 : positivity of `H` -/

/-- **T1.**  If all weights are nonnegative and at least one is strictly positive, then
`H(x) > 0` for every `x` (positivity of the mixture of spine
Eq. eq:exact-m-mixture, implicit in taking `log H` in
Eq. eq:exact-m-log-slope). -/
theorem H_pos (w c : Fin m → ℝ) (σ : ℝ) (hw : ∀ j, 0 ≤ w j) (hex : ∃ j, 0 < w j) (x : ℝ) :
    0 < H w c σ x := by
  obtain ⟨j0, hj0⟩ := hex
  have hnn : ∀ j ∈ (univ : Finset (Fin m)), 0 ≤ q w c σ j x := by
    intro j _; exact mul_nonneg (hw j) (Real.exp_pos _).le
  have hle : q w c σ j0 x ≤ H w c σ x := single_le_sum hnn (mem_univ j0)
  have hpos : 0 < q w c σ j0 x := mul_pos hj0 (Real.exp_pos _)
  exact lt_of_lt_of_le hpos hle

/-! ### Derivative of a single component -/

/-- The derivative of one component: `q_j'(x) = q_j(x)·(c_j − x)/σ²`.
Internal helper (calculus step for T2/T3); no direct paper display. -/
theorem hasDerivAt_q (w c : Fin m → ℝ) (σ : ℝ) (hσ : σ ≠ 0) (j : Fin m) (x : ℝ) :
    HasDerivAt (q w c σ j) (q w c σ j x * (c j - x) / σ ^ 2) x := by
  have hid : HasDerivAt (fun y : ℝ => y - c j) (1 : ℝ) x := by
    simpa using (hasDerivAt_id x).sub_const (c j)
  have hsq : HasDerivAt (fun y : ℝ => (y - c j) ^ 2) (2 * (x - c j)) x := by
    have h := hid.pow 2
    norm_num at h
    exact h
  have hpoly : HasDerivAt (fun y : ℝ => -(y - c j) ^ 2 / (2 * σ ^ 2)) ((c j - x) / σ ^ 2) x := by
    have h := (hsq.neg).div_const (2 * σ ^ 2)
    rw [show -(2 * (x - c j)) / (2 * σ ^ 2) = (c j - x) / σ ^ 2 from by field_simp; ring] at h
    exact h
  have hfull : HasDerivAt (fun y : ℝ => w j * Real.exp (-(y - c j) ^ 2 / (2 * σ ^ 2)))
      (w j * (Real.exp (-(x - c j) ^ 2 / (2 * σ ^ 2)) * ((c j - x) / σ ^ 2))) x :=
    (hpoly.exp).const_mul (w j)
  rw [show q w c σ j x * (c j - x) / σ ^ 2
      = w j * (Real.exp (-(x - c j) ^ 2 / (2 * σ ^ 2)) * ((c j - x) / σ ^ 2)) from by
        unfold q; ring]
  exact hfull

/-- The derivative of the mixture density: `H'(x) = ∑_j q_j(x)·(c_j − x)/σ²`.
Internal helper (term-by-term sum of `hasDerivAt_q`); no direct paper
display. -/
theorem hasDerivAt_H (w c : Fin m → ℝ) (σ : ℝ) (hσ : σ ≠ 0) (x : ℝ) :
    HasDerivAt (fun y => H w c σ y) (∑ j, q w c σ j x * (c j - x) / σ ^ 2) x := by
  show HasDerivAt (fun y => ∑ j, q w c σ j y) _ x
  exact HasDerivAt.fun_sum (fun j _ => hasDerivAt_q w c σ hσ j x)

/-! ### T2 : the log-derivative -/

/-- **T2.**  `(log H)'(x) = (c̄(x) − x)/σ²` — the first identity of spine
Eq. (eq:exact-m-log-slope) of exact_m_theorem_spine.tex
(= Eq. eq:exactmfull-log-slope of exact_m_theorem_full_proof.tex). -/
theorem logDeriv_H (w c : Fin m → ℝ) (σ : ℝ) (hσ : σ ≠ 0) (x : ℝ)
    (hH : H w c σ x ≠ 0) :
    deriv (fun x => Real.log (H w c σ x)) x = (cbar w c σ x - x) / σ ^ 2 := by
  have hlog := (hasDerivAt_H w c σ hσ x).log hH
  rw [hlog.deriv]
  have factorD : (∑ j, q w c σ j x * (c j - x) / σ ^ 2)
      = (∑ j, q w c σ j x * (c j - x)) / σ ^ 2 := by
    rw [Finset.sum_div]
  have hlin0 : (∑ j, q w c σ j x * (c j - x))
      = (∑ j, q w c σ j x * c j) - x * H w c σ x := by
    unfold H
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl (fun j _ => by ring)
  rw [factorD, hlin0]
  unfold cbar
  field_simp

/-! ### π-form bridges -/

/-- The softmax weights sum to one (normalization of the `π_j` of
Eq. eq:exactmfull-posterior).  Internal bridge lemma; no direct paper
display. -/
theorem sum_pi (w c : Fin m → ℝ) (σ : ℝ) (x : ℝ) (hH : H w c σ x ≠ 0) :
    ∑ j, piw w c σ x j = 1 := by
  have h : (∑ j, piw w c σ x j) = (∑ j, q w c σ j x) / H w c σ x := by
    unfold piw; rw [← Finset.sum_div]
  rw [h]
  show H w c σ x / H w c σ x = 1
  exact div_self hH

/-- `c̄` in π-weighted form, `c̄(x) = ∑_j π_j(x) c_j` — the form in which
`c̄` enters spine Eq. (eq:exact-m-log-slope).  Internal bridge lemma. -/
theorem cbar_eq_pi (w c : Fin m → ℝ) (σ : ℝ) (x : ℝ) :
    cbar w c σ x = ∑ j, piw w c σ x j * c j := by
  unfold cbar piw
  rw [Finset.sum_div]
  exact Finset.sum_congr rfl (fun j _ => by ring)

/-- `Var` in π-weighted form, `Var(x) = ∑_j π_j(x) c_j² − c̄(x)²` — the form
in which `Var` enters spine Eq. (eq:exact-m-log-slope).  Internal bridge
lemma. -/
theorem var_eq_pi (w c : Fin m → ℝ) (σ : ℝ) (x : ℝ) :
    varc w c σ x = (∑ j, piw w c σ x j * (c j) ^ 2) - (cbar w c σ x) ^ 2 := by
  unfold varc piw
  rw [Finset.sum_div]
  congr 1
  exact Finset.sum_congr rfl (fun j _ => by ring)

/-! ### T3 : the second log-derivative -/

/-- **T3.**  `(log H)''(x) = Var(x)/σ⁴ − 1/σ²` — the second identity of
spine Eq. (eq:exact-m-log-slope) of exact_m_theorem_spine.tex
(= Eq. eq:exactmfull-log-slope-derivative of
exact_m_theorem_full_proof.tex). -/
theorem second_logDeriv_H (w c : Fin m → ℝ) (σ : ℝ) (hσ : σ ≠ 0)
    (hw : ∀ j, 0 ≤ w j) (hex : ∃ j, 0 < w j) (x : ℝ) :
    deriv (deriv (fun x => Real.log (H w c σ x))) x = varc w c σ x / σ ^ 4 - 1 / σ ^ 2 := by
  have hHne : ∀ y, H w c σ y ≠ 0 := fun y => (H_pos w c σ hw hex y).ne'
  have hL : deriv (fun x => Real.log (H w c σ x)) = fun y => (cbar w c σ y - y) / σ ^ 2 := by
    funext y; exact logDeriv_H w c σ hσ y (hHne y)
  rw [hL]
  -- Derivative facts for numerator and denominator of `cbar`.
  have hN : HasDerivAt (fun y => ∑ j, q w c σ j y * c j)
      (∑ j, q w c σ j x * (c j - x) / σ ^ 2 * c j) x :=
    HasDerivAt.fun_sum (fun j _ => (hasDerivAt_q w c σ hσ j x).mul_const (c j))
  have hD : HasDerivAt (fun y => H w c σ y) (∑ j, q w c σ j x * (c j - x) / σ ^ 2) x :=
    hasDerivAt_H w c σ hσ x
  -- Linearity rewrites.
  have factorC : (∑ j, q w c σ j x * (c j - x) / σ ^ 2 * c j)
      = (∑ j, q w c σ j x * (c j - x) * c j) / σ ^ 2 := by
    rw [Finset.sum_div]; exact Finset.sum_congr rfl (fun j _ => by ring)
  have factorD : (∑ j, q w c σ j x * (c j - x) / σ ^ 2)
      = (∑ j, q w c σ j x * (c j - x)) / σ ^ 2 := by
    rw [Finset.sum_div]
  have hlin1 : (∑ j, q w c σ j x * (c j - x) * c j)
      = (∑ j, q w c σ j x * (c j) ^ 2) - x * (∑ j, q w c σ j x * c j) := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl (fun j _ => by ring)
  have hlin0 : (∑ j, q w c σ j x * (c j - x))
      = (∑ j, q w c σ j x * c j) - x * H w c σ x := by
    unfold H
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl (fun j _ => by ring)
  -- Derivative of `cbar` equals `Var/σ²`.
  have hcbar : HasDerivAt (fun y => cbar w c σ y) (varc w c σ x / σ ^ 2) x := by
    have hq := HasDerivAt.fun_div hN hD (hHne x)
    rw [show ((∑ j, q w c σ j x * (c j - x) / σ ^ 2 * c j) * H w c σ x
          - (∑ j, q w c σ j x * c j) * (∑ j, q w c σ j x * (c j - x) / σ ^ 2)) / H w c σ x ^ 2
        = varc w c σ x / σ ^ 2 from by
          rw [factorC, factorD, hlin1, hlin0]; unfold varc cbar; field_simp; ring] at hq
    exact hq
  -- Assemble the second derivative.
  have hval : (varc w c σ x / σ ^ 2 - 1) / σ ^ 2 = varc w c σ x / σ ^ 4 - 1 / σ ^ 2 := by
    ring
  have hLd : HasDerivAt (fun y => (cbar w c σ y - y) / σ ^ 2)
      (varc w c σ x / σ ^ 4 - 1 / σ ^ 2) x := by
    have hbase := (hcbar.sub (hasDerivAt_id x)).div_const (σ ^ 2)
    rw [hval] at hbase
    exact hbase
  exact hLd.deriv

/-! ### T4 : the crossover point of two components -/

/-- **T4.**  For two components with `c_j < c_k` and positive weights, `q_j(x) = q_k(x)`
holds exactly at `x = (c_j + c_k)/2 + σ²/(c_k − c_j)·log(w_j/w_k)` — the
weighted crossover of spine Eq. (eq:exact-m-crossover) of
exact_m_theorem_spine.tex, strengthened here to an iff
(the crossover is the UNIQUE equality point). -/
theorem crossover_point (w c : Fin m → ℝ) (σ : ℝ) (hσ : σ ≠ 0) (j k : Fin m)
    (hjk : c j < c k) (hwj : 0 < w j) (hwk : 0 < w k) (x : ℝ) :
    q w c σ j x = q w c σ k x ↔
      x = (c j + c k) / 2 + σ ^ 2 / (c k - c j) * Real.log (w j / w k) := by
  have h2 : (2 * σ ^ 2) ≠ 0 := mul_ne_zero two_ne_zero (pow_ne_zero 2 hσ)
  have hne : c k - c j ≠ 0 := sub_ne_zero.mpr (ne_of_gt hjk)
  have hlogdiv : Real.log (w j / w k) = Real.log (w j) - Real.log (w k) :=
    Real.log_div (ne_of_gt hwj) (ne_of_gt hwk)
  -- Write each component as a single exponential.
  have hexpj : q w c σ j x = Real.exp (Real.log (w j) - (x - c j) ^ 2 / (2 * σ ^ 2)) := by
    unfold q
    rw [show Real.log (w j) - (x - c j) ^ 2 / (2 * σ ^ 2)
        = Real.log (w j) + (-(x - c j) ^ 2 / (2 * σ ^ 2)) from by ring,
      Real.exp_add, Real.exp_log hwj]
  have hexpk : q w c σ k x = Real.exp (Real.log (w k) - (x - c k) ^ 2 / (2 * σ ^ 2)) := by
    unfold q
    rw [show Real.log (w k) - (x - c k) ^ 2 / (2 * σ ^ 2)
        = Real.log (w k) + (-(x - c k) ^ 2 / (2 * σ ^ 2)) from by ring,
      Real.exp_add, Real.exp_log hwk]
  rw [hexpj, hexpk, Real.exp_eq_exp, hlogdiv]
  -- Now a purely algebraic equivalence.
  constructor
  · intro hA
    have hlin : (c k - c j) * (2 * x - c j - c k)
        = 2 * σ ^ 2 * (Real.log (w j) - Real.log (w k)) := by
      have hA2 : (Real.log (w j) - (x - c j) ^ 2 / (2 * σ ^ 2)) * (2 * σ ^ 2)
               = (Real.log (w k) - (x - c k) ^ 2 / (2 * σ ^ 2)) * (2 * σ ^ 2) := by rw [hA]
      rw [sub_mul, sub_mul, div_mul_cancel₀ _ h2, div_mul_cancel₀ _ h2] at hA2
      linear_combination -hA2
    field_simp
    linear_combination hlin
  · intro hxs
    rw [hxs]
    field_simp
    ring

end FormalPRR.MixtureIndep

#print axioms FormalPRR.MixtureIndep.H_pos
#print axioms FormalPRR.MixtureIndep.logDeriv_H
#print axioms FormalPRR.MixtureIndep.second_logDeriv_H
#print axioms FormalPRR.MixtureIndep.crossover_point
#print axioms FormalPRR.MixtureIndep.sum_pi
#print axioms FormalPRR.MixtureIndep.cbar_eq_pi
#print axioms FormalPRR.MixtureIndep.var_eq_pi

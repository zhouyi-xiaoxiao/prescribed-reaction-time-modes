/-
FormalPRR/MixtureIdentities.lean — Target A2.

Source (paper): exact_m_theorem_spine.tex, Eq. (eq:exact-m-log-slope), with the
Gaussian mixture H defined in Eq. (eq:exact-m-mixture); full-proof version:
exact_m_theorem_full_proof.tex, Eqs. (eq:exactmfull-log-slope) and
(eq:exactmfull-log-slope-derivative), with q_j, π_j, c̄ as in
Eq. (eq:exactmfull-posterior).

Statements: for H(x) = Σ_j w_j exp(−(x−c_j)²/(2σ²)) with w_j > 0, σ > 0, m ≥ 1,
  (log H)'(x)  = (c̄(x) − x)/σ²,
  (log H)''(x) = Var_{π(x)}(c)/σ⁴ − 1/σ²,
where π_j(x) = q_j(x)/H(x), c̄(x) = Σ_j π_j(x) c_j, and
Var_{π(x)}(c) = Σ_j π_j(x) (c_j − c̄(x))².
-/
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

namespace FormalPRR
namespace Mixture

open Finset

/-- Mixture component `q_j(x) = w_j · exp(−(x−c_j)²/(2σ²))`
(Eq. eq:exactmfull-posterior of exact_m_theorem_full_proof.tex). -/
noncomputable def q {m : ℕ} (w c : Fin m → ℝ) (σ : ℝ) (j : Fin m) (x : ℝ) : ℝ :=
  w j * Real.exp (-(x - c j) ^ 2 / (2 * σ ^ 2))

/-- The common-variance Gaussian mixture `H_{σ,w}(x) = Σ_j q_j(x)`
(Eq. eq:exact-m-mixture of exact_m_theorem_spine.tex). -/
noncomputable def H {m : ℕ} (w c : Fin m → ℝ) (σ : ℝ) (x : ℝ) : ℝ :=
  ∑ j, q w c σ j x

/-- Posterior weight `π_j(x) = q_j(x)/H(x)`
(Eq. eq:exactmfull-posterior of exact_m_theorem_full_proof.tex). -/
noncomputable def post {m : ℕ} (w c : Fin m → ℝ) (σ : ℝ) (j : Fin m) (x : ℝ) : ℝ :=
  q w c σ j x / H w c σ x

/-- Posterior mean of the centres `c̄(x) = Σ_j π_j(x) c_j`
(Eq. eq:exactmfull-posterior of exact_m_theorem_full_proof.tex). -/
noncomputable def cbar {m : ℕ} (w c : Fin m → ℝ) (σ : ℝ) (x : ℝ) : ℝ :=
  ∑ j, post w c σ j x * c j

/-- Posterior variance of the centres `Var_{π(x)}(c) = Σ_j π_j(x)(c_j − c̄(x))²`
(as used in Eq. eq:exact-m-log-slope of exact_m_theorem_spine.tex). -/
noncomputable def varC {m : ℕ} (w c : Fin m → ℝ) (σ : ℝ) (x : ℝ) : ℝ :=
  ∑ j, post w c σ j x * (c j - cbar w c σ x) ^ 2

section Basic

variable {m : ℕ} (w c : Fin m → ℝ) (σ : ℝ)

/-- Positivity of a mixture component.  Internal helper; no direct paper
display. -/
lemma q_pos {j : Fin m} (hw : 0 < w j) (x : ℝ) : 0 < q w c σ j x :=
  mul_pos hw (Real.exp_pos _)

/-- Positivity of the mixture `H` ("H > 0 everywhere", implicit in taking
`log H` in Eq. eq:exact-m-log-slope).  Internal helper; no direct paper
display. -/
lemma H_pos (hm : 0 < m) (hw : ∀ j, 0 < w j) (x : ℝ) : 0 < H w c σ x := by
  have : Nonempty (Fin m) := Fin.pos_iff_nonempty.mp hm
  exact Finset.sum_pos (fun j _ => q_pos w c σ (hw j) x) Finset.univ_nonempty

/-- `c̄(x)` as a single quotient.  Internal helper; no direct paper
display. -/
lemma cbar_eq_div (x : ℝ) :
    cbar w c σ x = (∑ j, q w c σ j x * c j) / H w c σ x := by
  simp only [cbar, post]
  rw [Finset.sum_div]
  exact Finset.sum_congr rfl fun j _ => by ring

/-- The posterior weights sum to 1 (normalization of the `π_j` of
Eq. eq:exactmfull-posterior).  Internal helper; no direct paper display. -/
lemma sum_post (x : ℝ) (hH : H w c σ x ≠ 0) : ∑ j, post w c σ j x = 1 := by
  simp only [post]
  rw [← Finset.sum_div]
  have hHdef : (∑ j, q w c σ j x) = H w c σ x := rfl
  rw [hHdef, div_self hH]

/-- Posterior variance as second moment minus squared mean.  Internal
helper; no direct paper display. -/
lemma varC_eq (x : ℝ) (hH : H w c σ x ≠ 0) :
    varC w c σ x = (∑ j, q w c σ j x * c j ^ 2) / H w c σ x - cbar w c σ x ^ 2 := by
  have hexp : ∀ j ∈ (Finset.univ : Finset (Fin m)),
      post w c σ j x * (c j - cbar w c σ x) ^ 2
        = post w c σ j x * c j ^ 2
          - 2 * cbar w c σ x * (post w c σ j x * c j)
          + cbar w c σ x ^ 2 * post w c σ j x := fun j _ => by ring
  rw [varC, Finset.sum_congr rfl hexp, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum]
  have h1 : ∑ j, post w c σ j x * c j = cbar w c σ x := rfl
  have h2 : ∑ j, post w c σ j x * c j ^ 2
      = (∑ j, q w c σ j x * c j ^ 2) / H w c σ x := by
    simp only [post]
    rw [Finset.sum_div]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [h1, h2, sum_post w c σ x hH]
  ring

end Basic

section Deriv

variable {m : ℕ} (w c : Fin m → ℝ) (σ : ℝ)

/-- Transport a `HasDerivAt` along an equality of derivative values.
Internal helper (proof machinery); no direct paper display. -/
lemma hasDerivAt_deriv_congr {f : ℝ → ℝ} {a b x : ℝ}
    (h : HasDerivAt f a x) (hab : a = b) : HasDerivAt f b x := hab ▸ h

/-- Derivative of one mixture component: `q_j'(x) = q_j(x)·(c_j − x)/σ²`
("differentiation of q_j gives the first identity", exact_m_theorem_full_proof.tex,
proof of Eqs. eq:exactmfull-log-slope and eq:exactmfull-log-slope-derivative). -/
lemma hasDerivAt_q (hσ : σ ≠ 0) (j : Fin m) (x : ℝ) :
    HasDerivAt (fun y => q w c σ j y) (q w c σ j x * ((c j - x) / σ ^ 2)) x := by
  have h1 : HasDerivAt (fun y : ℝ => y - c j) 1 x := (hasDerivAt_id x).sub_const (c j)
  have hu : HasDerivAt (fun y : ℝ => -(y - c j) ^ 2 / (2 * σ ^ 2))
      ((c j - x) / σ ^ 2) x := by
    refine hasDerivAt_deriv_congr (((h1.fun_pow 2).neg).div_const (2 * σ ^ 2)) ?_
    simp only [mul_one, Nat.cast_ofNat]
    field_simp
    ring
  have h3 := hu.exp.const_mul (w j)
  have heq : q w c σ j x * ((c j - x) / σ ^ 2)
      = w j * (Real.exp (-(x - c j) ^ 2 / (2 * σ ^ 2)) * ((c j - x) / σ ^ 2)) := by
    simp only [q]; ring
  rw [show (fun y => q w c σ j y)
      = fun y => w j * Real.exp (-(y - c j) ^ 2 / (2 * σ ^ 2)) from rfl, heq]
  exact h3

/-- Derivative of the mixture `H`.  Internal helper (term-by-term sum of
`hasDerivAt_q`); no direct paper display. -/
lemma hasDerivAt_H (hσ : σ ≠ 0) (x : ℝ) :
    HasDerivAt (fun y => H w c σ y) (∑ j, q w c σ j x * ((c j - x) / σ ^ 2)) x := by
  exact HasDerivAt.fun_sum (u := Finset.univ)
    (fun j _ => hasDerivAt_q w c σ hσ j x)

/-- Rewriting `Σ_j q_j(x)(c_j−x)/σ²` through the first and zeroth moments.
Internal helper; no direct paper display. -/
lemma sum_deriv_terms (x : ℝ) :
    ∑ j, q w c σ j x * ((c j - x) / σ ^ 2)
      = ((∑ j, q w c σ j x * c j) - x * H w c σ x) / σ ^ 2 := by
  simp only [H]
  rw [Finset.mul_sum, ← Finset.sum_sub_distrib, Finset.sum_div]
  exact Finset.sum_congr rfl fun j _ => by ring

/-- Rewriting `Σ_j q_j(x)((c_j−x)/σ²)c_j` through the second and first
moments.  Internal helper; no direct paper display. -/
lemma sum_deriv_terms_c (x : ℝ) :
    ∑ j, q w c σ j x * ((c j - x) / σ ^ 2) * c j
      = ((∑ j, q w c σ j x * c j ^ 2) - x * ∑ j, q w c σ j x * c j) / σ ^ 2 := by
  rw [Finset.mul_sum, ← Finset.sum_sub_distrib, Finset.sum_div]
  exact Finset.sum_congr rfl fun j _ => by ring

/-- **First mixture log-derivative identity** — Eq. (eq:exact-m-log-slope), first
identity, of exact_m_theorem_spine.tex (= Eq. eq:exactmfull-log-slope of
exact_m_theorem_full_proof.tex):
`(log H)'(x) = (cbar(x) − x)/σ²` for positive weights, σ > 0 and m ≥ 1. -/
theorem deriv_log_H (hm : 0 < m) (hw : ∀ j, 0 < w j) (hσ : 0 < σ) (x : ℝ) :
    deriv (fun y => Real.log (H w c σ y)) x = (cbar w c σ x - x) / σ ^ 2 := by
  have hH : 0 < H w c σ x := H_pos w c σ hm hw x
  have hlog := (hasDerivAt_H w c σ (ne_of_gt hσ) x).log (ne_of_gt hH)
  rw [hlog.deriv, sum_deriv_terms w c σ x, cbar_eq_div w c σ x]
  have hσ2 : σ ^ 2 ≠ 0 := pow_ne_zero 2 (ne_of_gt hσ)
  field_simp

/-- Derivative of the posterior mean `c̄`.  Internal helper (quotient-rule
step for `deriv2_log_H`); no direct paper display. -/
lemma hasDerivAt_cbar (hm : 0 < m) (hw : ∀ j, 0 < w j) (hσ : 0 < σ) (x : ℝ) :
    HasDerivAt (fun y => cbar w c σ y)
      (((∑ j, q w c σ j x * ((c j - x) / σ ^ 2) * c j) * H w c σ x
          - (∑ j, q w c σ j x * c j) * (∑ j, q w c σ j x * ((c j - x) / σ ^ 2)))
        / (H w c σ x) ^ 2) x := by
  have hH : 0 < H w c σ x := H_pos w c σ hm hw x
  have hS1 : HasDerivAt (fun y => ∑ j, q w c σ j y * c j)
      (∑ j, q w c σ j x * ((c j - x) / σ ^ 2) * c j) x :=
    HasDerivAt.fun_sum (u := Finset.univ)
      (fun j _ => (hasDerivAt_q w c σ (ne_of_gt hσ) j x).mul_const (c j))
  have h := hS1.div (hasDerivAt_H w c σ (ne_of_gt hσ) x) (ne_of_gt hH)
  have hfe : (fun y => cbar w c σ y)
      = fun y => (∑ j, q w c σ j y * c j) / H w c σ y :=
    funext fun y => cbar_eq_div w c σ y
  rw [hfe]
  exact h

/-- **Second mixture log-derivative identity** — Eq. (eq:exact-m-log-slope), second
identity, of exact_m_theorem_spine.tex (= Eq. eq:exactmfull-log-slope-derivative of
exact_m_theorem_full_proof.tex):
`(log H)''(x) = Var_{π(x)}(c)/σ⁴ − 1/σ²` for positive weights, σ > 0 and m ≥ 1. -/
theorem deriv2_log_H (hm : 0 < m) (hw : ∀ j, 0 < w j) (hσ : 0 < σ) (x : ℝ) :
    deriv (deriv (fun y => Real.log (H w c σ y))) x
      = varC w c σ x / σ ^ 4 - 1 / σ ^ 2 := by
  have hH : 0 < H w c σ x := H_pos w c σ hm hw x
  have hfun : deriv (fun y => Real.log (H w c σ y))
      = fun y => (cbar w c σ y - y) / σ ^ 2 :=
    funext fun y => deriv_log_H w c σ hm hw hσ y
  rw [hfun]
  have hmain : HasDerivAt (fun y => (cbar w c σ y - y) / σ ^ 2)
      ((((∑ j, q w c σ j x * ((c j - x) / σ ^ 2) * c j) * H w c σ x
          - (∑ j, q w c σ j x * c j) * (∑ j, q w c σ j x * ((c j - x) / σ ^ 2)))
        / (H w c σ x) ^ 2 - 1) / σ ^ 2) x :=
    ((hasDerivAt_cbar w c σ hm hw hσ x).sub (hasDerivAt_id x)).div_const (σ ^ 2)
  rw [hmain.deriv, sum_deriv_terms w c σ x, sum_deriv_terms_c w c σ x,
    varC_eq w c σ x (ne_of_gt hH), cbar_eq_div w c σ x]
  have hσ2 : σ ≠ 0 := ne_of_gt hσ
  have hHne : H w c σ x ≠ 0 := ne_of_gt hH
  field_simp
  ring

end Deriv

end Mixture
end FormalPRR

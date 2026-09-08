/-
FormalPRR/WindowSignature.lean — Target B1 (partial: exhaustiveness half).

Source (paper): exact_m_theorem_full_proof.tex, Lemma (lem:exactmfull-zero-bound)
applied to the pure common-variance mixture via
Eq. (eq:exactmfull-exponential-polynomial):
  σ² e^{x²/(2σ²)} H'_{σ,w}(x) = Σ_j w_j (c_j − x) e^{−c_j²/(2σ²)} e^{c_j x/σ²},
so H' has at most 2m−1 real zeros and its zero set is finite; this is the
exhaustiveness step behind the 2m−1 window signature of
exact_m_theorem_spine.tex ("These constructions give 2m−1 simple stationary
points.  They are exhaustive: ...", around Eq. eq:exact-m-exp-polynomial).

-- SCOPE NOTE (B1 delivered only in part): this file proves the EXHAUSTIVENESS
-- (upper-bound) half of target B1: the pure mixture H_{σ,w} has at most 2m−1
-- stationary points, and its critical set is finite.  The EXISTENCE half of B1 —
-- that for centres with spacing ≥ Δ and explicit σ ≤ σ₀(Δ, m) the mixture has
-- exactly m local maxima and m−1 local minima on an interval containing all
-- centres with margin (which additionally needs A3's crossover/isolation bounds
-- and the sign analysis of Eq. eq:exact-m-log-slope near centres and crossovers)
-- is NOT formalized here.  Statements below say "at most", never "exactly".
-/
import FormalPRR.ExpPolyZeros
import FormalPRR.MixtureIdentities

namespace FormalPRR
namespace Window

open Finset

/-- The factorization Eq. (eq:exactmfull-exponential-polynomial) of
exact_m_theorem_full_proof.tex: `H'(x)` equals a nowhere-zero prefactor
`e^{−x²/(2σ²)}/σ²` times the exponential polynomial with frequencies `c_j/σ²`
and affine coefficients `(w_j e^{−c_j²/(2σ²)} c_j, −w_j e^{−c_j²/(2σ²)})`. -/
lemma deriv_H_factor (m : ℕ) (w c : Fin m → ℝ) (σ : ℝ) (hσ : σ ≠ 0) (x : ℝ) :
    deriv (fun y => Mixture.H w c σ y) x
      = (Real.exp (-x ^ 2 / (2 * σ ^ 2)) / σ ^ 2)
        * ExpPoly.expPoly m (fun j => c j / σ ^ 2)
            (fun j => w j * Real.exp (-(c j) ^ 2 / (2 * σ ^ 2)) * c j)
            (fun j => -(w j * Real.exp (-(c j) ^ 2 / (2 * σ ^ 2)))) x := by
  have hσ2 : (σ : ℝ) ^ 2 ≠ 0 := pow_ne_zero 2 hσ
  rw [(Mixture.hasDerivAt_H w c σ hσ x).deriv]
  simp only [ExpPoly.expPoly, Mixture.q]
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  have hexp : Real.exp (-(x - c j) ^ 2 / (2 * σ ^ 2))
      = Real.exp (-x ^ 2 / (2 * σ ^ 2)) * Real.exp (-(c j) ^ 2 / (2 * σ ^ 2))
        * Real.exp ((c j / σ ^ 2) * x) := by
    rw [← Real.exp_add, ← Real.exp_add]
    congr 1
    field_simp
    ring
  rw [hexp]
  ring

/-- **Stationary-point count of the pure mixture** (exhaustiveness half of B1;
Lemma lem:exactmfull-zero-bound of exact_m_theorem_full_proof.tex specialized to
`H'_{σ,w}` through Eq. eq:exactmfull-exponential-polynomial): for strictly
increasing centres, positive weights and σ > 0, every finite set of stationary
points of `H_{σ,w}` has cardinality at most `2m − 1`. -/
theorem mixture_deriv_zeros_card_le (m : ℕ) (w c : Fin m → ℝ) (σ : ℝ)
    (hw : ∀ j, 0 < w j) (hσ : 0 < σ) (hc : StrictMono c) (hm : 0 < m)
    (s : Finset ℝ) (hs : ∀ x ∈ s, deriv (fun y => Mixture.H w c σ y) x = 0) :
    s.card ≤ 2 * m - 1 := by
  have hσ' : σ ≠ 0 := ne_of_gt hσ
  have hσ2 : (0 : ℝ) < σ ^ 2 := by positivity
  have hlam : StrictMono (fun j => c j / σ ^ 2) := by
    intro i j hij
    simp only [div_eq_mul_inv]
    exact mul_lt_mul_of_pos_right (hc hij) (inv_pos.mpr hσ2)
  have hab : ∃ j : Fin m,
      (w j * Real.exp (-(c j) ^ 2 / (2 * σ ^ 2)) * c j ≠ 0)
        ∨ (-(w j * Real.exp (-(c j) ^ 2 / (2 * σ ^ 2))) ≠ 0) := by
    refine ⟨⟨0, hm⟩, Or.inr ?_⟩
    exact neg_ne_zero.mpr
      (mul_ne_zero (ne_of_gt (hw _)) (Real.exp_ne_zero _))
  refine ExpPoly.expPoly_card_le m (fun j => c j / σ ^ 2)
    (fun j => w j * Real.exp (-(c j) ^ 2 / (2 * σ ^ 2)) * c j)
    (fun j => -(w j * Real.exp (-(c j) ^ 2 / (2 * σ ^ 2)))) hlam hab s ?_
  intro x hx
  have h := hs x hx
  rw [deriv_H_factor m w c σ hσ' x] at h
  have hpre : Real.exp (-x ^ 2 / (2 * σ ^ 2)) / σ ^ 2 ≠ 0 :=
    div_ne_zero (Real.exp_ne_zero _) (ne_of_gt hσ2)
  exact (mul_eq_zero.mp h).resolve_left hpre

/-- The critical set of the pure mixture is finite, with at most `2m − 1` elements
("Its real zero set is finite", Lemma lem:exactmfull-zero-bound of
exact_m_theorem_full_proof.tex, for `H'_{σ,w}`). -/
theorem mixture_deriv_zeroSet_finite (m : ℕ) (w c : Fin m → ℝ) (σ : ℝ)
    (hw : ∀ j, 0 < w j) (hσ : 0 < σ) (hc : StrictMono c) (hm : 0 < m) :
    {x : ℝ | deriv (fun y => Mixture.H w c σ y) x = 0}.Finite ∧
      {x : ℝ | deriv (fun y => Mixture.H w c σ y) x = 0}.ncard ≤ 2 * m - 1 := by
  have hfin : {x : ℝ | deriv (fun y => Mixture.H w c σ y) x = 0}.Finite := by
    by_contra hinf
    rw [Set.not_finite] at hinf
    obtain ⟨t, hts, htcard⟩ := hinf.exists_subset_card_eq (2 * m)
    have := mixture_deriv_zeros_card_le m w c σ hw hσ hc hm t
      (fun x hx => hts (Finset.mem_coe.mpr hx))
    omega
  refine ⟨hfin, ?_⟩
  have hcard := mixture_deriv_zeros_card_le m w c σ hw hσ hc hm hfin.toFinset
    (fun x hx => hfin.mem_toFinset.mp hx)
  rwa [Set.ncard_eq_toFinset_card _ hfin]

end Window
end FormalPRR

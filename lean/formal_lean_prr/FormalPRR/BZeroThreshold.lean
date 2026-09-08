/-
  FormalPRR/BZeroThreshold.lean

  Budget-threshold well-definedness (paper: b0_quantitative_bound.tex, "B0 ASSEMBLY
  INVERSION" / A6).

  The budget threshold `B₀` is obtained by inverting a strictly monotone error
  function `E` against a margin `M`.  We prove, sorry-free:

  * `threshold_wd`      (B1)  existence + uniqueness of `B₀ > 0` with `E B₀ = M`,
                              and `E B < M` for all `B < B₀`  (IVT + strict monotonicity).
  * `margin_min`        (B2)  the two feasibility constraints reduce to `E < M` with
                              `M = min (τ·μ₁/2) (τ²·μ₂/8)`.
  * `explicit_log_form` (B3)  the closed form `B₀ = (1/a)·ln(1 + M/K)` for the concrete
                              `E B = K·(exp(a·B) − 1)`, with `E B₀ = M`, `B₀ > 0`, and the
                              full strict-monotone / below-threshold package.
-/
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Tactic

-- SCOPE: in-scope — B0 proposition of the present submission (budget-threshold
--  well-definedness, b0_quantitative_bound.tex, A6).

namespace FormalPRR.BZero

open Set

/-! ### B1 : well-definedness of the threshold -/

/-- **B1 (threshold well-definedness).**  Let `E : ℝ → ℝ` be continuous and strictly
increasing with `E 0 = 0`, let the margin `M > 0`, and suppose some point `b` witnesses
`M ≤ E b` (in the paper `E` is exp-type, hence unbounded, so such a `b` always exists).
Then there is a *unique* `B₀ > 0` with `E B₀ = M`, and `E B < M` for every `B < B₀`. -/
theorem threshold_wd (E : ℝ → ℝ) (hE : Continuous E) (hmono : StrictMono E)
    (hE0 : E 0 = 0) {M : ℝ} (hM : 0 < M) {b : ℝ} (hb : M ≤ E b) :
    ∃ B0 : ℝ, 0 < B0 ∧ E B0 = M ∧ (∀ B, B < B0 → E B < M) ∧
      (∀ B', E B' = M → B' = B0) := by
  -- The witness must lie strictly to the right of 0.
  have hb0 : 0 < b := by
    by_contra h
    rw [not_lt] at h
    have : E b ≤ E 0 := hmono.monotone h
    rw [hE0] at this
    linarith
  -- Intermediate value theorem on `[0, b]`.
  have hmem : M ∈ Icc (E 0) (E b) := by
    rw [hE0]; exact ⟨hM.le, hb⟩
  obtain ⟨c, hc_mem, hc⟩ := intermediate_value_Icc hb0.le hE.continuousOn hmem
  -- `c > 0` because `E c = M > 0 = E 0`.
  have hc0 : 0 < c := by
    rcases lt_or_eq_of_le hc_mem.1 with h | h
    · exact h
    · exfalso; rw [← h, hE0] at hc; linarith
  refine ⟨c, hc0, hc, ?_, ?_⟩
  · intro B hB
    have : E B < E c := hmono hB
    rw [hc] at this; exact this
  · intro B' hB'
    apply hmono.injective
    rw [hB', hc]

/-! ### B2 : the margin as a minimum of the two constraints -/

/-- **B2 (margin minimum).**  The two feasibility constraints of the assembly,
`(2/τ)·E < μ₁` and `2·(2/τ)²·E < μ₂`, hold simultaneously iff `E < M` where
`M = min (τ·μ₁/2) (τ²·μ₂/8)`.  Here `E` is the (nonnegative) error value. -/
theorem margin_min (τ μ1 μ2 e : ℝ) (hτ : 0 < τ) :
    ((2 / τ) * e < μ1 ∧ 2 * (2 / τ) ^ 2 * e < μ2) ↔
      e < min (τ * μ1 / 2) (τ ^ 2 * μ2 / 8) := by
  have hcomm1 : μ1 * τ = τ * μ1 := mul_comm _ _
  have e1 : (2 / τ) * e < μ1 ↔ e < τ * μ1 / 2 := by
    rw [div_mul_eq_mul_div, div_lt_iff₀ hτ]
    constructor <;> intro h <;> linarith [hcomm1]
  have e2 : 2 * (2 / τ) ^ 2 * e < μ2 ↔ e < τ ^ 2 * μ2 / 8 := by
    have hτ2 : (0 : ℝ) < τ ^ 2 := by positivity
    rw [show 2 * (2 / τ) ^ 2 * e = (8 * e) / τ ^ 2 by field_simp; ring, div_lt_iff₀ hτ2]
    constructor <;> intro h <;> nlinarith [h]
  rw [lt_min_iff, e1, e2]

/-! ### B3 : the explicit closed form -/

/-- The concrete error function `E B = K·(exp(a·B) − 1)`. -/
noncomputable def Eexp (K a B : ℝ) : ℝ := K * (Real.exp (a * B) - 1)

/-- `Eexp` is strictly increasing when `K, a > 0`. -/
theorem Eexp_strictMono {K a : ℝ} (hK : 0 < K) (ha : 0 < a) :
    StrictMono (Eexp K a) := by
  intro x y hxy
  have hexp : Real.exp (a * x) < Real.exp (a * y) := by
    apply Real.exp_lt_exp.mpr
    exact mul_lt_mul_of_pos_left hxy ha
  unfold Eexp
  nlinarith [hexp, hK]

/-- `Eexp` is continuous. -/
theorem Eexp_continuous {K a : ℝ} : Continuous (Eexp K a) := by
  unfold Eexp
  fun_prop

/-- `Eexp K a 0 = 0`. -/
@[simp] theorem Eexp_zero {K a : ℝ} : Eexp K a 0 = 0 := by
  unfold Eexp; simp

/-- **B3 (explicit closed form).**  For the concrete `E B = K·(exp(a·B) − 1)` with
`K, a > 0` and margin `M > 0`, the threshold is exactly
`B₀ = (1/a)·ln(1 + M/K)`:  it satisfies `E B₀ = M`, is strictly positive, and
`E B < M` for all `B < B₀`, and is the unique root. -/
theorem explicit_log_form {K a M : ℝ} (hK : 0 < K) (ha : 0 < a) (hM : 0 < M) :
    let B0 := (1 / a) * Real.log (1 + M / K)
    0 < B0 ∧ Eexp K a B0 = M ∧
      (∀ B, B < B0 → Eexp K a B < M) ∧
      (∀ B', Eexp K a B' = M → B' = B0) := by
  intro B0
  have hMK : 0 < M / K := div_pos hM hK
  have h1MK : (0 : ℝ) < 1 + M / K := by linarith
  have hgt1 : (1 : ℝ) < 1 + M / K := by linarith
  -- `E B0 = M`.
  have hroot : Eexp K a B0 = M := by
    have hlog : a * B0 = Real.log (1 + M / K) := by
      show a * ((1 / a) * Real.log (1 + M / K)) = Real.log (1 + M / K)
      field_simp
    unfold Eexp
    rw [hlog, Real.exp_log h1MK]
    field_simp
    ring
  -- `B0 > 0`.
  have hpos : 0 < B0 := by
    have hlogpos : 0 < Real.log (1 + M / K) := Real.log_pos hgt1
    have hainv : 0 < 1 / a := by positivity
    exact mul_pos hainv hlogpos
  refine ⟨hpos, hroot, ?_, ?_⟩
  · intro B hB
    have := (Eexp_strictMono hK ha) hB
    rw [hroot] at this; exact this
  · intro B' hB'
    apply (Eexp_strictMono hK ha).injective
    rw [hB', hroot]

/-- Corollary: the closed-form `B₀` of B3 is *the* threshold produced by the
IVT route B1.  The statement quantifies over a `threshold_wd` witness: it
runs `threshold_wd` on the concrete `Eexp K a` and identifies the produced
threshold with the closed form `(1/a)·ln(1 + M/K)` by strict-monotone
uniqueness, returning the witness together with its full B1 package
(positivity, root, below-threshold behaviour, uniqueness) AND the
closed-form identification. -/
theorem explicit_is_threshold {K a M : ℝ} (hK : 0 < K) (ha : 0 < a) (hM : 0 < M) :
    ∃ B0 : ℝ, 0 < B0 ∧ Eexp K a B0 = M ∧
      (∀ B, B < B0 → Eexp K a B < M) ∧
      (∀ B', Eexp K a B' = M → B' = B0) ∧
      B0 = (1 / a) * Real.log (1 + M / K) := by
  obtain ⟨hpos, hroot, hbelow, huniq⟩ := explicit_log_form hK ha hM
  -- Run the IVT route B1 on the concrete map, witnessing `M ≤ Eexp` at the
  -- closed form itself.
  obtain ⟨c, hc0, hcroot, hcbelow, hcuniq⟩ :=
    threshold_wd (Eexp K a) Eexp_continuous (Eexp_strictMono hK ha)
      Eexp_zero hM (le_of_eq hroot.symm)
  -- Identify the IVT threshold with the closed form by uniqueness of the root.
  exact ⟨c, hc0, hcroot, hcbelow, hcuniq, huniq c hcroot⟩

end FormalPRR.BZero

#print axioms FormalPRR.BZero.threshold_wd
#print axioms FormalPRR.BZero.margin_min
#print axioms FormalPRR.BZero.Eexp_strictMono
#print axioms FormalPRR.BZero.explicit_log_form
#print axioms FormalPRR.BZero.explicit_is_threshold

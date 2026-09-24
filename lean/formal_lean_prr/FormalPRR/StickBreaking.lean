/-
FormalPRR/StickBreaking.lean — gap-closure item L2(b) (plan TH-2 / TH-4).

Source (paper, CNSNS rebuild): the fixed-budget allocation law.  At fixed total
budget B the passage exposures are λ_j = B w_j / c_j with passage prices
c_j = W^{d−1} |μ'(t_j)| > 0 and allocation weights w_j ≥ 0, Σ w_j = 1.  The
limiting basin masses are the stick-breaking masses
    M_j = exp(−Σ_{i<j} λ_i) (1 − e^{−λ_j}),
with total window mass Σ_j M_j = 1 − exp(−Σ_j λ_j).  Inverse design: for target
masses p_j ≥ 0 with survivals S_j = 1 − Σ_{i<j} p_i and S_m > 0,
    λ_j = ln(S_j / S_{j+1}),   B = Σ_j c_j λ_j,   w_j = c_j λ_j / B.
Max–min: among all allocations at budget B the smallest mass is at most the
equal-mass level p*(B), defined by Σ_j c_j ln[(1 − j p*)/(1 − (j+1) p*)] = B
(indices from 0), with equality only for equal masses; p*(B) exists, is unique,
is strictly increasing in B and exceeds every q < 1/m for large B.

Indexing: passages are j = 0, …, m−1 (ℕ-indexed sequences restricted to
`range m`); the paper's 1-indexed formulas are the shift j ↦ j+1.

-- SCOPE NOTE: pure finite-sum algebra and one-variable real analysis.  The
-- identification of the limiting basin masses of the reaction-time law with the
-- stick-breaking masses (plan TH-2, a vanishing-noise limit theorem) is NOT
-- formalized here; this file checks the algebra that turns that limit into the
-- design law, and the max–min optimization over the design family.
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Topology.Algebra.Order.Field

namespace FormalPRR
namespace StickBreaking

open Finset

/-! ### Forward map: exposures → masses -/

/-- Cumulative exposure before passage `j`: `Σ_{i<j} λ_i`. -/
def cum (lam : ℕ → ℝ) (j : ℕ) : ℝ := ∑ i ∈ range j, lam i

/-- Stick-breaking basin mass `M_j = e^{−Σ_{i<j} λ_i} (1 − e^{−λ_j})` (plan TH-2). -/
noncomputable def mass (lam : ℕ → ℝ) (j : ℕ) : ℝ :=
  Real.exp (-cum lam j) * (1 - Real.exp (-lam j))

lemma cum_zero (lam : ℕ → ℝ) : cum lam 0 = 0 := by simp [cum]

lemma cum_succ (lam : ℕ → ℝ) (j : ℕ) : cum lam (j + 1) = cum lam j + lam j := by
  simp [cum, sum_range_succ]

/-- Each mass is a survival difference: `M_j = e^{−Σ_{i<j}λ_i} − e^{−Σ_{i≤j}λ_i}`. -/
theorem mass_eq_sub (lam : ℕ → ℝ) (j : ℕ) :
    mass lam j = Real.exp (-cum lam j) - Real.exp (-cum lam (j + 1)) := by
  rw [mass, cum_succ, neg_add, Real.exp_add]
  ring

/-- **Telescoping total mass** (plan TH-2): `Σ_{j<m} M_j = 1 − e^{−Σ_{j<m} λ_j}`. -/
theorem mass_sum (lam : ℕ → ℝ) (m : ℕ) :
    ∑ j ∈ range m, mass lam j = 1 - Real.exp (-cum lam m) := by
  induction m with
  | zero => simp [cum]
  | succ n ih =>
    rw [sum_range_succ, ih, mass_eq_sub]
    ring

/-- The window mass is strictly below one at every finite budget. -/
theorem mass_sum_lt_one (lam : ℕ → ℝ) (m : ℕ) : ∑ j ∈ range m, mass lam j < 1 := by
  rw [mass_sum]
  linarith [Real.exp_pos (-cum lam m)]

theorem mass_nonneg {lam : ℕ → ℝ} {j : ℕ} (h : 0 ≤ lam j) : 0 ≤ mass lam j := by
  have : Real.exp (-lam j) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  exact mul_nonneg (Real.exp_pos _).le (by linarith)

theorem mass_pos {lam : ℕ → ℝ} {j : ℕ} (h : 0 < lam j) : 0 < mass lam j := by
  have : Real.exp (-lam j) < 1 := Real.exp_lt_one_iff.mpr (by linarith)
  exact mul_pos (Real.exp_pos _) (by linarith)

/-- Masses up to `m` depend only on the exposures of the first `m` passages. -/
lemma mass_congr {lam lam' : ℕ → ℝ} {m : ℕ} (h : ∀ j < m, lam j = lam' j) :
    ∀ j < m, mass lam j = mass lam' j := by
  intro j hj
  have hc : cum lam j = cum lam' j :=
    sum_congr rfl fun i hi => h i (lt_trans (mem_range.1 hi) hj)
  simp only [mass, hc, h j hj]

/-! ### Inverse map: masses → exposures -/

/-- Survival before passage `j` for target masses `p`: `S_j = 1 − Σ_{i<j} p_i`. -/
def surv (p : ℕ → ℝ) (j : ℕ) : ℝ := 1 - ∑ i ∈ range j, p i

lemma surv_zero (p : ℕ → ℝ) : surv p 0 = 1 := by simp [surv]

lemma surv_succ (p : ℕ → ℝ) (j : ℕ) : surv p (j + 1) = surv p j - p j := by
  simp [surv, sum_range_succ]
  ring

/-- Survivals are nonincreasing along the window when the masses are nonnegative. -/
lemma surv_le_of_le {p : ℕ → ℝ} {m : ℕ} (hp : ∀ j < m, 0 ≤ p j) :
    ∀ j ≤ m, surv p m ≤ surv p j := by
  induction m with
  | zero => intro j hj; rw [Nat.le_zero.1 hj]
  | succ n ih =>
    intro j hj
    rcases Nat.lt_or_ge j (n + 1) with h | h
    · have h1 := ih (fun i hi => hp i (Nat.lt_succ_of_lt hi)) j (Nat.lt_succ_iff.1 h)
      rw [surv_succ]
      linarith [hp n (Nat.lt_succ_self n)]
    · rw [le_antisymm hj h]

lemma surv_pos {p : ℕ → ℝ} {m : ℕ} (hp : ∀ j < m, 0 ≤ p j) (hS : 0 < surv p m) :
    ∀ j ≤ m, 0 < surv p j :=
  fun j hj => lt_of_lt_of_le hS (surv_le_of_le hp j hj)

/-- Inverse design exposure `λ_j = ln(S_j / S_{j+1})` (plan TH-4). -/
noncomputable def lamOf (p : ℕ → ℝ) (j : ℕ) : ℝ := Real.log (surv p j / surv p (j + 1))

/-- The cumulative inverse-design exposure is `−ln S_j`. -/
theorem cum_lamOf {p : ℕ → ℝ} {m : ℕ} (hp : ∀ j < m, 0 ≤ p j) (hS : 0 < surv p m) :
    ∀ j ≤ m, cum (lamOf p) j = -Real.log (surv p j) := by
  intro j
  induction j with
  | zero => intro _; simp [cum, surv_zero]
  | succ n ih =>
    intro hn
    have h1 := surv_pos hp hS n (Nat.le_of_succ_le hn)
    have h2 := surv_pos hp hS (n + 1) hn
    rw [cum_succ, ih (Nat.le_of_succ_le hn), lamOf, Real.log_div h1.ne' h2.ne']
    ring

theorem exp_neg_cum_lamOf {p : ℕ → ℝ} {m : ℕ} (hp : ∀ j < m, 0 ≤ p j)
    (hS : 0 < surv p m) : ∀ j ≤ m, Real.exp (-cum (lamOf p) j) = surv p j := by
  intro j hj
  rw [cum_lamOf hp hS j hj, neg_neg, Real.exp_log (surv_pos hp hS j hj)]

/-- **Inverse design reproduces the targets** (plan TH-4): with
`λ_j = ln(S_j/S_{j+1})`, the stick-breaking masses equal the target masses. -/
theorem mass_lamOf {p : ℕ → ℝ} {m : ℕ} (hp : ∀ j < m, 0 ≤ p j) (hS : 0 < surv p m) :
    ∀ j < m, mass (lamOf p) j = p j := by
  intro j hj
  rw [mass_eq_sub, exp_neg_cum_lamOf hp hS j hj.le,
    exp_neg_cum_lamOf hp hS (j + 1) hj, surv_succ]
  ring

/-- Survival of the stick-breaking masses is `e^{−Σ_{i<j} λ_i}`. -/
theorem surv_mass (lam : ℕ → ℝ) (j : ℕ) : surv (mass lam) j = Real.exp (-cum lam j) := by
  rw [surv, mass_sum]
  ring

/-- **The forward map is inverted exactly** (plan TH-4): `λ_j = ln(S_j/S_{j+1})`
recovers every exposure from its stick-breaking masses (no hypotheses). -/
theorem lamOf_mass (lam : ℕ → ℝ) (j : ℕ) : lamOf (mass lam) j = lam j := by
  rw [lamOf, surv_mass, surv_mass, cum_succ, ← Real.exp_sub]
  rw [show -cum lam j - -(cum lam j + lam j) = lam j by ring, Real.log_exp]

theorem lamOf_nonneg {p : ℕ → ℝ} {j : ℕ} (hpj : 0 ≤ p j) (hS : 0 < surv p (j + 1)) :
    0 ≤ lamOf p j := by
  apply Real.log_nonneg
  rw [le_div_iff₀ hS, one_mul, surv_succ] at *
  linarith

theorem lamOf_pos {p : ℕ → ℝ} {j : ℕ} (hpj : 0 < p j) (hS : 0 < surv p (j + 1)) :
    0 < lamOf p j := by
  apply Real.log_pos
  rw [lt_div_iff₀ hS, one_mul]
  rw [surv_succ]
  linarith

/-! ### Budget identity -/

/-- Exposure from an allocation: `λ_j = B w_j / c_j` (plan TH-2). -/
noncomputable def alloc (B : ℝ) (c w : ℕ → ℝ) (j : ℕ) : ℝ := B * w j / c j

/-- **Budget identity** (plan TH-4): `Σ_j c_j λ_j = B` for every allocation
`Σ_j w_j = 1` — the budget buys log-survival at price `c_j`. -/
theorem budget_of_alloc {m : ℕ} {B : ℝ} {c w : ℕ → ℝ} (hc : ∀ j < m, c j ≠ 0)
    (hw : ∑ j ∈ range m, w j = 1) : ∑ j ∈ range m, c j * alloc B c w j = B := by
  calc ∑ j ∈ range m, c j * alloc B c w j = ∑ j ∈ range m, B * w j := by
        refine sum_congr rfl fun j hj => ?_
        have := hc j (mem_range.1 hj)
        rw [alloc]
        field_simp
    _ = B * ∑ j ∈ range m, w j := (mul_sum _ _ _).symm
    _ = B := by rw [hw, mul_one]

/-- **Allocation from exposures** (plan TH-4): for exposures `λ` with
`B = Σ c_j λ_j > 0`, the weights `w_j = c_j λ_j / B` sum to one and reproduce `λ`. -/
theorem alloc_of_lam {m : ℕ} {c lam : ℕ → ℝ} (hc : ∀ j < m, c j ≠ 0)
    (hB : 0 < ∑ j ∈ range m, c j * lam j) :
    (∑ j ∈ range m, c j * lam j / (∑ i ∈ range m, c i * lam i) = 1) ∧
    ∀ j < m, alloc (∑ i ∈ range m, c i * lam i) c
      (fun j => c j * lam j / (∑ i ∈ range m, c i * lam i)) j = lam j := by
  refine ⟨?_, ?_⟩
  · rw [← sum_div, div_self hB.ne']
  · intro j hj
    have := hc j hj
    simp only [alloc]
    field_simp

/-! ### Design cost and its monotonicity -/

/-- Budget needed to realize target masses `p`: `C(p) = Σ_{j<m} c_j ln(S_j/S_{j+1})`. -/
noncomputable def cost (m : ℕ) (c p : ℕ → ℝ) : ℝ := ∑ j ∈ range m, c j * lamOf p j

/-- Survivals are antitone in the target masses. -/
lemma surv_le_surv {p q : ℕ → ℝ} {m : ℕ} (hpq : ∀ j < m, p j ≤ q j) :
    ∀ j ≤ m, surv q j ≤ surv p j := by
  intro j hj
  simp only [surv]
  have : ∑ i ∈ range j, p i ≤ ∑ i ∈ range j, q i :=
    sum_le_sum fun i hi => hpq i (lt_of_lt_of_le (mem_range.1 hi) hj)
  linarith

/-- **Componentwise monotonicity of the inverse design** (plan TH-4): raising any
target mass never lowers any exposure. -/
theorem lamOf_le_of_le {p q : ℕ → ℝ} {m : ℕ} (hp : ∀ j < m, 0 ≤ p j)
    (hpq : ∀ j < m, p j ≤ q j) (hSq : 0 < surv q m) :
    ∀ j < m, lamOf p j ≤ lamOf q j := by
  have hq : ∀ j < m, 0 ≤ q j := fun j hj => le_trans (hp j hj) (hpq j hj)
  have hSqpos := surv_pos hq hSq
  intro j hj
  have hb := hSqpos j hj.le
  have hb1 := hSqpos (j + 1) hj
  have hab := surv_le_surv hpq j hj.le
  have hab1 := surv_le_surv hpq (j + 1) hj
  have ha1 : 0 < surv p (j + 1) := lt_of_lt_of_le hb1 hab1
  rw [lamOf, lamOf]
  apply Real.log_le_log (div_pos (lt_of_lt_of_le hb hab) ha1)
  rw [div_le_div_iff₀ ha1 hb1, surv_succ, surv_succ]
  rw [surv_succ] at ha1 hb1
  have hx := hp j hj
  have hxy := hpq j hj
  nlinarith [mul_le_mul_of_nonneg_right hab hx,
    mul_le_mul_of_nonneg_left hxy (lt_of_lt_of_le hb hab).le]

/-- Strict version: raising target mass `j` strictly raises exposure `j`. -/
theorem lamOf_lt_of_lt {p q : ℕ → ℝ} {m : ℕ} (hp : ∀ j < m, 0 ≤ p j)
    (hpq : ∀ j < m, p j ≤ q j) (hSq : 0 < surv q m) {k : ℕ} (hk : k < m)
    (hlt : p k < q k) : lamOf p k < lamOf q k := by
  have hq : ∀ j < m, 0 ≤ q j := fun j hj => le_trans (hp j hj) (hpq j hj)
  have hSqpos := surv_pos hq hSq
  have hb := hSqpos k hk.le
  have hb1 := hSqpos (k + 1) hk
  have hab := surv_le_surv hpq k hk.le
  have hab1 := surv_le_surv hpq (k + 1) hk
  have ha1 : 0 < surv p (k + 1) := lt_of_lt_of_le hb1 hab1
  have ha : 0 < surv p k := lt_of_lt_of_le hb hab
  rw [lamOf, lamOf]
  apply Real.log_lt_log (div_pos ha ha1)
  rw [div_lt_div_iff₀ ha1 hb1, surv_succ, surv_succ]
  rw [surv_succ] at ha1 hb1
  have hx := hp k hk
  nlinarith [mul_le_mul_of_nonneg_right hab hx, mul_lt_mul_of_pos_left hlt ha]

theorem cost_le_of_le {m : ℕ} {c p q : ℕ → ℝ} (hc : ∀ j < m, 0 ≤ c j)
    (hp : ∀ j < m, 0 ≤ p j) (hpq : ∀ j < m, p j ≤ q j) (hSq : 0 < surv q m) :
    cost m c p ≤ cost m c q :=
  sum_le_sum fun j hj => mul_le_mul_of_nonneg_left
    (lamOf_le_of_le hp hpq hSq j (mem_range.1 hj)) (hc j (mem_range.1 hj))

theorem cost_lt_of_lt {m : ℕ} {c p q : ℕ → ℝ} (hc : ∀ j < m, 0 < c j)
    (hp : ∀ j < m, 0 ≤ p j) (hpq : ∀ j < m, p j ≤ q j) (hSq : 0 < surv q m)
    {k : ℕ} (hk : k < m) (hlt : p k < q k) : cost m c p < cost m c q := by
  apply sum_lt_sum
  · exact fun j hj => mul_le_mul_of_nonneg_left
      (lamOf_le_of_le hp hpq hSq j (mem_range.1 hj)) (hc j (mem_range.1 hj)).le
  · exact ⟨k, mem_range.2 hk,
      mul_lt_mul_of_pos_left (lamOf_lt_of_lt hp hpq hSq hk hlt) (hc k hk)⟩

/-! ### Max–min ⇒ equal masses -/

/-- Survival of the equal-mass vector: `S_j = 1 − j q`. -/
lemma surv_const (q : ℝ) (j : ℕ) : surv (fun _ => q) j = 1 - j * q := by
  simp [surv, sum_const, card_range, nsmul_eq_mul]

/-- **Closed form of the equal-mass budget** (plan TH-4):
`C(q·1) = Σ_{j<m} c_j ln[(1 − j q)/(1 − (j+1) q)]`. -/
theorem cost_const (m : ℕ) (c : ℕ → ℝ) (q : ℝ) :
    cost m c (fun _ => q) =
      ∑ j ∈ range m, c j * Real.log ((1 - j * q) / (1 - (j + 1) * q)) := by
  refine sum_congr rfl fun j _ => ?_
  rw [lamOf, surv_const, surv_const]
  push_cast
  ring_nf

/-- **Max–min bound** (plan TH-4): if target masses `p` are realizable at the
budget `C(p)` and the equal-mass level `q ≥ 0` has the same budget,
`C(q·1) = C(p)`, then every lower bound `r` of the masses satisfies `r ≤ q`;
i.e. `min_j p_j ≤ p*`. -/
theorem maxmin_le {m : ℕ} (hm : 0 < m) {c p : ℕ → ℝ} (hc : ∀ j < m, 0 < c j)
    (hS : 0 < surv p m) {q r : ℝ} (hq : 0 ≤ q)
    (hbudget : cost m c (fun _ => q) = cost m c p) (hr : ∀ j < m, r ≤ p j) : r ≤ q := by
  by_contra h
  have hqr : q < r := not_le.mp h
  have hr0 : 0 ≤ r := le_trans hq hqr.le
  have hSr : 0 < surv (fun _ => r) m := lt_of_lt_of_le hS (surv_le_surv hr m le_rfl)
  have h1 : cost m c (fun _ => q) < cost m c (fun _ => r) :=
    cost_lt_of_lt hc (fun _ _ => hq) (fun _ _ => hqr.le) hSr hm hqr
  have h2 : cost m c (fun _ => r) ≤ cost m c p :=
    cost_le_of_le (fun j hj => (hc j hj).le) (fun _ _ => hr0) hr hS
  linarith

/-- **Equality case** (plan TH-4): if all masses are at least the equal-mass
level `q` at the same budget, they all equal `q` — the max–min optimum has
equal masses and is unique. -/
theorem maxmin_eq {m : ℕ} {c p : ℕ → ℝ} (hc : ∀ j < m, 0 < c j)
    (hS : 0 < surv p m) {q : ℝ} (hq : 0 ≤ q)
    (hbudget : cost m c (fun _ => q) = cost m c p) (hr : ∀ j < m, q ≤ p j) :
    ∀ j < m, p j = q := by
  intro j hj
  by_contra hne
  have hlt : q < p j := lt_of_le_of_ne (hr j hj) (Ne.symm hne)
  have := cost_lt_of_lt hc (fun _ _ => hq) hr hS hj hlt
  linarith

/-- **Max–min over allocations** (plan TH-4): for any allocation `w` with
`Σ w = 1` (nonnegativity of `w` is not even needed), at budget `B` with prices `c > 0`, every lower bound of the
resulting basin masses is at most the equal-mass level `p*` defined by
`C(p*·1) = B`. -/
theorem maxmin_alloc {m : ℕ} (hm : 0 < m) {B : ℝ} {c w : ℕ → ℝ}
    (hc : ∀ j < m, 0 < c j) (hw1 : ∑ j ∈ range m, w j = 1)
    {q : ℝ} (hq : 0 ≤ q) (hstar : cost m c (fun _ => q) = B)
    {r : ℝ} (hr : ∀ j < m, r ≤ mass (alloc B c w) j) : r ≤ q := by
  have hcost : cost m c (mass (alloc B c w)) = B := by
    simp only [cost, lamOf_mass]
    exact budget_of_alloc (fun j hj => (hc j hj).ne') hw1
  refine maxmin_le hm hc ?_ hq ?_ hr
  · rw [surv_mass]; exact Real.exp_pos _
  · rw [hstar, hcost]

/-- **The max–min allocation is unique and explicit** (plan TH-4): if every
basin mass of the allocation `w` is at least `p*`, then all masses equal `p*`
and `w_j = c_j ln[(1 − j p*)/(1 − (j+1) p*)] / B`. -/
theorem maxmin_alloc_eq {m : ℕ} {B : ℝ} {c w : ℕ → ℝ}
    (hc : ∀ j < m, 0 < c j) (hw1 : ∑ j ∈ range m, w j = 1)
    (hB : 0 < B) {q : ℝ} (hq : 0 ≤ q) (hstar : cost m c (fun _ => q) = B)
    (hr : ∀ j < m, q ≤ mass (alloc B c w) j) :
    (∀ j < m, mass (alloc B c w) j = q) ∧
    ∀ j < m, w j = c j * Real.log ((1 - j * q) / (1 - (j + 1) * q)) / B := by
  have hcost : cost m c (mass (alloc B c w)) = B := by
    simp only [cost, lamOf_mass]
    exact budget_of_alloc (fun j hj => (hc j hj).ne') hw1
  have hS : 0 < surv (mass (alloc B c w)) m := by rw [surv_mass]; exact Real.exp_pos _
  have heq := maxmin_eq hc hS hq (by rw [hstar, hcost]) hr
  refine ⟨heq, fun j hj => ?_⟩
  -- the exposure of passage `j` is determined by the (equal) masses up to `j`
  have hsurv : ∀ i ≤ m, surv (mass (alloc B c w)) i = surv (fun _ => q) i := by
    intro i hi
    simp only [surv]
    congr 1
    exact sum_congr rfl fun k hk => heq k (lt_of_lt_of_le (mem_range.1 hk) hi)
  have hlam : alloc B c w j = lamOf (fun _ => q) j := by
    rw [← lamOf_mass (alloc B c w) j, lamOf, lamOf, hsurv j hj.le, hsurv (j + 1) hj]
  have hlam' : lamOf (fun _ => q) j = Real.log ((1 - j * q) / (1 - (j + 1) * q)) := by
    rw [lamOf, surv_const, surv_const]
    push_cast
    ring_nf
  rw [alloc, hlam'] at hlam
  have hcj := (hc j hj).ne'
  field_simp
  field_simp at hlam
  linarith

/-! ### The equal-mass level p*(B): existence, uniqueness, monotonicity -/

/-- **Strict monotonicity of the equal-mass budget** (plan TH-4): for admissible
levels `0 ≤ q, q'` with `m q' < 1`, `C(q·1) < C(q'·1) ↔ q < q'`.  Hence `p*(B)`
is unique and strictly increasing in `B`, and `p*(B) > q` as soon as
`B > C(q·1)` (so `p*(B) → 1/m` as `B → ∞`). -/
theorem cost_const_lt_iff {m : ℕ} (hm : 0 < m) {c : ℕ → ℝ} (hc : ∀ j < m, 0 < c j)
    {q q' : ℝ} (hq : 0 ≤ q) (hq' : 0 ≤ q') (hqm : (m : ℝ) * q < 1)
    (hq'm : (m : ℝ) * q' < 1) :
    cost m c (fun _ => q) < cost m c (fun _ => q') ↔ q < q' := by
  have hS : ∀ x : ℝ, (m : ℝ) * x < 1 → 0 < surv (fun _ => x) m := by
    intro x hx; rw [surv_const]; linarith
  constructor
  · intro h
    by_contra hle
    have hle : q' ≤ q := not_lt.mp hle
    have := cost_le_of_le (fun j hj => (hc j hj).le) (fun _ _ => hq') (fun _ _ => hle)
      (hS q hqm)
    linarith
  · intro h
    exact cost_lt_of_lt hc (fun _ _ => hq) (fun _ _ => h.le) (hS q' hq'm) hm h

/-- Uniqueness of the equal-mass level at a given budget. -/
theorem pstar_unique {m : ℕ} (hm : 0 < m) {c : ℕ → ℝ} (hc : ∀ j < m, 0 < c j)
    {q q' : ℝ} (hq : 0 ≤ q) (hq' : 0 ≤ q') (hqm : (m : ℝ) * q < 1)
    (hq'm : (m : ℝ) * q' < 1) (h : cost m c (fun _ => q) = cost m c (fun _ => q')) :
    q = q' := by
  rcases lt_trichotomy q q' with hlt | heq | hgt
  · have := (cost_const_lt_iff hm hc hq hq' hqm hq'm).2 hlt; linarith
  · exact heq
  · have := (cost_const_lt_iff hm hc hq' hq hq'm hqm).2 hgt; linarith

/-- **Existence of the equal-mass level** (plan TH-4): for every budget `B ≥ 0`
there is `q ∈ [0, 1/m)` with `Σ_{j<m} c_j ln[(1 − j q)/(1 − (j+1) q)] = B`. -/
theorem pstar_exists {m : ℕ} (hm : 0 < m) {c : ℕ → ℝ} (hc : ∀ j < m, 0 < c j)
    {B : ℝ} (hB : 0 ≤ B) :
    ∃ q, 0 ≤ q ∧ (m : ℝ) * q < 1 ∧ cost m c (fun _ => q) = B := by
  obtain ⟨n, rfl⟩ : ∃ n, m = n + 1 := ⟨m - 1, by omega⟩
  have hcn : 0 < c n := hc n (Nat.lt_succ_self n)
  set E := Real.exp (B / c n) with hE
  have hE1 : 1 ≤ E := Real.one_le_exp (div_nonneg hB hcn.le)
  have hN : ((n + 1 : ℕ) : ℝ) = (n : ℝ) + 1 := by push_cast; ring
  obtain ⟨D, hDdef⟩ : ∃ D : ℝ, D = E * ((n : ℝ) + 1) - ((n : ℝ) + 1) + 1 := ⟨_, rfl⟩
  have hden : 0 < D := by
    have : 0 ≤ (E - 1) * ((n : ℝ) + 1) := mul_nonneg (by linarith) (by positivity)
    rw [hDdef]
    nlinarith
  obtain ⟨q₁, hq₁⟩ : ∃ q : ℝ, q = (E - 1) / D := ⟨_, rfl⟩
  have hq₁0 : 0 ≤ q₁ := by rw [hq₁]; exact div_nonneg (by linarith) hden.le
  have hq₁m : ((n + 1 : ℕ) : ℝ) * q₁ < 1 := by
    rw [hN, hq₁, mul_div_assoc', div_lt_one hden, hDdef]
    nlinarith
  -- admissibility on [0, q₁]
  have hsurvpos : ∀ x ∈ Set.Icc 0 q₁, ∀ j ≤ n + 1, 0 < 1 - (j : ℝ) * x := by
    intro x hx j hj
    have hj' : (j : ℝ) ≤ (n : ℝ) + 1 := by exact_mod_cast hj
    have : (j : ℝ) * x ≤ ((n : ℝ) + 1) * q₁ :=
      mul_le_mul hj' hx.2 hx.1 (by positivity)
    rw [hN] at hq₁m
    linarith
  -- continuity of the closed form on [0, q₁]
  have hcont : ContinuousOn (fun x => cost (n + 1) c (fun _ => x)) (Set.Icc 0 q₁) := by
    have : (fun x => cost (n + 1) c (fun _ => x)) = fun x =>
        ∑ j ∈ range (n + 1), c j * Real.log ((1 - j * x) / (1 - (j + 1) * x)) := by
      funext x; exact cost_const _ _ _
    rw [this]
    apply continuousOn_finsetSum
    intro j hj
    apply ContinuousOn.mul continuousOn_const
    apply ContinuousOn.log
    · apply ContinuousOn.div (by fun_prop) (by fun_prop)
      intro x hx
      have := hsurvpos x hx (j + 1) (mem_range.1 hj)
      push_cast at this
      exact this.ne'
    · intro x hx
      have h1 := hsurvpos x hx j (mem_range.1 hj).le
      have h2 := hsurvpos x hx (j + 1) (mem_range.1 hj)
      push_cast at h2
      exact (div_pos h1 h2).ne'
  -- endpoint values
  have hval0 : cost (n + 1) c (fun _ => (0 : ℝ)) = 0 := by
    rw [cost_const]; simp
  have hvalq : B ≤ cost (n + 1) c (fun _ => q₁) := by
    have hnn : ∀ j ∈ range (n + 1), 0 ≤ c j * lamOf (fun _ => q₁) j := by
      intro j hj
      refine mul_nonneg (hc j (mem_range.1 hj)).le (lamOf_nonneg hq₁0 ?_)
      rw [surv_const]
      have := hsurvpos q₁ ⟨hq₁0, le_rfl⟩ (j + 1) (mem_range.1 hj)
      exact this
    have hlast : c n * lamOf (fun _ => q₁) n = B := by
      rw [lamOf, surv_const, surv_const]
      have hD := hden.ne'
      have h1 : 1 - ((n : ℝ) + 1) * q₁ = 1 / D := by
        rw [hq₁]; field_simp; linarith
      have h2 : 1 - (n : ℝ) * q₁ = E / D := by
        rw [hq₁]; field_simp; linarith
      have hratio : (1 - (n : ℝ) * q₁) / (1 - ((n + 1 : ℕ) : ℝ) * q₁) = E := by
        rw [hN, h1, h2]
        field_simp
      rw [hratio, hE, Real.log_exp]
      field_simp
    calc B = c n * lamOf (fun _ => q₁) n := hlast.symm
      _ ≤ ∑ j ∈ range (n + 1), c j * lamOf (fun _ => q₁) j :=
          single_le_sum hnn (mem_range.2 (Nat.lt_succ_self n))
      _ = cost (n + 1) c (fun _ => q₁) := rfl
  obtain ⟨q, hq, hqB⟩ := intermediate_value_Icc hq₁0 hcont
    ⟨by rw [hval0]; exact hB, hvalq⟩
  refine ⟨q, hq.1, ?_, hqB⟩
  have : ((n + 1 : ℕ) : ℝ) * q ≤ ((n + 1 : ℕ) : ℝ) * q₁ :=
    mul_le_mul_of_nonneg_left hq.2 (by positivity)
  linarith

end StickBreaking
end FormalPRR

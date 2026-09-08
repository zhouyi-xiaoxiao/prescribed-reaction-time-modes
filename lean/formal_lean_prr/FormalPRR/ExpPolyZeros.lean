/-
FormalPRR/ExpPolyZeros.lean — Target A1 (distinct-zeros exponential-polynomial bound).

Source (paper): exact_m_theorem_spine.tex, Eq. (eq:exact-m-exp-polynomial):
the exponential polynomial Σ_{j=1..m} (a_j + b_j x) e^{λ_j x}, λ_1 < ... < λ_m,
"has at most 2m−1 real zeros counted with multiplicity".
Full proof source: exact_m_theorem_full_proof.tex, Lemma (lem:exactmfull-zero-bound),
Eqs. (eq:exactmfull-general-exp-polynomial) and (eq:exactmfull-rolle-count):
generalized Rolle induction — multiply by e^{−λ₁x}, differentiate twice to kill the
lowest affine term, induct on m.

This file delivers BOTH versions: the distinct-zeros bound (any finite set of
real zeros has cardinality ≤ 2m−1) AND, in the `Multiplicity` section, the
paper's full count with multiplicity (`expPoly_zeros_with_multiplicity_le`).
The original distinct-only SCOPE NOTE is resolved.
-/
import Mathlib.Analysis.Calculus.LocalExtr.Rolle
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

namespace FormalPRR
namespace ExpPoly

open Finset

/-- The exponential polynomial `P_m(x) = Σ_{j} (a_j + b_j x) e^{λ_j x}`
(Eq. eq:exactmfull-general-exp-polynomial of exact_m_theorem_full_proof.tex,
Eq. eq:exact-m-exp-polynomial of exact_m_theorem_spine.tex). -/
noncomputable def expPoly (m : ℕ) (lam a b : Fin m → ℝ) (x : ℝ) : ℝ :=
  ∑ j, (a j + b j * x) * Real.exp (lam j * x)

/-- Transport a `HasDerivAt` along an equality of derivative values.
Internal helper (proof machinery); no direct paper display. -/
lemma hasDerivAt_deriv_congr {f : ℝ → ℝ} {u v x : ℝ}
    (h : HasDerivAt f u x) (huv : u = v) : HasDerivAt f v x := huv ▸ h

/-- Derivative of one affine–exponential term `(p + q y) e^{μ y}`.
Internal helper (calculus step for the Rolle induction); no direct paper
display. -/
lemma hasDerivAt_affine_exp (p q mu x : ℝ) :
    HasDerivAt (fun y => (p + q * y) * Real.exp (mu * y))
      ((p * mu + q + q * mu * x) * Real.exp (mu * x)) x := by
  have h1 : HasDerivAt (fun y : ℝ => p + q * y) q x := by
    simpa using ((hasDerivAt_id x).const_mul q).const_add p
  have h2 : HasDerivAt (fun y : ℝ => Real.exp (mu * y)) (mu * Real.exp (mu * x)) x := by
    simpa [mul_comm] using ((hasDerivAt_id x).const_mul mu).exp
  refine hasDerivAt_deriv_congr (h1.mul h2) ?_
  ring

/-- An affine function with a nonvanishing coefficient pair has at most one
real zero.  Internal helper (base case of the Rolle induction of Lemma
lem:exactmfull-zero-bound); no direct paper display. -/
lemma affine_zeros_card_le (p q : ℝ) (hpq : p ≠ 0 ∨ q ≠ 0) (s : Finset ℝ)
    (hs : ∀ x ∈ s, p + q * x = 0) : s.card ≤ 1 := by
  rcases eq_or_ne q 0 with hq | hq
  · have hp : p ≠ 0 := by
      rcases hpq with h | h
      · exact h
      · exact absurd hq h
    have hempty : s = ∅ := by
      refine Finset.eq_empty_of_forall_notMem fun x hx => ?_
      have := hs x hx
      rw [hq, zero_mul, add_zero] at this
      exact hp this
    simp [hempty]
  · refine Finset.card_le_one.mpr fun x hx y hy => ?_
    have h1 := hs x hx
    have h2 := hs y hy
    have : q * x = q * y := by linarith
    exact mul_left_cancel₀ hq this

/-- **One generalized-Rolle step** ("ordinary Rolle supplies a root between each pair of
distinct roots", proof of Lemma lem:exactmfull-zero-bound in
exact_m_theorem_full_proof.tex): a finite set of distinct zeros of `g` produces a finite
set of `s.card − 1` distinct zeros of its derivative `g'`. -/
lemma exists_zeros_deriv (g g' : ℝ → ℝ) (hg : ∀ x, HasDerivAt g (g' x) x)
    (s : Finset ℝ) (hs : ∀ x ∈ s, g x = 0) :
    ∃ t : Finset ℝ, t.card = s.card - 1 ∧ ∀ y ∈ t, g' y = 0 := by
  rcases Nat.lt_or_ge s.card 2 with hsmall | hbig
  · refine ⟨∅, ?_, by simp⟩
    simp only [Finset.card_empty]
    omega
  · obtain ⟨K, hK⟩ : ∃ K, s.card = K + 1 := ⟨s.card - 1, by omega⟩
    set e := s.orderEmbOfFin hK with he_def
    have hgcont : Continuous g := continuous_iff_continuousAt.mpr fun y => (hg y).continuousAt
    have key : ∀ i : Fin K, ∃ y ∈ Set.Ioo (e i.castSucc) (e i.succ), g' y = 0 := by
      intro i
      have hlt : e i.castSucc < e i.succ := e.strictMono (Fin.castSucc_lt_succ (i := i))
      have h1 : g (e i.castSucc) = 0 := hs _ (Finset.orderEmbOfFin_mem s hK _)
      have h2 : g (e i.succ) = 0 := hs _ (Finset.orderEmbOfFin_mem s hK _)
      exact exists_hasDerivAt_eq_zero hlt hgcont.continuousOn (h1.trans h2.symm)
        (fun z _ => hg z)
    choose Y hYmem hY0 using key
    have hYmono : StrictMono Y := by
      intro i j hij
      have hij' : (i.succ : Fin (K + 1)) ≤ j.castSucc := by
        rw [Fin.le_def]
        simp only [Fin.val_succ, Fin.val_castSucc]
        exact hij
      have h3 : e i.succ ≤ e j.castSucc := e.monotone hij'
      have h4 := (hYmem i).2
      have h5 := (hYmem j).1
      linarith
    refine ⟨Finset.image Y Finset.univ, ?_, ?_⟩
    · rw [Finset.card_image_of_injective _ hYmono.injective, Finset.card_univ,
        Fintype.card_fin, hK]
      omega
    · intro y hy
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.mp hy
      exact hY0 i

/-- Derivative of the reduced function `g(x) = (a₀ + b₀ x) + Σ_j (α_j + β_j x) e^{μ_j x}`
(the function `Q = e^{−λ₁x} P_m` of Lemma lem:exactmfull-zero-bound). -/
lemma hasDerivAt_headTail (m : ℕ) (mu α β : Fin m → ℝ) (a0 b0 x : ℝ) :
    HasDerivAt
      (fun y => (a0 + b0 * y) + ∑ j, (α j + β j * y) * Real.exp (mu j * y))
      (b0 + ∑ j, (α j * mu j + β j + β j * mu j * x) * Real.exp (mu j * x)) x := by
  have hhead : HasDerivAt (fun y : ℝ => a0 + b0 * y) b0 x := by
    simpa using ((hasDerivAt_id x).const_mul b0).const_add a0
  have htail := HasDerivAt.fun_sum (u := (Finset.univ : Finset (Fin m)))
    (fun j _ => hasDerivAt_affine_exp (α j) (β j) (mu j) x)
  exact hhead.add htail

/-- Second-derivative step: `g'` is again a head-plus-affine-exponential sum, and its
derivative annihilates the constant head ("two derivatives annihilate the first affine
term", Lemma lem:exactmfull-zero-bound). -/
lemma hasDerivAt_tail1 (m : ℕ) (mu α β : Fin m → ℝ) (b0 x : ℝ) :
    HasDerivAt
      (fun y => b0 + ∑ j, (α j * mu j + β j + β j * mu j * y) * Real.exp (mu j * y))
      (∑ j, (α j * mu j ^ 2 + 2 * β j * mu j + β j * mu j ^ 2 * x)
        * Real.exp (mu j * x)) x := by
  have htail := HasDerivAt.fun_sum (u := (Finset.univ : Finset (Fin m)))
    (fun j _ => hasDerivAt_affine_exp (α j * mu j + β j) (β j * mu j) (mu j) x)
  have h := htail.const_add b0
  refine hasDerivAt_deriv_congr h ?_
  exact Finset.sum_congr rfl fun j _ => by ring

/-- A1 core (distinct-zeros version; Eq. eq:exact-m-exp-polynomial of
exact_m_theorem_spine.tex / Lemma lem:exactmfull-zero-bound of
exact_m_theorem_full_proof.tex), stated over the `expPoly` wrapper: any
finite set of real zeros of `P_m` has cardinality at most `2m − 1`.
The quotable public form is `expPoly_distinct_zeros_card_le`. -/
theorem expPoly_card_le :
    ∀ (m : ℕ) (lam a b : Fin m → ℝ), StrictMono lam →
      (∃ j, a j ≠ 0 ∨ b j ≠ 0) →
      ∀ s : Finset ℝ, (∀ x ∈ s, expPoly m lam a b x = 0) → s.card ≤ 2 * m - 1 := by
  intro m
  induction m with
  | zero =>
    intro lam a b _ hab s _
    obtain ⟨j, -⟩ := hab
    exact j.elim0
  | succ m ih =>
    intro lam a b hlam hab s hs
    by_cases hrest : ∀ j : Fin m, a j.succ = 0 ∧ b j.succ = 0
    · -- Tail coefficients all vanish: P = (a₀ + b₀x) e^{λ₀x}, at most one zero.
      have hzero : ∀ x ∈ s, a 0 + b 0 * x = 0 := by
        intro x hx
        have h := hs x hx
        simp only [expPoly, Fin.sum_univ_succ] at h
        have htail : ∑ j : Fin m,
            (a j.succ + b j.succ * x) * Real.exp (lam j.succ * x) = 0 :=
          Finset.sum_eq_zero fun j _ => by rw [(hrest j).1, (hrest j).2]; ring
        rw [htail, add_zero] at h
        exact (mul_eq_zero.mp h).resolve_right (Real.exp_ne_zero _)
      have hab0 : a 0 ≠ 0 ∨ b 0 ≠ 0 := by
        obtain ⟨j, hj⟩ := hab
        rcases eq_or_ne j 0 with rfl | hj0
        · exact hj
        · exfalso
          have h := hrest (j.pred hj0)
          rw [Fin.succ_pred] at h
          rcases hj with h1 | h1
          · exact h1 h.1
          · exact h1 h.2
      have := affine_zeros_card_le (a 0) (b 0) hab0 s hzero
      omega
    · -- Genuine tail: multiply by e^{−λ₀x}, differentiate twice, apply the IH.
      rw [not_forall] at hrest
      obtain ⟨j0, hj0⟩ := hrest
      have hm1 : 0 < m := j0.pos
      set mu : Fin m → ℝ := fun j => lam j.succ - lam 0 with hmu_def
      have hmu_pos : ∀ j, 0 < mu j := fun j => sub_pos.mpr (hlam (Fin.succ_pos j))
      -- Zeros of P are zeros of g = e^{−λ₀x}·P.
      have hg_eq : ∀ x : ℝ,
          ((a 0 + b 0 * x) + ∑ j, (a j.succ + b j.succ * x) * Real.exp (mu j * x))
            = Real.exp (-lam 0 * x) * expPoly (m + 1) lam a b x := by
        intro x
        simp only [expPoly]
        rw [Fin.sum_univ_succ, mul_add, Finset.mul_sum]
        congr 1
        · rw [mul_comm (Real.exp (-lam 0 * x)), mul_assoc, ← Real.exp_add]
          have hz : lam 0 * x + -lam 0 * x = 0 := by ring
          rw [hz, Real.exp_zero, mul_one]
        · refine Finset.sum_congr rfl fun j _ => ?_
          rw [mul_comm (Real.exp (-lam 0 * x)), mul_assoc, ← Real.exp_add]
          have hz : lam j.succ * x + -lam 0 * x = mu j * x := by
            simp only [hmu_def]; ring
          rw [hz]
      have hgs : ∀ x ∈ s,
          ((a 0 + b 0 * x) + ∑ j, (a j.succ + b j.succ * x) * Real.exp (mu j * x)) = 0 := by
        intro x hx
        rw [hg_eq x, hs x hx, mul_zero]
      -- First Rolle step: g ↦ g'.
      obtain ⟨t, ht_card, ht0⟩ := exists_zeros_deriv
        (fun x => (a 0 + b 0 * x) + ∑ j, (a j.succ + b j.succ * x) * Real.exp (mu j * x))
        (fun x => b 0 + ∑ j,
          (a j.succ * mu j + b j.succ + b j.succ * mu j * x) * Real.exp (mu j * x))
        (fun x => hasDerivAt_headTail m mu (fun j => a j.succ) (fun j => b j.succ)
          (a 0) (b 0) x)
        s hgs
      -- Second Rolle step: g' ↦ g''.
      obtain ⟨u, hu_card, hu0⟩ := exists_zeros_deriv
        (fun x => b 0 + ∑ j,
          (a j.succ * mu j + b j.succ + b j.succ * mu j * x) * Real.exp (mu j * x))
        (fun x => ∑ j,
          (a j.succ * mu j ^ 2 + 2 * b j.succ * mu j + b j.succ * mu j ^ 2 * x)
            * Real.exp (mu j * x))
        (fun x => hasDerivAt_tail1 m mu (fun j => a j.succ) (fun j => b j.succ) (b 0) x)
        t ht0
      -- g'' is an exponential polynomial with m terms; apply the induction hypothesis.
      have hmu_mono : StrictMono mu := by
        intro i j hij
        have h := hlam (Fin.succ_lt_succ_iff.mpr hij)
        simp only [hmu_def]
        exact sub_lt_sub_right h (lam 0)
      have hABne : ∃ j : Fin m,
          (a j.succ * mu j ^ 2 + 2 * b j.succ * mu j ≠ 0) ∨ (b j.succ * mu j ^ 2 ≠ 0) := by
        have hmu0 : mu j0 ≠ 0 := ne_of_gt (hmu_pos j0)
        rcases eq_or_ne (b j0.succ) 0 with hb0 | hb0
        · have ha0 : a j0.succ ≠ 0 := fun ha => hj0 ⟨ha, hb0⟩
          refine ⟨j0, Or.inl ?_⟩
          rw [hb0]
          simpa using mul_ne_zero ha0 (pow_ne_zero 2 hmu0)
        · exact ⟨j0, Or.inr (mul_ne_zero hb0 (pow_ne_zero 2 hmu0))⟩
      have hIH := ih mu (fun j => a j.succ * mu j ^ 2 + 2 * b j.succ * mu j)
        (fun j => b j.succ * mu j ^ 2) hmu_mono hABne u (fun x hx => hu0 x hx)
      omega

/-- **Distinct-real-zeros bound for exponential polynomials** — the distinct-zeros
version of Eq. (eq:exact-m-exp-polynomial) of exact_m_theorem_spine.tex and of
Lemma (lem:exactmfull-zero-bound) / Eq. (eq:exactmfull-general-exp-polynomial) of
exact_m_theorem_full_proof.tex: for λ₁ < ⋯ < λ_m and coefficients (a_j, b_j) not all
zero, `Σ_j (a_j + b_j x) e^{λ_j x}` has at most `2m − 1` distinct real zeros — i.e.
every finite set of its real zeros has cardinality ≤ 2m − 1.

The hypothesis `∃ j, a j ≠ 0 ∨ b j ≠ 0` is weaker than the paper's "no identically
zero summand" (some summand nonzero vs. every summand nonzero), so this statement is
more general than the paper's.

-- SCOPE NOTE (resolved): this theorem counts DISTINCT zeros only, while the paper's
-- Lemma (lem:exactmfull-zero-bound) counts zeros WITH MULTIPLICITY.  That stronger
-- with-multiplicity count (target B2) IS delivered in this file as
-- `expPoly_zeros_with_multiplicity_le` (take ν ≡ 1 to recover this statement), so no
-- gap remains; this distinct-zeros form is kept as the directly quotable A1 deliverable. -/
theorem expPoly_distinct_zeros_card_le (m : ℕ) (lam a b : Fin m → ℝ)
    (hlam : StrictMono lam) (hab : ∃ j, a j ≠ 0 ∨ b j ≠ 0) (s : Finset ℝ)
    (hs : ∀ x ∈ s, ∑ j, (a j + b j * x) * Real.exp (lam j * x) = 0) :
    s.card ≤ 2 * m - 1 :=
  expPoly_card_le m lam a b hlam hab s hs

/-- The real zero set of a nonzero exponential polynomial is finite with at most
`2m − 1` elements ("Its real zero set is finite", Lemma lem:exactmfull-zero-bound of
exact_m_theorem_full_proof.tex; finiteness here follows from the zero-count bound
rather than from the paper's compactness-plus-analyticity argument). -/
theorem expPoly_zeroSet_finite (m : ℕ) (lam a b : Fin m → ℝ)
    (hlam : StrictMono lam) (hab : ∃ j, a j ≠ 0 ∨ b j ≠ 0) :
    {x : ℝ | expPoly m lam a b x = 0}.Finite ∧
      {x : ℝ | expPoly m lam a b x = 0}.ncard ≤ 2 * m - 1 := by
  have hm1 : 0 < m := by
    obtain ⟨j, -⟩ := hab
    exact j.pos
  have hfin : {x : ℝ | expPoly m lam a b x = 0}.Finite := by
    by_contra hinf
    rw [Set.not_finite] at hinf
    obtain ⟨t, hts, htcard⟩ := hinf.exists_subset_card_eq (2 * m)
    have := expPoly_card_le m lam a b hlam hab t
      (fun x hx => hts (Finset.mem_coe.mpr hx))
    omega
  refine ⟨hfin, ?_⟩
  have hcard := expPoly_card_le m lam a b hlam hab hfin.toFinset
    (fun x hx => hfin.mem_toFinset.mp hx)
  rwa [Set.ncard_eq_toFinset_card _ hfin]

/-! ## Target B2: zeros counted with multiplicity

The paper's Lemma (lem:exactmfull-zero-bound) counts zeros WITH multiplicity.
Multiplicity is encoded analytically: `f` vanishes to order `≥ ν x` at `x` iff
`iteratedDeriv i f x = 0` for every `i < ν x`.  The total count over a finite set
of points is `Σ_{x ∈ t} ν x`.  This section proves that total is `≤ 2m − 1`,
via the paper's generalized Rolle counting ("differentiation reduces the
multiplicity at each root, and ordinary Rolle supplies a root between each pair
of distinct roots", Eq. eq:exactmfull-rolle-count). -/

section Multiplicity

/-- Derivative of `expPoly` stays in the family (explicit coefficient step).
Internal helper for the multiplicity count; no direct paper display. -/
lemma hasDerivAt_expPoly (M : ℕ) (lam a b : Fin M → ℝ) (x : ℝ) :
    HasDerivAt (fun y => expPoly M lam a b y)
      (expPoly M lam (fun j => a j * lam j + b j) (fun j => b j * lam j) x) x :=
  HasDerivAt.fun_sum (u := (Finset.univ : Finset (Fin M)))
    (fun j _ => hasDerivAt_affine_exp (a j) (b j) (lam j) x)

/-- Every iterated derivative of `expPoly` is again an `expPoly` with the same
frequencies (the key structural fact behind the multiplicity count).
Internal helper; no direct paper display. -/
lemma iteratedDeriv_expPoly_rep (M : ℕ) (lam a b : Fin M → ℝ) :
    ∀ n : ℕ, ∃ a' b' : Fin M → ℝ,
      iteratedDeriv n (fun y => expPoly M lam a b y) = fun y => expPoly M lam a' b' y := by
  intro n
  induction n with
  | zero => exact ⟨a, b, by rw [iteratedDeriv_zero]⟩
  | succ n ihn =>
    obtain ⟨a', b', h⟩ := ihn
    refine ⟨fun j => a' j * lam j + b' j, fun j => b' j * lam j, ?_⟩
    rw [iteratedDeriv_succ, h]
    funext y
    exact (hasDerivAt_expPoly M lam a' b' y).deriv

/-- `iteratedDeriv k` of an `expPoly` is differentiable, with derivative
`iteratedDeriv (k+1)` of it.  Internal helper; no direct paper display. -/
lemma hasDerivAt_iteratedDeriv_expPoly (M : ℕ) (lam a b : Fin M → ℝ) (k : ℕ) (x : ℝ) :
    HasDerivAt (iteratedDeriv k (fun y => expPoly M lam a b y))
      (iteratedDeriv (k + 1) (fun y => expPoly M lam a b y) x) x := by
  obtain ⟨a', b', h⟩ := iteratedDeriv_expPoly_rep M lam a b k
  rw [iteratedDeriv_succ, h]
  have hd := hasDerivAt_expPoly M lam a' b' x
  rw [show deriv (fun y => expPoly M lam a' b' y) x
      = expPoly M lam (fun j => a' j * lam j + b' j) (fun j => b' j * lam j) x from hd.deriv]
  exact hd

/-- Each iterated derivative of `e^{c x}·P(x)` is `e^{c x}` times a fixed linear
combination of the iterated derivatives of `P` up to the same order.  This encodes
"multiplication by a nowhere-zero function preserves both zeros and their
multiplicities" (proof of Lemma lem:exactmfull-zero-bound) without a general
Leibniz formula. -/
lemma exists_exp_mul_expansion (cc : ℝ) (M : ℕ) (lam a b : Fin M → ℝ) :
    ∀ i : ℕ, ∃ γ : ℕ → ℝ,
      iteratedDeriv i (fun y => Real.exp (cc * y) * expPoly M lam a b y)
        = fun y => Real.exp (cc * y) *
            ∑ k ∈ Finset.range (i + 1),
              γ k * iteratedDeriv k (fun z => expPoly M lam a b z) y := by
  intro i
  induction i with
  | zero =>
    refine ⟨fun _ => 1, ?_⟩
    rw [iteratedDeriv_zero]
    funext y
    rw [Finset.sum_range_one, iteratedDeriv_zero, one_mul]
  | succ i ihn =>
    obtain ⟨γ, hγ⟩ := ihn
    set γ' : ℕ → ℝ :=
      fun k => cc * (if k ≤ i then γ k else 0) + (if k = 0 then 0 else γ (k - 1))
      with hγ'_def
    refine ⟨γ', ?_⟩
    rw [iteratedDeriv_succ, hγ]
    funext y
    have hE : HasDerivAt (fun z : ℝ => Real.exp (cc * z)) (cc * Real.exp (cc * y)) y := by
      simpa [mul_comm] using ((hasDerivAt_id y).const_mul cc).exp
    have hS : HasDerivAt
        (fun z => ∑ k ∈ Finset.range (i + 1),
          γ k * iteratedDeriv k (fun w => expPoly M lam a b w) z)
        (∑ k ∈ Finset.range (i + 1),
          γ k * iteratedDeriv (k + 1) (fun w => expPoly M lam a b w) y) y :=
      HasDerivAt.fun_sum
        (fun k _ => (hasDerivAt_iteratedDeriv_expPoly M lam a b k y).const_mul (γ k))
    have hprod : HasDerivAt
        (fun z => Real.exp (cc * z) * ∑ k ∈ Finset.range (i + 1),
          γ k * iteratedDeriv k (fun w => expPoly M lam a b w) z)
        (cc * Real.exp (cc * y) * (∑ k ∈ Finset.range (i + 1),
            γ k * iteratedDeriv k (fun w => expPoly M lam a b w) y)
          + Real.exp (cc * y) * ∑ k ∈ Finset.range (i + 1),
              γ k * iteratedDeriv (k + 1) (fun w => expPoly M lam a b w) y) y :=
      hE.mul hS
    rw [hprod.deriv]
    have hsplit : (∑ k ∈ Finset.range (i + 1 + 1),
          γ' k * iteratedDeriv k (fun w => expPoly M lam a b w) y)
        = cc * (∑ k ∈ Finset.range (i + 1),
            γ k * iteratedDeriv k (fun w => expPoly M lam a b w) y)
          + ∑ k ∈ Finset.range (i + 1),
              γ k * iteratedDeriv (k + 1) (fun w => expPoly M lam a b w) y := by
      simp only [hγ'_def, add_mul]
      rw [Finset.sum_add_distrib]
      congr 1
      · rw [Finset.sum_range_succ]
        have hlast : cc * (if i + 1 ≤ i then γ (i + 1) else 0)
            * iteratedDeriv (i + 1) (fun w => expPoly M lam a b w) y = 0 := by
          rw [if_neg (by omega)]
          ring
        rw [hlast, add_zero, Finset.mul_sum]
        refine Finset.sum_congr rfl fun k hk => ?_
        rw [if_pos (Nat.lt_succ_iff.mp (Finset.mem_range.mp hk))]
        ring
      · rw [Finset.sum_range_succ']
        have h0 : (if (0 : ℕ) = 0 then (0 : ℝ) else γ (0 - 1))
            * iteratedDeriv 0 (fun w => expPoly M lam a b w) y = 0 := by
          rw [if_pos rfl]
          ring
        rw [h0, add_zero]
        refine Finset.sum_congr rfl fun k _ => ?_
        rw [if_neg (Nat.succ_ne_zero k), Nat.add_sub_cancel]
    rw [hsplit]
    ring

/-- Vanishing-order transfer: if all derivatives of `P` up to order `n` vanish at
`x₀`, so do those of `e^{c x}·P` ("multiplication by a nowhere-zero function
preserves both zeros and their multiplicities", Lemma lem:exactmfull-zero-bound). -/
lemma vanish_exp_mul (cc : ℝ) (M : ℕ) (lam a b : Fin M → ℝ) (x0 : ℝ) (n : ℕ)
    (h : ∀ i < n, iteratedDeriv i (fun y => expPoly M lam a b y) x0 = 0) :
    ∀ i < n, iteratedDeriv i (fun y => Real.exp (cc * y) * expPoly M lam a b y) x0 = 0 := by
  intro i hi
  obtain ⟨γ, hγ⟩ := exists_exp_mul_expansion cc M lam a b i
  rw [hγ]
  have hzero : ∑ k ∈ Finset.range (i + 1),
      γ k * iteratedDeriv k (fun z => expPoly M lam a b z) x0 = 0 :=
    Finset.sum_eq_zero fun k hk => by
      rw [h k (lt_of_le_of_lt (Nat.lt_succ_iff.mp (Finset.mem_range.mp hk)) hi), mul_zero]
  simp only [hzero, mul_zero]

/-- **Generalized Rolle step with multiplicities** (Eq. eq:exactmfull-rolle-count of
exact_m_theorem_full_proof.tex): a multiplicity certificate of total `N` for `g`
yields one of total `≥ N − 1` for `g'` — differentiation drops each root's
multiplicity by one, and ordinary Rolle supplies a fresh simple root between each
pair of adjacent distinct roots. -/
lemma exists_zeros_deriv_mult (g g' : ℝ → ℝ) (hg : ∀ x, HasDerivAt g (g' x) x)
    (t : Finset ℝ) (ν : ℝ → ℕ)
    (hz : ∀ x ∈ t, ∀ i < ν x, iteratedDeriv i g x = 0) :
    ∃ (t' : Finset ℝ) (ν' : ℝ → ℕ),
      ∑ x ∈ t, ν x ≤ (∑ x ∈ t', ν' x) + 1 ∧
      ∀ x ∈ t', ∀ i < ν' x, iteratedDeriv i g' x = 0 := by
  classical
  have hderiv : deriv g = g' := funext fun x => (hg x).deriv
  set tp := t.filter (fun x => 1 ≤ ν x) with htp_def
  have hsum_tp : ∑ x ∈ tp, ν x = ∑ x ∈ t, ν x := by
    rw [htp_def]
    exact Finset.sum_filter_of_ne fun x _ hne => Nat.one_le_iff_ne_zero.mpr hne
  have htp_mem : ∀ x ∈ tp, x ∈ t ∧ 1 ≤ ν x := by
    intro x hx
    rw [htp_def, Finset.mem_filter] at hx
    exact hx
  have htp_zero : ∀ x ∈ tp, g x = 0 := by
    intro x hx
    obtain ⟨hxt, hν⟩ := htp_mem x hx
    have h0 := hz x hxt 0 (by omega)
    rwa [iteratedDeriv_zero] at h0
  have htp_mult : ∀ x ∈ tp, ∀ i < ν x - 1, iteratedDeriv i g' x = 0 := by
    intro x hx i hi
    obtain ⟨hxt, hν⟩ := htp_mem x hx
    rw [← hderiv, ← iteratedDeriv_succ']
    exact hz x hxt (i + 1) (by omega)
  have hsplit : ∑ x ∈ tp, ν x = (∑ x ∈ tp, (ν x - 1)) + tp.card := by
    have hcongr : ∀ x ∈ tp, ν x = (ν x - 1) + 1 := by
      intro x hx
      have := (htp_mem x hx).2
      omega
    rw [Finset.sum_congr rfl hcongr, Finset.sum_add_distrib, Finset.sum_const,
      smul_eq_mul, mul_one]
  rcases Nat.lt_or_ge tp.card 2 with hsmall | hbig
  · refine ⟨tp, fun x => ν x - 1, ?_, htp_mult⟩
    show ∑ x ∈ t, ν x ≤ (∑ x ∈ tp, (ν x - 1)) + 1
    omega
  · obtain ⟨K, hK⟩ : ∃ K, tp.card = K + 1 := ⟨tp.card - 1, by omega⟩
    set e := tp.orderEmbOfFin hK with he_def
    have hgcont : Continuous g := continuous_iff_continuousAt.mpr fun y => (hg y).continuousAt
    have key : ∀ i : Fin K, ∃ y ∈ Set.Ioo (e i.castSucc) (e i.succ), g' y = 0 := by
      intro i
      have hlt : e i.castSucc < e i.succ := e.strictMono (Fin.castSucc_lt_succ (i := i))
      have h1 : g (e i.castSucc) = 0 := htp_zero _ (Finset.orderEmbOfFin_mem tp hK _)
      have h2 : g (e i.succ) = 0 := htp_zero _ (Finset.orderEmbOfFin_mem tp hK _)
      exact exists_hasDerivAt_eq_zero hlt hgcont.continuousOn (h1.trans h2.symm)
        (fun z _ => hg z)
    choose Y hYmem hY0 using key
    have hYmono : StrictMono Y := by
      intro i j hij
      have hij' : (i.succ : Fin (K + 1)) ≤ j.castSucc := by
        rw [Fin.le_def]
        simp only [Fin.val_succ, Fin.val_castSucc]
        exact hij
      have h3 : e i.succ ≤ e j.castSucc := e.monotone hij'
      have h4 := (hYmem i).2
      have h5 := (hYmem j).1
      linarith
    have hYnotintp : ∀ i : Fin K, Y i ∉ tp := by
      intro i hmem
      have hrange : Y i ∈ Set.range e := by
        rw [he_def, Finset.range_orderEmbOfFin]
        exact hmem
      obtain ⟨j, hj⟩ := hrange
      rcases Nat.lt_or_ge i.val j.val with hji | hji
      · have hle : i.succ ≤ j := by
          rw [Fin.le_def]
          simpa [Fin.val_succ] using hji
        have hmon := e.monotone hle
        rw [hj] at hmon
        have h5 := (hYmem i).2
        linarith
      · have hle : (j : Fin (K + 1)) ≤ i.castSucc := by
          rw [Fin.le_def]
          simpa [Fin.val_castSucc] using hji
        have hmon := e.monotone hle
        rw [hj] at hmon
        have h4 := (hYmem i).1
        linarith
    set R := Finset.image Y Finset.univ with hR_def
    have hdisj : Disjoint tp R := by
      rw [Finset.disjoint_left]
      intro x hx hxR
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.mp hxR
      exact hYnotintp i hx
    have hRcard : R.card = K := by
      rw [hR_def, Finset.card_image_of_injective _ hYmono.injective, Finset.card_univ,
        Fintype.card_fin]
    refine ⟨tp ∪ R, fun x => if x ∈ R then 1 else ν x - 1, ?_, ?_⟩
    · have hsum_union : ∑ x ∈ tp ∪ R, (if x ∈ R then 1 else ν x - 1)
          = (∑ x ∈ tp, (ν x - 1)) + K := by
        rw [Finset.sum_union hdisj]
        congr 1
        · exact Finset.sum_congr rfl fun x hx =>
            if_neg (Finset.disjoint_left.mp hdisj hx)
        · rw [Finset.sum_congr rfl (fun x hx => if_pos hx), Finset.sum_const,
            smul_eq_mul, mul_one, hRcard]
      show ∑ x ∈ t, ν x ≤ (∑ x ∈ tp ∪ R, (if x ∈ R then 1 else ν x - 1)) + 1
      rw [hsum_union]
      omega
    · intro x hx i hi
      have hi' : i < (if x ∈ R then 1 else ν x - 1) := hi
      rcases Finset.mem_union.mp hx with hxtp | hxR
      · have hxnR : x ∉ R := Finset.disjoint_left.mp hdisj hxtp
        rw [if_neg hxnR] at hi'
        exact htp_mult x hxtp i hi'
      · rw [if_pos hxR] at hi'
        obtain ⟨iY, -, rfl⟩ := Finset.mem_image.mp hxR
        have hi0 : i = 0 := by omega
        subst hi0
        rw [iteratedDeriv_zero]
        exact hY0 iY

/-- Base case with multiplicity: a nonzero affine–exponential term
`(p + q x) e^{λ₀ x}` has at most one real zero counted with multiplicity
(the `m = 1` "affine zero bound" of Lemma lem:exactmfull-zero-bound). -/
lemma affine_exp_mult_le (p q lam0 : ℝ) (hpq : p ≠ 0 ∨ q ≠ 0)
    (t : Finset ℝ) (ν : ℝ → ℕ)
    (hz : ∀ x ∈ t, ∀ i < ν x,
      iteratedDeriv i (fun y => (p + q * y) * Real.exp (lam0 * y)) x = 0) :
    ∑ x ∈ t, ν x ≤ 1 := by
  classical
  set tp := t.filter (fun x => 1 ≤ ν x) with htp_def
  have hsum_tp : ∑ x ∈ tp, ν x = ∑ x ∈ t, ν x := by
    rw [htp_def]
    exact Finset.sum_filter_of_ne fun x _ hne => Nat.one_le_iff_ne_zero.mpr hne
  have htp_mem : ∀ x ∈ tp, x ∈ t ∧ 1 ≤ ν x := by
    intro x hx
    rw [htp_def, Finset.mem_filter] at hx
    exact hx
  have hzero : ∀ x ∈ tp, p + q * x = 0 := by
    intro x hx
    obtain ⟨hxt, hν⟩ := htp_mem x hx
    have h0 := hz x hxt 0 (by omega)
    rw [iteratedDeriv_zero] at h0
    exact (mul_eq_zero.mp h0).resolve_right (Real.exp_ne_zero _)
  have hν1 : ∀ x ∈ tp, ν x ≤ 1 := by
    intro x hx
    obtain ⟨hxt, hν⟩ := htp_mem x hx
    by_contra hgt
    have h1 := hz x hxt 1 (by omega)
    rw [iteratedDeriv_one] at h1
    rw [(hasDerivAt_affine_exp p q lam0 x).deriv] at h1
    have haff : p * lam0 + q + q * lam0 * x = 0 :=
      (mul_eq_zero.mp h1).resolve_right (Real.exp_ne_zero _)
    have hzx := hzero x hx
    have hq : q = 0 := by linear_combination haff - lam0 * hzx
    have hp : p = 0 := by linear_combination hzx - x * hq
    rcases hpq with h | h
    · exact h hp
    · exact h hq
  have hcard : tp.card ≤ 1 := affine_zeros_card_le p q hpq tp hzero
  have hbound : ∑ x ∈ tp, ν x ≤ ∑ x ∈ tp, 1 := Finset.sum_le_sum hν1
  rw [Finset.sum_const, smul_eq_mul, mul_one] at hbound
  omega

/-- B2 core (Lemma lem:exactmfull-zero-bound / Eq. eq:exactmfull-rolle-count
of exact_m_theorem_full_proof.tex): with-multiplicity bound over the
`expPoly` wrapper.  The quotable public form is
`expPoly_zeros_with_multiplicity_le`. -/
theorem expPoly_mult_le :
    ∀ (m : ℕ) (lam a b : Fin m → ℝ), StrictMono lam →
      (∃ j, a j ≠ 0 ∨ b j ≠ 0) →
      ∀ (t : Finset ℝ) (ν : ℝ → ℕ),
        (∀ x ∈ t, ∀ i < ν x, iteratedDeriv i (fun y => expPoly m lam a b y) x = 0) →
        ∑ x ∈ t, ν x ≤ 2 * m - 1 := by
  intro m
  induction m with
  | zero =>
    intro lam a b _ hab t ν _
    obtain ⟨j, -⟩ := hab
    exact j.elim0
  | succ m ih =>
    intro lam a b hlam hab t ν hz
    by_cases hrest : ∀ j : Fin m, a j.succ = 0 ∧ b j.succ = 0
    · -- Tail coefficients all vanish: P = (a₀ + b₀x) e^{λ₀x}.
      have hcollapse : (fun y => expPoly (m + 1) lam a b y)
          = fun y => (a 0 + b 0 * y) * Real.exp (lam 0 * y) := by
        funext y
        simp only [expPoly, Fin.sum_univ_succ]
        have htail : ∑ j : Fin m,
            (a j.succ + b j.succ * y) * Real.exp (lam j.succ * y) = 0 :=
          Finset.sum_eq_zero fun j _ => by rw [(hrest j).1, (hrest j).2]; ring
        rw [htail, add_zero]
      rw [hcollapse] at hz
      have hab0 : a 0 ≠ 0 ∨ b 0 ≠ 0 := by
        obtain ⟨j, hj⟩ := hab
        rcases eq_or_ne j 0 with rfl | hj0
        · exact hj
        · exfalso
          have h := hrest (j.pred hj0)
          rw [Fin.succ_pred] at h
          rcases hj with h1 | h1
          · exact h1 h.1
          · exact h1 h.2
      have := affine_exp_mult_le (a 0) (b 0) (lam 0) hab0 t ν hz
      omega
    · -- Genuine tail: multiply by e^{−λ₀x} (multiplicities preserved),
      -- take two multiplicity-Rolle steps, apply the IH.
      rw [not_forall] at hrest
      obtain ⟨j0, hj0⟩ := hrest
      have hm1 : 0 < m := j0.pos
      set mu : Fin m → ℝ := fun j => lam j.succ - lam 0 with hmu_def
      have hmu_pos : ∀ j, 0 < mu j := fun j => sub_pos.mpr (hlam (Fin.succ_pos j))
      have hgEq : (fun y => Real.exp (-lam 0 * y) * expPoly (m + 1) lam a b y)
          = fun y => (a 0 + b 0 * y)
              + ∑ j, (a j.succ + b j.succ * y) * Real.exp (mu j * y) := by
        funext y
        symm
        simp only [expPoly]
        rw [Fin.sum_univ_succ, mul_add, Finset.mul_sum]
        congr 1
        · rw [mul_comm (Real.exp (-lam 0 * y)), mul_assoc, ← Real.exp_add]
          have hzadd : lam 0 * y + -lam 0 * y = 0 := by ring
          rw [hzadd, Real.exp_zero, mul_one]
        · refine Finset.sum_congr rfl fun j _ => ?_
          rw [mul_comm (Real.exp (-lam 0 * y)), mul_assoc, ← Real.exp_add]
          have hzadd : lam j.succ * y + -lam 0 * y = mu j * y := by
            simp only [hmu_def]
            ring
          rw [hzadd]
      have htransfer : ∀ x ∈ t, ∀ i < ν x,
          iteratedDeriv i (fun y => (a 0 + b 0 * y)
            + ∑ j, (a j.succ + b j.succ * y) * Real.exp (mu j * y)) x = 0 := by
        intro x hx i hi
        have hv := vanish_exp_mul (-lam 0) (m + 1) lam a b x (ν x) (hz x hx) i hi
        rwa [hgEq] at hv
      obtain ⟨t1, ν1, hle1, hcert1⟩ := exists_zeros_deriv_mult
        (fun x => (a 0 + b 0 * x)
          + ∑ j, (a j.succ + b j.succ * x) * Real.exp (mu j * x))
        (fun x => b 0 + ∑ j,
          (a j.succ * mu j + b j.succ + b j.succ * mu j * x) * Real.exp (mu j * x))
        (fun x => hasDerivAt_headTail m mu (fun j => a j.succ) (fun j => b j.succ)
          (a 0) (b 0) x)
        t ν htransfer
      obtain ⟨t2, ν2, hle2, hcert2⟩ := exists_zeros_deriv_mult
        (fun x => b 0 + ∑ j,
          (a j.succ * mu j + b j.succ + b j.succ * mu j * x) * Real.exp (mu j * x))
        (fun x => ∑ j,
          (a j.succ * mu j ^ 2 + 2 * b j.succ * mu j + b j.succ * mu j ^ 2 * x)
            * Real.exp (mu j * x))
        (fun x => hasDerivAt_tail1 m mu (fun j => a j.succ) (fun j => b j.succ) (b 0) x)
        t1 ν1 hcert1
      have hmu_mono : StrictMono mu := by
        intro i j hij
        have h := hlam (Fin.succ_lt_succ_iff.mpr hij)
        simp only [hmu_def]
        exact sub_lt_sub_right h (lam 0)
      have hABne : ∃ j : Fin m,
          (a j.succ * mu j ^ 2 + 2 * b j.succ * mu j ≠ 0) ∨ (b j.succ * mu j ^ 2 ≠ 0) := by
        have hmu0 : mu j0 ≠ 0 := ne_of_gt (hmu_pos j0)
        rcases eq_or_ne (b j0.succ) 0 with hb0 | hb0
        · have ha0 : a j0.succ ≠ 0 := fun ha => hj0 ⟨ha, hb0⟩
          refine ⟨j0, Or.inl ?_⟩
          rw [hb0]
          simpa using mul_ne_zero ha0 (pow_ne_zero 2 hmu0)
        · exact ⟨j0, Or.inr (mul_ne_zero hb0 (pow_ne_zero 2 hmu0))⟩
      have hIH := ih mu (fun j => a j.succ * mu j ^ 2 + 2 * b j.succ * mu j)
        (fun j => b j.succ * mu j ^ 2) hmu_mono hABne t2 ν2
        (fun x hx i hi => hcert2 x hx i hi)
      omega

/-- **With-multiplicity zero bound for exponential polynomials** (target B2) —
Lemma (lem:exactmfull-zero-bound) / Eq. (eq:exactmfull-general-exp-polynomial) of
exact_m_theorem_full_proof.tex and Eq. (eq:exact-m-exp-polynomial) of
exact_m_theorem_spine.tex, WITH multiplicity: for λ₁ < ⋯ < λ_m and coefficients
(a_j, b_j) not all zero, the total vanishing order of
`Σ_j (a_j + b_j x) e^{λ_j x}` over any finite set of points is at most `2m − 1`.

Multiplicity is encoded analytically: the function vanishes to order `≥ ν x` at
`x` iff `iteratedDeriv i f x = 0` for all `i < ν x`; the count is `Σ_{x ∈ t} ν x`.
As in the distinct-zeros version, the hypothesis `∃ j, a j ≠ 0 ∨ b j ≠ 0` is
weaker than the paper's "no identically zero summand", so this statement is more
general than the paper's. -/
theorem expPoly_zeros_with_multiplicity_le (m : ℕ) (lam a b : Fin m → ℝ)
    (hlam : StrictMono lam) (hab : ∃ j, a j ≠ 0 ∨ b j ≠ 0)
    (t : Finset ℝ) (ν : ℝ → ℕ)
    (hz : ∀ x ∈ t, ∀ i < ν x,
      iteratedDeriv i (fun y => ∑ j, (a j + b j * y) * Real.exp (lam j * y)) x = 0) :
    ∑ x ∈ t, ν x ≤ 2 * m - 1 :=
  expPoly_mult_le m lam a b hlam hab t ν hz

end Multiplicity

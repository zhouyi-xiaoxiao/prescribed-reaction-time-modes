import Mathlib.Analysis.Calculus.LocalExtr.Rolle
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Data.Set.Card
import Mathlib.Data.Finset.Sort
import Mathlib.Order.Fin.Basic
import Mathlib.Data.Fin.SuccPred

/-!
# Sharp real-zero count of an exponential polynomial

This file formalizes the hard analytic kernel behind the "exactly `m` modes" count in the
PRR manuscript (spine Eq. `exact-m-exp-polynomial`; full proof, the `2m-1` count lemma).

After multiplying `H'(x)` by a positive Gaussian factor, the stationary points of a Gaussian
mixture `H` are the real zeros of an **exponential polynomial** of the form
`P(x) = ∑_{j=1}^m (a_j + b_j x) · exp(λ_j x)` with `λ_1 < ... < λ_m` real and distinct.
The main theorem `expPoly_atMostZeros` proves that a not-identically-zero `P` of this form has
**at most `2m - 1` real zeros** (counted without multiplicity).

## Proof strategy (generalized Rolle induction — the paper's own argument)

* Base / degenerate cases collapse to an affine function via a constant-of-zero-derivative
  argument (no linear independence of exponentials required).
* Inductive step: multiply by `exp(-λ_1 x)` (`hfg`), so the lowest exponent becomes `0`.
  Differentiating twice kills the affine (`μ_0 = 0`) term, leaving an exponential polynomial in
  the `m-1` exponents `μ_j = λ_j - λ_1 > 0` (`hg''eq`). Either that second derivative is
  identically zero (⟹ the shifted function is affine ⟹ `≤ 1` zero), or the induction hypothesis
  bounds its zeros by `2(m-1) - 1`; two Rolle steps then give `≤ 2m - 1`.

The Rolle bookkeeping is packaged as `atMost_succ_of_deriv` (a clean wrapper over mathlib's
Rolle theorem `exists_hasDerivAt_eq_zero`), which is the reusable "`#zeros f ≤ #zeros f' + 1`"
engine requested as a companion lemma.

## Main statements

* `affine_times_exp_zeros` : `(a + b·x)·exp(λx)` has at most one real zero (`Set.Subsingleton`).
* `atMost_succ_of_deriv`   : Rolle-count engine, `AtMostZeros f' M → AtMostZeros f (M+1)`.
* `expPoly_hasDerivAt`      : the derivative of an exponential polynomial is one of the same shape.
* `expPoly_atMostZeros`     : the general `≤ 2m - 1` bound (finset form).
* `expPoly_zeros_ncard_le`  : `Set.Finite ∧ Set.ncard ≤ 2m - 1` (the finiteness half is part of
  the type, since `Set.ncard` of an infinite set is `0`).
* `expPoly_..._le_{one,three,five}` : the paper's `m = 1, 2, 3` designs (bimodal / trimodal /
  the `m = 3` phase diagram) as corollaries, each also carrying `Set.Finite`.

No `sorry`, no `admit`, no new axioms: see the `#print axioms` block at the end of the file.
-/

open Real

noncomputable section

/-- An exponential polynomial `∑_{j} (a_j + b_j x) · exp(λ_j x)` on `Fin m`.
Encodes spine Eq. `exact-m-exp-polynomial`. -/
noncomputable def expPoly {m : ℕ} (a b lam : Fin m → ℝ) : ℝ → ℝ :=
  fun x => ∑ j : Fin m, (a j + b j * x) * Real.exp (lam j * x)

/-- The derivative of an exponential polynomial is an exponential polynomial with the **same**
exponents `lam` and transformed affine coefficients `(a', b') = (λ·a + b, λ·b)`.  This is the
"differentiation stays in shape" step of the generalized-Rolle induction.  Internal helper;
no direct paper display. -/
theorem expPoly_hasDerivAt {m : ℕ} (a b lam : Fin m → ℝ) (x : ℝ) :
    HasDerivAt (expPoly a b lam)
      (expPoly (fun j => lam j * a j + b j) (fun j => lam j * b j) lam x) x := by
  have hterm : ∀ j : Fin m,
      HasDerivAt (fun y : ℝ => (a j + b j * y) * Real.exp (lam j * y))
        ((lam j * a j + b j + lam j * b j * x) * Real.exp (lam j * x)) x := by
    intro j
    have h1 : HasDerivAt (fun y : ℝ => a j + b j * y) (b j) x := by
      simpa using ((hasDerivAt_id x).const_mul (b j)).const_add (a j)
    have hlin : HasDerivAt (fun y : ℝ => lam j * y) (lam j) x := by
      simpa using (hasDerivAt_id x).const_mul (lam j)
    have h2 : HasDerivAt (fun y : ℝ => Real.exp (lam j * y)) (Real.exp (lam j * x) * lam j) x :=
      hlin.exp
    have h3 := h1.mul h2
    have hval : b j * Real.exp (lam j * x) + (a j + b j * x) * (Real.exp (lam j * x) * lam j)
        = (lam j * a j + b j + lam j * b j * x) * Real.exp (lam j * x) := by ring
    rw [hval] at h3
    exact h3
  have hsum : HasDerivAt (fun y : ℝ => ∑ j : Fin m, (a j + b j * y) * Real.exp (lam j * y))
      (∑ j : Fin m, (lam j * a j + b j + lam j * b j * x) * Real.exp (lam j * x)) x :=
    HasDerivAt.fun_sum (fun j _ => hterm j)
  have e1 : expPoly a b lam
      = fun y : ℝ => ∑ j : Fin m, (a j + b j * y) * Real.exp (lam j * y) := rfl
  have e2 : expPoly (fun j => lam j * a j + b j) (fun j => lam j * b j) lam x
      = ∑ j : Fin m, (lam j * a j + b j + lam j * b j * x) * Real.exp (lam j * x) := rfl
  rw [e1, e2]
  exact hsum

/-- **Base lemma.** `(a + b·x)·exp(λx)` has at most one real zero whenever `(a, b) ≠ (0, 0)`:
its zero set is a subsingleton.  This is the `m = 1` heart of the count (the affine factor has
`≤ 1` zero, and `exp` never vanishes).  Anchor: base case of Lemma
(lem:exactmfull-zero-bound), exact_m_theorem_full_proof.tex. -/
theorem affine_times_exp_zeros (a b lam : ℝ) (hab : a ≠ 0 ∨ b ≠ 0) :
    {x : ℝ | (a + b * x) * Real.exp (lam * x) = 0}.Subsingleton := by
  intro x hx y hy
  simp only [Set.mem_setOf_eq, mul_eq_zero] at hx hy
  have hx' : a + b * x = 0 := hx.resolve_right (Real.exp_ne_zero _)
  have hy' : a + b * y = 0 := hy.resolve_right (Real.exp_ne_zero _)
  have hb : b ≠ 0 := by
    rcases hab with ha | hb
    · rintro rfl; rw [zero_mul, add_zero] at hx'; exact ha hx'
    · exact hb
  have hbxy : b * x = b * y := by linarith
  exact mul_left_cancel₀ hb hbxy

/-- `(a + b·x)·exp(λx)` has a FINITE zero set of cardinality `≤ 1`: the
conclusion carries `Set.Finite` explicitly alongside the `ncard` bound,
since `Set.ncard` of an infinite set is `0` and the cardinality inequality
alone would not assert finiteness.  Corollary packaging of the base case of
Lemma (lem:exactmfull-zero-bound); no separate paper display. -/
theorem affine_times_exp_ncard_le (a b lam : ℝ) (hab : a ≠ 0 ∨ b ≠ 0) :
    {x : ℝ | (a + b * x) * Real.exp (lam * x) = 0}.Finite ∧
      {x : ℝ | (a + b * x) * Real.exp (lam * x) = 0}.ncard ≤ 1 := by
  have hsub := affine_times_exp_zeros a b lam hab
  have hfin : {x : ℝ | (a + b * x) * Real.exp (lam * x) = 0}.Finite := hsub.finite
  exact ⟨hfin, (Set.ncard_le_one hfin).mpr (fun p hp q hq => hsub hp hq)⟩

/-- `f` has at most `N` real zeros: every finite set of zeros has cardinality `≤ N`.
This is a faithful "at most `N` zeros" statement that also yields finiteness and an
`ncard` bound (see `AtMostZeros.setFinite`, `AtMostZeros.ncard_le`). -/
def AtMostZeros (f : ℝ → ℝ) (N : ℕ) : Prop :=
  ∀ s : Finset ℝ, (∀ x ∈ s, f x = 0) → s.card ≤ N

/-- **Rolle interleaving.** If `f` is everywhere differentiable with derivative `f'`, and `p` is a
strictly increasing list of `n+1` zeros of `f`, then there are `n` strictly increasing zeros of
`f'`, one strictly between each consecutive pair (mathlib Rolle `exists_hasDerivAt_eq_zero`).
Internal helper (generic proof machinery); no direct paper display. -/
theorem rolle_interleave {f f' : ℝ → ℝ} (hf : ∀ x, HasDerivAt f (f' x) x)
    {n : ℕ} (p : Fin (n + 1) → ℝ) (hp : StrictMono p) (hz : ∀ i, f (p i) = 0) :
    ∃ q : Fin n → ℝ, StrictMono q ∧ ∀ i, f' (q i) = 0 := by
  have hdiff : Differentiable ℝ f := fun x => (hf x).differentiableAt
  have hcont : Continuous f := hdiff.continuous
  have hex : ∀ i : Fin n, ∃ c ∈ Set.Ioo (p i.castSucc) (p i.succ), f' c = 0 := by
    intro i
    have hlt : p i.castSucc < p i.succ := hp (Fin.castSucc_lt_succ_iff.mpr le_rfl)
    exact exists_hasDerivAt_eq_zero hlt hcont.continuousOn
      (by rw [hz i.castSucc, hz i.succ]) (fun x _ => hf x)
  choose c hc hc0 using hex
  refine ⟨c, ?_, hc0⟩
  intro i j hij
  have h1 : c i < p i.succ := (hc i).2
  have h2 : p j.castSucc < c j := (hc j).1
  have hsc : i.succ ≤ j.castSucc := by
    have hval := Fin.lt_def.mp hij
    simp only [Fin.le_def, Fin.val_succ, Fin.val_castSucc]
    omega
  have h3 : p i.succ ≤ p j.castSucc := hp.monotone hsc
  linarith

/-- **Rolle-count engine** (`#zeros f ≤ #zeros f' + 1`).  If `f` is everywhere differentiable with
derivative `f'`, and `f'` has at most `M` zeros, then `f` has at most `M + 1` zeros.
Internal helper (generic proof machinery); no direct paper display. -/
theorem atMost_succ_of_deriv {f f' : ℝ → ℝ} (hf : ∀ x, HasDerivAt f (f' x) x)
    {M : ℕ} (hM : AtMostZeros f' M) : AtMostZeros f (M + 1) := by
  intro s hs
  rcases Nat.eq_zero_or_pos s.card with hc | hc
  · omega
  · obtain ⟨n, hn⟩ : ∃ n, s.card = n + 1 := ⟨s.card - 1, by omega⟩
    have hpmono : StrictMono (⇑(s.orderEmbOfFin hn)) := (s.orderEmbOfFin hn).strictMono
    have hpz : ∀ i, f (s.orderEmbOfFin hn i) = 0 := fun i => hs _ (s.orderEmbOfFin_mem hn i)
    obtain ⟨q, hqmono, hqz⟩ := rolle_interleave hf (⇑(s.orderEmbOfFin hn)) hpmono hpz
    have hcard : (Finset.image q Finset.univ).card = n := by
      rw [Finset.card_image_of_injective _ hqmono.injective, Finset.card_univ, Fintype.card_fin]
    have htz : ∀ x ∈ Finset.image q Finset.univ, f' x = 0 := by
      intro x hx
      rw [Finset.mem_image] at hx
      obtain ⟨i, _, rfl⟩ := hx
      exact hqz i
    have hle := hM _ htz
    rw [hcard] at hle
    omega

/-- An `AtMostZeros` bound makes the zero set finite.  Internal helper
(generic proof machinery); no direct paper display. -/
theorem AtMostZeros.setFinite {f : ℝ → ℝ} {N : ℕ} (h : AtMostZeros f N) :
    {x : ℝ | f x = 0}.Finite := by
  by_contra hinf
  rw [Set.not_finite] at hinf
  obtain ⟨t, hts, htc⟩ := hinf.exists_subset_card_eq (N + 1)
  have hle : t.card ≤ N := h t (fun x hx => hts (Finset.mem_coe.mpr hx))
  omega

/-- An `AtMostZeros f N` bound gives `ncard {x | f x = 0} ≤ N`.  Internal
helper (generic proof machinery); no direct paper display. -/
theorem AtMostZeros.ncard_le {f : ℝ → ℝ} {N : ℕ} (h : AtMostZeros f N) :
    {x : ℝ | f x = 0}.ncard ≤ N := by
  have hfin := h.setFinite
  rw [Set.ncard_eq_toFinset_card _ hfin]
  exact h _ (fun x hx => (Set.Finite.mem_toFinset hfin).mp hx)

/-- **Main theorem** (spine Eq. `exact-m-exp-polynomial`; full proof, the `2m-1` count).
A not-identically-zero exponential polynomial `∑_{j=1}^m (a_j + b_j x) exp(λ_j x)` with strictly
increasing exponents `λ` has at most `2m - 1` real zeros (counted without multiplicity).

SCOPE NOTE: This is the DISTINCT-zeros count (`Set.ncard`, i.e. `≤ 2m - 1` distinct real roots),
which is exactly what makes the exhaustive mode count go through (distinct stationary points ⟹
distinct maxima). The manuscript states the bound with multiplicity; the with-multiplicity
refinement (via iterated-derivative vanishing) is not formalized here. The distinct-zeros bound
holds unconditionally for every `m` and is a genuine, publishable formal fact. -/
theorem expPoly_atMostZeros :
    ∀ (m : ℕ) (a b lam : Fin m → ℝ), StrictMono lam → expPoly a b lam ≠ 0 →
      AtMostZeros (expPoly a b lam) (2 * m - 1) := by
  intro m
  induction m with
  | zero =>
    intro a b lam _ hne
    exact absurd (by funext x; simp [expPoly]) hne
  | succ k ih =>
    intro a b lam hlam hne
    rw [show 2 * (k + 1) - 1 = 2 * k + 1 from by omega]
    -- Shift the lowest exponent to `0`.
    set μ : Fin (k + 1) → ℝ := fun j => lam j - lam 0 with hμ_def
    have hμmono : StrictMono μ := by
      intro i j hij; simp only [hμ_def]; exact sub_lt_sub_right (hlam hij) _
    have hμ0 : μ 0 = 0 := by simp [hμ_def]
    -- `g = P · exp(-λ_0 x)`, and its first two derivatives, all of exp-polynomial shape.
    set g := expPoly a b μ with hg_def
    set g' := expPoly (fun j => μ j * a j + b j) (fun j => μ j * b j) μ with hg'_def
    set A2 : Fin (k + 1) → ℝ := fun j => μ j * (μ j * a j + b j) + μ j * b j with hA2_def
    set B2 : Fin (k + 1) → ℝ := fun j => μ j * (μ j * b j) with hB2_def
    set g'' := expPoly A2 B2 μ with hg''_def
    have hderiv_g : ∀ x, HasDerivAt g (g' x) x := fun x => expPoly_hasDerivAt a b μ x
    have hderiv_g' : ∀ x, HasDerivAt g' (g'' x) x :=
      fun x => expPoly_hasDerivAt (fun j => μ j * a j + b j) (fun j => μ j * b j) μ x
    -- Zeros of `P` = zeros of `g` since `exp(λ_0 x) > 0`.
    have hfg : ∀ x, expPoly a b lam x = Real.exp (lam 0 * x) * g x := by
      intro x
      rw [hg_def]
      simp only [expPoly, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro j _
      have hexp : Real.exp (lam 0 * x) * Real.exp (μ j * x) = Real.exp (lam j * x) := by
        rw [← Real.exp_add]; congr 1; simp only [hμ_def]; ring
      rw [← hexp]; ring
    -- It suffices to bound the zeros of `g`.
    suffices hgb : AtMostZeros g (2 * k + 1) by
      intro s hs
      exact hgb s (fun x hx =>
        (mul_eq_zero.mp (hfg x ▸ hs x hx)).resolve_left (Real.exp_ne_zero _))
    by_cases hg''0 : g'' = 0
    · -- Degenerate case: `g'' ≡ 0`, so `g` is affine, hence has `≤ 1` zero.
      have hg'const : ∀ x y, g' x = g' y :=
        fun x y => is_const_of_deriv_eq_zero (fun z => (hderiv_g' z).differentiableAt)
          (fun z => by simp [(hderiv_g' z).deriv, hg''0]) x y
      have hg'c : ∀ x, g' x = g' 0 := fun x => hg'const x 0
      have hgc : ∀ x, HasDerivAt g (g' 0) x := fun x => (hg'c x ▸ hderiv_g x)
      by_cases hc0 : g' 0 = 0
      · -- `g` is constant; it is nonzero, so it has no zeros.
        have hgconst : ∀ x y, g x = g y :=
          fun x y => is_const_of_deriv_eq_zero (fun z => (hgc z).differentiableAt)
            (fun z => by rw [(hgc z).deriv]; exact hc0) x y
        have hgne : g ≠ 0 := by
          intro h0; apply hne; funext x; rw [hfg x, h0]; simp
        obtain ⟨x0, hx0⟩ := Function.ne_iff.mp hgne
        simp only [Pi.zero_apply] at hx0
        intro s hs
        rcases s.eq_empty_or_nonempty with rfl | ⟨z, hz⟩
        · simp
        · exact absurd (hgconst z x0 ▸ hs z hz) hx0
      · -- `g` has a nonzero constant derivative, hence is injective: `≤ 1` zero.
        have hg'nz : AtMostZeros g' 0 := by
          intro s hs
          rcases s.eq_empty_or_nonempty with rfl | ⟨z, hz⟩
          · simp
          · exact absurd ((hg'c z).symm.trans (hs z hz)) hc0
        have hb : AtMostZeros g (0 + 1) := atMost_succ_of_deriv hderiv_g hg'nz
        intro s hs
        have := hb s hs
        omega
    · -- Main case: `g''` is a nonzero exp-polynomial in the `k` exponents `μ_1 < ... < μ_k`.
      have hg''eq : g'' = expPoly
          (fun i : Fin k => A2 i.succ) (fun i : Fin k => B2 i.succ) (fun i : Fin k => μ i.succ) := by
        funext x
        rw [hg''_def]
        simp only [expPoly]
        rw [Fin.sum_univ_succ]
        have hA20 : A2 0 = 0 := by simp only [hA2_def, hμ0]; ring
        have hB20 : B2 0 = 0 := by simp only [hB2_def, hμ0]; ring
        rw [hA20, hB20]
        simp
      have hμsucc : StrictMono (fun i : Fin k => μ i.succ) := hμmono.comp Fin.strictMono_succ
      have hne'' : expPoly (fun i : Fin k => A2 i.succ) (fun i : Fin k => B2 i.succ)
          (fun i : Fin k => μ i.succ) ≠ 0 := hg''eq ▸ hg''0
      have hk1 : 1 ≤ k := by
        rcases Nat.eq_zero_or_pos k with hk | hk
        · subst hk; exact absurd (by funext x; simp [expPoly]) hne''
        · exact hk
      have ihres := ih (fun i : Fin k => A2 i.succ) (fun i : Fin k => B2 i.succ)
        (fun i : Fin k => μ i.succ) hμsucc hne''
      rw [← hg''eq] at ihres
      have hb1 : AtMostZeros g' (2 * k - 1 + 1) := atMost_succ_of_deriv hderiv_g' ihres
      have hb2 : AtMostZeros g (2 * k - 1 + 1 + 1) := atMost_succ_of_deriv hderiv_g hb1
      rwa [show 2 * k - 1 + 1 + 1 = 2 * k + 1 from by omega] at hb2

/-- The zero set of a not-identically-zero exponential polynomial is finite.
Corollary packaging of `expPoly_atMostZeros`, anchored to Lemma
(lem:exactmfull-zero-bound); no separate paper display. -/
theorem expPoly_zeros_finite {m : ℕ} (a b lam : Fin m → ℝ) (hlam : StrictMono lam)
    (hne : expPoly a b lam ≠ 0) : {x : ℝ | expPoly a b lam x = 0}.Finite :=
  (expPoly_atMostZeros m a b lam hlam hne).setFinite

/-- `Set.ncard` form of the main theorem: the zero set is FINITE with at
most `2m - 1` distinct real zeros.  The conclusion carries `Set.Finite`
explicitly alongside the `ncard` bound, since `Set.ncard` of an infinite
set is `0` and the cardinality inequality alone would not assert
finiteness.
Corollary of `expPoly_atMostZeros` (Lemma lem:exactmfull-zero-bound); no
separate paper display. -/
theorem expPoly_zeros_ncard_le {m : ℕ} (a b lam : Fin m → ℝ) (hlam : StrictMono lam)
    (hne : expPoly a b lam ≠ 0) :
    {x : ℝ | expPoly a b lam x = 0}.Finite ∧
      {x : ℝ | expPoly a b lam x = 0}.ncard ≤ 2 * m - 1 :=
  ⟨expPoly_zeros_finite a b lam hlam hne,
    (expPoly_atMostZeros m a b lam hlam hne).ncard_le⟩

/-- `m = 1` (single exponent): a finite zero set with at most `1` real
zero (finiteness carried explicitly; see `expPoly_zeros_ncard_le`).
Corollary of `expPoly_atMostZeros` (Lemma lem:exactmfull-zero-bound); no
separate paper display. -/
theorem expPoly_one_zeros_le_one (a b lam : Fin 1 → ℝ) (hlam : StrictMono lam)
    (hne : expPoly a b lam ≠ 0) :
    {x : ℝ | expPoly a b lam x = 0}.Finite ∧
      {x : ℝ | expPoly a b lam x = 0}.ncard ≤ 1 := by
  have h := expPoly_zeros_ncard_le a b lam hlam hne
  exact ⟨h.1, by simpa using h.2⟩

/-- `m = 2` (the bimodal design): a finite zero set with at most `3` real
zeros (finiteness carried explicitly; see `expPoly_zeros_ncard_le`).
Corollary of `expPoly_atMostZeros` (Lemma lem:exactmfull-zero-bound); no
separate paper display. -/
theorem expPoly_two_zeros_le_three (a b lam : Fin 2 → ℝ) (hlam : StrictMono lam)
    (hne : expPoly a b lam ≠ 0) :
    {x : ℝ | expPoly a b lam x = 0}.Finite ∧
      {x : ℝ | expPoly a b lam x = 0}.ncard ≤ 3 := by
  have h := expPoly_zeros_ncard_le a b lam hlam hne
  exact ⟨h.1, by simpa using h.2⟩

/-- `m = 3` (the trimodal design / `m = 3` phase diagram): a finite zero set
with at most `5` real zeros (finiteness carried explicitly; see
`expPoly_zeros_ncard_le`).
Corollary of `expPoly_atMostZeros` (Lemma lem:exactmfull-zero-bound); no
separate paper display. -/
theorem expPoly_three_zeros_le_five (a b lam : Fin 3 → ℝ) (hlam : StrictMono lam)
    (hne : expPoly a b lam ≠ 0) :
    {x : ℝ | expPoly a b lam x = 0}.Finite ∧
      {x : ℝ | expPoly a b lam x = 0}.ncard ≤ 5 := by
  have h := expPoly_zeros_ncard_le a b lam hlam hne
  exact ⟨h.1, by simpa using h.2⟩

end

-- Axiom audit: every delivered theorem must reduce to mathlib's standard axioms only
-- (`propext`, `Classical.choice`, `Quot.sound`) with no `sorryAx`.
#print axioms expPoly_hasDerivAt
#print axioms affine_times_exp_zeros
#print axioms affine_times_exp_ncard_le
#print axioms rolle_interleave
#print axioms atMost_succ_of_deriv
#print axioms expPoly_atMostZeros
#print axioms expPoly_zeros_finite
#print axioms expPoly_zeros_ncard_le
#print axioms expPoly_one_zeros_le_one
#print axioms expPoly_two_zeros_le_three
#print axioms expPoly_three_zeros_le_five

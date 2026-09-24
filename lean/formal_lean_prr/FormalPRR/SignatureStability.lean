/-
FormalPRR/SignatureStability.lean — gap-closure item L3.

Source (paper): the perturbation-stable signature lemma, i.e. the last paragraph
of the proof of Theorem thm:exactmfull-doi (per-tube sign preservation) and Step 2
of prop:exactmfull-b0 ("margins suffice ⇒ exactly one root per tube"), reused for
the local count of the fixed-budget passage theorem (plan TH-3).

Statement.  Let [a, b] contain finitely many pairwise disjoint closed tubes
[α_k, β_k] (α_k < β_k).  Suppose the reference derivative G' satisfies
  • |G'| ≥ μ₁ at every point of [a, b] outside the open tubes,
  • G' changes sign across each tube: G'(α_k) G'(β_k) < 0,
  • |G''| ≥ μ₂ on each closed tube,
and the perturbed derivative F' (with derivative F'' on the tubes) satisfies
  • |F' − G'| < μ₁ outside the open tubes,  • |F'' − G''| < μ₂ on the tubes.
Then F' has exactly one zero in each closed tube; it lies in the open tube, is
nondegenerate (F'' ≠ 0) and has the curvature sign of G''; F' has no zero in
[a, b] outside the open tubes (endpoints a, b and tube endpoints included) and
has the sign of G' there; and the zero set of F' in [a, b] has exactly n points.
No continuity of G'' is assumed: F'' has constant sign on each tube by Darboux.

The corollary `signature_of_margins` combines this with B0ChainKernel.margins_iff:
jet bounds |F'−G'| ≤ E/r₀, |F''−G''| ≤ 2E/r₀² and E < min(r₀μ₁, r₀²μ₂/2) give the
exact signature; `signature_of_budget` further chains BudgetThreshold.Ec_lt_iff
(B < B_cert ⇔ E_c(B) < M̂).

-- SCOPE NOTE: F', F'', G', G'' are abstract real functions.  The Cauchy/Dyson jet
-- bounds that produce |F'−G'| and |F''−G''| (Step 1 of prop:exactmfull-b0, the
-- flat Dyson chain), the construction of the tubes and margins μ₁, μ₂ from the
-- pure-mixture/slow-factor analysis, and the passage-profile C² convergence of
-- plan TH-3 are NOT formalized; they enter only as hypotheses.
-/
import Mathlib.Analysis.Calculus.Darboux
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Data.Set.Card
import Mathlib.Data.Sign.Basic
import FormalPRR.B0ChainKernel
import FormalPRR.BudgetThreshold

namespace FormalPRR
namespace Signature

open Set

/-- **Sign transfer** (internal helper): `|f − g| < μ ≤ |g|` forces
`sign f = sign g` (and `g ≠ 0`). -/
lemma sign_eq_of_abs_sub_lt {f g μ : ℝ} (hg : μ ≤ |g|) (hfg : |f - g| < μ) :
    SignType.sign f = SignType.sign g := by
  have h0 : 0 ≤ |f - g| := abs_nonneg _
  obtain ⟨h1, h2⟩ := abs_sub_lt_iff.1 hfg
  rcases lt_trichotomy g 0 with hneg | hzero | hpos
  · rw [abs_of_neg hneg] at hg
    rw [sign_neg hneg, sign_neg (by linarith)]
  · rw [hzero, abs_zero] at hg
    linarith
  · rw [abs_of_pos hpos] at hg
    rw [sign_pos hpos, sign_pos (by linarith)]

lemma ne_zero_of_abs_sub_lt {f g μ : ℝ} (hg : μ ≤ |g|) (hfg : |f - g| < μ) : f ≠ 0 := by
  intro hf
  rw [hf, zero_sub, abs_neg] at hfg
  linarith

/-- Two sign-matched pairs: `g₁ g₂ < 0` transfers to `f₁ f₂ < 0` (internal helper). -/
lemma mul_neg_of_sign_eq {f₁ f₂ g₁ g₂ : ℝ} (h₁ : SignType.sign f₁ = SignType.sign g₁)
    (h₂ : SignType.sign f₂ = SignType.sign g₂) (hg : g₁ * g₂ < 0) : f₁ * f₂ < 0 := by
  have : SignType.sign (f₁ * f₂) = -1 := by
    rw [sign_mul, h₁, h₂, ← sign_mul, sign_neg hg]
  exact sign_eq_neg_one_iff.1 this

/-- **Single tube** (plan L3): if `F'` has derivative `F''` on `[α, β]`, `F''` never
vanishes there, and `F'(α) F'(β) < 0`, then `F'` has exactly one zero on `[α, β]`. -/
theorem tube_unique_zero {α β : ℝ} (hαβ : α < β) {F' F'' : ℝ → ℝ}
    (hF : ∀ t ∈ Icc α β, HasDerivAt F' (F'' t) t) (hne : ∀ t ∈ Icc α β, F'' t ≠ 0)
    (hsign : F' α * F' β < 0) :
    ∃! t, t ∈ Icc α β ∧ F' t = 0 := by
  have hcont : ContinuousOn F' (Icc α β) :=
    fun t ht => (hF t ht).continuousAt.continuousWithinAt
  have hderiv : ∀ t ∈ interior (Icc α β), HasDerivWithinAt F' (F'' t) (interior (Icc α β)) t := by
    intro t ht
    rw [interior_Icc] at ht
    exact (hF t (Ioo_subset_Icc_self ht)).hasDerivWithinAt
  rcases hasDerivWithinAt_forall_lt_or_forall_gt_of_forall_ne (convex_Icc α β)
      (fun t ht => (hF t ht).hasDerivWithinAt) hne with hlt | hgt
  · -- F'' < 0: F' strictly decreasing
    have hanti : StrictAntiOn F' (Icc α β) :=
      strictAntiOn_of_hasDerivWithinAt_neg (convex_Icc α β) hcont hderiv
        (fun t ht => hlt t (interior_subset ht))
    have hab := hanti (left_mem_Icc.2 hαβ.le) (right_mem_Icc.2 hαβ.le) hαβ
    have hβ : F' β < 0 ∧ 0 < F' α := by
      rcases mul_neg_iff.1 hsign with h | h
      · exact ⟨h.2, h.1⟩
      · exact absurd hab (by linarith [h.1, h.2])
    obtain ⟨t, ht, hzero⟩ := intermediate_value_Icc' hαβ.le hcont ⟨hβ.1.le, hβ.2.le⟩
    refine ⟨t, ⟨ht, hzero⟩, fun s hs => hanti.injOn hs.1 ht (by rw [hs.2, hzero])⟩
  · -- F'' > 0: F' strictly increasing
    have hmono : StrictMonoOn F' (Icc α β) :=
      strictMonoOn_of_hasDerivWithinAt_pos (convex_Icc α β) hcont hderiv
        (fun t ht => hgt t (interior_subset ht))
    have hab := hmono (left_mem_Icc.2 hαβ.le) (right_mem_Icc.2 hαβ.le) hαβ
    have hα : F' α < 0 ∧ 0 < F' β := by
      rcases mul_neg_iff.1 hsign with h | h
      · exact absurd hab (by linarith [h.1, h.2])
      · exact h
    obtain ⟨t, ht, hzero⟩ := intermediate_value_Icc hαβ.le hcont ⟨hα.1.le, hα.2.le⟩
    refine ⟨t, ⟨ht, hzero⟩, fun s hs => hmono.injOn hs.1 ht (by rw [hs.2, hzero])⟩

/-- Tube endpoints lie outside every open tube when the closed tubes are disjoint. -/
lemma endpoint_not_mem {n : ℕ} {α β : Fin n → ℝ}
    (hdisj : ∀ k l, k ≠ l → Disjoint (Icc (α k) (β k)) (Icc (α l) (β l)))
    {k : Fin n} {t : ℝ} (ht : t ∈ Icc (α k) (β k)) (hend : t = α k ∨ t = β k) :
    ∀ l, t ∉ Ioo (α l) (β l) := by
  intro l hl
  by_cases hkl : k = l
  · subst hkl
    rcases hend with h | h <;> rw [h] at hl
    · exact lt_irrefl _ hl.1
    · exact lt_irrefl _ hl.2
  · exact (Set.disjoint_left.1 (hdisj k l hkl)) ht (Ioo_subset_Icc_self hl)

/-- **Perturbation-stable signature lemma** (plan L3; proof of thm:exactmfull-doi,
last paragraph; prop:exactmfull-b0 Step 2; plan TH-3 local count). -/
theorem signature_stable {a b μ₁ μ₂ : ℝ} {n : ℕ} {α β : Fin n → ℝ}
    {F' F'' G' G'' : ℝ → ℝ}
    (hαβ : ∀ k, α k < β k) (hsub : ∀ k, Icc (α k) (β k) ⊆ Icc a b)
    (hdisj : ∀ k l, k ≠ l → Disjoint (Icc (α k) (β k)) (Icc (α l) (β l)))
    (hF : ∀ k, ∀ t ∈ Icc (α k) (β k), HasDerivAt F' (F'' t) t)
    (hoff : ∀ t ∈ Icc a b, (∀ k, t ∉ Ioo (α k) (β k)) → μ₁ ≤ |G' t|)
    (hon : ∀ k, ∀ t ∈ Icc (α k) (β k), μ₂ ≤ |G'' t|)
    (hcross : ∀ k, G' (α k) * G' (β k) < 0)
    (hF1 : ∀ t ∈ Icc a b, (∀ k, t ∉ Ioo (α k) (β k)) → |F' t - G' t| < μ₁)
    (hF2 : ∀ k, ∀ t ∈ Icc (α k) (β k), |F'' t - G'' t| < μ₂) :
    (∀ k, ∃! t, t ∈ Icc (α k) (β k) ∧ F' t = 0) ∧
    (∀ k, ∀ t ∈ Icc (α k) (β k), F' t = 0 →
      t ∈ Ioo (α k) (β k) ∧ F'' t ≠ 0 ∧ SignType.sign (F'' t) = SignType.sign (G'' t)) ∧
    (∀ t ∈ Icc a b, (∀ k, t ∉ Ioo (α k) (β k)) →
      F' t ≠ 0 ∧ SignType.sign (F' t) = SignType.sign (G' t)) := by
  -- off the open tubes: F' ≠ 0 with the sign of G'
  have hoffF : ∀ t ∈ Icc a b, (∀ k, t ∉ Ioo (α k) (β k)) →
      F' t ≠ 0 ∧ SignType.sign (F' t) = SignType.sign (G' t) :=
    fun t ht hnot => ⟨ne_zero_of_abs_sub_lt (hoff t ht hnot) (hF1 t ht hnot),
      sign_eq_of_abs_sub_lt (hoff t ht hnot) (hF1 t ht hnot)⟩
  -- endpoint signs
  have hends : ∀ k, F' (α k) * F' (β k) < 0 := by
    intro k
    have hαm : α k ∈ Icc (α k) (β k) := left_mem_Icc.2 (hαβ k).le
    have hβm : β k ∈ Icc (α k) (β k) := right_mem_Icc.2 (hαβ k).le
    have h1 := hoffF (α k) (hsub k hαm) (endpoint_not_mem hdisj hαm (Or.inl rfl))
    have h2 := hoffF (β k) (hsub k hβm) (endpoint_not_mem hdisj hβm (Or.inr rfl))
    exact mul_neg_of_sign_eq h1.2 h2.2 (hcross k)
  -- on the tubes: F'' ≠ 0 with the sign of G''
  have honF : ∀ k, ∀ t ∈ Icc (α k) (β k),
      F'' t ≠ 0 ∧ SignType.sign (F'' t) = SignType.sign (G'' t) :=
    fun k t ht => ⟨ne_zero_of_abs_sub_lt (hon k t ht) (hF2 k t ht),
      sign_eq_of_abs_sub_lt (hon k t ht) (hF2 k t ht)⟩
  refine ⟨fun k => tube_unique_zero (hαβ k) (hF k) (fun t ht => (honF k t ht).1) (hends k),
    ?_, hoffF⟩
  intro k t ht hzero
  refine ⟨?_, honF k t ht⟩
  rcases eq_or_lt_of_le ht.1 with h | h
  · have := hends k
    rw [h, hzero, zero_mul] at this
    exact absurd this (lt_irrefl 0)
  rcases eq_or_lt_of_le ht.2 with h' | h'
  · have := hends k
    rw [← h', hzero, mul_zero] at this
    exact absurd this (lt_irrefl 0)
  exact ⟨h, h'⟩

/-- **Exact count** (plan L3): under the hypotheses of `signature_stable`, the zero
set of `F'` in `[a, b]` has exactly `n` points (one per tube). -/
theorem signature_count {a b μ₁ μ₂ : ℝ} {n : ℕ} {α β : Fin n → ℝ}
    {F' F'' G' G'' : ℝ → ℝ}
    (hαβ : ∀ k, α k < β k) (hsub : ∀ k, Icc (α k) (β k) ⊆ Icc a b)
    (hdisj : ∀ k l, k ≠ l → Disjoint (Icc (α k) (β k)) (Icc (α l) (β l)))
    (hF : ∀ k, ∀ t ∈ Icc (α k) (β k), HasDerivAt F' (F'' t) t)
    (hoff : ∀ t ∈ Icc a b, (∀ k, t ∉ Ioo (α k) (β k)) → μ₁ ≤ |G' t|)
    (hon : ∀ k, ∀ t ∈ Icc (α k) (β k), μ₂ ≤ |G'' t|)
    (hcross : ∀ k, G' (α k) * G' (β k) < 0)
    (hF1 : ∀ t ∈ Icc a b, (∀ k, t ∉ Ioo (α k) (β k)) → |F' t - G' t| < μ₁)
    (hF2 : ∀ k, ∀ t ∈ Icc (α k) (β k), |F'' t - G'' t| < μ₂) :
    {t | t ∈ Icc a b ∧ F' t = 0}.ncard = n := by
  obtain ⟨huniq, -, hoffF⟩ :=
    signature_stable hαβ hsub hdisj hF hoff hon hcross hF1 hF2
  choose z hz hzu using huniq
  have hinj : Function.Injective z := by
    intro k l hkl
    by_contra hne
    exact Set.disjoint_left.1 (hdisj k l hne) (hz k).1 (hkl ▸ (hz l).1)
  have hset : {t | t ∈ Icc a b ∧ F' t = 0} = range z := by
    ext t
    constructor
    · rintro ⟨ht, hzero⟩
      by_contra hnot
      have hout : ∀ k, t ∉ Ioo (α k) (β k) := by
        intro k hk
        exact hnot ⟨k, (hzu k t ⟨Ioo_subset_Icc_self hk, hzero⟩).symm⟩
      exact (hoffF t ht hout).1 hzero
    · rintro ⟨k, rfl⟩
      exact ⟨hsub k (hz k).1, (hz k).2⟩
  rw [hset, Set.ncard_range_of_injective hinj, Nat.card_eq_fintype_card, Fintype.card_fin]

/-- **Margins suffice** (plan L3 + B0ChainKernel.margins_iff): uniform jet bounds
`|F' − G'| ≤ E/r₀` on `[a, b]` and `|F'' − G''| ≤ 2E/r₀²` on the tubes, with
`E < min(r₀ μ₁, r₀² μ₂/2)`, give the exact signature (one nondegenerate zero of `F'`
per tube, none elsewhere, count `n`). -/
theorem signature_of_margins {a b μ₁ μ₂ E r0 : ℝ} (hr : 0 < r0) {n : ℕ}
    {α β : Fin n → ℝ} {F' F'' G' G'' : ℝ → ℝ}
    (hαβ : ∀ k, α k < β k) (hsub : ∀ k, Icc (α k) (β k) ⊆ Icc a b)
    (hdisj : ∀ k l, k ≠ l → Disjoint (Icc (α k) (β k)) (Icc (α l) (β l)))
    (hF : ∀ k, ∀ t ∈ Icc (α k) (β k), HasDerivAt F' (F'' t) t)
    (hoff : ∀ t ∈ Icc a b, (∀ k, t ∉ Ioo (α k) (β k)) → μ₁ ≤ |G' t|)
    (hon : ∀ k, ∀ t ∈ Icc (α k) (β k), μ₂ ≤ |G'' t|)
    (hcross : ∀ k, G' (α k) * G' (β k) < 0)
    (hJ1 : ∀ t ∈ Icc a b, |F' t - G' t| ≤ E / r0)
    (hJ2 : ∀ k, ∀ t ∈ Icc (α k) (β k), |F'' t - G'' t| ≤ 2 * E / r0 ^ 2)
    (hE : E < min (r0 * μ₁) (r0 ^ 2 * μ₂ / 2)) :
    (∀ k, ∃! t, t ∈ Icc (α k) (β k) ∧ F' t = 0) ∧
    (∀ k, ∀ t ∈ Icc (α k) (β k), F' t = 0 →
      t ∈ Ioo (α k) (β k) ∧ F'' t ≠ 0 ∧ SignType.sign (F'' t) = SignType.sign (G'' t)) ∧
    (∀ t ∈ Icc a b, (∀ k, t ∉ Ioo (α k) (β k)) →
      F' t ≠ 0 ∧ SignType.sign (F' t) = SignType.sign (G' t)) ∧
    {t | t ∈ Icc a b ∧ F' t = 0}.ncard = n := by
  obtain ⟨hm1, hm2⟩ := (B0ChainKernel.margins_iff hr E μ₁ μ₂).2 hE
  have hF1 : ∀ t ∈ Icc a b, (∀ k, t ∉ Ioo (α k) (β k)) → |F' t - G' t| < μ₁ :=
    fun t ht _ => lt_of_le_of_lt (hJ1 t ht) hm1
  have hF2 : ∀ k, ∀ t ∈ Icc (α k) (β k), |F'' t - G'' t| < μ₂ :=
    fun k t ht => lt_of_le_of_lt (hJ2 k t ht) hm2
  obtain ⟨h1, h2, h3⟩ := signature_stable hαβ hsub hdisj hF hoff hon hcross hF1 hF2
  exact ⟨h1, h2, h3, signature_count hαβ hsub hdisj hF hoff hon hcross hF1 hF2⟩

/-- **Budget form** (plan L3 + BudgetThreshold.Ec_lt_iff): if the jet bounds hold
with `E = E_c(B)` and `0 < M̂ = min(r₀ μ₁, r₀² μ₂/2)`, then every `B < B_cert(M̂)`
yields the exact signature.  This is the machine-checked logical skeleton
"jet bound `E_c(B)` and `B < B_cert` ⇒ exact window signature"; the jet bound
itself (Cauchy step, flat Dyson chain) is a hypothesis. -/
theorem signature_of_budget {a b μ₁ μ₂ r0 : ℝ} (hr : 0 < r0) {n : ℕ}
    {α β : Fin n → ℝ} {F' F'' G' G'' : ℝ → ℝ}
    {kappaHat vInf Pi delta T B : ℝ}
    (hk : 0 < kappaHat) (hv : 0 < vInf) (hd : 0 ≤ delta) (hTr : 0 < T + r0)
    (hM : 0 < min (r0 * μ₁) (r0 ^ 2 * μ₂ / 2))
    (hαβ : ∀ k, α k < β k) (hsub : ∀ k, Icc (α k) (β k) ⊆ Icc a b)
    (hdisj : ∀ k l, k ≠ l → Disjoint (Icc (α k) (β k)) (Icc (α l) (β l)))
    (hF : ∀ k, ∀ t ∈ Icc (α k) (β k), HasDerivAt F' (F'' t) t)
    (hoff : ∀ t ∈ Icc a b, (∀ k, t ∉ Ioo (α k) (β k)) → μ₁ ≤ |G' t|)
    (hon : ∀ k, ∀ t ∈ Icc (α k) (β k), μ₂ ≤ |G'' t|)
    (hcross : ∀ k, G' (α k) * G' (β k) < 0)
    (hJ1 : ∀ t ∈ Icc a b,
      |F' t - G' t| ≤ BudgetThreshold.Ec kappaHat vInf Pi delta T r0 B / r0)
    (hJ2 : ∀ k, ∀ t ∈ Icc (α k) (β k),
      |F'' t - G'' t| ≤ 2 * BudgetThreshold.Ec kappaHat vInf Pi delta T r0 B / r0 ^ 2)
    (hB : B < BudgetThreshold.Bcert kappaHat vInf Pi delta T r0
      (min (r0 * μ₁) (r0 ^ 2 * μ₂ / 2))) :
    (∀ k, ∃! t, t ∈ Icc (α k) (β k) ∧ F' t = 0) ∧
    {t | t ∈ Icc a b ∧ F' t = 0}.ncard = n := by
  have hE := (BudgetThreshold.Ec_lt_iff Pi hk hv hd hTr hM B).2 hB
  obtain ⟨h1, -, -, h4⟩ :=
    signature_of_margins hr hαβ hsub hdisj hF hoff hon hcross hJ1 hJ2 hE
  exact ⟨h1, h4⟩

end Signature
end FormalPRR

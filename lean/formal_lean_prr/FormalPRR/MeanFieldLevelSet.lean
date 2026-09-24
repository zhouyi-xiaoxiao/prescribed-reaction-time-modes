/-
FormalPRR/MeanFieldLevelSet.lean — gap-closure item L2(a) (plan TH-5).

Source (paper, CNSNS rebuild): the mean-field level-set proposition.  With the
mean exposure rate G(t) = E V_t ≥ 0 and the mean cumulative exposure
Λ(t) = ∫₀ᵗ G, the mean-field reaction-time density at total budget B is
    f₁(t) = B G(t) e^{−B Λ(t)}.
Its derivative is f₁' = B e^{−BΛ} (G' − B G²), hence
    sign f₁'(t) = sign (G'(t) − B G(t)²),
and, where G > 0, f₁ rises exactly on the level set {G'/G² > B}.  A rising flank
of f₁ therefore survives at budget B iff B < sup_flank G'/G², which is the
definition B_top^mf = min over rising flanks of max G'/G² (plan TH-5).

-- SCOPE NOTE: this file is pure one-variable calculus on given functions G, Λ
-- with Λ' = G.  It does NOT formalize the identification of G with the model's
-- exposure rate, the asymptotics ln B_top^mf = Δ*²/(8σ²) + ln(1/ε) + C₀ + o(1),
-- or any statement about the exact (non-mean-field) process.
-/
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Topology.Order.Basic

namespace FormalPRR
namespace MeanField

open Set Filter Topology

/-- The mean-field density `f₁(t) = B G(t) e^{−B Λ(t)}` (plan TH-5). -/
noncomputable def f1 (B : ℝ) (G Λ : ℝ → ℝ) (t : ℝ) : ℝ :=
  B * G t * Real.exp (-(B * Λ t))

/-- **Derivative of the mean-field density** (plan TH-5 / L2(a)):
if `Λ' = G` and `G' = g'` at `t`, then
`f₁'(t) = B e^{−BΛ(t)} (g' − B G(t)²)`. -/
theorem hasDerivAt_f1 {B : ℝ} {G Λ : ℝ → ℝ} {g' t : ℝ}
    (hΛ : HasDerivAt Λ (G t) t) (hG : HasDerivAt G g' t) :
    HasDerivAt (f1 B G Λ) (B * Real.exp (-(B * Λ t)) * (g' - B * G t ^ 2)) t := by
  have h1 : HasDerivAt (fun s => Real.exp (-(B * Λ s)))
      (Real.exp (-(B * Λ t)) * (-(B * G t))) t :=
    ((hΛ.const_mul B).neg).exp
  have h2 := (hG.const_mul B).mul h1
  have hfun : f1 B G Λ = fun s => B * G s * Real.exp (-(B * Λ s)) := rfl
  rw [hfun]
  exact h2.congr_deriv (by ring)

/-- The prefactor `B e^{−BΛ}` is positive for `B > 0` (internal helper). -/
lemma prefactor_pos {B L : ℝ} (hB : 0 < B) : 0 < B * Real.exp (-(B * L)) :=
  mul_pos hB (Real.exp_pos _)

/-- **Sign law, positive case** (plan TH-5): for `B > 0`,
`0 < B e^{−BΛ}(g' − BG²) ↔ B G² < g'`. -/
theorem deriv_pos_iff {B L g G : ℝ} (hB : 0 < B) :
    0 < B * Real.exp (-(B * L)) * (g - B * G ^ 2) ↔ B * G ^ 2 < g := by
  have hc := prefactor_pos (L := L) hB
  constructor
  · intro h
    by_contra hle
    have hle : g ≤ B * G ^ 2 := not_lt.mp hle
    have : B * Real.exp (-(B * L)) * (g - B * G ^ 2) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos hc.le (by linarith)
    linarith
  · intro h
    exact mul_pos hc (by linarith)

/-- **Sign law, negative case** (plan TH-5): for `B > 0`,
`B e^{−BΛ}(g' − BG²) < 0 ↔ g' < B G²`. -/
theorem deriv_neg_iff {B L g G : ℝ} (hB : 0 < B) :
    B * Real.exp (-(B * L)) * (g - B * G ^ 2) < 0 ↔ g < B * G ^ 2 := by
  have hc := prefactor_pos (L := L) hB
  constructor
  · intro h
    by_contra hle
    have hle : B * G ^ 2 ≤ g := not_lt.mp hle
    have : 0 ≤ B * Real.exp (-(B * L)) * (g - B * G ^ 2) :=
      mul_nonneg hc.le (by linarith)
    linarith
  · intro h
    exact mul_neg_of_pos_of_neg hc (by linarith)

/-- **Sign law, zero case** (plan TH-5): for `B > 0`,
`B e^{−BΛ}(g' − BG²) = 0 ↔ g' = B G²`. -/
theorem deriv_eq_zero_iff {B L g G : ℝ} (hB : 0 < B) :
    B * Real.exp (-(B * L)) * (g - B * G ^ 2) = 0 ↔ g = B * G ^ 2 := by
  have hc := prefactor_pos (L := L) hB
  rw [mul_eq_zero, sub_eq_zero]
  constructor
  · rintro (h | h)
    · exact absurd h hc.ne'
    · exact h
  · intro h; exact Or.inr h

/-- **Sign equivalence** `sign f₁' = sign (G' − B G²)` as one statement
(plan TH-5 / L2(a)): for `B > 0`, `Λ' = G` and `G' = g'` at `t`,
`SignType.sign (f₁'(t)) = SignType.sign (g' − B G(t)²)`. -/
theorem sign_deriv_f1 {B : ℝ} {G Λ : ℝ → ℝ} {g' t : ℝ} (hB : 0 < B)
    (hΛ : HasDerivAt Λ (G t) t) (hG : HasDerivAt G g' t) :
    SignType.sign (deriv (f1 B G Λ) t) = SignType.sign (g' - B * G t ^ 2) := by
  rw [(hasDerivAt_f1 (B := B) hΛ hG).deriv]
  have hc := prefactor_pos (L := Λ t) hB
  rw [sign_mul, sign_pos hc, one_mul]

/-- **Level-set form** (plan TH-5): where `G(t) > 0` and `B > 0`,
`f₁` has positive derivative at `t` iff `B < G'(t)/G(t)²`. -/
theorem deriv_f1_pos_iff_levelset {B : ℝ} {G Λ : ℝ → ℝ} {g' t : ℝ} (hB : 0 < B)
    (hΛ : HasDerivAt Λ (G t) t) (hG : HasDerivAt G g' t) (hGpos : 0 < G t) :
    0 < deriv (f1 B G Λ) t ↔ B < g' / G t ^ 2 := by
  rw [(hasDerivAt_f1 (B := B) hΛ hG).deriv, deriv_pos_iff hB,
    lt_div_iff₀ (pow_pos hGpos 2)]

/-- **Downward closure in B** (plan TH-5): a point of the rising level set at
budget `B` stays in it for every smaller nonnegative budget `B' ≤ B`. Hence the
set of budgets at which a given flank still rises is an interval `[0, B_flank)`,
with `B_flank = sup_flank G'/G²`. -/
theorem rising_mono_budget {B B' g G : ℝ} (hBB' : B' ≤ B) (h : B * G ^ 2 < g) :
    B' * G ^ 2 < g :=
  lt_of_le_of_lt (mul_le_mul_of_nonneg_right hBB' (sq_nonneg G)) h

/-- **No rise on a flank above its threshold** (plan TH-5): if `B ≥ 0` and
`G' ≤ B G²` throughout the open flank `(a, b)`, then `f₁` is antitone on
`[a, b]` (the mean-field mode of that flank is lost). -/
theorem f1_antitoneOn {B a b : ℝ} {G Λ G' : ℝ → ℝ} (hB : 0 ≤ B)
    (hΛ : ∀ t ∈ Icc a b, HasDerivAt Λ (G t) t)
    (hG : ∀ t ∈ Icc a b, HasDerivAt G (G' t) t)
    (hle : ∀ t ∈ Ioo a b, G' t ≤ B * G t ^ 2) :
    AntitoneOn (f1 B G Λ) (Icc a b) := by
  have hd : ∀ t ∈ Icc a b, HasDerivAt (f1 B G Λ)
      (B * Real.exp (-(B * Λ t)) * (G' t - B * G t ^ 2)) t :=
    fun t ht => hasDerivAt_f1 (hΛ t ht) (hG t ht)
  apply antitoneOn_of_hasDerivWithinAt_nonpos (convex_Icc a b)
  · exact fun t ht => (hd t ht).continuousAt.continuousWithinAt
  · intro t ht
    rw [interior_Icc] at ht
    exact (hd t (Ioo_subset_Icc_self ht)).hasDerivWithinAt
  · intro t ht
    rw [interior_Icc] at ht
    exact mul_nonpos_of_nonneg_of_nonpos (mul_nonneg hB (Real.exp_pos _).le)
      (by linarith [hle t ht])

/-- **Strict rise below the threshold** (plan TH-5): if `B > 0` and
`B G² < G'` throughout `(a, b)`, then `f₁` is strictly increasing on `[a, b]`. -/
theorem f1_strictMonoOn {B a b : ℝ} {G Λ G' : ℝ → ℝ} (hB : 0 < B)
    (hΛ : ∀ t ∈ Icc a b, HasDerivAt Λ (G t) t)
    (hG : ∀ t ∈ Icc a b, HasDerivAt G (G' t) t)
    (hlt : ∀ t ∈ Ioo a b, B * G t ^ 2 < G' t) :
    StrictMonoOn (f1 B G Λ) (Icc a b) := by
  have hd : ∀ t ∈ Icc a b, HasDerivAt (f1 B G Λ)
      (B * Real.exp (-(B * Λ t)) * (G' t - B * G t ^ 2)) t :=
    fun t ht => hasDerivAt_f1 (hΛ t ht) (hG t ht)
  apply strictMonoOn_of_hasDerivWithinAt_pos (convex_Icc a b)
  · exact fun t ht => (hd t ht).continuousAt.continuousWithinAt
  · intro t ht
    rw [interior_Icc] at ht
    exact (hd t (Ioo_subset_Icc_self ht)).hasDerivWithinAt
  · intro t ht
    rw [interior_Icc] at ht
    exact (deriv_pos_iff hB).2 (hlt t ht)

/-- **Local rise at a point of the level set** (plan TH-5): if `B > 0`, `Λ' = G`
and `G'` exist near `t₀`, `G` and `G'` are continuous at `t₀`, and
`B G(t₀)² < G'(t₀)`, then `f₁` is strictly increasing on some interval
`[t₀ − δ, t₀ + δ]`, `δ > 0`.  Together with `f1_antitoneOn` this is the
level-set characterization "a flank rises at budget `B` iff `B < sup G'/G²`". -/
theorem f1_rises_near {B t₀ : ℝ} {G Λ G' : ℝ → ℝ} (hB : 0 < B)
    (hΛ : ∀ᶠ t in 𝓝 t₀, HasDerivAt Λ (G t) t)
    (hG : ∀ᶠ t in 𝓝 t₀, HasDerivAt G (G' t) t)
    (hGc : ContinuousAt G t₀) (hG'c : ContinuousAt G' t₀)
    (hlt : B * G t₀ ^ 2 < G' t₀) :
    ∃ δ > 0, StrictMonoOn (f1 B G Λ) (Icc (t₀ - δ) (t₀ + δ)) := by
  have hcont : ContinuousAt (fun t => G' t - B * G t ^ 2) t₀ :=
    hG'c.sub ((hGc.pow 2).const_mul B)
  have hpos : ∀ᶠ t in 𝓝 t₀, 0 < G' t - B * G t ^ 2 :=
    hcont.eventually (lt_mem_nhds (by linarith))
  have hall := (hΛ.and hG).and hpos
  obtain ⟨ε, hε, hball⟩ := Metric.eventually_nhds_iff.1 hall
  refine ⟨ε / 2, half_pos hε, ?_⟩
  have hsub : ∀ t ∈ Icc (t₀ - ε / 2) (t₀ + ε / 2), dist t t₀ < ε := by
    intro t ht
    rw [Real.dist_eq, abs_lt]
    constructor <;> linarith [ht.1, ht.2]
  apply f1_strictMonoOn hB
  · exact fun t ht => (hball (hsub t ht)).1.1
  · exact fun t ht => (hball (hsub t ht)).1.2
  · intro t ht
    have := (hball (hsub t (Ioo_subset_Icc_self ht))).2
    linarith

end MeanField
end FormalPRR

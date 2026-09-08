/-
FormalPRR/CrossoverBounds.lean  (target A3: crossover ratio bounds)

Tex source (docstring anchors):
  prr_submission/exact_m_theorem_spine.tex —
  Eq. (eq:exact-m-crossover): the weighted crossover
    s_j = (c_j + c_{j+1})/2 + sigma^2/(c_{j+1} - c_j) log(w_j/w_{j+1}),
  and the sentence following it: "The adjacent ratio obeys
    q_{j+1}/q_j = 1/9 at s_j - sigma^2 log 9/(c_{j+1} - c_j)
  and q_{j+1}/q_j = 9 at the corresponding plus edge", with
  q_j(x) = w_j exp[-(x - c_j)^2/(2 sigma^2)] as in
  Eq. (eq:exact-m-mixture) / the paragraph before Eq. (eq:exact-m-log-slope);
  "all nonadjacent components are exponentially small" (same paragraph).
  prr_submission/exact_m_theorem_full_proof.tex —
  Eq. (eq:exactmfull-logistic-posterior) via the adjacent-odds identity
    q_{j+1}(x)/q_j(x) = exp[Delta_j (x - s_j)/sigma^2]
  (display before Eq. eq:exactmfull-crossover-interval), and the crossover
  interval C_j of width sigma^2-scale around s_j.

Contents:
  * `crossover`  — Eq. (eq:exact-m-crossover), verbatim.
  * `adjacent_odds` — the exact odds identity
    q_2(x) = q_1(x) exp[(c_2 - c_1)(x - s)/sigma^2] (full proof display).
  * `crossover_ratio_one` / `_eq_one` — q_2(s)/q_1(s) = 1.
  * `crossover_ratio_minus` / `crossover_ratio_plus` — ratio 1/9 and 9 at
    x = s ∓ sigma^2 log 9/(c_2 - c_1).  All exact exp-algebra identities.
  * `nonadjacent_small` — nonadjacent smallness with explicit constants:
    under min-spacing Delta > 0, for x in the window
    [s_j - sigma^2 log 9/Delta, s_j + sigma^2 log 9/Delta] and i not in
    {j, j+1}:  q_i(x)/max(q_j(x), q_{j+1}(x))
                 <= (w_i / min(w_j, w_{j+1})) exp(-(3/4) Delta^2/sigma^2),
    i.e. the constants of the target statement are C = w_i/min(w_j, w_{j+1})
    and c = 4/3 (in the form C exp(-Delta^2/(c sigma^2))).
    -- SCOPE NOTE: the spine asserts nonadjacent smallness "for all
    sufficiently small sigma" with unspecified constants; here the smallness
    regime is made explicit by the hypothesis
    sigma^2 (|log(w_j/w_{j+1})| + log 9) <= Delta^2/4, which keeps the
    crossover window inside the (c_j, c_{j+1}) gap.  The constants C, c are
    our own explicit choices, as FORMALIZATION_TARGETS.md A3 instructs
    ("they need not match the prose's implicit ones but must be stated and
    proved").
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace FormalPRR
namespace CrossoverBounds

noncomputable section
open Real

/-- Mixture component `q_j(x) = w_j exp[-(x - c_j)^2/(2 sigma^2)]`
(spine, paragraph before Eq. eq:exact-m-log-slope). -/
def q (w c sigma x : ℝ) : ℝ := w * Real.exp (-(x - c) ^ 2 / (2 * sigma ^ 2))

/-- Positivity of a mixture component with positive weight.  Internal helper
(elementary property of the `q_j` of the paragraph before
Eq. eq:exact-m-log-slope); no direct paper display. -/
theorem q_pos {w : ℝ} (hw : 0 < w) (c sigma x : ℝ) : 0 < q w c sigma x :=
  mul_pos hw (Real.exp_pos _)

/-- Nonnegativity of a mixture component with nonnegative weight.  Internal
helper; no direct paper display. -/
theorem q_nonneg {w : ℝ} (hw : 0 ≤ w) (c sigma x : ℝ) : 0 ≤ q w c sigma x :=
  mul_nonneg hw (Real.exp_pos _).le

/-- The weighted crossover of Eq. (eq:exact-m-crossover), verbatim:
`s = (c_1 + c_2)/2 + sigma^2/(c_2 - c_1) log(w_1/w_2)`. -/
def crossover (w1 w2 c1 c2 sigma : ℝ) : ℝ :=
  (c1 + c2) / 2 + sigma ^ 2 / (c2 - c1) * Real.log (w1 / w2)

/-- The exact adjacent-odds identity (full proof, display before
Eq. eq:exactmfull-crossover-interval):
`q_2(x) = q_1(x) exp[(c_2 - c_1)(x - s)/sigma^2]` for every `x`.
Pure exp-algebra; needs only `sigma ≠ 0`, `c_1 ≠ c_2`, `w_1, w_2 > 0`. -/
theorem adjacent_odds {sigma c1 c2 w1 w2 : ℝ} (hs : sigma ≠ 0) (hc : c1 ≠ c2)
    (hw1 : 0 < w1) (hw2 : 0 < w2) (x : ℝ) :
    q w2 c2 sigma x =
      q w1 c1 sigma x *
        Real.exp ((c2 - c1) * (x - crossover w1 w2 c1 c2 sigma) / sigma ^ 2) := by
  have hc' : c2 - c1 ≠ 0 := sub_ne_zero.mpr (Ne.symm hc)
  have hL : Real.exp (Real.log (w1 / w2)) = w1 / w2 := Real.exp_log (by positivity)
  unfold q crossover
  rw [mul_assoc, ← Real.exp_add]
  have hexp : -(x - c1) ^ 2 / (2 * sigma ^ 2) +
      (c2 - c1) * (x - ((c1 + c2) / 2 + sigma ^ 2 / (c2 - c1) * Real.log (w1 / w2))) /
        sigma ^ 2 =
      -(x - c2) ^ 2 / (2 * sigma ^ 2) + -Real.log (w1 / w2) := by
    field_simp
    ring
  rw [hexp, Real.exp_add, Real.exp_neg, hL]
  field_simp

/-- `q_2(s) = q_1(s)`: the two adjacent components agree at the crossover
(spine Eq. eq:exact-m-crossover; the "weighted crossover" is where the
adjacent odds are one). -/
theorem crossover_ratio_one {sigma c1 c2 w1 w2 : ℝ} (hs : sigma ≠ 0)
    (hc : c1 ≠ c2) (hw1 : 0 < w1) (hw2 : 0 < w2) :
    q w2 c2 sigma (crossover w1 w2 c1 c2 sigma) =
      q w1 c1 sigma (crossover w1 w2 c1 c2 sigma) := by
  rw [adjacent_odds hs hc hw1 hw2, sub_self, mul_zero, zero_div, Real.exp_zero,
    mul_one]

/-- Ratio form of `crossover_ratio_one`: `q_2(s)/q_1(s) = 1`
(FORMALIZATION_TARGETS.md A3, first claim). -/
theorem crossover_ratio_eq_one {sigma c1 c2 w1 w2 : ℝ} (hs : sigma ≠ 0)
    (hc : c1 ≠ c2) (hw1 : 0 < w1) (hw2 : 0 < w2) :
    q w2 c2 sigma (crossover w1 w2 c1 c2 sigma) /
        q w1 c1 sigma (crossover w1 w2 c1 c2 sigma) = 1 := by
  rw [crossover_ratio_one hs hc hw1 hw2]
  exact div_self (q_pos hw1 _ _ _).ne'

/-- The minus edge (spine: "The adjacent ratio obeys `q_{j+1}/q_j = 1/9` at
`s_j - sigma^2 log(9)/(c_{j+1} - c_j)`"). -/
theorem crossover_ratio_minus {sigma c1 c2 w1 w2 : ℝ} (hs : sigma ≠ 0)
    (hc : c1 ≠ c2) (hw1 : 0 < w1) (hw2 : 0 < w2) :
    q w2 c2 sigma (crossover w1 w2 c1 c2 sigma - sigma ^ 2 * Real.log 9 / (c2 - c1)) /
        q w1 c1 sigma (crossover w1 w2 c1 c2 sigma - sigma ^ 2 * Real.log 9 / (c2 - c1)) =
      1 / 9 := by
  have hc' : c2 - c1 ≠ 0 := sub_ne_zero.mpr (Ne.symm hc)
  set xm := crossover w1 w2 c1 c2 sigma - sigma ^ 2 * Real.log 9 / (c2 - c1) with hxm
  rw [adjacent_odds hs hc hw1 hw2 xm,
    mul_div_cancel_left₀ _ (q_pos hw1 c1 sigma xm).ne']
  have harg : (c2 - c1) * (xm - crossover w1 w2 c1 c2 sigma) / sigma ^ 2 =
      -Real.log 9 := by
    rw [hxm]
    field_simp
    ring
  rw [harg, Real.exp_neg, Real.exp_log (by norm_num : (0:ℝ) < 9), one_div]

/-- The plus edge (spine: "`q_{j+1}/q_j = 9` at the corresponding plus
edge"). -/
theorem crossover_ratio_plus {sigma c1 c2 w1 w2 : ℝ} (hs : sigma ≠ 0)
    (hc : c1 ≠ c2) (hw1 : 0 < w1) (hw2 : 0 < w2) :
    q w2 c2 sigma (crossover w1 w2 c1 c2 sigma + sigma ^ 2 * Real.log 9 / (c2 - c1)) /
        q w1 c1 sigma (crossover w1 w2 c1 c2 sigma + sigma ^ 2 * Real.log 9 / (c2 - c1)) =
      9 := by
  have hc' : c2 - c1 ≠ 0 := sub_ne_zero.mpr (Ne.symm hc)
  set xp := crossover w1 w2 c1 c2 sigma + sigma ^ 2 * Real.log 9 / (c2 - c1) with hxp
  rw [adjacent_odds hs hc hw1 hw2 xp,
    mul_div_cancel_left₀ _ (q_pos hw1 c1 sigma xp).ne']
  have harg : (c2 - c1) * (xp - crossover w1 w2 c1 c2 sigma) / sigma ^ 2 =
      Real.log 9 := by
    rw [hxp]
    field_simp
    ring
  rw [harg, Real.exp_log (by norm_num : (0:ℝ) < 9)]

/-! ## Nonadjacent smallness -/

/-- Spacing accumulates: if consecutive gaps are at least `Delta > 0`, then
`c i + Delta <= c j` for `i < j < m`.  Internal helper for
`nonadjacent_small` (the min-spacing hypothesis of FORMALIZATION_TARGETS.md
A3); no direct paper display. -/
theorem spacing_mono {m : ℕ} {c : ℕ → ℝ} {Delta : ℝ} (hD : 0 < Delta)
    (hspace : ∀ k, k + 1 < m → Delta ≤ c (k + 1) - c k) :
    ∀ i j, i < j → j < m → c i + Delta ≤ c j := by
  intro i j hij hjm
  induction j with
  | zero => omega
  | succ k ih =>
    rcases Nat.lt_succ_iff_lt_or_eq.mp hij with hik | hik
    · have h1 : c i + Delta ≤ c k := ih hik (by omega)
      have h2 : Delta ≤ c (k + 1) - c k := hspace k hjm
      linarith
    · subst hik
      have h2 : Delta ≤ c (i + 1) - c i := hspace i hjm
      linarith

/-- Exponent geometry, left case: a third centre at least `Delta` to the left
of `c_j`, with `x` at least `Delta/4` to the right of `c_j`, satisfies
`(3/2) Delta^2 + (x - c_j)^2 <= (x - c_i)^2`.  Internal helper for
`nonadjacent_small`; no direct paper display. -/
theorem far_exponent_left {Delta ci cj x : ℝ} (hD : 0 < Delta)
    (hA : Delta ≤ cj - ci) (hB : Delta / 4 ≤ x - cj) :
    3 / 2 * Delta ^ 2 + (x - cj) ^ 2 ≤ (x - ci) ^ 2 := by
  have hAB : Delta * (Delta / 4) ≤ (cj - ci) * (x - cj) :=
    mul_le_mul hA hB (by positivity) (le_trans hD.le hA)
  nlinarith [hA, hD]

/-- Exponent geometry, right case (mirror image of `far_exponent_left`).
Internal helper for `nonadjacent_small`; no direct paper display. -/
theorem far_exponent_right {Delta ci cj x : ℝ} (hD : 0 < Delta)
    (hA : Delta ≤ ci - cj) (hB : Delta / 4 ≤ cj - x) :
    3 / 2 * Delta ^ 2 + (x - cj) ^ 2 ≤ (x - ci) ^ 2 := by
  have hAB : Delta * (Delta / 4) ≤ (ci - cj) * (cj - x) :=
    mul_le_mul hA hB (by positivity) (le_trans hD.le hA)
  nlinarith [hA, hD]

/-- Gaussian-component domination from the exponent geometry:
`q_i(x) <= (w_i/w_ref) exp(-(3/4) Delta^2/sigma^2) q_ref(x)`.  Internal
helper for `nonadjacent_small` (the "exponentially small" mechanism of the
spine's nonadjacent-smallness sentence); no direct paper display. -/
theorem q_far_le {sigma Delta wi wref ci cref x : ℝ} (hs : sigma ≠ 0)
    (hwi : 0 ≤ wi) (hwref : 0 < wref)
    (hfact : 3 / 2 * Delta ^ 2 + (x - cref) ^ 2 ≤ (x - ci) ^ 2) :
    q wi ci sigma x ≤
      wi / wref * Real.exp (-(3 / 4) * Delta ^ 2 / sigma ^ 2) *
        q wref cref sigma x := by
  have h2s : (0 : ℝ) < 2 * sigma ^ 2 := by positivity
  have hexp : Real.exp (-(x - ci) ^ 2 / (2 * sigma ^ 2)) ≤
      Real.exp (-(3 / 4) * Delta ^ 2 / sigma ^ 2) *
        Real.exp (-(x - cref) ^ 2 / (2 * sigma ^ 2)) := by
    rw [← Real.exp_add]
    apply Real.exp_le_exp.mpr
    rw [show -(3 / 4) * Delta ^ 2 / sigma ^ 2 + -(x - cref) ^ 2 / (2 * sigma ^ 2) =
        (-(3 / 2 * Delta ^ 2) - (x - cref) ^ 2) / (2 * sigma ^ 2) by
      field_simp
      ring]
    rw [div_le_div_iff_of_pos_right h2s]
    linarith
  calc q wi ci sigma x = wi * Real.exp (-(x - ci) ^ 2 / (2 * sigma ^ 2)) := rfl
    _ ≤ wi * (Real.exp (-(3 / 4) * Delta ^ 2 / sigma ^ 2) *
          Real.exp (-(x - cref) ^ 2 / (2 * sigma ^ 2))) :=
        mul_le_mul_of_nonneg_left hexp hwi
    _ = wi / wref * Real.exp (-(3 / 4) * Delta ^ 2 / sigma ^ 2) *
          (wref * Real.exp (-(x - cref) ^ 2 / (2 * sigma ^ 2))) := by
        field_simp
    _ = _ := rfl

/-- Ratio bound against a dominating reference component inside the maximum:
if `q_ref(x) <= M`, `0 < M`, `w_min <= w_ref`, then
`q_i(x)/M <= (w_i/w_min) exp(-(3/4) Delta^2/sigma^2)`.  Internal helper for
`nonadjacent_small`; no direct paper display. -/
theorem ratio_bound_of_far {sigma Delta wi wref wmin ci cref x M : ℝ}
    (hs : sigma ≠ 0) (hwi : 0 ≤ wi) (hwref : 0 < wref) (hwmin : 0 < wmin)
    (hminref : wmin ≤ wref)
    (hfact : 3 / 2 * Delta ^ 2 + (x - cref) ^ 2 ≤ (x - ci) ^ 2)
    (hrefM : q wref cref sigma x ≤ M) (hM : 0 < M) :
    q wi ci sigma x / M ≤ wi / wmin * Real.exp (-(3 / 4) * Delta ^ 2 / sigma ^ 2) := by
  have hE : (0 : ℝ) < Real.exp (-(3 / 4) * Delta ^ 2 / sigma ^ 2) := Real.exp_pos _
  have hdivle : wi / wref ≤ wi / wmin := by
    rw [div_le_div_iff₀ hwref hwmin]
    exact mul_le_mul_of_nonneg_left hminref hwi
  have h1 : q wi ci sigma x ≤ wi / wref * Real.exp (-(3 / 4) * Delta ^ 2 / sigma ^ 2) * M := by
    calc q wi ci sigma x ≤
          wi / wref * Real.exp (-(3 / 4) * Delta ^ 2 / sigma ^ 2) * q wref cref sigma x :=
        q_far_le hs hwi hwref hfact
      _ ≤ wi / wref * Real.exp (-(3 / 4) * Delta ^ 2 / sigma ^ 2) * M := by
        apply mul_le_mul_of_nonneg_left hrefM
        positivity
  rw [div_le_iff₀ hM]
  calc q wi ci sigma x ≤ wi / wref * Real.exp (-(3 / 4) * Delta ^ 2 / sigma ^ 2) * M := h1
    _ ≤ wi / wmin * Real.exp (-(3 / 4) * Delta ^ 2 / sigma ^ 2) * M := by
        apply mul_le_mul_of_nonneg_right _ hM.le
        exact mul_le_mul_of_nonneg_right hdivle hE.le

/-- Position of the crossover window inside the gap: under the explicit
smallness hypothesis `sigma^2 (|L| + log 9) <= Delta^2/4` (with
`L = log(w_1/w_2)`) and gap `c_2 - c_1 >= Delta`, every `x` in the window
`[s - sigma^2 log 9/Delta, s + sigma^2 log 9/Delta]` satisfies
`c_1 + Delta/4 <= x <= c_2 - Delta/4`.  Internal helper for
`nonadjacent_small` (the crossover-interval containment behind
Eq. eq:exactmfull-crossover-interval of exact_m_theorem_full_proof.tex);
the explicit constants are ours, per the file-header SCOPE NOTE. -/
theorem window_inside_gap {sigma Delta c1 c2 L x : ℝ} (hs : sigma ≠ 0)
    (hD : 0 < Delta) (hgap : Delta ≤ c2 - c1)
    (hsmall : sigma ^ 2 * (|L| + Real.log 9) ≤ Delta ^ 2 / 4)
    (hx_lo : (c1 + c2) / 2 + sigma ^ 2 / (c2 - c1) * L - sigma ^ 2 * Real.log 9 / Delta ≤ x)
    (hx_hi : x ≤ (c1 + c2) / 2 + sigma ^ 2 / (c2 - c1) * L + sigma ^ 2 * Real.log 9 / Delta) :
    Delta / 4 ≤ x - c1 ∧ Delta / 4 ≤ c2 - x := by
  have hDj : (0 : ℝ) < c2 - c1 := lt_of_lt_of_le hD hgap
  have hs2 : (0 : ℝ) < sigma ^ 2 := by positivity
  have hlog9 : (0 : ℝ) < Real.log 9 := Real.log_pos (by norm_num)
  -- |offset| <= sigma^2 |L| / Delta
  have habs : |sigma ^ 2 / (c2 - c1) * L| = sigma ^ 2 * |L| / (c2 - c1) := by
    rw [abs_mul, abs_div, abs_of_nonneg hs2.le, abs_of_nonneg hDj.le, div_mul_eq_mul_div]
  have hoff : |sigma ^ 2 / (c2 - c1) * L| ≤ sigma ^ 2 * |L| / Delta := by
    rw [habs, div_le_div_iff₀ hDj hD]
    exact mul_le_mul_of_nonneg_left hgap (by positivity)
  have hoff_lo : -(sigma ^ 2 * |L| / Delta) ≤ sigma ^ 2 / (c2 - c1) * L :=
    neg_le_of_abs_le hoff
  have hoff_hi : sigma ^ 2 / (c2 - c1) * L ≤ sigma ^ 2 * |L| / Delta :=
    le_of_abs_le hoff
  -- sigma^2 (|L| + log 9)/Delta <= Delta/4
  have hbudget : sigma ^ 2 * |L| / Delta + sigma ^ 2 * Real.log 9 / Delta ≤ Delta / 4 := by
    rw [← add_div, div_le_iff₀ hD]
    calc sigma ^ 2 * |L| + sigma ^ 2 * Real.log 9 = sigma ^ 2 * (|L| + Real.log 9) := by
          ring
      _ ≤ Delta ^ 2 / 4 := hsmall
      _ = Delta / 4 * Delta := by ring
  have hmid1 : Delta / 2 ≤ (c1 + c2) / 2 - c1 := by linarith
  have hmid2 : Delta / 2 ≤ c2 - (c1 + c2) / 2 := by linarith
  constructor
  · linarith
  · linarith

/-- A3 main theorem, nonadjacent smallness with explicit constants
(spine: "all nonadjacent components are exponentially small" on the crossover
window; constants per FORMALIZATION_TARGETS.md A3): for a family of `m`
components with min-spacing `Delta > 0` and positive weights, indices
`j + 1 < m`, `i < m`, `i ∉ {j, j+1}`, and `x` in the crossover window
`[s_j - sigma^2 log 9/Delta, s_j + sigma^2 log 9/Delta]`, under the explicit
smallness condition `sigma^2 (|log(w_j/w_{j+1})| + log 9) <= Delta^2/4`:

  `q_i(x) / max(q_j(x), q_{j+1}(x))
     <= (w_i / min(w_j, w_{j+1})) * exp(-(3/4) Delta^2 / sigma^2)`,

i.e. `C exp(-Delta^2/(c sigma^2))` with `C = w_i/min(w_j, w_{j+1})`,
`c = 4/3`. -/
theorem nonadjacent_small {m : ℕ} {c w : ℕ → ℝ} {sigma Delta : ℝ} {j i : ℕ} {x : ℝ}
    (hs : sigma ≠ 0) (hD : 0 < Delta)
    (hspace : ∀ k, k + 1 < m → Delta ≤ c (k + 1) - c k)
    (hw : ∀ k, k < m → 0 < w k)
    (hj : j + 1 < m) (hi : i < m) (hij : i ≠ j) (hij1 : i ≠ j + 1)
    (hsmall : sigma ^ 2 * (|Real.log (w j / w (j + 1))| + Real.log 9) ≤ Delta ^ 2 / 4)
    (hx_lo : crossover (w j) (w (j + 1)) (c j) (c (j + 1)) sigma -
        sigma ^ 2 * Real.log 9 / Delta ≤ x)
    (hx_hi : x ≤ crossover (w j) (w (j + 1)) (c j) (c (j + 1)) sigma +
        sigma ^ 2 * Real.log 9 / Delta) :
    q (w i) (c i) sigma x /
        max (q (w j) (c j) sigma x) (q (w (j + 1)) (c (j + 1)) sigma x) ≤
      w i / min (w j) (w (j + 1)) *
        Real.exp (-(3 / 4) * Delta ^ 2 / sigma ^ 2) := by
  have hwj : 0 < w j := hw j (by omega)
  have hwj1 : 0 < w (j + 1) := hw (j + 1) hj
  have hwi : 0 < w i := hw i hi
  have hwmin : 0 < min (w j) (w (j + 1)) := lt_min hwj hwj1
  have hMpos : 0 < max (q (w j) (c j) sigma x) (q (w (j + 1)) (c (j + 1)) sigma x) :=
    lt_of_lt_of_le (q_pos hwj (c j) sigma x) (le_max_left _ _)
  -- window position inside the gap
  have hwin : Delta / 4 ≤ x - c j ∧ Delta / 4 ≤ c (j + 1) - x := by
    apply window_inside_gap hs hD (hspace j hj) hsmall
    · simpa [crossover] using hx_lo
    · simpa [crossover] using hx_hi
  have hcase : i < j ∨ j + 1 < i := by omega
  rcases hcase with hlt | hgt
  · -- third centre to the left: compare against component j
    have hA : Delta ≤ c j - c i := by
      have := spacing_mono hD hspace i j hlt (by omega)
      linarith
    exact ratio_bound_of_far hs hwi.le hwj hwmin (min_le_left _ _)
      (far_exponent_left hD hA hwin.1) (le_max_left _ _) hMpos
  · -- third centre to the right: compare against component j+1
    have hA : Delta ≤ c i - c (j + 1) := by
      have := spacing_mono hD hspace (j + 1) i hgt hi
      linarith
    exact ratio_bound_of_far hs hwi.le hwj1 hwmin (min_le_right _ _)
      (far_exponent_right hD hA hwin.2) (le_max_right _ _) hMpos

end
end CrossoverBounds
end FormalPRR

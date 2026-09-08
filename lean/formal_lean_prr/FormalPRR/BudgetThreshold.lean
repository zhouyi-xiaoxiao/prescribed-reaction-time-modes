/-
FormalPRR/BudgetThreshold.lean  (target A6: B0 assembly inversion)

Tex source (docstring anchors):
  prr_submission/prr_assets/b0_quantitative_bound.tex
  * Weighted-norm route: remark `rem:exactmfull-b0-baseline`,
    Eq. (eq:exactmfull-b0-baseline-jet)
        E_wt(B) = v_* kappa_pi (e^{(3/2) B v_inf T} - 1) ||q_0||_X
    and Eq. (eq:exactmfull-b0-baseline-threshold)
        B_cert^wt = (2/(3 v_inf T)) log(1 + M/(v_* kappa_pi ||q_0||_X)),
        M = min(tau mu_1/2, tau^2 mu_2/8).
  * Sharpened route: proposition `prop:exactmfull-b0`,
    Eq. (eq:exactmfull-b0-Ec)
        E_c(B) = kappa_hat v_inf e^Pi (e^{B v_eff (T+r_0)} - 1),
        v_eff = kappa_hat (1+delta_v) v_inf,
    Eq. (eq:exactmfull-b0-threshold)
        B_cert = log(1 + M_hat/(kappa_hat v_inf e^Pi)) / (v_eff (T+r_0)),
    and Step 3 of its proof: `E_c(B) < M_hat` if and only if `B < B_cert`;
    combined with Step 1's "E_c is a continuous, strictly increasing function
    of B with E_c(0) = 0".

This file proves the pure real-analytic inversion skeleton shared by both
final displays: the assembly map E(B) = a (e^{cB} - 1) is strictly increasing
with E(0) = 0, the closed-form threshold B0 = log(1 + M/a)/c satisfies
E(B0) = M, and E(B) < M holds if and only if B < B0.  The two concrete routes
are then instantiated with their exact constants from the tex displays.

The abstract constants here stand for (weighted route)
a = v_* kappa_pi ||q_0||_X > 0, c = (3/2) v_inf T > 0, and (sharpened route)
a = kappa_hat v_inf e^Pi, c = v_eff (T + r_0).  The margin M (resp. M_hat) is
an arbitrary positive real; the specific min(...) inventories of the tex are
carried as opaque positive inputs, which is exactly how the displays use them.
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace FormalPRR
namespace BudgetThreshold

noncomputable section
open Real

/-! ## Generic assembly map and its inversion -/

/-- Generic budget-assembly map `E(B) = a (e^{cB} - 1)`.
Both final displays of `b0_quantitative_bound.tex` have this form:
Eq. (eq:exactmfull-b0-baseline-jet) with `a = v_* kappa_pi ||q_0||_X`,
`c = (3/2) v_inf T`, and Eq. (eq:exactmfull-b0-Ec) with
`a = kappa_hat v_inf e^Pi`, `c = v_eff (T + r_0)`. -/
def E (a c B : ℝ) : ℝ := a * (Real.exp (c * B) - 1)

/-- Generic closed-form threshold `B0 = log(1 + M/a) / c`, the common shape of
Eq. (eq:exactmfull-b0-baseline-threshold) and Eq. (eq:exactmfull-b0-threshold). -/
def B0 (a c M : ℝ) : ℝ := Real.log (1 + M / a) / c

/-- `E(0) = 0` (proof of `prop:exactmfull-b0`, Step 1: "with E_c(0)=0"). -/
theorem E_zero (a c : ℝ) : E a c 0 = 0 := by
  simp [E]

/-- `E` is strictly increasing in the budget (the strictly-increasing half of
`prop:exactmfull-b0` Step 1, "E_c is a continuous, strictly increasing
function of B"; the continuity half is `continuous_E`). -/
theorem strictMono_E {a c : ℝ} (ha : 0 < a) (hc : 0 < c) :
    StrictMono (E a c) := by
  intro B₁ B₂ h
  have hexp : Real.exp (c * B₁) < Real.exp (c * B₂) :=
    Real.exp_lt_exp.mpr (by nlinarith)
  have : Real.exp (c * B₁) - 1 < Real.exp (c * B₂) - 1 := by linarith
  exact mul_lt_mul_of_pos_left this ha

/-- `E` is continuous in the budget (the continuity half of
`prop:exactmfull-b0` Step 1, "E_c is a continuous, strictly increasing
function of B"; composition of `Real.exp` with affine maps).  No sign
hypotheses are needed. -/
theorem continuous_E (a c : ℝ) : Continuous (E a c) := by
  unfold E
  exact continuous_const.mul
    (((continuous_const.mul continuous_id).rexp).sub continuous_const)

/-- The threshold is positive: `0 < B0` for positive data. -/
theorem B0_pos {a c M : ℝ} (ha : 0 < a) (hc : 0 < c) (hM : 0 < M) :
    0 < B0 a c M := by
  have h1 : (1 : ℝ) < 1 + M / a := by
    have := div_pos hM ha
    linarith
  exact div_pos (Real.log_pos h1) hc

/-- Exact inversion: `E(B0) = M`
(`prop:exactmfull-b0` Step 3: the threshold saturates the margin). -/
theorem E_B0 {a c M : ℝ} (ha : 0 < a) (hc : 0 < c) (hM : 0 < M) :
    E a c (B0 a c M) = M := by
  have hpos : (0 : ℝ) < 1 + M / a := by
    have := div_pos hM ha
    linarith
  have hexp : Real.exp (c * (Real.log (1 + M / a) / c)) = 1 + M / a := by
    rw [mul_div_cancel₀ _ (ne_of_gt hc)]
    exact Real.exp_log hpos
  unfold E B0
  rw [hexp]
  field_simp
  ring

/-- Two-sided inversion (`prop:exactmfull-b0`, Step 3, verbatim:
"`E_c(B) < M_hat` holds if and only if ... `B < B_cert`").  Stated for every
real `B`; the paper's restriction `0 < B` is not needed. -/
theorem E_lt_iff {a c M : ℝ} (ha : 0 < a) (hc : 0 < c) (hM : 0 < M) (B : ℝ) :
    E a c B < M ↔ B < B0 a c M := by
  conv_lhs => rw [← E_B0 ha hc hM]
  exact (strictMono_E ha hc).lt_iff_lt

/-- Sub-threshold budgets stay strictly below the margin:
`0 ≤ B < B0 → E(B) < M` (the direction used by Step 2 of the proof of
`prop:exactmfull-b0`; the hypothesis `0 ≤ B` is recorded to mirror the
paper's budget range but is not needed by the proof). -/
theorem E_lt_of_lt {a c M : ℝ} (ha : 0 < a) (hc : 0 < c) (hM : 0 < M)
    {B : ℝ} (_hB0 : 0 ≤ B) (hB : B < B0 a c M) : E a c B < M :=
  (E_lt_iff ha hc hM B).mpr hB

/-! ## Weighted-norm route (remark `rem:exactmfull-b0-baseline`)

Constants: `vStar = v_*`, `kappa = kappa_pi`, `Q = ||q_0||_X`, `vInf = v_inf`,
`T` the window endpoint, `M = min(tau mu_1/2, tau^2 mu_2/8)` (opaque here). -/

/-- Weighted-route assembly map, Eq. (eq:exactmfull-b0-baseline-jet):
`E_wt(B) = v_* kappa_pi ||q_0||_X (e^{(3/2) B v_inf T} - 1)`. -/
def Ewt (vStar kappa Q vInf T B : ℝ) : ℝ :=
  vStar * kappa * Q * (Real.exp ((3 / 2) * B * vInf * T) - 1)

/-- Weighted-route threshold, Eq. (eq:exactmfull-b0-baseline-threshold):
`B_cert^wt = (2/(3 v_inf T)) log(1 + M/(v_* kappa_pi ||q_0||_X))`. -/
def B0wt (vStar kappa Q vInf T M : ℝ) : ℝ :=
  (2 / (3 * vInf * T)) * Real.log (1 + M / (vStar * kappa * Q))

/-- The weighted-route map is the generic map at
`a = v_* kappa_pi ||q_0||_X`, `c = (3/2) v_inf T`. -/
theorem Ewt_eq_E (vStar kappa Q vInf T B : ℝ) :
    Ewt vStar kappa Q vInf T B = E (vStar * kappa * Q) ((3 / 2) * vInf * T) B := by
  unfold Ewt E
  ring_nf

/-- The weighted-route threshold is the generic threshold at the same data. -/
theorem B0wt_eq_B0 {vInf T : ℝ} (hv : 0 < vInf) (hT : 0 < T)
    (vStar kappa Q M : ℝ) :
    B0wt vStar kappa Q vInf T M = B0 (vStar * kappa * Q) ((3 / 2) * vInf * T) M := by
  unfold B0wt B0
  field_simp

/-- `E_wt(0) = 0`. -/
theorem Ewt_zero (vStar kappa Q vInf T : ℝ) : Ewt vStar kappa Q vInf T 0 = 0 := by
  simp [Ewt]

/-- `E_wt` is continuous in `B` (remark `rem:exactmfull-b0-baseline`: Step 1
of the proof of `prop:exactmfull-b0` applies verbatim to `E_wt`;
instantiation of `continuous_E`). -/
theorem continuous_Ewt (vStar kappa Q vInf T : ℝ) :
    Continuous (Ewt vStar kappa Q vInf T) := by
  have h : Ewt vStar kappa Q vInf T =
      E (vStar * kappa * Q) (3 / 2 * vInf * T) :=
    funext fun B => Ewt_eq_E vStar kappa Q vInf T B
  rw [h]
  exact continuous_E _ _

/-- `E_wt` is strictly increasing in `B` for positive constants. -/
theorem strictMono_Ewt {vStar kappa Q vInf T : ℝ}
    (h1 : 0 < vStar) (h2 : 0 < kappa) (h3 : 0 < Q) (h4 : 0 < vInf) (h5 : 0 < T) :
    StrictMono (Ewt vStar kappa Q vInf T) := by
  have := strictMono_E (a := vStar * kappa * Q) (c := (3 / 2) * vInf * T)
    (by positivity) (by positivity)
  intro B₁ B₂ h
  rw [Ewt_eq_E, Ewt_eq_E]
  exact this h

/-- The weighted-route threshold saturates the margin: `E_wt(B0wt) = M`. -/
theorem Ewt_B0wt {vStar kappa Q vInf T M : ℝ}
    (h1 : 0 < vStar) (h2 : 0 < kappa) (h3 : 0 < Q) (h4 : 0 < vInf) (h5 : 0 < T)
    (hM : 0 < M) :
    Ewt vStar kappa Q vInf T (B0wt vStar kappa Q vInf T M) = M := by
  rw [Ewt_eq_E, B0wt_eq_B0 h4 h5]
  exact E_B0 (by positivity) (by positivity) hM

/-- Weighted-route inversion: `E_wt(B) < M ↔ B < B_cert^wt`
(remark `rem:exactmfull-b0-baseline`: "Steps 2 and 3 of the proof of
proposition `prop:exactmfull-b0` apply verbatim with `(E_c, r_0)` replaced by
`(E_wt, tau/2)`"). -/
theorem Ewt_lt_iff {vStar kappa Q vInf T M : ℝ}
    (h1 : 0 < vStar) (h2 : 0 < kappa) (h3 : 0 < Q) (h4 : 0 < vInf) (h5 : 0 < T)
    (hM : 0 < M) (B : ℝ) :
    Ewt vStar kappa Q vInf T B < M ↔ B < B0wt vStar kappa Q vInf T M := by
  rw [Ewt_eq_E, B0wt_eq_B0 h4 h5]
  exact E_lt_iff (by positivity) (by positivity) hM B

/-- Sub-threshold budgets on the weighted route: `0 ≤ B < B0wt → E_wt(B) < M`. -/
theorem Ewt_lt_of_lt {vStar kappa Q vInf T M : ℝ}
    (h1 : 0 < vStar) (h2 : 0 < kappa) (h3 : 0 < Q) (h4 : 0 < vInf) (h5 : 0 < T)
    (hM : 0 < M) {B : ℝ} (_hB0 : 0 ≤ B) (hB : B < B0wt vStar kappa Q vInf T M) :
    Ewt vStar kappa Q vInf T B < M :=
  (Ewt_lt_iff h1 h2 h3 h4 h5 hM B).mpr hB

/-! ## Sharpened route (proposition `prop:exactmfull-b0`)

Constants: `kappaHat = kappa_hat` (Eq. eq:exactmfull-b0-kappahat),
`vInf = v_inf`, `Pi` the bookkeeping exponent (Eq. eq:exactmfull-b0-Pi),
`delta = delta_v ≥ 0` (Eq. eq:exactmfull-b0-deltav), `T + r0 > 0` the Cauchy
horizon, `M = M_hat = min(r_0 mu_1, r_0^2 mu_2 / 2)` (opaque here). -/

/-- Effective rate `v_eff = kappa_hat (1 + delta_v) v_inf`
(Eq. eq:exactmfull-b0-remainderbound). -/
def veff (kappaHat delta vInf : ℝ) : ℝ := kappaHat * (1 + delta) * vInf

/-- Sharpened assembly map, Eq. (eq:exactmfull-b0-Ec):
`E_c(B) = kappa_hat v_inf e^Pi (e^{B v_eff (T + r_0)} - 1)`. -/
def Ec (kappaHat vInf Pi delta T r0 B : ℝ) : ℝ :=
  kappaHat * vInf * Real.exp Pi *
    (Real.exp (B * veff kappaHat delta vInf * (T + r0)) - 1)

/-- Sharpened threshold, Eq. (eq:exactmfull-b0-threshold):
`B_cert = log(1 + M_hat/(kappa_hat v_inf e^Pi)) / (kappa_hat (1+delta_v) v_inf (T+r_0))`. -/
def Bcert (kappaHat vInf Pi delta T r0 M : ℝ) : ℝ :=
  Real.log (1 + M / (kappaHat * vInf * Real.exp Pi)) /
    (veff kappaHat delta vInf * (T + r0))

/-- The sharpened map is the generic map at `a = kappa_hat v_inf e^Pi`,
`c = v_eff (T + r_0)`. -/
theorem Ec_eq_E (kappaHat vInf Pi delta T r0 B : ℝ) :
    Ec kappaHat vInf Pi delta T r0 B =
      E (kappaHat * vInf * Real.exp Pi) (veff kappaHat delta vInf * (T + r0)) B := by
  unfold Ec E
  ring_nf

/-- The sharpened threshold is the generic threshold at the same data. -/
theorem Bcert_eq_B0 (kappaHat vInf Pi delta T r0 M : ℝ) :
    Bcert kappaHat vInf Pi delta T r0 M =
      B0 (kappaHat * vInf * Real.exp Pi) (veff kappaHat delta vInf * (T + r0)) M :=
  rfl

/-- `E_c(0) = 0` (proof of `prop:exactmfull-b0`, Step 1). -/
theorem Ec_zero (kappaHat vInf Pi delta T r0 : ℝ) :
    Ec kappaHat vInf Pi delta T r0 0 = 0 := by
  simp [Ec]

/-- Positivity of the effective rate for the stated data (`delta_v ≥ 0` by
Eq. eq:exactmfull-b0-deltav). -/
theorem veff_pos {kappaHat delta vInf : ℝ}
    (hk : 0 < kappaHat) (hd : 0 ≤ delta) (hv : 0 < vInf) :
    0 < veff kappaHat delta vInf := by
  unfold veff
  positivity

/-- `E_c` is continuous in `B` (the continuity half of
`prop:exactmfull-b0` Step 1, "E_c is a continuous, strictly increasing
function of B with E_c(0) = 0"; instantiation of `continuous_E`).  No sign
hypotheses are needed. -/
theorem continuous_Ec (kappaHat vInf Pi delta T r0 : ℝ) :
    Continuous (Ec kappaHat vInf Pi delta T r0) := by
  have h : Ec kappaHat vInf Pi delta T r0 =
      E (kappaHat * vInf * Real.exp Pi) (veff kappaHat delta vInf * (T + r0)) :=
    funext fun B => Ec_eq_E kappaHat vInf Pi delta T r0 B
  rw [h]
  exact continuous_E _ _

/-- `E_c` is strictly increasing in `B` (the strictly-increasing half of
`prop:exactmfull-b0` Step 1, "E_c is a continuous, strictly increasing
function of B with E_c(0) = 0"; the continuity half is `continuous_Ec` and
the value at zero is `Ec_zero`).  Hypotheses: `kappa_hat > 0`, `v_inf > 0`,
`delta_v ≥ 0`, `T + r_0 > 0`; the exponent `Pi` is an arbitrary real. -/
theorem strictMono_Ec {kappaHat vInf delta T r0 : ℝ} (Pi : ℝ)
    (hk : 0 < kappaHat) (hv : 0 < vInf) (hd : 0 ≤ delta) (hTr : 0 < T + r0) :
    StrictMono (Ec kappaHat vInf Pi delta T r0) := by
  have := strictMono_E (a := kappaHat * vInf * Real.exp Pi)
    (c := veff kappaHat delta vInf * (T + r0))
    (by positivity) (mul_pos (veff_pos hk hd hv) hTr)
  intro B₁ B₂ h
  rw [Ec_eq_E, Ec_eq_E]
  exact this h

/-- The sharpened threshold saturates the margin: `E_c(B_cert) = M_hat`. -/
theorem Ec_Bcert {kappaHat vInf delta T r0 M : ℝ} (Pi : ℝ)
    (hk : 0 < kappaHat) (hv : 0 < vInf) (hd : 0 ≤ delta) (hTr : 0 < T + r0)
    (hM : 0 < M) :
    Ec kappaHat vInf Pi delta T r0 (Bcert kappaHat vInf Pi delta T r0 M) = M := by
  rw [Ec_eq_E, Bcert_eq_B0]
  exact E_B0 (by positivity) (mul_pos (veff_pos hk hd hv) hTr) hM

/-- Sharpened inversion, `prop:exactmfull-b0` Step 3 verbatim:
"`E_c(B) < M_hat` holds if and only if
`e^{B v_eff (T+r_0)} - 1 < M_hat/(kappa_hat v_inf e^Pi)`, that is, if and only
if `B < B_cert`". -/
theorem Ec_lt_iff {kappaHat vInf delta T r0 M : ℝ} (Pi : ℝ)
    (hk : 0 < kappaHat) (hv : 0 < vInf) (hd : 0 ≤ delta) (hTr : 0 < T + r0)
    (hM : 0 < M) (B : ℝ) :
    Ec kappaHat vInf Pi delta T r0 B < M ↔ B < Bcert kappaHat vInf Pi delta T r0 M := by
  rw [Ec_eq_E, Bcert_eq_B0]
  exact E_lt_iff (by positivity) (mul_pos (veff_pos hk hd hv) hTr) hM B

/-- Sub-threshold budgets on the sharpened route
(`prop:exactmfull-b0` Step 3: "By strict monotonicity of `E_c`,
Eq. (eq:exactmfull-b0-conditions) holds for every `0 < B < B_cert`"). -/
theorem Ec_lt_of_lt {kappaHat vInf delta T r0 M : ℝ} (Pi : ℝ)
    (hk : 0 < kappaHat) (hv : 0 < vInf) (hd : 0 ≤ delta) (hTr : 0 < T + r0)
    (hM : 0 < M) {B : ℝ} (_hB0 : 0 ≤ B)
    (hB : B < Bcert kappaHat vInf Pi delta T r0 M) :
    Ec kappaHat vInf Pi delta T r0 B < M :=
  (Ec_lt_iff Pi hk hv hd hTr hM B).mpr hB

end
end BudgetThreshold
end FormalPRR

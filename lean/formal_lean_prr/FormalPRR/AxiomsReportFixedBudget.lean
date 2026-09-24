/-
FormalPRR/AxiomsReportFixedBudget.lean — axiom audit of the modules added on
2026-09-23 for the fixed-budget design law (gap-closure items L2, L3).
Every `#print axioms` below must list at most propext, Classical.choice, Quot.sound.
-/
import FormalPRR.MeanFieldLevelSet
import FormalPRR.StickBreaking
import FormalPRR.SurvivalSandwich
import FormalPRR.SignatureStability
import FormalPRR.PassageProfile

/-! ## MeanFieldLevelSet -/

#print axioms FormalPRR.MeanField.hasDerivAt_f1
#print axioms FormalPRR.MeanField.prefactor_pos
#print axioms FormalPRR.MeanField.deriv_pos_iff
#print axioms FormalPRR.MeanField.deriv_neg_iff
#print axioms FormalPRR.MeanField.deriv_eq_zero_iff
#print axioms FormalPRR.MeanField.sign_deriv_f1
#print axioms FormalPRR.MeanField.deriv_f1_pos_iff_levelset
#print axioms FormalPRR.MeanField.rising_mono_budget
#print axioms FormalPRR.MeanField.f1_antitoneOn
#print axioms FormalPRR.MeanField.f1_strictMonoOn
#print axioms FormalPRR.MeanField.f1_rises_near

/-! ## StickBreaking -/

#print axioms FormalPRR.StickBreaking.cum_zero
#print axioms FormalPRR.StickBreaking.cum_succ
#print axioms FormalPRR.StickBreaking.mass_eq_sub
#print axioms FormalPRR.StickBreaking.mass_sum
#print axioms FormalPRR.StickBreaking.mass_sum_lt_one
#print axioms FormalPRR.StickBreaking.mass_nonneg
#print axioms FormalPRR.StickBreaking.mass_pos
#print axioms FormalPRR.StickBreaking.mass_congr
#print axioms FormalPRR.StickBreaking.surv_zero
#print axioms FormalPRR.StickBreaking.surv_succ
#print axioms FormalPRR.StickBreaking.surv_le_of_le
#print axioms FormalPRR.StickBreaking.surv_pos
#print axioms FormalPRR.StickBreaking.cum_lamOf
#print axioms FormalPRR.StickBreaking.exp_neg_cum_lamOf
#print axioms FormalPRR.StickBreaking.mass_lamOf
#print axioms FormalPRR.StickBreaking.surv_mass
#print axioms FormalPRR.StickBreaking.lamOf_mass
#print axioms FormalPRR.StickBreaking.lamOf_nonneg
#print axioms FormalPRR.StickBreaking.lamOf_pos
#print axioms FormalPRR.StickBreaking.budget_of_alloc
#print axioms FormalPRR.StickBreaking.alloc_of_lam
#print axioms FormalPRR.StickBreaking.surv_le_surv
#print axioms FormalPRR.StickBreaking.lamOf_le_of_le
#print axioms FormalPRR.StickBreaking.lamOf_lt_of_lt
#print axioms FormalPRR.StickBreaking.cost_le_of_le
#print axioms FormalPRR.StickBreaking.cost_lt_of_lt
#print axioms FormalPRR.StickBreaking.surv_const
#print axioms FormalPRR.StickBreaking.cost_const
#print axioms FormalPRR.StickBreaking.maxmin_le
#print axioms FormalPRR.StickBreaking.maxmin_eq
#print axioms FormalPRR.StickBreaking.maxmin_alloc
#print axioms FormalPRR.StickBreaking.maxmin_alloc_eq
#print axioms FormalPRR.StickBreaking.cost_const_lt_iff
#print axioms FormalPRR.StickBreaking.pstar_unique
#print axioms FormalPRR.StickBreaking.pstar_exists

/-! ## SurvivalSandwich -/

#print axioms FormalPRR.Sandwich.exp_neg_sub_le_sq
#print axioms FormalPRR.Sandwich.exp_sub_one_sub_le
#print axioms FormalPRR.Sandwich.key_quadratic
#print axioms FormalPRR.Sandwich.tangent_lower
#print axioms FormalPRR.Sandwich.quad_upper
#print axioms FormalPRR.Sandwich.integrable_exp_neg_mul
#print axioms FormalPRR.Sandwich.integral_tangent
#print axioms FormalPRR.Sandwich.integrable_tangent
#print axioms FormalPRR.Sandwich.sandwich_lower
#print axioms FormalPRR.Sandwich.sandwich_upper
#print axioms FormalPRR.Sandwich.sandwich_variance
#print axioms FormalPRR.Sandwich.event_mass_floor

/-! ## SignatureStability -/

#print axioms FormalPRR.Signature.sign_eq_of_abs_sub_lt
#print axioms FormalPRR.Signature.ne_zero_of_abs_sub_lt
#print axioms FormalPRR.Signature.mul_neg_of_sign_eq
#print axioms FormalPRR.Signature.tube_unique_zero
#print axioms FormalPRR.Signature.endpoint_not_mem
#print axioms FormalPRR.Signature.signature_stable
#print axioms FormalPRR.Signature.signature_count
#print axioms FormalPRR.Signature.signature_of_margins
#print axioms FormalPRR.Signature.signature_of_budget

/-! ## PassageProfile -/

#print axioms FormalPRR.PassageProfile.sqrt_two_pi_pos
#print axioms FormalPRR.PassageProfile.phi_pos
#print axioms FormalPRR.PassageProfile.phi_le
#print axioms FormalPRR.PassageProfile.hasDerivAt_phi
#print axioms FormalPRR.PassageProfile.hasDerivAt_hfun
#print axioms FormalPRR.PassageProfile.hfun_one_sign_change
#print axioms FormalPRR.PassageProfile.pbeta_pos
#print axioms FormalPRR.PassageProfile.hasDerivAt_pbeta
#print axioms FormalPRR.PassageProfile.curvature_integrand_neg
#print axioms FormalPRR.PassageProfile.curvature_identity
#print axioms FormalPRR.PassageProfile.integral_neg_of_neg_off_point
#print axioms FormalPRR.PassageProfile.A8_curvature_neg
#print axioms FormalPRR.PassageProfile.zero_unique_of_downcrossing

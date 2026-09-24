import FormalPRR
/-! Axiom audit of the fixed-budget paper package, 2026-09-23 (gap-closure items L1-L3).
Part A: the 101 inherited paper declarations of audit_all_138_20260903.lean (the 37
companion declarations moved to the non-default library FormalPRRCompanion are dropped).
Part B: the 80 theorem/lemma declarations of the five modules added 2026-09-23.
Part C: a meta-level sweep over EVERY theorem declared in a FormalPRR.* module
(including private and auxiliary ones), checking that its axioms are among
propext, Classical.choice, Quot.sound; it prints one summary line per module and
an overall PASS/FAIL line. -/

-- Part A (101 inherited)
#print axioms FormalPRR.B0ChainKernel.OmegaZ_mul_h
#print axioms FormalPRR.B0ChainKernel.budget_sum
#print axioms FormalPRR.B0ChainKernel.complete_square_identity
#print axioms FormalPRR.B0ChainKernel.div_one_sub_le
#print axioms FormalPRR.B0ChainKernel.far_from_one_center
#print axioms FormalPRR.B0ChainKernel.gaussN_le_max
#print axioms FormalPRR.B0ChainKernel.gaussN_le_of_far
#print axioms FormalPRR.B0ChainKernel.gaussN_nonneg
#print axioms FormalPRR.B0ChainKernel.initial_law_integral
#print axioms FormalPRR.B0ChainKernel.initial_law_integral_parallel
#print axioms FormalPRR.B0ChainKernel.integral_gaussN_eq_one
#print axioms FormalPRR.B0ChainKernel.inv_sqrt_one_sub_le
#print axioms FormalPRR.B0ChainKernel.kappa_block_pow
#print axioms FormalPRR.B0ChainKernel.kappa_block_prod_pow
#print axioms FormalPRR.B0ChainKernel.kappa_hat_display_bound
#print axioms FormalPRR.B0ChainKernel.kappa_hat_pow_four
#print axioms FormalPRR.B0ChainKernel.log_inv_one_sub_le
#print axioms FormalPRR.B0ChainKernel.margins_iff
#print axioms FormalPRR.B0ChainKernel.mean_multiplier_im_sq_le
#print axioms FormalPRR.B0ChainKernel.one_sub_exp_neg_cos_ge
#print axioms FormalPRR.B0ChainKernel.one_sub_exp_neg_ge
#print axioms FormalPRR.B0ChainKernel.penalty_O1_of_R1
#print axioms FormalPRR.B0ChainKernel.penalty_exponent_cancel
#print axioms FormalPRR.B0ChainKernel.penalty_exponent_le
#print axioms FormalPRR.B0ChainKernel.prod_inv_sqrt_le
#print axioms FormalPRR.B0ChainKernel.radius_penalty_closed_form
#print axioms FormalPRR.B0ChainKernel.radius_penalty_uniform_O1
#print axioms FormalPRR.B0ChainKernel.re_vZ_lower
#print axioms FormalPRR.B0ChainKernel.sec_le_sqrt_one_add_tan_sq
#print axioms FormalPRR.B0ChainKernel.sec_sq_le_one_add_tan_sq
#print axioms FormalPRR.B0ChainKernel.sqrt_sqrt_pow_four
#print axioms FormalPRR.B0ChainKernel.two_slab_sup_bound
#print axioms FormalPRR.B0ChainKernel.young_cross_term
#print axioms FormalPRR.BZero.Eexp_strictMono
#print axioms FormalPRR.BZero.explicit_is_threshold
#print axioms FormalPRR.BZero.explicit_log_form
#print axioms FormalPRR.BZero.margin_min
#print axioms FormalPRR.BZero.threshold_wd
#print axioms FormalPRR.BudgetThreshold.B0_pos
#print axioms FormalPRR.BudgetThreshold.B0wt_eq_B0
#print axioms FormalPRR.BudgetThreshold.Bcert_eq_B0
#print axioms FormalPRR.BudgetThreshold.E_B0
#print axioms FormalPRR.BudgetThreshold.E_lt_iff
#print axioms FormalPRR.BudgetThreshold.E_lt_of_lt
#print axioms FormalPRR.BudgetThreshold.E_zero
#print axioms FormalPRR.BudgetThreshold.Ec_Bcert
#print axioms FormalPRR.BudgetThreshold.Ec_eq_E
#print axioms FormalPRR.BudgetThreshold.Ec_lt_iff
#print axioms FormalPRR.BudgetThreshold.Ec_lt_of_lt
#print axioms FormalPRR.BudgetThreshold.Ec_zero
#print axioms FormalPRR.BudgetThreshold.Ewt_B0wt
#print axioms FormalPRR.BudgetThreshold.Ewt_eq_E
#print axioms FormalPRR.BudgetThreshold.Ewt_lt_iff
#print axioms FormalPRR.BudgetThreshold.Ewt_lt_of_lt
#print axioms FormalPRR.BudgetThreshold.Ewt_zero
#print axioms FormalPRR.BudgetThreshold.continuous_E
#print axioms FormalPRR.BudgetThreshold.continuous_Ec
#print axioms FormalPRR.BudgetThreshold.continuous_Ewt
#print axioms FormalPRR.BudgetThreshold.strictMono_E
#print axioms FormalPRR.BudgetThreshold.strictMono_Ec
#print axioms FormalPRR.BudgetThreshold.strictMono_Ewt
#print axioms FormalPRR.BudgetThreshold.veff_pos
#print axioms FormalPRR.CrossoverBounds.adjacent_odds
#print axioms FormalPRR.CrossoverBounds.crossover_ratio_eq_one
#print axioms FormalPRR.CrossoverBounds.crossover_ratio_minus
#print axioms FormalPRR.CrossoverBounds.crossover_ratio_one
#print axioms FormalPRR.CrossoverBounds.crossover_ratio_plus
#print axioms FormalPRR.CrossoverBounds.far_exponent_left
#print axioms FormalPRR.CrossoverBounds.far_exponent_right
#print axioms FormalPRR.CrossoverBounds.nonadjacent_small
#print axioms FormalPRR.CrossoverBounds.q_far_le
#print axioms FormalPRR.CrossoverBounds.q_nonneg
#print axioms FormalPRR.CrossoverBounds.q_pos
#print axioms FormalPRR.CrossoverBounds.ratio_bound_of_far
#print axioms FormalPRR.CrossoverBounds.spacing_mono
#print axioms FormalPRR.CrossoverBounds.window_inside_gap
#print axioms FormalPRR.ExpPoly.expPoly_distinct_zeros_card_le
#print axioms FormalPRR.ExpPoly.expPoly_zeroSet_finite
#print axioms FormalPRR.ExpPoly.expPoly_zeros_with_multiplicity_le
#print axioms FormalPRR.Mixture.deriv2_log_H
#print axioms FormalPRR.Mixture.deriv_log_H
#print axioms FormalPRR.MixtureIndep.H_pos
#print axioms FormalPRR.MixtureIndep.cbar_eq_pi
#print axioms FormalPRR.MixtureIndep.crossover_point
#print axioms FormalPRR.MixtureIndep.logDeriv_H
#print axioms FormalPRR.MixtureIndep.second_logDeriv_H
#print axioms FormalPRR.MixtureIndep.sum_pi
#print axioms FormalPRR.MixtureIndep.var_eq_pi
#print axioms FormalPRR.Window.mixture_deriv_zeroSet_finite
#print axioms FormalPRR.Window.mixture_deriv_zeros_card_le
#print axioms affine_times_exp_ncard_le
#print axioms affine_times_exp_zeros
#print axioms atMost_succ_of_deriv
#print axioms expPoly_atMostZeros
#print axioms expPoly_hasDerivAt
#print axioms expPoly_one_zeros_le_one
#print axioms expPoly_three_zeros_le_five
#print axioms expPoly_two_zeros_le_three
#print axioms expPoly_zeros_finite
#print axioms expPoly_zeros_ncard_le
#print axioms rolle_interleave

-- Part B (80 new)
-- MeanFieldLevelSet
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
-- StickBreaking
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
-- SurvivalSandwich
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
-- SignatureStability
#print axioms FormalPRR.Signature.sign_eq_of_abs_sub_lt
#print axioms FormalPRR.Signature.ne_zero_of_abs_sub_lt
#print axioms FormalPRR.Signature.mul_neg_of_sign_eq
#print axioms FormalPRR.Signature.tube_unique_zero
#print axioms FormalPRR.Signature.endpoint_not_mem
#print axioms FormalPRR.Signature.signature_stable
#print axioms FormalPRR.Signature.signature_count
#print axioms FormalPRR.Signature.signature_of_margins
#print axioms FormalPRR.Signature.signature_of_budget
-- PassageProfile
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

-- Part C (sweep)
open Lean Elab Command in
elab "#audit_formalprr_sweep" : command => do
  let env ← getEnv
  let std : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut perMod : Std.HashMap Name (Nat × Nat × Nat × Nat) := {}
  let mut badList : Array String := #[]
  let mut nThm := 0
  let mut nThmUser := 0
  let mut nOther := 0
  for (n, ci) in env.constants.map₁.toList do
    match env.getModuleIdxFor? n with
    | none => pure ()
    | some idx =>
      let modName := env.header.moduleNames[idx.toNat]!
      if (`FormalPRR).isPrefixOf modName && modName != `FormalPRR then
        let axs ← collectAxioms n
        let ok := axs.all (fun a => std.contains a)
        let isThm := match ci with | .thmInfo _ => true | _ => false
        let user := !n.isInternal || isPrivateName n
        let (t, u, o, b) := perMod.getD modName (0, 0, 0, 0)
        perMod := perMod.insert modName
          (t + (if isThm then 1 else 0), u + (if isThm && user then 1 else 0),
           o + (if isThm then 0 else 1), b + (if ok then 0 else 1))
        if isThm then
          nThm := nThm + 1
          if user then nThmUser := nThmUser + 1
        else
          nOther := nOther + 1
        if !ok then badList := badList.push s!"{n} : {axs}"
  let mods := perMod.toList.toArray.qsort (fun a b => a.1.toString < b.1.toString)
  for (m, (t, u, o, b)) in mods do
    logInfo m!"SWEEP module {m}: theorems {t} (non-auxiliary {u}), other constants {o}, non-standard footprints {b}"
  if badList.isEmpty then
    logInfo m!"SWEEP PASS: {nThm} theorems ({nThmUser} non-auxiliary) and {nOther} other constants in FormalPRR.* modules; every axiom footprint is a subset of [propext, Classical.choice, Quot.sound]; sorryAx occurrences 0"
  else
    logInfo m!"SWEEP FAIL: {badList}"

#audit_formalprr_sweep

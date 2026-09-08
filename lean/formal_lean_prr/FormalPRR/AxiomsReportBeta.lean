/-
FormalPRR/AxiomsReportBeta.lean

Per-theorem axiom audit for the Beta half of the package:
A3 (CrossoverBounds), A4 (SeedConditioning, companion kernel),
A5 (B0ChainKernel; NewtonKernel companion kernel), A6 (BudgetThreshold).

Every `#print axioms` below must list at most
`propext`, `Classical.choice`, `Quot.sound`.
The captured output lives in axioms_report_beta.txt at the project root.
-/
import FormalPRR.BudgetThreshold
import FormalPRR.SeedConditioning
import FormalPRR.NewtonKernel
import FormalPRR.B0ChainKernel
import FormalPRR.CrossoverBounds

/-! ## A6 BudgetThreshold -/

#print axioms FormalPRR.BudgetThreshold.E_zero
#print axioms FormalPRR.BudgetThreshold.strictMono_E
#print axioms FormalPRR.BudgetThreshold.continuous_E
#print axioms FormalPRR.BudgetThreshold.continuous_Ewt
#print axioms FormalPRR.BudgetThreshold.continuous_Ec
#print axioms FormalPRR.BudgetThreshold.B0_pos
#print axioms FormalPRR.BudgetThreshold.E_B0
#print axioms FormalPRR.BudgetThreshold.E_lt_iff
#print axioms FormalPRR.BudgetThreshold.E_lt_of_lt
#print axioms FormalPRR.BudgetThreshold.Ewt_eq_E
#print axioms FormalPRR.BudgetThreshold.B0wt_eq_B0
#print axioms FormalPRR.BudgetThreshold.Ewt_zero
#print axioms FormalPRR.BudgetThreshold.strictMono_Ewt
#print axioms FormalPRR.BudgetThreshold.Ewt_B0wt
#print axioms FormalPRR.BudgetThreshold.Ewt_lt_iff
#print axioms FormalPRR.BudgetThreshold.Ewt_lt_of_lt
#print axioms FormalPRR.BudgetThreshold.veff_pos
#print axioms FormalPRR.BudgetThreshold.Ec_eq_E
#print axioms FormalPRR.BudgetThreshold.Bcert_eq_B0
#print axioms FormalPRR.BudgetThreshold.Ec_zero
#print axioms FormalPRR.BudgetThreshold.strictMono_Ec
#print axioms FormalPRR.BudgetThreshold.Ec_Bcert
#print axioms FormalPRR.BudgetThreshold.Ec_lt_iff
#print axioms FormalPRR.BudgetThreshold.Ec_lt_of_lt

/-! ## A4 SeedConditioning (companion kernel) -/

#print axioms FormalPRR.SeedConditioning.frobSq_nonneg
#print axioms FormalPRR.SeedConditioning.discr_eq
#print axioms FormalPRR.SeedConditioning.discr_nonneg
#print axioms FormalPRR.SeedConditioning.sqrt_discr_le_frobSq
#print axioms FormalPRR.SeedConditioning.sigmaMinSq_nonneg
#print axioms FormalPRR.SeedConditioning.sigmaMaxSq_nonneg
#print axioms FormalPRR.SeedConditioning.sigmaMin_nonneg
#print axioms FormalPRR.SeedConditioning.sigmaMax_nonneg
#print axioms FormalPRR.SeedConditioning.sigmaMin_sq
#print axioms FormalPRR.SeedConditioning.sigmaMax_sq
#print axioms FormalPRR.SeedConditioning.sq_add_sq
#print axioms FormalPRR.SeedConditioning.sigmaMinSq_mul_sigmaMaxSq
#print axioms FormalPRR.SeedConditioning.sigmaMin_mul_sigmaMax
#print axioms FormalPRR.SeedConditioning.sigmaMin_le_sigmaMax
#print axioms FormalPRR.SeedConditioning.sigmaMin_eq_abs_det_div
#print axioms FormalPRR.SeedConditioning.sigmaMax_le_frobNorm
#print axioms FormalPRR.SeedConditioning.singularValues_unique
#print axioms FormalPRR.SeedConditioning.div_sqrt_mono
#print axioms FormalPRR.SeedConditioning.sigmaMin_lower_bound

/-! ## A5 companion kernel: NewtonKernel -/

#print axioms FormalPRR.NewtonKernel.neumann_inverse_bound
#print axioms FormalPRR.NewtonKernel.inverse_times_quarter
#print axioms FormalPRR.NewtonKernel.contraction_factor_le
#print axioms FormalPRR.NewtonKernel.alpha_div_arith
#print axioms FormalPRR.NewtonKernel.contraction_maps_closedBall
#print axioms FormalPRR.NewtonKernel.newton_kernel_fixed_point
#print axioms FormalPRR.NewtonKernel.newton_kernel_root_bound

/-! ## A5 present-paper deliverable: B0ChainKernel -/

#print axioms FormalPRR.B0ChainKernel.young_cross_term
#print axioms FormalPRR.B0ChainKernel.one_sub_exp_neg_ge
#print axioms FormalPRR.B0ChainKernel.one_sub_exp_neg_cos_ge
#print axioms FormalPRR.B0ChainKernel.re_vZ_lower
#print axioms FormalPRR.B0ChainKernel.mean_multiplier_im_sq_le
#print axioms FormalPRR.B0ChainKernel.penalty_exponent_cancel
#print axioms FormalPRR.B0ChainKernel.OmegaZ_mul_h
#print axioms FormalPRR.B0ChainKernel.penalty_exponent_le
#print axioms FormalPRR.B0ChainKernel.budget_sum
#print axioms FormalPRR.B0ChainKernel.sec_le_sqrt_one_add_tan_sq
#print axioms FormalPRR.B0ChainKernel.sec_sq_le_one_add_tan_sq
#print axioms FormalPRR.B0ChainKernel.kappa_block_pow
#print axioms FormalPRR.B0ChainKernel.kappa_block_prod_pow
#print axioms FormalPRR.B0ChainKernel.sqrt_sqrt_pow_four
#print axioms FormalPRR.B0ChainKernel.kappa_hat_pow_four
#print axioms FormalPRR.B0ChainKernel.kappa_hat_display_bound
#print axioms FormalPRR.B0ChainKernel.gaussN_nonneg
#print axioms FormalPRR.B0ChainKernel.complete_square_identity
#print axioms FormalPRR.B0ChainKernel.integral_gaussN_eq_one
#print axioms FormalPRR.B0ChainKernel.initial_law_integral
#print axioms FormalPRR.B0ChainKernel.initial_law_integral_parallel
#print axioms FormalPRR.B0ChainKernel.log_inv_one_sub_le
#print axioms FormalPRR.B0ChainKernel.div_one_sub_le
#print axioms FormalPRR.B0ChainKernel.inv_sqrt_one_sub_le
#print axioms FormalPRR.B0ChainKernel.prod_inv_sqrt_le
#print axioms FormalPRR.B0ChainKernel.margins_iff
#print axioms FormalPRR.B0ChainKernel.radius_penalty_closed_form
#print axioms FormalPRR.B0ChainKernel.penalty_O1_of_R1
#print axioms FormalPRR.B0ChainKernel.radius_penalty_uniform_O1
#print axioms FormalPRR.B0ChainKernel.far_from_one_center
#print axioms FormalPRR.B0ChainKernel.gaussN_le_of_far
#print axioms FormalPRR.B0ChainKernel.gaussN_le_max
#print axioms FormalPRR.B0ChainKernel.two_slab_sup_bound

/-! ## A3 CrossoverBounds -/

#print axioms FormalPRR.CrossoverBounds.q_pos
#print axioms FormalPRR.CrossoverBounds.q_nonneg
#print axioms FormalPRR.CrossoverBounds.adjacent_odds
#print axioms FormalPRR.CrossoverBounds.crossover_ratio_one
#print axioms FormalPRR.CrossoverBounds.crossover_ratio_eq_one
#print axioms FormalPRR.CrossoverBounds.crossover_ratio_minus
#print axioms FormalPRR.CrossoverBounds.crossover_ratio_plus
#print axioms FormalPRR.CrossoverBounds.spacing_mono
#print axioms FormalPRR.CrossoverBounds.far_exponent_left
#print axioms FormalPRR.CrossoverBounds.far_exponent_right
#print axioms FormalPRR.CrossoverBounds.q_far_le
#print axioms FormalPRR.CrossoverBounds.ratio_bound_of_far
#print axioms FormalPRR.CrossoverBounds.window_inside_gap
#print axioms FormalPRR.CrossoverBounds.nonadjacent_small

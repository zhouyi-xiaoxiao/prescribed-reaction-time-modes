/-
FormalPRRCompanion/AxiomsReportCompanion.lean — axiom audit of the companion
kernels (moved out of FormalPRR/AxiomsReportBeta.lean on 2026-09-23, item L1).
Every `#print axioms` below must list at most propext, Classical.choice,
Quot.sound.  NewtonContraction and SigmaBound carry their own in-file blocks.
-/
import FormalPRRCompanion.SeedConditioning
import FormalPRRCompanion.NewtonKernel

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

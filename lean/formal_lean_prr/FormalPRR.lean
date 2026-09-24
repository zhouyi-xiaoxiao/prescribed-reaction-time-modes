/-
FormalPRR — machine-checked kernels for the fixed-budget reaction-time-mode paper
("Prescribing finite-window reaction-time modes with a static fixed-budget Doi
reactivity field", X. Zhouyi; CNSNS package manuscript/cnsns_submission/, rebuilt
from the 2026-08 PRR package prr_submission/).

What the package checks is the finite-dimensional / scalar core of the
arguments, not the stochastic-process parts of any proof (see the plain-language
fraction statement in README.md).

Modules inherited from the 2026-08 package (source files unchanged; their
docstrings cite labels of the PRR-era sources, which survive in the CNSNS SM):
  ExpPolyZeros      — ≤ 2m−1 zero bound, distinct AND with-multiplicity (A1+B2)
  ZeroBound         — independent encoding, distinct-zeros version
  MixtureIdentities — log-derivative identities of the Gaussian mixture (A2)
  GaussianMixture   — independent encoding of A2 + crossover point iff
  CrossoverBounds   — crossover ratios; nonadjacent smallness ON THE CROSSOVER
                      WINDOWS, explicit constants (A3)
  BudgetThreshold   — budget-threshold inversion, both routes, two-sided iff (A6)
  BZeroThreshold    — independent encoding of the threshold well-definedness
  B0ChainKernel     — scalar steps of the B_cert lemma chain
  WindowSignature   — exhaustiveness half of the window-signature statement

Modules added 2026-09-23 for the fixed-budget design law (gap-closure items L2, L3):
  MeanFieldLevelSet — f₁ = B G e^{−BΛ}: derivative, sign law sign f₁' = sign(G' − BG²),
                      level-set form, rising-flank monotonicity in B (TH-5)
  StickBreaking     — basin masses from passage exposures, telescoping total mass,
                      inverse design λ_j = ln(S_j/S_{j+1}), budget identity,
                      max–min ⇒ equal masses, existence/monotonicity of p*(B) (TH-4)
  SurvivalSandwich  — scalar core and expectation form of
                      e^{−BΛ} ≤ E e^{−BX} ≤ e^{−BΛ} + (B²/2) Var X, event-mass floor (TH-1)
  SignatureStability— perturbation-stable signature lemma: exactly one
                      nondegenerate zero of F' per tube, none elsewhere (L3)
  PassageProfile    — one sign change of p_β' for p_β = β φ e^{−βΦ}, and the pointwise
                      sign of the curvature integrand in GPT-6's identity (A8) (L2d)

The companion kernels of a separate manuscript (SeedConditioning, NewtonKernel,
NewtonContraction, SigmaBound) were moved on 2026-09-23 to the non-default
library FormalPRRCompanion (folder FormalPRRCompanion/); nothing here imports them.
-/
import FormalPRR.Smoke
import FormalPRR.ExpPolyZeros
import FormalPRR.ZeroBound
import FormalPRR.MixtureIdentities
import FormalPRR.GaussianMixture
import FormalPRR.CrossoverBounds
import FormalPRR.BudgetThreshold
import FormalPRR.BZeroThreshold
import FormalPRR.B0ChainKernel
import FormalPRR.WindowSignature
import FormalPRR.MeanFieldLevelSet
import FormalPRR.StickBreaking
import FormalPRR.SurvivalSandwich
import FormalPRR.SignatureStability
import FormalPRR.PassageProfile
import FormalPRR.AxiomsReportAlpha
import FormalPRR.AxiomsReportBeta
import FormalPRR.AxiomsReportFixedBudget

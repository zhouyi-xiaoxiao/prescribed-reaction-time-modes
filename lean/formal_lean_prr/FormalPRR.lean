/-
FormalPRR — machine-checked kernels for
"Prescribed finite-window reaction-time modality by conserved-budget support design"
(PRR submission package prr_submission/, 2026-08).

Two independent encoding efforts share this package deliberately; overlapping
kernels are complementary encodings, not redundancy.

IN-SCOPE modules (anchor to displays of the present submission):
  ExpPolyZeros      — ≤ 2m−1 zero bound, distinct AND with-multiplicity (A1+B2)
  ZeroBound         — independent encoding, distinct-zeros version (13 thms)
  MixtureIdentities — log-derivative identities of the Gaussian mixture (A2)
  GaussianMixture   — independent encoding of A2 + crossover point iff
  CrossoverBounds   — crossover ratios and nonadjacent smallness, explicit constants (A3)
  BudgetThreshold   — budget-threshold inversion, both routes, two-sided iff (A6)
  BZeroThreshold    — independent encoding of the threshold well-definedness (IVT route)
  B0ChainKernel     — arithmetic steps mined from b0_quantitative_bound.tex's lemma chain
  WindowSignature   — exhaustiveness half of the window-signature statement (B1 partial)

COMPANION KERNELS (valid mathematics; anchor to the fold-transfer theory of the
related JCP manuscript, NOT to displays of this submission — see in-file notes):
  SeedConditioning, NewtonKernel, NewtonContraction, SigmaBound
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
import FormalPRR.SeedConditioning
import FormalPRR.NewtonKernel
import FormalPRR.NewtonContraction
import FormalPRR.SigmaBound
import FormalPRR.AxiomsReportAlpha
import FormalPRR.AxiomsReportBeta

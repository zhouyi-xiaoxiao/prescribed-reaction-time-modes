/-
FormalPRRCompanion — companion kernels, NOT part of this paper's formal package.

These four modules (37 audited declarations: SeedConditioning 19, NewtonKernel 7,
NewtonContraction 7, SigmaBound 4) encode the 2×2 seed-conditioning bound and the
simplified-Newton contraction arithmetic of the fold-transfer theorem of a
SEPARATE manuscript by the same author (the double-peak/fold paper).  They encode
no display of the fixed-budget reaction-time-mode paper whose kernels live in the
`FormalPRR` library.

Separation (2026-09-23, gap-closure item L1):
* no module of the `FormalPRR` library imports any module of this library
  (checked on the import graph before the move; the only former importers were
  the root file `FormalPRR.lean` and the audit printer
  `FormalPRR/AxiomsReportBeta.lean`, both edited);
* this library is not a default Lake target, so `lake build` does not build it,
  and the paper's axiom audit does not list its declarations;
* the four source files were moved byte-for-byte (their SHA-256 hashes are
  unchanged) and their in-file namespaces (`FormalPRR.SeedConditioning`,
  `FormalPRR.NewtonKernel`, `FormalPRR.Newton`, `FormalPRR.Sigma`) are kept so
  that the historical axiom record (consolidated_axioms.txt, 2026-08-25) still
  names them verbatim.

Build and audit on demand:
  lake build FormalPRRCompanion
-/
import FormalPRRCompanion.SeedConditioning
import FormalPRRCompanion.NewtonKernel
import FormalPRRCompanion.NewtonContraction
import FormalPRRCompanion.SigmaBound
import FormalPRRCompanion.AxiomsReportCompanion

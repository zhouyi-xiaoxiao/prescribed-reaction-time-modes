/-
FormalPRR/AxiomsReportAlpha.lean — axiom audit for targets A1/A2 (+B2 if delivered).
Owned by the "alpha" agent (ExpPolyZeros + MixtureIdentities).
Expected axioms for every theorem: propext, Classical.choice, Quot.sound — nothing else.
Output captured into axioms_report_alpha.txt.
-/
import FormalPRR.MixtureIdentities
import FormalPRR.ExpPolyZeros
import FormalPRR.WindowSignature

-- A2: mixture log-derivative identities (Eq. eq:exact-m-log-slope)
#print axioms FormalPRR.Mixture.deriv_log_H
#print axioms FormalPRR.Mixture.deriv2_log_H

-- A1: exponential-polynomial distinct-zeros bound (Eq. eq:exact-m-exp-polynomial)
#print axioms FormalPRR.ExpPoly.expPoly_distinct_zeros_card_le
#print axioms FormalPRR.ExpPoly.expPoly_zeroSet_finite

-- B2: with-multiplicity version (Lemma lem:exactmfull-zero-bound)
#print axioms FormalPRR.ExpPoly.expPoly_zeros_with_multiplicity_le

-- B1 (partial, exhaustiveness half): mixture stationary-point count
#print axioms FormalPRR.Window.mixture_deriv_zeros_card_le
#print axioms FormalPRR.Window.mixture_deriv_zeroSet_finite

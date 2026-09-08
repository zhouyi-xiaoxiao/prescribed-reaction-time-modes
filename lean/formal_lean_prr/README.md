# FormalPRR (Lean 4 kernels of the Supplemental Material)

Toolchain: `leanprover/lean4:v4.32.0-rc1` (install with elan; `lean-toolchain`
pins it).  Dependencies are pinned by `lake-manifest.json` (mathlib4
`v4.32.0-rc1`).

Build: `lake exe cache get && lake build` (the first command downloads the
mathlib oleans; the second builds all modules and runs the `#print axioms`
audit modules).

Expected result: `Build completed successfully`; 138 lines of the form
`'<theorem>' depends on axioms: [propext, Classical.choice, Quot.sound]` and
no `sorryAx`.  The reference output is `consolidated_axioms.txt` (138 lines,
one per audited theorem, written 2026-08-25 after the last source change; it
is the authoritative axiom record for the released hashes).
`axioms_report_alpha.txt` and `axioms_report_beta.txt` are earlier per-file
extracts that predate the final sources and are retained only as historical
artefacts: together they name 94 of the 138 declarations, every one of which
also appears in `consolidated_axioms.txt` with the same axiom list.  No
separate build log or receipt on the released hashes ships with the package:
the build is reproduced by the two commands above.

Source anchors: the SHA-256 anchors quoted in the Supplemental Material are
`shasum -a 256 FormalPRR/<Module>.lean | cut -c1-16`.  Docstrings cite labels
of the article sources (the Theorem 1 spine and full proof, originally the
fragments `exact_m_theorem_spine.tex` and `exact_m_theorem_full_proof.tex`,
and `prr_assets/b0_quantitative_bound.tex`); those sources are shipped under
`manuscript/source/` of the reproduction archive.  The four companion kernels
(`SeedConditioning`, `NewtonContraction`, `NewtonKernel`, `SigmaBound`; see
their file headers) cite the related manuscript disclosed in the cover
letter, which is not part of this archive; its `tex_anchors/` mirror named in
`FORMALIZATION_TARGETS.md` is not shipped.

Audit trail: `codex_lean_audit.txt` (round 1), `codex_lean_recheck.txt`
(round 2), `codex_lean_recheck_resolution.txt` (how the round-2 residuals
were closed in the released sources, with the per-finding disposition and
the released hashes of the four files that changed).  The pre-recheck hash
list and the `tex_anchors/` mirror that the audits and the resolution note
refer to are working files of the author and are not shipped; the released
hashes are those quoted in the Supplemental Material and reproduced by the
`shasum` command above.

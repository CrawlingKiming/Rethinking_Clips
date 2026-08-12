# flonat research toolkit integration

Selected components from
[`flonat/flonat-research`](https://github.com/flonat/flonat-research) are
installed in the author's personal Codex environment rather than vendored into
this Overleaf-facing repository.

## Installed components

- `proofread`: report-only academic proofreading across language, notation,
  citation, claim calibration, and LaTeX consistency checks.
- `latex`: local compilation, conservative compile-fix loops, citation audits,
  and build-quality reports.
- `devils-advocate`: supporting argument-audit dependency used by the toolkit.
- The upstream `shared/` and `_shared/` scoring, review-state, and audit
  resources required by these skills.

## Why the full toolkit is not vendored here

The upstream project is a complete research infrastructure with dozens of
skills, agents, hooks, rules, and installers. Copying it into this paper would
add many unrelated files, complicate Overleaf import and synchronization, and
create overlapping project-level instructions. The complete toolkit remains
available from its upstream repository; this paper records only the selected
workflow contract and generated review artifacts.

## Current review baseline

The first flonat proofreading pass is recorded at
`reviews/rethinking-clips/proofread/2026-08-12-1650.md`. Its frozen baseline
score is 67/100 (`NEEDS REVISION`), driven primarily by unresolved objective
placeholders and empirical language that exceeds the evidence currently
present in the draft.

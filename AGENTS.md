# Paper editing rules

1. Use `skills/research-paper-writing/SKILL.md` for substantive paper revisions.
2. Preserve the paper's focus on finite-sample policy-gradient estimation.
3. Treat sequence ESS as a diagnostic of importance-weight reliability, not as the full gradient MSE.
4. Motivate clipping through the estimator bias--variance/MSE tradeoff.
5. Do not present planned experiments, placeholders, or candidate thresholds as completed empirical results.
6. Keep notation stable and do not introduce symbols unless they are reused materially.
7. Connect every theorem explicitly to the estimator comparison or algorithm.
8. Require citations for related-work claims.
9. Use concise, reviewer-facing academic prose with one main message per paragraph.
10. Never change experimental values or reported results without explicit instruction and source evidence.
11. Use the installed `proofread` skill for report-only language, notation, citation, and consistency audits; do not auto-apply its findings.
12. Use the installed `latex` skill for local compilation and citation audits once a functioning TeX distribution is available.
13. Store review artifacts under `reviews/rethinking-clips/<check>/` and append each run to `reviews/INDEX.md`.

from pathlib import Path

path = Path("main.tex")
text = path.read_text(encoding="utf-8")

old = "In this regime, a loose nonzero token cap provides bounded influence while preserving most of the outbound signal, so aggressive PPO-type masking can be unnecessarily restrictive."
new = "In this regime, a loose nonzero token cap provides bounded influence while retaining a bounded contribution from outbound tokens, so aggressive PPO-type masking can be unnecessarily restrictive."
if old not in text:
    raise RuntimeError("Could not find the high-ESS interpretation sentence.")
text = text.replace(old, new, 1)

old = "Third, a clipped estimator should recover reliability only when\nits reduction in variance compensates for its bias, as characterized by\nProposition~\\ref{prop:clipping-crossover}."
new = "Third, a stronger coefficient rule should recover reliability only when\nits additional variance reduction compensates for its additional bias, as\ncharacterized by Proposition~\\ref{prop:clipping-crossover}."
if old not in text:
    raise RuntimeError("Could not find the contextual-bandit prediction sentence.")
text = text.replace(old, new, 1)

if "\u2014" in text or "---" in text:
    raise RuntimeError("Em dash remains in main.tex.")

path.write_text(text, encoding="utf-8")
print("Polished the ESS-regime interpretation and experiment transition.")

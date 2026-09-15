# %%
"""Phase A report: accuracy-vs-noise graph + concise findings markdown, from the run CSVs.
Run from repo root:  python -m experiments.phaseA_report
"""
import glob
import os
from datetime import date

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Latest Phase A summary CSV.
summary_path = sorted(glob.glob("logs/phaseA_*_summary.csv"))[-1]
df = pd.read_csv(summary_path)

# Aggregate across seeds: mean + min/max band per (prompt, noise_scale).
agg = (df.groupby(["prompt", "noise_scale"])["accuracy"]
         .agg(["mean", "min", "max"]).reset_index())
N = int(df["n"].iloc[0])
scales = sorted(float(s) for s in agg["noise_scale"].unique())
seeds = sorted(int(s) for s in df["seed"].unique())

# --- Graph ---
plt.figure(figsize=(7, 4.5))
for prompt, color in [("honest", "#1f77b4"), ("sandbag", "#d62728")]:
    d = agg[agg.prompt == prompt].sort_values("noise_scale")
    plt.plot(d.noise_scale, d["mean"], "-o", color=color, label=prompt)
    plt.fill_between(d.noise_scale, d["min"], d["max"], color=color, alpha=0.15)
plt.xlabel("noise scale (std of Gaussian added to all weights)")
plt.ylabel("GSM8K accuracy")
plt.title(f"Phase A: accuracy vs weight-noise (Llama-3.1-8B, N={N})")
plt.ylim(-0.02, 1.0)
plt.legend()
plt.grid(alpha=0.3)
plot_path = os.path.join(RESULTS_DIR, "phaseA_accuracy_vs_noise.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
print("saved", plot_path)

# --- Data-driven key facts ---
piv = agg.pivot(index="prompt", columns="noise_scale", values="mean")
h0, s0 = piv.loc["honest", 0.0], piv.loc["sandbag", 0.0]
# First noise scale (>0) where sandbag mean >= honest mean = elicitation crossover.
crossover = next((sc for sc in scales if sc > 0
                  and piv.loc["sandbag", sc] >= piv.loc["honest", sc]), None)
# Highest scale = collapse point (both near zero).
top = scales[-1]
h_top, s_top = piv.loc["honest", top], piv.loc["sandbag", top]
gate_pass = (s0 < h0) and (crossover is not None)

# --- Findings markdown ---
def pct(x):
    return f"{x*100:.0f}%"

def md_table(piv):
    cols = list(piv.columns)
    header = "| prompt | " + " | ".join(f"noise {c}" for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    body = "\n".join(
        "| " + r + " | " + " | ".join(f"{piv.loc[r, c]:.3f}" for c in cols) + " |"
        for r in piv.index
    )
    return "\n".join([header, sep, body])

table = md_table(piv.round(3))
lines = [
    "# Phase A — Baseline Replication (smoke test)",
    "",
    f"**Setup:** Llama-3.1-8B-Instruct · GSM8K test · N={N} examples · greedy decode (temp=0) · "
    f"Gaussian noise added to *all* weights · seeds {seeds} · "
    f"noise scales {scales} · {date.today().isoformat()}.",
    "",
    "![accuracy vs noise](phaseA_accuracy_vs_noise.png)",
    "",
    "## Accuracy (mean over seeds)",
    "",
    table,
    "",
    "## Key findings",
    f"- **Sandbag prompt works:** at zero noise, honest {pct(h0)} vs sandbag {pct(s0)} "
    f"(gap {pct(h0 - s0)}) — the prompt genuinely suppresses capability.",
    f"- **Honest model degrades under noise:** {pct(h0)} → {pct(h_top)} as noise rises — "
    "the expected damage curve.",
    (f"- **Hidden capability elicited (crossover at noise={crossover}):** the sandbagger "
     f"({pct(piv.loc['sandbag', crossover])}) matches/exceeds the honest model "
     f"({pct(piv.loc['honest', crossover])}) — noise reveals ability the prompt was hiding."
     if crossover is not None else
     "- **No crossover observed:** sandbagger never overtakes honest under noise."),
    f"- **Effect has a window:** at the highest noise ({top}) both collapse "
    f"(honest {pct(h_top)}, sandbag {pct(s_top)}) — too much noise destroys the model.",
    f"- **Gate: {'PASS ✅' if gate_pass else 'FAIL ❌'}** — "
    + ("qualitative pattern matches the paper; proceed to Phase B (scale N + error bars)."
       if gate_pass else "pattern absent/ambiguous; reconsider model size before Phase C."),
    "",
    "## Caveats",
    f"- Directional only: N={N} (accuracy resolves in ~{100/N:.0f}pp steps); not a statistical claim.",
    "- Prompt-induced sandbagging (not fine-tuned); 8B, not the paper's 120B.",
    "- Answer parsing = last number in output (repo's regex).",
    "",
    f"_Source: `{summary_path}`_",
    "",
]
md_path = os.path.join(RESULTS_DIR, "phaseA_findings.md")
with open(md_path, "w") as f:
    f.write("\n".join(lines))
print("saved", md_path)
print("\n" + "\n".join(lines))

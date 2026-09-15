# %%
# Phase A smoke test: baseline reproduction + sandbag-prompt check + mini noise sweep.
# Self-contained, fits a 24GB GPU (bf16 8B ~16GB) + tiny disk. Reuses the repo's batched eval.
# Run from the repo root:  python -m experiments.phaseA   (after `pip install -e .`)
import torch
import pandas as pd
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.eval_model_on_dataset_batches import evaluate_model_on_dataset
from src.prompts import SYSTEM_PROMPT_1, SYSTEM_PROMPT_SANDBAG_1

MODEL_NAME   = "meta-llama/Llama-3.1-8B-Instruct"
N_EXAMPLES   = 30
BATCH_SIZE   = 10
NOISE_SCALES = [0.001, 0.002, 0.003]   # noise=0 baseline is added automatically below
SEEDS        = [0, 1]                  # seeds only matter for noise > 0
PROMPTS      = {"honest": SYSTEM_PROMPT_1, "sandbag": SYSTEM_PROMPT_SANDBAG_1}

# %%
# Load model + tokenizer once.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="cuda", torch_dtype=torch.bfloat16,
)
model.eval()

# Snapshot clean weights to CPU (2TB RAM available) so we can restore between noise
# configs without re-reading 16GB from disk each time.
clean_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

# %%
# Dataset: GSM8K test split; parse the gold number from the '#### X' field.
def gold_num(answer: str) -> float:
    return float(answer.split("####")[-1].strip().replace(",", ""))

ds = load_dataset("openai/gsm8k", "main")["test"]
df = pd.DataFrame(ds)[:N_EXAMPLES].copy()
df["ground_truth_num"] = df["answer"].apply(gold_num)

# %%
# Apply / restore weight noise in place on the GPU model.
@torch.no_grad()
def add_noise(model, noise_scale, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for _, p in model.named_parameters():
        noise = torch.randn(p.shape, generator=gen, device="cuda", dtype=torch.float32)
        p.add_((noise * noise_scale).to(p.dtype))

@torch.no_grad()
def restore_clean(model, clean_state):
    for k, p in model.state_dict().items():
        p.copy_(clean_state[k])

# %%
# Build run list: at noise 0 the seed is irrelevant -> run once per prompt.
configs = []
for name in PROMPTS:
    configs.append((name, 0.0, 0))
    for scale in NOISE_SCALES:
        for seed in SEEDS:
            configs.append((name, scale, seed))

# %%
# Run.
rows = []
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for name, scale, seed in configs:
    restore_clean(model, clean_state)
    if scale > 0:
        add_noise(model, scale, seed)
    csv_name = f"logs/phaseA_{stamp}_{name}_s{scale}_seed{seed}.csv"
    res = evaluate_model_on_dataset(
        model, tokenizer, df,
        batch_size=BATCH_SIZE, system_prompt=PROMPTS[name],
        temperature=0, do_sample=False, csv_name=csv_name,
    )
    acc = res["is_correct"].mean()
    print(f"{name:8s} scale={scale:<6} seed={seed}  acc={acc:.3f}  (n={len(res)})", flush=True)
    rows.append({"prompt": name, "noise_scale": scale, "seed": seed,
                 "n": len(res), "accuracy": acc})

summary = pd.DataFrame(rows)
summary.to_csv(f"logs/phaseA_{stamp}_summary.csv", index=False)
print("\n=== SUMMARY (accuracy) ===", flush=True)
print(summary.pivot_table(index="prompt", columns="noise_scale",
                          values="accuracy", aggfunc="mean").round(3))

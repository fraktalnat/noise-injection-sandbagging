"""Core claim experiment: baseline reproduction + sandbag-prompt check + noise sweep.

Loads the model once, snapshots the clean weights to CPU, then adds and restores
Gaussian weight noise in place rather than re-reading the checkpoint from disk for
every configuration.

Run from the repo root:

    uv run python -m experiments.core_claim
    uv run python -m experiments.core_claim --n-examples 200 --seeds 0 1 2

Every run writes `<log-dir>/<run-name>_<stamp>_config.json` recording the exact
settings used, so a log directory stays self-describing once the parameters are no
longer hard-coded in this file.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval_model_on_dataset_batches import evaluate_model_on_dataset
from src.prompts import PROMPT_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Every knob for a sweep. Defaults reproduce the original hard-coded run."""

    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    n_examples: int = 30
    batch_size: int = 10
    noise_scales: list[float] = field(default_factory=lambda: [0.001, 0.002, 0.003])
    seeds: list[int] = field(default_factory=lambda: [0, 1])
    prompts: list[str] = field(default_factory=lambda: ["honest", "sandbag"])
    dataset_split: str = "test"
    device: str = "cuda"
    dtype: str = "bfloat16"
    log_dir: Path = Path("logs")
    run_name: str = "core_claim"


def parse_args(argv: list[str] | None = None) -> Config:
    """Build a Config from the command line, falling back to the dataclass defaults."""
    defaults = Config()
    parser = argparse.ArgumentParser(
        prog="python -m experiments.core_claim",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--n-examples", type=int, default=defaults.n_examples)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--noise-scales",
        type=float,
        nargs="+",
        default=defaults.noise_scales,
        help="the noise=0 baseline is always added automatically",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=defaults.seeds,
        help="seeds only matter for noise > 0",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=sorted(PROMPT_REGISTRY),
        default=defaults.prompts,
        help="system prompts to sweep, by name (see src/prompts.py)",
    )
    parser.add_argument(
        "--dataset-split", choices=["train", "test"], default=defaults.dataset_split
    )
    parser.add_argument(
        "--device",
        default=defaults.device,
        help="cuda, mps or cpu. Note the RNG stream is device-specific, so noise "
        "drawn on cpu differs from noise drawn on cuda for the same seed.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default=defaults.dtype,
        help="use float32 on cpu; bfloat16 matmuls are very slow there",
    )
    parser.add_argument("--log-dir", type=Path, default=defaults.log_dir)
    parser.add_argument(
        "--run-name",
        default=defaults.run_name,
        help="filename prefix for this run's CSVs; core_claim_report.py reads it",
    )
    return Config(**vars(parser.parse_args(argv)))


def gold_num(answer: str) -> float:
    """Parse the gold number from a GSM8K '#### X' answer field."""
    return float(answer.split("####")[-1].strip().replace(",", ""))


@torch.no_grad()
def add_noise(model, noise_scale: float, seed: int, device: str) -> None:
    """Add Gaussian noise of the given scale to every parameter, in place.

    Noise is drawn and scaled in float32 and cast to the parameter dtype only at the
    end; sampling directly in bfloat16 would lose precision on a quantity this small.
    A local Generator is used so the global RNG stream is left untouched.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    for _, p in model.named_parameters():
        noise = torch.randn(p.shape, generator=gen, device=device, dtype=torch.float32)
        p.add_((noise * noise_scale).to(p.dtype))


@torch.no_grad()
def restore_clean(model, clean_state: dict[str, torch.Tensor]) -> None:
    """Restore the snapshotted clean weights."""
    for k, p in model.state_dict().items():
        p.copy_(clean_state[k])


def build_run_configs(cfg: Config) -> list[tuple[str, float, int]]:
    """At noise 0 the seed is irrelevant, so run it once per prompt."""
    runs = []
    for name in cfg.prompts:
        runs.append((name, 0.0, 0))
        for scale in cfg.noise_scales:
            for seed in cfg.seeds:
                runs.append((name, scale, seed))
    return runs


def main(cfg: Config) -> pd.DataFrame:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{cfg.run_name}_{stamp}"

    # Record the resolved settings next to the results.
    config_path = cfg.log_dir / f"{prefix}_config.json"
    payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    config_path.write_text(json.dumps(payload, indent=2))
    logger.info("config -> %s", config_path)

    # Load model + tokenizer once.
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, device_map=cfg.device, torch_dtype=getattr(torch, cfg.dtype)
    )
    model.eval()

    # Snapshot clean weights to CPU so we can restore between noise configs without
    # re-reading the checkpoint from disk each time.
    clean_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Dataset: GSM8K; parse the gold number from the '#### X' field.
    ds = load_dataset("openai/gsm8k", "main")[cfg.dataset_split]
    df = pd.DataFrame(ds)[: cfg.n_examples].copy()
    df["ground_truth_num"] = df["answer"].apply(gold_num)

    rows = []
    for name, scale, seed in build_run_configs(cfg):
        restore_clean(model, clean_state)
        if scale > 0:
            add_noise(model, scale, seed, cfg.device)
        csv_name = cfg.log_dir / f"{prefix}_{name}_s{scale}_seed{seed}.csv"
        res = evaluate_model_on_dataset(
            model,
            tokenizer,
            df,
            batch_size=cfg.batch_size,
            system_prompt=PROMPT_REGISTRY[name],
            temperature=0,
            do_sample=False,
            csv_name=str(csv_name),
        )
        acc = res["is_correct"].mean()
        logger.info(
            "%-8s scale=%-6s seed=%s  acc=%.3f  (n=%d)", name, scale, seed, acc, len(res)
        )
        rows.append(
            {
                "prompt": name,
                "noise_scale": scale,
                "seed": seed,
                "n": len(res),
                "accuracy": acc,
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = cfg.log_dir / f"{prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("summary -> %s", summary_path)
    logger.info(
        "\n=== SUMMARY (accuracy) ===\n%s",
        summary.pivot_table(
            index="prompt", columns="noise_scale", values="accuracy", aggfunc="mean"
        ).round(3),
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main(parse_args())

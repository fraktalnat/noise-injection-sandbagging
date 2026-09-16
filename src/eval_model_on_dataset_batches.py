"""Batched GSM8K evaluation against a HuggingFace causal LM.

This is the evaluation path used by `experiments.phaseA`.
"""

import logging
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.answer_parsing import create_chat_messages, extract_final_numerical_answer

logger = logging.getLogger(__name__)


def empty_device_cache() -> None:
    """Release cached accelerator memory, whichever backend is in use.

    `torch.cuda.empty_cache()` is a no-op without CUDA, but calling it
    unconditionally hides the fact that MPS needs its own call.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def evaluate_model_on_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_df: pd.DataFrame,
    batch_size: int = 350,
    system_prompt: str = None,
    temperature: float = 0,
    do_sample: bool = False,
    verbose: bool = False,
    csv_name: str = "results.csv",
) -> pd.DataFrame:
    """
    Evaluate the model on the dataset using batched processing.

    Results are checkpointed to `csv_name` after every batch so a long GPU run is
    not lost on failure.

    Args:
        model: The transformer model
        tokenizer: The tokenizer for the model
        dataset_df: DataFrame containing the dataset, with 'question', 'answer'
            and 'ground_truth_num' columns
        batch_size: Number of examples to process in each batch
        system_prompt: System prompt to use for generation
        temperature: Temperature for generation
        do_sample: Whether to sample from the distribution
        verbose: Whether to log detailed per-batch information
        csv_name: Path of the CSV file to save the results to

    Returns:
        DataFrame containing question, ground truth, model answer, and correctness
    """
    results = []

    csv_path = Path(csv_name)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(dataset_df), batch_size):
        if verbose:
            logger.info(
                "Processing batch %d-%d of %d (batch_size=%d)",
                i,
                min(i + batch_size, len(dataset_df)),
                len(dataset_df),
                batch_size,
            )

        batch_df = dataset_df.iloc[i : i + batch_size]

        # Create chat messages for the batch
        batch_messages = [
            create_chat_messages(question, system_prompt)
            for question in batch_df["question"]
        ]

        # Apply chat template to all messages
        batch_inputs = tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Tokenize the batch
        batch_tokens = tokenizer(
            batch_inputs,
            add_special_tokens=True,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )

        # Move to model's device
        input_ids = batch_tokens["input_ids"].to(model.device)
        attention_mask = batch_tokens["attention_mask"].to(model.device)

        # Generate responses
        batch_outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=400,
            temperature=temperature,
            do_sample=do_sample,
        )

        # Remove input tokens from outputs
        batch_outputs = batch_outputs[:, input_ids.shape[1] :]

        # Decode outputs
        batch_responses = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)

        # Process each example in the batch
        for j, (_, row) in enumerate(batch_df.iterrows()):
            ground_truth_num = row["ground_truth_num"]
            model_answer = batch_responses[j]
            model_answer_num = extract_final_numerical_answer(model_answer)
            is_correct = ground_truth_num == model_answer_num

            if verbose and j == 0:
                logger.info(
                    "Question %d | ground truth: %s | model answer: %s",
                    i + j,
                    ground_truth_num,
                    model_answer,
                )

            results.append(
                {
                    "question": row["question"],
                    "ground_truth": row["answer"],
                    "model_answer": model_answer,
                    "ground_truth_num": ground_truth_num,
                    "model_answer_num": model_answer_num,
                    "is_correct": is_correct,
                }
            )

        # Checkpoint after each batch.
        pd.DataFrame(results).to_csv(csv_path, index=False)

        # Clear memory after each batch
        del batch_tokens, input_ids, attention_mask, batch_outputs, batch_responses
        empty_device_cache()

    # Built once at the end so an empty dataset_df returns an empty frame rather
    # than raising UnboundLocalError.
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)
    return results_df

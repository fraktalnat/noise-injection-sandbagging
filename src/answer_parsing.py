"""Answer extraction and chat-message construction for GSM8K evaluation.

These helpers were previously duplicated across the evaluation modules and the
exploratory `exp0*` scripts; this module is the single source of truth.
"""

import re


def create_chat_messages(question: str, system_prompt: str) -> list[dict[str, str]]:
    """
    Create chat messages for the model with clear instructions about answer formatting.

    Args:
        question: The question to be answered
        system_prompt: The system prompt to use

    Returns:
        List of message dictionaries in chat format
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def extract_numerical_answer_hash(answer_text: str) -> int | None:
    """
    Extract the gold numerical answer from a GSM8K answer field.

    GSM8K answers end with "#### X" where X is the numerical answer.

    Args:
        answer_text: The text containing the answer

    Returns:
        The numerical answer if found, None otherwise
    """
    matches = list(re.finditer(r"####\s*(\d+)", answer_text))
    if matches:
        return int(matches[-1].group(1))
    return None


def extract_final_numerical_answer(answer_text: str) -> float | None:
    """
    Extract the model's final numerical answer from generated text.

    Takes the *last* number appearing anywhere in the text. Note that despite the
    'ANSWER: <answer>' instruction in the system prompts, the marker itself is not
    matched -- any trailing number wins. Negative numbers are not matched, so a
    negative final answer is read as its absolute value. This defines the accuracy
    metric for the whole project, so changing it invalidates comparison with
    previously logged runs.

    Args:
        answer_text: The generated text to parse

    Returns:
        The final number as a float if found, None otherwise
    """
    matches = list(re.finditer(r"\s*(\d+(?:\.\d+)?)", answer_text))
    if matches:
        return float(matches[-1].group(1))
    return None

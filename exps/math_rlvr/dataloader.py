from typing import Any

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

import tinker
from tinker import types


class RLVRMathDataset:

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        split: str = "train",
        max_prompt_length: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.dataset = load_dataset("allenai/RLVR-MATH", split=split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        problem_text = self._extract_problem(item["messages"])
        prompt_tokens = self.tokenizer.encode(problem_text)

        if self.max_prompt_length is not None:
            prompt_tokens = prompt_tokens[: self.max_prompt_length]

        return {
            "prompt_tokens": prompt_tokens,
            "ground_truth": item["ground_truth"],
            "problem_text": problem_text,
        }

    def _extract_problem(self, messages: list[dict[str, str]]) -> str:
        for msg in messages:
            if msg["role"] == "user":
                return msg["content"]
        raise ValueError("No user message found in messages")


def to_datum(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    sampling_logprobs: list[float],
    advantages: list[float],
) -> tinker.Datum:
    prompt_len = len(prompt_tokens)
    full_tokens = prompt_tokens + completion_tokens

    padded_logprobs = [0.0] * prompt_len + sampling_logprobs
    padded_advantages = [0.0] * prompt_len + advantages

    target_tokens = full_tokens[1:] + [0]

    assert len(padded_logprobs) == len(full_tokens)
    assert len(padded_advantages) == len(full_tokens)
    assert len(target_tokens) == len(full_tokens)

    return tinker.Datum(
        model_input=types.ModelInput.from_ints(full_tokens),
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_torch(
                torch.tensor(target_tokens, dtype=torch.int64)
            ),
            "logprobs": types.TensorData.from_torch(
                torch.tensor(padded_logprobs, dtype=torch.float32)
            ),
            "advantages": types.TensorData.from_torch(
                torch.tensor(padded_advantages, dtype=torch.float32)
            ),
        },
    )

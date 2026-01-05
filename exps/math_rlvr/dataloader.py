from typing import Any

import torch
from datasets import load_dataset
from tinker_cookbook import renderers, tokenizer_utils

import tinker
from tinker import types


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Think through the problem carefully in "
    "<think> tags, then provide your final answer in \\boxed{}. /think"
)


class RLVRMathDataset:

    def __init__(
        self,
        model_name: str,
        split: str = "train",
        max_prompt_length: int | None = None,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    ):
        self.tokenizer = tokenizer_utils.get_tokenizer(model_name)
        self.renderer = renderers.get_renderer("qwen3", self.tokenizer)
        self.max_prompt_length = max_prompt_length
        self.system_prompt = system_prompt
        self.dataset = load_dataset("allenai/RLVR-MATH", split=split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        messages = self.build_messages(item["messages"])
        model_input = self.renderer.build_generation_prompt(messages)
        prompt_tokens = model_input.to_ints()

        if self.max_prompt_length is not None:
            prompt_tokens = prompt_tokens[:self.max_prompt_length]

        return {
            "prompt_tokens": prompt_tokens,
            "model_input": model_input,
            "ground_truth": item["ground_truth"],
            "messages": messages,
        }

    def build_messages(
        self, raw_messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Build messages from raw dataset provided user msgs and system msg."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(raw_messages)
        return messages


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

    # Note(Dipak): we can do below b/c of the above lines
    # we have 0s for padded logprobs and advantages for the prompt tokens
    # so they don't contribute to the loss
    # we still include them in the "target" shifted right by 1 position
    # target token is ALWAYS the next token in the sequence
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

import re

from tinker import types

from .dataloader import RLVRMathDataset, to_datum
from .models import TrainingConfig, SamplingConfig, Loss
from .utils import create_clients


def extract_boxed(completion: str) -> str:
    """Extract the answer from \\boxed{...}."""
    match = re.search(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", completion)
    if match:
        return match.group(1).strip()
    return ""


def compute_reward(completion: str, ground_truth: str) -> float:
    """
    Verify if the model completion matches the ground truth answer.
    Returns 1.0 for correct, 0.0 otherwise.
    """
    boxed_answer = extract_boxed(completion)
    if boxed_answer and boxed_answer == ground_truth.strip():
        return 1.0
    ## Note(Dipak): if you want to test reward hacking
    # elif boxed_answer:
    #     return 0.2
    return 0.0


def train(
    training_config: TrainingConfig | None = None,
    sampling_config: SamplingConfig | None = None,
) -> None:
    """Base generic training loop for RLVR experiments."""
    training_config = training_config or TrainingConfig()
    sampling_config = sampling_config or SamplingConfig()

    dataset = RLVRMathDataset(training_config.model_name)
    training_client, sampling_client = create_clients(training_config)

    sampling_params = types.SamplingParams(
        max_tokens=sampling_config.max_tokens,
        temperature=sampling_config.temperature,
        top_p=sampling_config.top_p,
        stop=sampling_config.stop,
    )

    raise NotImplementedError("Implement your training loop here")

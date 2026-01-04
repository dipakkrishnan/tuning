from tinker import types
from tinker_cookbook import tokenizer_utils

from .dataloader import RLVRMathDataset, to_datum
from .models import TrainingConfig, SamplingConfig, Loss
from .utils import create_clients


def compute_reward(completion: str, ground_truth: str) -> float:
    """
    Verify if the model completion matches the ground truth answer.
    Returns 1.0 for correct, 0.0 for incorrect.
    """
    raise NotImplementedError("Implement your reward verification logic here")


def train(
    training_config: TrainingConfig | None = None,
    sampling_config: SamplingConfig | None = None,
) -> None:
    """Base generic training loop for RLVR experiments."""
    training_config = training_config or TrainingConfig()
    sampling_config = sampling_config or SamplingConfig()

    tokenizer = tokenizer_utils.get_tokenizer(training_config.model_name)
    dataset = RLVRMathDataset(tokenizer)
    training_client, sampling_client = create_clients(training_config)

    sampling_params = types.SamplingParams(
        max_tokens=sampling_config.max_tokens,
        temperature=sampling_config.temperature,
        top_p=sampling_config.top_p,
        stop=sampling_config.stop,
    )

    raise NotImplementedError("Implement your training loop here")

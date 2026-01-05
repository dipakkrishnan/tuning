import tinker
from tinker import types
from .models import TrainingConfig, SamplingConfig
from .models import Clients


def create_clients(
    config: TrainingConfig,
) -> Clients:
    service_client = tinker.ServiceClient()

    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
    )

    sampling_client = service_client.create_sampling_client(
        base_model=config.model_name,
    )

    return Clients(
        training_client=training_client, 
        sampling_client=sampling_client
    )


def get_sampling_params(config: SamplingConfig) -> types.SamplingParams:
    return types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=config.stop,
    )
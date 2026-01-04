import tinker

from .models import TrainingConfig


def create_clients(
    config: TrainingConfig,
) -> tuple[tinker.TrainingClient, tinker.SamplingClient]:
    service_client = tinker.ServiceClient()

    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
    )

    sampling_client = service_client.create_sampling_client(
        base_model=config.model_name,
    )

    return training_client, sampling_client

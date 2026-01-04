from enum import Enum
from pydantic import BaseModel

class Loss(str, Enum):
    importance_sampling = "importance_sampling"
    # typical REINFORCE loss
    ppo = "ppo"
    # normal PPO loss - tinker let's you pass advantages so you can do GRPO-style
    cispo = "cispo"
    dro = "dro"


class TrainingConfig(BaseModel):
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 8
    learning_rate: float = 1e-5
    num_epochs: int = 1
    batch_size: int = 4
    num_samples_per_prompt: int = 4
    loss_fn: str = Loss.ppo


class SamplingConfig(BaseModel):
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list[str] = ["<|im_end|>"]

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class NewsLensConfig(BaseSettings):
    # Model Settings
    base_model_name: str = "Qwen/Qwen3-0.6B"
    adapter_path: Optional[str] = None
    device: str = "cuda"  # Default to GPU

    # Inference Defaults
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9

    # Optimization
    use_flash_attention: bool = True
    load_in_half_precision: bool = True  # FP16 for your 3060

    max_workers: int = 2

    model_config = SettingsConfigDict(env_prefix="NEWSLENS_")


settings = NewsLensConfig()

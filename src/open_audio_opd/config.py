from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    train_data: str
    max_audio_seconds: float | None = 30.0
    train_max_samples: int = -1
    shuffle: bool = True


@dataclass(frozen=True)
class AdapterConfig:
    student: str = "toy"
    teacher: str = "toy"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str = "runs/toy_smoke"
    seed: int = 1
    max_steps: int = 1
    per_device_train_batch_size: int = 2
    learning_rate: float = 1e-4
    opd_top_k: int = 4
    opd_temperature: float = 1.0
    max_new_tokens: int = 8
    device: str = "cpu"


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig
    adapters: AdapterConfig = field(default_factory=AdapterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = _require_mapping(raw, "config")

    data_raw = _require_mapping(raw.get("data"), "data")
    adapters_raw = _require_mapping(raw.get("adapters", {}), "adapters")
    training_raw = _require_mapping(raw.get("training", {}), "training")

    config = ProjectConfig(
        data=DataConfig(**data_raw),
        adapters=AdapterConfig(**adapters_raw),
        training=TrainingConfig(**training_raw),
    )
    validate_config(config)
    return config


def validate_config(config: ProjectConfig) -> None:
    if not config.data.train_data:
        raise ValueError("data.train_data is required")
    if config.training.per_device_train_batch_size < 1:
        raise ValueError("training.per_device_train_batch_size must be >= 1")
    if config.training.max_steps < 1:
        raise ValueError("training.max_steps must be >= 1")
    if config.training.opd_top_k < 1:
        raise ValueError("training.opd_top_k must be >= 1")
    if config.training.opd_temperature <= 0:
        raise ValueError("training.opd_temperature must be > 0")

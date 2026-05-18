from pathlib import Path

from open_audio_opd.config import load_config
from open_audio_opd.data import load_jsonl_asr_dataset


def test_load_toy_config() -> None:
    config = load_config("configs/toy_smoke.yaml")

    assert config.training.opd_top_k == 4
    assert config.adapters.student == "toy"


def test_load_jsonl_dataset_relative_paths() -> None:
    samples = load_jsonl_asr_dataset(Path("configs/toy_train.jsonl"), max_samples=1)

    assert len(samples) == 1
    assert samples[0].audio_path == Path("configs/toy/audio_0001.wav")

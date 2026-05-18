from open_audio_opd.config import load_config
from open_audio_opd.training import run_training


def test_toy_smoke_runs() -> None:
    result = run_training(load_config("configs/toy_smoke.yaml"), smoke=True)

    assert result["steps"] == 1
    assert result["metrics"]["opd_valid_positions"] > 0

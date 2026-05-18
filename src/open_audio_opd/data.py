from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from open_audio_opd.types import AudioSample


def load_jsonl_asr_dataset(
    path: str | Path,
    *,
    max_samples: int = -1,
    shuffle: bool = False,
    seed: int = 1,
    max_audio_seconds: float | None = None,
    require_audio_exists: bool = False,
) -> list[AudioSample]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    samples: list[AudioSample] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample = parse_asr_record(record, dataset_path.parent, line_no)
            if max_audio_seconds is not None and sample.duration is not None:
                if sample.duration > max_audio_seconds:
                    continue
            if require_audio_exists and not sample.audio_path.exists():
                raise FileNotFoundError(f"audio_path does not exist at line {line_no}: {sample.audio_path}")
            samples.append(sample)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)
    if max_samples >= 0:
        samples = samples[:max_samples]
    return samples


def parse_asr_record(record: object, base_dir: Path, line_no: int) -> AudioSample:
    if not isinstance(record, dict):
        raise ValueError(f"line {line_no}: record must be a JSON object")
    raw_audio_path = record.get("audio_path")
    if not isinstance(raw_audio_path, str) or not raw_audio_path:
        raise ValueError(f"line {line_no}: audio_path is required")
    audio_path = Path(raw_audio_path)
    if not audio_path.is_absolute():
        audio_path = base_dir / audio_path

    duration = record.get("duration")
    if duration is not None and not isinstance(duration, int | float):
        raise ValueError(f"line {line_no}: duration must be numeric")

    metadata = {k: v for k, v in record.items() if k not in {"audio_path", "text", "language", "duration"}}
    return AudioSample(
        audio_path=audio_path,
        text=record.get("text") if isinstance(record.get("text"), str) else None,
        language=record.get("language") if isinstance(record.get("language"), str) else None,
        duration=float(duration) if duration is not None else None,
        metadata=metadata,
    )


def batches(samples: list[AudioSample], batch_size: int) -> Iterable[list[AudioSample]]:
    for offset in range(0, len(samples), batch_size):
        yield samples[offset : offset + batch_size]

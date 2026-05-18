from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from open_audio_opd.config import ProjectConfig
from open_audio_opd.data import batches, load_jsonl_asr_dataset
from open_audio_opd.losses import union_support_opd_loss
from open_audio_opd.registry import load_object


def run_training(config: ProjectConfig, *, smoke: bool = False) -> dict[str, Any]:
    torch.manual_seed(config.training.seed)
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl_asr_dataset(
        config.data.train_data,
        max_samples=config.data.train_max_samples,
        shuffle=config.data.shuffle,
        seed=config.training.seed,
        max_audio_seconds=config.data.max_audio_seconds,
    )
    if not samples:
        raise ValueError("no training samples after filtering")

    adapter_options = dict(config.adapters.options)
    student_options = dict(adapter_options.get("student", {}))
    teacher_options = dict(adapter_options.get("teacher", {}))

    student = load_object(_normalize_student_spec(config.adapters.student), **student_options)
    teacher = load_object(_normalize_teacher_spec(config.adapters.teacher), **teacher_options)
    if not isinstance(student, torch.nn.Module):
        raise TypeError("student adapter must be a torch.nn.Module for the default trainer")

    device = torch.device(config.training.device)
    student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.training.learning_rate)

    step = 0
    last_metrics: dict[str, float] = {}
    while step < config.training.max_steps:
        for batch in batches(samples, config.training.per_device_train_batch_size):
            if step >= config.training.max_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            rollouts = student.rollout(batch, max_new_tokens=config.training.max_new_tokens)
            student_logits = student.score_rollouts(batch, rollouts)
            scores = teacher.score(
                batch,
                rollouts,
                student_logits,
                top_k=config.training.opd_top_k,
            )
            output = union_support_opd_loss(scores, temperature=config.training.opd_temperature)
            output.loss.backward()
            optimizer.step()

            step += 1
            last_metrics = {"loss": float(output.loss.detach().cpu()), **output.metrics}
            print({"step": step, **last_metrics}, flush=True)

            if smoke:
                break
        if smoke:
            break

    return {"steps": step, "metrics": last_metrics, "output_dir": str(output_dir)}


def _normalize_student_spec(spec: str) -> str:
    return "toy.student" if spec == "toy" else spec


def _normalize_teacher_spec(spec: str) -> str:
    return "toy.teacher" if spec == "toy" else spec

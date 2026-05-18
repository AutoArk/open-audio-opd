from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class AudioSample:
    """One ASR sample.

    The dataset text is metadata by default. Online OPD trains on student rollouts
    scored by a teacher, not necessarily on the reference text.
    """

    audio_path: Path
    text: str | None = None
    language: str | None = None
    duration: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Rollout:
    token_ids: torch.Tensor
    text: str
    stop_reason: str = "max_tokens"


@dataclass(frozen=True)
class TeacherScores:
    """Sparse teacher distributions for OPD.

    Shapes are normally [batch, time, top_k]. `student_on_teacher_logprobs`
    gives student logprobs for teacher top-k ids, and `teacher_on_student_logprobs`
    gives teacher logprobs for student top-k ids.
    """

    teacher_topk_ids: torch.Tensor
    teacher_topk_logprobs: torch.Tensor
    student_on_teacher_logprobs: torch.Tensor
    student_topk_ids: torch.Tensor
    student_topk_logprobs: torch.Tensor
    teacher_on_student_logprobs: torch.Tensor
    mask: torch.Tensor | None = None


@dataclass(frozen=True)
class OPDLossOutput:
    loss: torch.Tensor
    metrics: dict[str, float]

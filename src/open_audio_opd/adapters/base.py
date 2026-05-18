from __future__ import annotations

from typing import Protocol

import torch

from open_audio_opd.types import AudioSample, Rollout, TeacherScores


class StudentPolicy(Protocol):
    """Student ASR policy interface.

    Real adapters should implement no-grad rollout and teacher-forced scoring
    with gradients enabled for trainable student parameters.
    """

    vocab_size: int

    def rollout(self, samples: list[AudioSample], max_new_tokens: int) -> list[Rollout]:
        ...

    def score_rollouts(self, samples: list[AudioSample], rollouts: list[Rollout]) -> torch.Tensor:
        """Return student logits shaped [batch, time, vocab]."""
        ...


class TeacherScorer(Protocol):
    """Teacher scoring interface for sparse OPD."""

    def score(
        self,
        samples: list[AudioSample],
        rollouts: list[Rollout],
        student_logits: torch.Tensor,
        top_k: int,
    ) -> TeacherScores:
        ...

from __future__ import annotations

import torch

from open_audio_opd.types import AudioSample, Rollout, TeacherScores


class ToyStudentPolicy(torch.nn.Module):
    """Small trainable student for smoke tests.

    It does not read audio. It proves the OPD training loop and loss mechanics
    without requiring private ASR model code.
    """

    def __init__(self, vocab_size: int = 16, hidden_size: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))
        self.proj = torch.nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def rollout(self, samples: list[AudioSample], max_new_tokens: int) -> list[Rollout]:
        token_count = max(1, min(max_new_tokens, 4))
        rollouts = []
        for idx, sample in enumerate(samples):
            base = 1 + (idx % max(self.vocab_size - 2, 1))
            token_ids = torch.tensor(
                [(base + step) % self.vocab_size for step in range(token_count)], dtype=torch.long
            )
            text = sample.text or "toy transcript"
            rollouts.append(Rollout(token_ids=token_ids, text=text, stop_reason="toy_fixed"))
        return rollouts

    def score_rollouts(self, samples: list[AudioSample], rollouts: list[Rollout]) -> torch.Tensor:
        del samples
        max_len = max(int(r.token_ids.numel()) for r in rollouts)
        features = []
        for rollout in rollouts:
            rows = []
            for pos in range(max_len):
                value = float(pos + 1)
                rows.append(torch.full((self.hidden_size,), value / self.hidden_size))
            features.append(torch.stack(rows))
        hidden = torch.stack(features).to(self.bias.device)
        return self.proj(hidden) + self.bias


class ToyTeacherScorer:
    """Deterministic sparse teacher that prefers rollout tokens and neighbors."""

    def __init__(self, vocab_size: int = 16) -> None:
        self.vocab_size = vocab_size

    def score(
        self,
        samples: list[AudioSample],
        rollouts: list[Rollout],
        student_logits: torch.Tensor,
        top_k: int,
    ) -> TeacherScores:
        del samples
        batch, time, _ = student_logits.shape
        k = min(top_k, self.vocab_size)
        device = student_logits.device

        student_logprobs = torch.log_softmax(student_logits, dim=-1)
        student_topk_logprobs, student_topk_ids = torch.topk(student_logprobs, k=k, dim=-1)

        teacher_ids = torch.full((batch, time, k), -1, dtype=torch.long, device=device)
        teacher_lps = torch.full((batch, time, k), -20.0, dtype=student_logits.dtype, device=device)
        student_on_teacher_lps = torch.empty_like(teacher_lps)
        teacher_on_student_lps = torch.empty_like(student_topk_logprobs)
        mask = torch.zeros((batch, time), dtype=torch.bool, device=device)

        for b, rollout in enumerate(rollouts):
            ids = rollout.token_ids.to(device)
            for t in range(min(time, int(ids.numel()))):
                target = int(ids[t].item())
                support = [(target + offset) % self.vocab_size for offset in range(k)]
                support_ids = torch.tensor(support, dtype=torch.long, device=device)
                teacher_ids[b, t] = support_ids
                logits = torch.linspace(0.0, -2.0, steps=k, device=device, dtype=student_logits.dtype)
                teacher_lps[b, t] = torch.log_softmax(logits, dim=-1)
                student_on_teacher_lps[b, t] = student_logprobs[b, t, support_ids]
                mask[b, t] = True
                for j, token_id in enumerate(student_topk_ids[b, t]):
                    distance = min(
                        (int(token_id.item()) - target) % self.vocab_size,
                        (target - int(token_id.item())) % self.vocab_size,
                    )
                    teacher_on_student_lps[b, t, j] = -float(distance)

        return TeacherScores(
            teacher_topk_ids=teacher_ids,
            teacher_topk_logprobs=teacher_lps,
            student_on_teacher_logprobs=student_on_teacher_lps,
            student_topk_ids=student_topk_ids,
            student_topk_logprobs=student_topk_logprobs,
            teacher_on_student_logprobs=teacher_on_student_lps,
            mask=mask,
        )

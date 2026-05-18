from __future__ import annotations

import math

import torch

from open_audio_opd.types import OPDLossOutput, TeacherScores


def _as_batched_positions(scores: TeacherScores) -> tuple[torch.Tensor, ...]:
    tensors = (
        scores.teacher_topk_ids,
        scores.teacher_topk_logprobs,
        scores.student_on_teacher_logprobs,
        scores.student_topk_ids,
        scores.student_topk_logprobs,
        scores.teacher_on_student_logprobs,
    )
    if any(t.ndim != 3 for t in tensors):
        raise ValueError("all score tensors must be shaped [batch, time, top_k]")
    shapes = {tuple(t.shape[:2]) for t in tensors}
    if len(shapes) != 1:
        raise ValueError("all score tensors must share [batch, time] shape")
    if scores.mask is not None and tuple(scores.mask.shape) != tuple(tensors[0].shape[:2]):
        raise ValueError("mask must be shaped [batch, time]")
    return tensors


def union_support_opd_loss(scores: TeacherScores, temperature: float = 1.0) -> OPDLossOutput:
    """Compute sparse KL(teacher || student) over teacher/student union support.

    This intentionally avoids dense full-vocabulary KL. It also avoids assuming
    teacher and student token ids are globally interchangeable outside the
    adapter-provided comparable support.
    """

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    (
        teacher_ids,
        teacher_logprobs,
        student_on_teacher_logprobs,
        student_ids,
        student_logprobs,
        teacher_on_student_logprobs,
    ) = _as_batched_positions(scores)

    device = teacher_logprobs.device
    dtype = teacher_logprobs.dtype
    mask = scores.mask
    if mask is None:
        mask = torch.ones(teacher_ids.shape[:2], dtype=torch.bool, device=device)
    else:
        mask = mask.to(device=device, dtype=torch.bool)

    losses: list[torch.Tensor] = []
    support_sizes: list[int] = []
    skipped = 0

    batch, time, _ = teacher_ids.shape
    neg_inf = torch.tensor(-math.inf, device=device, dtype=dtype)

    for b in range(batch):
        for t in range(time):
            if not bool(mask[b, t].item()):
                skipped += 1
                continue

            support: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

            for idx in range(teacher_ids.shape[-1]):
                token_id = int(teacher_ids[b, t, idx].item())
                if token_id < 0:
                    continue
                support[token_id] = (teacher_logprobs[b, t, idx], student_on_teacher_logprobs[b, t, idx])

            for idx in range(student_ids.shape[-1]):
                token_id = int(student_ids[b, t, idx].item())
                if token_id < 0:
                    continue
                teacher_lp = teacher_on_student_logprobs[b, t, idx]
                student_lp = student_logprobs[b, t, idx]
                old_teacher_lp, _ = support.get(token_id, (teacher_lp, student_lp))
                support[token_id] = (torch.maximum(old_teacher_lp, teacher_lp), student_lp)

            if not support:
                skipped += 1
                continue

            teacher_lp_vec = torch.stack([pair[0] for pair in support.values()]) / temperature
            student_lp_vec = torch.stack([pair[1] for pair in support.values()]) / temperature
            teacher_dist = torch.softmax(teacher_lp_vec, dim=-1)
            student_log_dist = torch.log_softmax(student_lp_vec, dim=-1)
            teacher_log_dist = torch.log_softmax(teacher_lp_vec, dim=-1)
            losses.append(torch.sum(teacher_dist * (teacher_log_dist - student_log_dist)))
            support_sizes.append(len(support))

    if losses:
        loss = torch.stack(losses).mean()
        mean_support = float(sum(support_sizes) / len(support_sizes))
    else:
        loss = torch.zeros((), device=device, dtype=dtype)
        mean_support = 0.0

    total_positions = int(mask.numel())
    valid_positions = len(losses)
    metrics = {
        "opd_valid_positions": float(valid_positions),
        "opd_skipped_positions": float(skipped),
        "opd_valid_ratio": float(valid_positions / max(total_positions, 1)),
        "opd_valid_topk_mean": mean_support,
    }
    return OPDLossOutput(loss=loss, metrics=metrics)

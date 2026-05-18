import torch

from open_audio_opd.losses import union_support_opd_loss
from open_audio_opd.types import TeacherScores


def test_union_support_exceeds_teacher_topk_when_student_adds_tokens() -> None:
    scores = TeacherScores(
        teacher_topk_ids=torch.tensor([[[1, 2]]]),
        teacher_topk_logprobs=torch.log_softmax(torch.tensor([[[2.0, 1.0]]]), dim=-1),
        student_on_teacher_logprobs=torch.log_softmax(torch.tensor([[[0.5, 1.5]]]), dim=-1),
        student_topk_ids=torch.tensor([[[2, 3]]]),
        student_topk_logprobs=torch.log_softmax(torch.tensor([[[1.5, 1.0]]]), dim=-1),
        teacher_on_student_logprobs=torch.log_softmax(torch.tensor([[[1.0, 0.1]]]), dim=-1),
        mask=torch.tensor([[True]]),
    )

    output = union_support_opd_loss(scores)

    assert output.loss.item() >= 0
    assert output.metrics["opd_valid_topk_mean"] == 3.0


def test_loss_skips_masked_positions() -> None:
    scores = TeacherScores(
        teacher_topk_ids=torch.tensor([[[1, 2]]]),
        teacher_topk_logprobs=torch.zeros(1, 1, 2),
        student_on_teacher_logprobs=torch.zeros(1, 1, 2),
        student_topk_ids=torch.tensor([[[2, 3]]]),
        student_topk_logprobs=torch.zeros(1, 1, 2),
        teacher_on_student_logprobs=torch.zeros(1, 1, 2),
        mask=torch.tensor([[False]]),
    )

    output = union_support_opd_loss(scores)

    assert output.loss.item() == 0
    assert output.metrics["opd_valid_positions"] == 0

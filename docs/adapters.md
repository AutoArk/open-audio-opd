# Adapter Guide

The public project does not include private model code. Real audio task support
is added through adapter factories referenced as `module:attr` in YAML. ASR is
the first documented task; TTS should use the same sparse OPD score contract with
task-specific samples and rollout semantics.

## Student Adapter

A student adapter must be a `torch.nn.Module` for the default trainer.

```python
def build_student(model_name_or_path: str, **kwargs):
    return MyStudent(model_name_or_path, **kwargs)
```

Required methods:

- `rollout(samples, max_new_tokens)`: no-grad generation from audio prompts.
- `score_rollouts(samples, rollouts)`: teacher-forced forward with gradients,
  returning logits shaped `[batch, time, vocab]`.

For ASR, the rollout phase should block non-text/audio codec tokens when the
student vocabulary includes them. For TTS, adapters should define which acoustic
or codec token ranges are valid output targets and expose only comparable support
to the OPD loss.

## Teacher Adapter

```python
def build_teacher(model_name_or_path: str, **kwargs):
    return MyTeacher(model_name_or_path, **kwargs)
```

Required method:

- `score(samples, rollouts, student_logits, top_k)`: return `TeacherScores`.

The teacher adapter owns model-specific alignment. For audio-aware teachers,
target positions must be computed from the actual processor output, not from a
plain tokenizer prefix length.

For stable union-support KL, return both cross-support directions:

- student logprobs on teacher top-k ids;
- teacher logprobs on student top-k ids.

## Token Mapping

Do not assume teacher and student token ids match. The adapter should map only
comparable text tokens into the student support. Student-only audio/TTS tokens
should not receive teacher supervision unless the teacher has matching semantics.

## Qwen3-ASR-Style Teachers

If a teacher requires a forced prefix such as `language <lang><asr_text>`, keep
that prefix teacher-only. The student rollout should remain the transcript text.
Compute OPD only on tokens after the teacher text marker.

## Future TTS Adapters

TTS adapters should keep the core contract unchanged:

- student rollout produces text-conditioned acoustic/code/token output;
- teacher scores comparable output positions;
- adapters map teacher/student token spaces before creating `TeacherScores`;
- OPD loss sees only aligned support and does not need to know task-specific
  audio preprocessing details.

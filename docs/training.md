# Training Notes

## Objective

The default objective is online OPD-only:

```text
loss = opd_loss
```

Reference transcripts can be present in the dataset, but the core loop trains
against teacher-scored student rollouts.

## Union Support

Use union support, not teacher-only top-k:

```text
support = teacher_top_k union student_top_k
opd_loss = KL(teacher || student) on support
```

This is important because teacher-only top-k can ignore bad tokens that the
student assigns high probability to.

## Monitoring

Track at least:

- `loss`
- `opd_valid_topk_mean`
- generated token length
- non-empty rollout ratio
- teacher fallback/alignment metrics from real adapters

For `opd_top_k=32`, `opd_valid_topk_mean` should be greater than `32` if union
support is active.

## Multi-Node Pattern

Multi-node launchers should:

- read hosts from a hostfile;
- verify all nodes use the same Python and dependency versions;
- clean stale distributed processes before launch;
- keep per-node logs;
- make all ranks run matching forward/backward collectives;
- export only complete checkpoints.

The v1 repository keeps this as documentation because cluster networking,
interfaces, and model stacks are site-specific.

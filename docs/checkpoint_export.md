# Checkpoint Export

FSDP checkpoints are usually not inference-ready model directories.

Recommended workflow:

1. Pick a complete `global_step_*` checkpoint.
2. Copy the original inference-compatible model folder as a template.
3. Merge all `model_world_size_*_rank_*.pt` shards into the template.
4. Save to a new target directory with the step in its name.
5. Validate by loading the exported model and running a short ASR example.

The CLI command below currently prints the expected contract for stack-specific
exporters:

```bash
open-audio-opd export-fsdp \
  --checkpoint-dir runs/my_run/checkpoints/global_step_1000 \
  --template-model-dir /path/to/base_student \
  --target-dir /path/to/exported_student_step1000
```

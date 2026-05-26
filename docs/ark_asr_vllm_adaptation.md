# Ark-ASR vLLM Adaptation Notes

This note records the practical details needed to serve Ark-ASR checkpoints
with vLLM.

## Target Stack

- Model checkpoint: `/data/yumu/model/trained_model/ark_asr_td_opd`
- Repository: `/data/yumu/open-audio-opd`
- Conda environment: `/root/miniforge3/envs/asr_vlm`
- Verified runtime: Python 3.10, PyTorch 2.9.0+cu128, Transformers 4.57.3,
  vLLM 0.12.0

The vLLM integration lives in `scripts/vllm/ark_asr_vllm/`.

## vLLM 0.12 Compatibility Points

- `BaseDummyInputsBuilder` is imported from `vllm.multimodal.profiling`.
- `MultiModalDataParser` accepts `target_sr`; do not pass `target_channels`.
- Prompt replacement should use `PromptReplacement(..., target="<|audio|>", ...)`
  and return `PromptUpdateDetails.select_token_id(...)`.
- The model class should expose `get_language_model()`.
- Initialize the text model with `init_vllm_registered_model` using
  `architectures=["Qwen2ForCausalLM"]`.
- Do not rely on older `_mark_tower_model` or `_mark_language_model` helpers.

## Architecture Mapping

Ark-ASR uses a Whisper-style audio encoder plus an MLP adapter that injects
audio embeddings into a Qwen2 causal LM at `<|audio|>` placeholder positions.
The vLLM wrapper keeps the same split:

- `audio_encoder.*`: Ark-ASR audio encoder and adapter weights.
- `language_model.model.*`: mapped from Hugging Face `model.*`.
- `language_model.lm_head.*`: mapped from Hugging Face `lm_head.*`.

## Bad Token Masking

ASR decoding must block non-ASR control tokens during generation, matching the
Transformers README example that uses `bad_words_ids`.

The service builds `allowed_token_ids` once at startup:

- Keep eos `<|im_end|>`.
- Block every other `tokenizer.all_special_ids`.
- Block added control tokens whose text matches `<...>`.
- Block token ids `>= 151670` by default.

This prevents tokens such as `<|user|>`, `<|assistant|>`, `<|audio|>`,
`<|begin_of_audio|>`, and `<|end_of_audio|>` from being sampled. Output text is
still normalized as a fallback, but correctness should not depend on cleanup.

Use `/token-mask` to inspect the active mask.

## Service Operations

Deploy the online service:

```bash
cd /data/yumu/open-audio-opd
GPU=2 PORT=8025 scripts/vllm/deploy_ark_asr_vllm_service.sh start
```

Check status:

```bash
scripts/vllm/deploy_ark_asr_vllm_service.sh status
curl -sS http://127.0.0.1:8025/health
curl -sS http://127.0.0.1:8025/token-mask
```

Run an ASR request:

```bash
curl -sS -X POST http://127.0.0.1:8025/asr \
  -F file=@assets/libai.wav \
  -F max_new_tokens=64
```

Stop the service:

```bash
scripts/vllm/deploy_ark_asr_vllm_service.sh stop
```

Logs and PID files are written under `runs/vllm/`.

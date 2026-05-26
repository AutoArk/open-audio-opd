from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from vllm import AsyncLLMEngine, SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs

from .local_hf import load_local_processor
from .register import register

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

logger = logging.getLogger(__name__)

ASR_INSTRUCTION = "Please transcribe this audio."
TURN_END_MARKERS = ("<|user|>", "<|assistant|>", "<|im_end|>")
CONTROL_TOKEN_PATTERN = re.compile(r"^<.*>$")
LEADING_NOISE_PATTERN = re.compile(r"^[\s,.;:!?-]+")


def normalize_asr_text(text: str) -> str:
    if not text:
        return ""
    cut = len(text)
    for marker in TURN_END_MARKERS:
        index = text.find(marker)
        if index != -1 and index < cut:
            cut = index
    text = text[:cut].strip()
    return LEADING_NOISE_PATTERN.sub("", text).strip()


def build_conversation(audio_path: str, begin_time: float, end_time: float) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": audio_path,
                    "begin_time": begin_time,
                    "end_time": end_time,
                },
                {"type": "text", "text": ASR_INSTRUCTION},
            ],
        }
    ]


def build_prompt_text() -> str:
    return (
        "<|user|><|begin_of_audio|><|audio|><|end_of_audio|>"
        f"{ASR_INSTRUCTION}<|assistant|>"
    )


def normalize_token_ids(token_ids: Any) -> list[int]:
    if token_ids is None:
        return []
    if isinstance(token_ids, (list, tuple, set)):
        return [int(token_id) for token_id in token_ids if token_id is not None]
    return [int(token_ids)]


def build_eos_token_ids(tokenizer: Any) -> list[int]:
    ids: list[int] = []
    ids.extend(normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        ids.append(int(im_end_id))
    return list(dict.fromkeys(ids))


def build_blocked_token_ids(
    tokenizer: Any,
    *,
    keep_token_ids: set[int],
    block_from_id: int,
) -> set[int]:
    blocked = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None)
    added = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    for token_id, token_meta in added.items():
        token_id = int(token_id)
        content = getattr(token_meta, "content", None)
        if content is None and isinstance(token_meta, dict):
            content = token_meta.get("content")
        if content and CONTROL_TOKEN_PATTERN.match(str(content)):
            blocked.add(token_id)
    if block_from_id >= 0:
        blocked.update(range(int(block_from_id), int(len(tokenizer))))
    blocked.difference_update(int(token_id) for token_id in keep_token_ids)
    return blocked


def build_allowed_token_ids(
    tokenizer: Any,
    *,
    eos_token_ids: list[int],
    block_from_id: int,
) -> tuple[list[int], set[int]]:
    keep = set(int(token_id) for token_id in eos_token_ids)
    blocked = build_blocked_token_ids(
        tokenizer,
        keep_token_ids=keep,
        block_from_id=block_from_id,
    )
    allowed = [token_id for token_id in range(int(len(tokenizer))) if token_id not in blocked]
    return allowed, blocked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ark-ASR vLLM ASR service")
    parser.add_argument("--model", default=os.getenv("ARK_ASR_MODEL_PATH", "/data/yumu/model/trained_model/ark_asr_td_opd"))
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--dtype", default=os.getenv("VLLM_DTYPE", "bfloat16"))
    parser.add_argument("--max-model-len", type=int, default=int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")))
    parser.add_argument("--max-num-seqs", type=int, default=int(os.getenv("VLLM_MAX_NUM_SEQS", "16")))
    parser.add_argument("--max-num-batched-tokens", type=int, default=int(os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "16384")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85")))
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    parser.add_argument("--enforce-eager", action="store_true", default=os.getenv("VLLM_ENFORCE_EAGER", "0") == "1")
    parser.add_argument("--disable-log-stats", action="store_true", default=os.getenv("VLLM_DISABLE_LOG_STATS", "0") == "1")
    parser.add_argument("--max-audio-seconds", type=int, default=int(os.getenv("ARK_ASR_MAX_AUDIO_SECONDS", "30")))
    parser.add_argument("--sampling-rate", type=int, default=int(os.getenv("ARK_ASR_SAMPLING_RATE", "16000")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("ARK_ASR_MAX_NEW_TOKENS", "256")))
    parser.add_argument("--asr-block-token-id-from", type=int, default=int(os.getenv("ARK_ASR_BLOCK_TOKEN_ID_FROM", "151670")))
    return parser.parse_args()


class AppState:
    args: argparse.Namespace
    processor: Any
    engine: AsyncLLMEngine
    eos_token_ids: list[int]
    allowed_token_ids: list[int]
    blocked_token_ids: set[int]


state = AppState()
processor_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    args = parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    register()
    state.args = args
    state.processor = load_local_processor(args.model)
    tokenizer = state.processor.tokenizer
    state.eos_token_ids = build_eos_token_ids(tokenizer)
    state.allowed_token_ids, state.blocked_token_ids = build_allowed_token_ids(
        tokenizer,
        eos_token_ids=state.eos_token_ids,
        block_from_id=args.asr_block_token_id_from,
    )
    logger.info(
        "Ark-ASR token mask: vocab=%s allowed=%s blocked=%s eos=%s block_from=%s",
        len(tokenizer),
        len(state.allowed_token_ids),
        len(state.blocked_token_ids),
        state.eos_token_ids,
        args.asr_block_token_id_from,
    )

    engine_args = AsyncEngineArgs(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        config_format="arkasr",
        tokenizer_mode="hf",
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        disable_log_stats=args.disable_log_stats,
        enable_log_requests=True,
        limit_mm_per_prompt={"audio": 1},
        enable_mm_embeds=True,
        mm_processor_kwargs={
            "audio_max_length": args.max_audio_seconds * args.sampling_rate,
            "audio_padding": "longest",
            "sampling_rate": args.sampling_rate,
            "text_kwargs": {"padding": "longest"},
        },
    )
    state.engine = AsyncLLMEngine.from_engine_args(engine_args)
    yield


app = FastAPI(title="Ark-ASR vLLM ASR", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/token-mask")
async def token_mask() -> dict[str, Any]:
    tokenizer = state.processor.tokenizer
    probes = [
        "<|user|>",
        "<|assistant|>",
        "<|im_end|>",
        "<|audio|>",
        "<|begin_of_audio|>",
        "<|end_of_audio|>",
    ]
    return {
        "allowed_count": len(state.allowed_token_ids),
        "blocked_count": len(state.blocked_token_ids),
        "eos_token_ids": state.eos_token_ids,
        "tokens": {
            token: {
                "id": tokenizer.convert_tokens_to_ids(token),
                "blocked": tokenizer.convert_tokens_to_ids(token) in state.blocked_token_ids,
                "allowed": tokenizer.convert_tokens_to_ids(token) in state.allowed_token_ids,
            }
            for token in probes
        },
    }


def prepare_prompt(
    audio_path: str,
    begin_time: float,
    end_time: float,
) -> TokensPrompt:
    args = state.args
    inputs = state.processor.apply_chat_template(
        build_conversation(audio_path, begin_time, end_time),
        return_tensors="pt",
        sampling_rate=args.sampling_rate,
        audio_padding="longest",
        add_generation_prompt=True,
        text_kwargs={"padding": "longest"},
        audio_max_length=args.max_audio_seconds * args.sampling_rate,
    )
    if "audios" not in inputs:
        raise RuntimeError(f"Processor output missing audios, keys={list(inputs.keys())}")

    input_ids = state.processor.tokenizer.encode(
        build_prompt_text(),
        add_special_tokens=False,
    )
    audios = inputs["audios"]
    if torch.is_tensor(audios) and audios.ndim == 2:
        audios = audios.unsqueeze(0)

    return TokensPrompt(
        prompt_token_ids=input_ids,
        multi_modal_data={"audio": {"audios": audios}},
    )


async def run_generation(prompt: TokensPrompt, max_new_tokens: int) -> str:
    request_id = f"ark-asr-{uuid.uuid4().hex}"
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop_token_ids=state.eos_token_ids,
        skip_special_tokens=False,
        allowed_token_ids=state.allowed_token_ids,
    )
    final_output = None
    async for output in state.engine.generate(prompt, sampling_params, request_id):
        final_output = output
    if final_output is None or not final_output.outputs:
        return ""
    return normalize_asr_text(final_output.outputs[0].text)


@app.post("/asr")
async def asr(
    file: UploadFile = File(...),
    begin_time: float = Form(-1),
    end_time: float = Form(-1),
    max_new_tokens: int | None = Form(None),
) -> JSONResponse:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    started = time.perf_counter()
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        async with processor_lock:
            prompt = await asyncio.to_thread(
                prepare_prompt,
                tmp_path,
                begin_time,
                end_time,
            )
        text = await run_generation(
            prompt,
            max_new_tokens=max_new_tokens or state.args.max_new_tokens,
        )
        latency = time.perf_counter() - started
        return JSONResponse(
            {
                "text": text,
                "latency_s": latency,
                "prompt_tokens": len(prompt["prompt_token_ids"]),
            }
        )
    except Exception as exc:
        logger.exception("ASR request failed")
        detail = f"{exc.__class__.__name__}: {exc}"
        if exc.__cause__ is not None:
            detail += f"; cause={exc.__cause__.__class__.__name__}: {exc.__cause__}"
        raise HTTPException(status_code=500, detail=detail) from exc
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    response_format: str | None = Form(None),
    temperature: float | None = Form(None),
) -> JSONResponse:
    del model, response_format, temperature
    return await asr(file=file, begin_time=-1, end_time=-1, max_new_tokens=None)


def main() -> None:
    import uvicorn

    args = parse_args()
    uvicorn.run("scripts.vllm.ark_asr_vllm.service:app", host=args.host, port=args.port, factory=False)


if __name__ == "__main__":
    main()

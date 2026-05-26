from __future__ import annotations

import argparse
import asyncio
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

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ARK_ASR_CORS_ALLOW_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


UI_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Ark-ASR vLLM Test</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #1769aa;
      --accent-strong: #0f4f85;
      --danger: #b42318;
      --ok: #167647;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    main {
      width: min(1040px, calc(100vw - 32px));
      margin: 28px auto;
    }
    h1 {
      margin: 0 0 18px;
      font-size: 24px;
      line-height: 1.2;
      font-weight: 700;
      letter-spacing: 0;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 420px);
      gap: 16px;
      align-items: start;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 15px;
      line-height: 1.3;
    }
    label {
      display: block;
      margin: 12px 0 6px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }
    input[type="file"],
    input[type="number"],
    input[type="text"] {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
      color: var(--text);
      font-size: 14px;
    }
    .row {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .buttons {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }
    button {
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 12px;
      background: #fff;
      color: var(--text);
      font-weight: 650;
      cursor: pointer;
    }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    button.primary:hover { background: var(--accent-strong); }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    audio {
      width: 100%;
      margin-top: 12px;
    }
    .status {
      min-height: 22px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }
    .status.ok { color: var(--ok); }
    .status.err { color: var(--danger); }
    .result {
      min-height: 150px;
      white-space: pre-wrap;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      background: #fbfcfe;
      font-size: 16px;
      line-height: 1.55;
    }
    pre {
      overflow: auto;
      min-height: 110px;
      max-height: 320px;
      margin: 12px 0 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #111827;
      color: #e5e7eb;
      font-size: 12px;
      line-height: 1.45;
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 13px;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 9px;
      background: #fff;
    }
    @media (max-width: 820px) {
      main { width: min(100vw - 20px, 720px); margin: 16px auto; }
      .grid { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <h1>Ark-ASR vLLM Test</h1>
    <div class="grid">
      <section>
        <h2>Input</h2>
        <label for="endpoint">Endpoint</label>
        <input id="endpoint" type="text" value="/asr" />

        <label for="fileInput">Audio file</label>
        <input id="fileInput" type="file" accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm" />

        <div class="row">
          <div>
            <label for="beginTime">Begin time</label>
            <input id="beginTime" type="number" value="-1" step="0.1" />
          </div>
          <div>
            <label for="endTime">End time</label>
            <input id="endTime" type="number" value="-1" step="0.1" />
          </div>
          <div>
            <label for="maxTokens">Max tokens</label>
            <input id="maxTokens" type="number" value="256" min="1" max="2048" />
          </div>
        </div>

        <div class="buttons">
          <button id="recordBtn">Start recording</button>
          <button id="stopBtn" disabled>Stop</button>
          <button id="clearBtn">Clear</button>
          <button id="submitBtn" class="primary">Transcribe</button>
        </div>

        <audio id="player" controls></audio>
        <div id="inputStatus" class="status">No audio selected.</div>
      </section>

      <section>
        <h2>Result</h2>
        <div class="meta">
          <div class="pill" id="latency">latency: -</div>
          <div class="pill" id="promptTokens">prompt tokens: -</div>
        </div>
        <div id="result" class="result"></div>
        <pre id="raw">{}</pre>
      </section>
    </div>
  </main>

  <script>
    const fileInput = document.getElementById("fileInput");
    const endpoint = document.getElementById("endpoint");
    const beginTime = document.getElementById("beginTime");
    const endTime = document.getElementById("endTime");
    const maxTokens = document.getElementById("maxTokens");
    const recordBtn = document.getElementById("recordBtn");
    const stopBtn = document.getElementById("stopBtn");
    const clearBtn = document.getElementById("clearBtn");
    const submitBtn = document.getElementById("submitBtn");
    const player = document.getElementById("player");
    const inputStatus = document.getElementById("inputStatus");
    const result = document.getElementById("result");
    const raw = document.getElementById("raw");
    const latency = document.getElementById("latency");
    const promptTokens = document.getElementById("promptTokens");

    let selectedBlob = null;
    let selectedName = "audio.webm";
    let objectUrl = "";
    let recorder = null;
    let chunks = [];
    let startedAt = 0;

    function setStatus(text, kind = "") {
      inputStatus.textContent = text;
      inputStatus.className = "status" + (kind ? " " + kind : "");
    }

    function setAudio(blob, name) {
      selectedBlob = blob;
      selectedName = name || "audio.webm";
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(blob);
      player.src = objectUrl;
      setStatus(`${selectedName} (${Math.round(blob.size / 1024)} KB)`, "ok");
    }

    fileInput.addEventListener("change", () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) return;
      setAudio(file, file.name || "audio");
    });

    recordBtn.addEventListener("click", async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        chunks = [];
        recorder = new MediaRecorder(stream);
        recorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) chunks.push(event.data);
        };
        recorder.onstop = () => {
          const mime = recorder.mimeType || "audio/webm";
          const blob = new Blob(chunks, { type: mime });
          stream.getTracks().forEach((track) => track.stop());
          setAudio(blob, "microphone.webm");
          recordBtn.disabled = false;
          stopBtn.disabled = true;
        };
        startedAt = Date.now();
        recorder.start();
        recordBtn.disabled = true;
        stopBtn.disabled = false;
        setStatus("Recording...", "");
      } catch (error) {
        setStatus(error.message || String(error), "err");
      }
    });

    stopBtn.addEventListener("click", () => {
      if (recorder && recorder.state !== "inactive") {
        recorder.stop();
        setStatus(`Recorded ${Math.max(1, Math.round((Date.now() - startedAt) / 1000))}s`, "ok");
      }
    });

    clearBtn.addEventListener("click", () => {
      selectedBlob = null;
      selectedName = "audio.webm";
      fileInput.value = "";
      player.removeAttribute("src");
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = "";
      result.textContent = "";
      raw.textContent = "{}";
      latency.textContent = "latency: -";
      promptTokens.textContent = "prompt tokens: -";
      setStatus("No audio selected.");
    });

    submitBtn.addEventListener("click", async () => {
      if (!selectedBlob) {
        setStatus("Select a file or record from the microphone first.", "err");
        return;
      }
      submitBtn.disabled = true;
      setStatus("Uploading and transcribing...", "");
      result.textContent = "";
      raw.textContent = "{}";
      const started = performance.now();
      try {
        const form = new FormData();
        form.append("file", selectedBlob, selectedName);
        form.append("begin_time", beginTime.value || "-1");
        form.append("end_time", endTime.value || "-1");
        form.append("max_new_tokens", maxTokens.value || "256");
        const response = await fetch(endpoint.value || "/asr", {
          method: "POST",
          body: form,
        });
        const text = await response.text();
        let data;
        try {
          data = JSON.parse(text);
        } catch {
          throw new Error(text || `HTTP ${response.status}`);
        }
        if (!response.ok) {
          throw new Error(data.detail || `HTTP ${response.status}`);
        }
        result.textContent = data.text || "";
        raw.textContent = JSON.stringify(data, null, 2);
        latency.textContent = `latency: ${Number(data.latency_s || 0).toFixed(3)}s`;
        promptTokens.textContent = `prompt tokens: ${data.prompt_tokens ?? "-"}`;
        setStatus(`Done in ${((performance.now() - started) / 1000).toFixed(3)}s`, "ok");
      } catch (error) {
        result.textContent = "";
        raw.textContent = JSON.stringify({ error: error.message || String(error) }, null, 2);
        setStatus(error.message || String(error), "err");
      } finally {
        submitBtn.disabled = false;
      }
    });
  </script>
</body>
</html>"""


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(UI_HTML)


@app.get("/ui", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    return HTMLResponse(UI_HTML)


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


def clamp_sampling_value(name: str, value: float, minimum: float, maximum: float) -> float:
    if value < minimum or value > maximum:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be between {minimum} and {maximum}, got {value}",
        )
    return value


async def run_generation(
    prompt: TokensPrompt,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    request_id = f"ark-asr-{uuid.uuid4().hex}"
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
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
    temperature: float = Form(0.0),
    top_p: float = Form(1.0),
    top_k: int = Form(-1),
    repetition_penalty: float = Form(1.0),
) -> JSONResponse:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    started = time.perf_counter()
    tmp_path = ""
    try:
        max_tokens = max_new_tokens or state.args.max_new_tokens
        if max_tokens <= 0:
            raise HTTPException(status_code=400, detail=f"max_new_tokens must be positive, got {max_tokens}")
        temperature = clamp_sampling_value("temperature", temperature, 0.0, 2.0)
        top_p = clamp_sampling_value("top_p", top_p, 0.0, 1.0)
        if top_k < -1 or top_k == 0:
            raise HTTPException(status_code=400, detail=f"top_k must be -1 or a positive integer, got {top_k}")
        repetition_penalty = clamp_sampling_value("repetition_penalty", repetition_penalty, 0.1, 2.0)

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
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        latency = time.perf_counter() - started
        return JSONResponse(
            {
                "text": text,
                "latency_s": latency,
                "prompt_tokens": len(prompt["prompt_token_ids"]),
                "sampling": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                },
            }
        )
    except HTTPException:
        raise
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
    max_new_tokens: int | None = Form(None),
    top_p: float = Form(1.0),
    top_k: int = Form(-1),
    repetition_penalty: float = Form(1.0),
) -> JSONResponse:
    del model, response_format
    return await asr(
        file=file,
        begin_time=-1,
        end_time=-1,
        max_new_tokens=max_new_tokens,
        temperature=0.0 if temperature is None else temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )


def main() -> None:
    import uvicorn

    args = parse_args()
    uvicorn.run("scripts.vllm.ark_asr_vllm.service:app", host=args.host, port=args.port, factory=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

SPECIAL_TOKEN_PATTERN = re.compile(
    r"<\|(?:"
    r"bicodec_(?:semantic|global)_\d+|"
    r"(?:start|end)_(?:global_token|glm_token|semantic_token|content)"
    r")\|>"
)
TURN_END_MARKERS = ("<|user|>", "<|assistant|>", "<|im_end|>")
LEADING_NOISE_PATTERN = re.compile(r"^[\s,.;:!?-]+")
CONTROL_TOKEN_PATTERN = re.compile(r"^<.*>$")


@dataclass
class AsrRecord:
    audio: str
    text: str = ""
    begin_time: float = -1.0
    end_time: float = -1.0
    metadata: dict[str, Any] | None = None


class BlockTokenIdsFromLogitsProcessor(LogitsProcessor):
    """Mask token ids >= block_from_id and explicit token ids during ASR generation."""

    def __init__(self, block_from_id: int | None, block_token_ids: Iterable[int] | None = None):
        self.block_from_id = None if block_from_id is None or int(block_from_id) < 0 else int(block_from_id)
        self.block_token_ids = sorted(set(int(token_id) for token_id in (block_token_ids or [])))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_size = scores.shape[-1]
        if self.block_from_id is not None and self.block_from_id < vocab_size:
            scores[:, self.block_from_id :] = -float("inf")
        valid_token_ids = [token_id for token_id in self.block_token_ids if 0 <= token_id < vocab_size]
        if valid_token_ids:
            scores[:, valid_token_ids] = -float("inf")
        return scores


def truncate_generation_text(text: str) -> str:
    if not text:
        return ""
    cut = len(text)
    for marker in TURN_END_MARKERS:
        index = text.find(marker)
        if index != -1 and index < cut:
            cut = index
    return text[:cut].strip()


def remove_special_tokens(text: str) -> str:
    if not text:
        return ""
    if "<|text|>" in text:
        text = text.split("<|text|>", 1)[1]
    return SPECIAL_TOKEN_PATTERN.sub("", text).strip()


def normalize_prediction_text(text: str) -> str:
    if not text:
        return ""
    text = truncate_generation_text(text)
    text = remove_special_tokens(text)
    text = re.sub(r"\s+", " ", text).strip()
    return LEADING_NOISE_PATTERN.sub("", text).strip()


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "keys") and hasattr(value, "__getitem__"):
        return {key: value[key] for key in value.keys()}
    raise TypeError(f"Unexpected processor output type: {type(value)}")


def normalize_token_ids(token_ids: Any) -> list[int]:
    if token_ids is None:
        return []
    if isinstance(token_ids, (list, tuple, set)):
        return [int(token_id) for token_id in token_ids if token_id is not None]
    return [int(token_ids)]


def build_eos_token_ids(tokenizer: Any) -> list[int]:
    eos_ids = []
    eos_ids.extend(normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))
    for marker in TURN_END_MARKERS:
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(token_id, int) and token_id >= 0:
            eos_ids.append(int(token_id))
    return list(dict.fromkeys(eos_ids))


def build_asr_keep_token_ids(model: Any, tokenizer: Any) -> list[int]:
    keep_token_ids = set()
    keep_token_ids.update(normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))
    keep_token_ids.update(normalize_token_ids(getattr(getattr(model, "config", None), "eos_token_id", None)))
    keep_token_ids.update(
        normalize_token_ids(getattr(getattr(model, "generation_config", None), "eos_token_id", None))
    )
    return sorted(keep_token_ids)


def build_asr_extra_block_token_ids(
    tokenizer: Any,
    keep_token_ids: Iterable[int] | None = None,
    block_from_id: int | None = None,
) -> list[int]:
    keep = set(int(token_id) for token_id in (keep_token_ids or []))
    max_control_token_id = None if block_from_id is None or int(block_from_id) < 0 else int(block_from_id)
    block_token_ids = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None)
    added_tokens_decoder = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    for token_id, token_meta in added_tokens_decoder.items():
        token_id = int(token_id)
        if max_control_token_id is not None and token_id >= max_control_token_id:
            continue
        token_content = getattr(token_meta, "content", None)
        if token_content is None and isinstance(token_meta, dict):
            token_content = token_meta.get("content")
        if token_content and CONTROL_TOKEN_PATTERN.match(token_content):
            block_token_ids.add(token_id)
    block_token_ids.difference_update(keep)
    return sorted(block_token_ids)


def apply_audio_gain(audios: torch.Tensor, gain: float) -> torch.Tensor:
    if not torch.is_tensor(audios):
        return audios
    gain = float(gain)
    if gain == 1.0:
        return audios
    return torch.clamp(audios.float() * gain, min=-1.0, max=1.0)


def resolve_torch_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    if device != "cuda" and mapping[dtype_name] != torch.float32:
        print(f"Warning: dtype={dtype_name} on {device} is not well supported. Falling back to float32.")
        return torch.float32
    return mapping[dtype_name]


def load_model(model_path: str, device: str, torch_dtype: torch.dtype, attn_impl: str):
    if attn_impl == "auto":
        candidates = ["flash_attention_2", "sdpa"] if device == "cuda" else ["eager"]
    else:
        candidates = [attn_impl]
        if attn_impl == "flash_attention_2":
            candidates.extend(["sdpa", "eager"] if device == "cuda" else ["eager"])

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                attn_implementation=candidate,
            ).to(device)
            model.eval()
            return model, candidate
        except (ImportError, RuntimeError, ValueError) as exc:
            message = str(exc)
            can_fallback = candidate == "flash_attention_2" and (
                "flash_attn" in message or "FlashAttention" in message
            )
            if not can_fallback:
                raise
            print(f"Warning: attn_impl={candidate} unavailable ({message.splitlines()[0]}). Falling back.")
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to load model with any attention implementation.")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: JSONL record must be an object")
            records.append(record)
    return records


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunked(items: list[Any], batch_size: int):
    for offset in range(0, len(items), batch_size):
        yield items[offset : offset + batch_size]


def parse_record(
    item: dict[str, Any],
    audio_field: str,
    text_field: str,
    begin_field: str,
    end_field: str,
) -> AsrRecord:
    audio = item.get(audio_field)
    if not isinstance(audio, str) or not audio:
        raise ValueError(f"Input record missing audio field {audio_field!r}: {item}")
    text = item.get(text_field, "")
    metadata = {key: value for key, value in item.items() if key not in {audio_field, text_field, begin_field, end_field}}
    return AsrRecord(
        audio=audio,
        text=str(text) if text is not None else "",
        begin_time=float(item.get(begin_field, -1)),
        end_time=float(item.get(end_field, -1)),
        metadata=metadata,
    )


class ArkAsrTransformerInferencer:
    def __init__(
        self,
        model_path: str,
        processor_path: str | None = None,
        *,
        dtype: str = "auto",
        attn_impl: str = "auto",
        padding_side: str = "left",
        asr_block_token_id_from: int = 151670,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = resolve_torch_dtype(dtype, self.device)
        self.model, self.resolved_attn_impl = load_model(model_path, self.device, self.torch_dtype, attn_impl)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = padding_side
        self.processor = AutoProcessor.from_pretrained(
            processor_path or model_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        if hasattr(self.processor, "tokenizer"):
            if self.processor.tokenizer.pad_token_id is None:
                self.processor.tokenizer.pad_token_id = self.tokenizer.pad_token_id
            self.processor.tokenizer.padding_side = padding_side
        self.eos_token_ids = build_eos_token_ids(self.tokenizer)
        keep_token_ids = build_asr_keep_token_ids(self.model, self.tokenizer)
        self.extra_block_token_ids = build_asr_extra_block_token_ids(
            self.tokenizer,
            keep_token_ids=keep_token_ids,
            block_from_id=asr_block_token_id_from,
        )
        self.asr_block_token_id_from = asr_block_token_id_from

    def infer_batch(
        self,
        records: list[AsrRecord],
        *,
        target_sr: int = 16000,
        max_audio_seconds: int = 40,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float | None = 0.5,
        repetition_penalty: float | None = 1.4,
        audio_gain: float = 1.0,
    ) -> list[dict[str, Any]]:
        conversations = []
        for record in records:
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio",
                                "path": record.audio,
                                "begin_time": record.begin_time,
                                "end_time": record.end_time,
                            },
                            {"type": "text", "text": "Please transcribe this audio."},
                        ],
                    }
                ]
            )

        inputs_raw = self.processor.apply_chat_template(
            conversations,
            return_tensors="pt",
            sampling_rate=target_sr,
            audio_padding="longest",
            add_generation_prompt=True,
            text_kwargs={"padding": "longest"},
            audio_max_length=int(max_audio_seconds * target_sr),
        )
        if torch.is_tensor(inputs_raw):
            raise RuntimeError("ASR apply_chat_template returned Tensor-only; audio was not encoded.")
        inputs = as_dict(inputs_raw)
        if "audios" not in inputs:
            raise RuntimeError(f"ASR inputs missing 'audios'; processor keys={list(inputs.keys())}")
        if "attention_mask" not in inputs and "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long)

        for key, value in list(inputs.items()):
            if not torch.is_tensor(value):
                continue
            if key == "audios":
                inputs[key] = apply_audio_gain(value, audio_gain).to(device=self.device, dtype=self.torch_dtype)
            else:
                inputs[key] = value.to(self.device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.eos_token_ids:
            generate_kwargs["eos_token_id"] = self.eos_token_ids
        if do_sample and temperature is not None:
            generate_kwargs["temperature"] = temperature
        if repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = repetition_penalty
        if self.asr_block_token_id_from >= 0 or self.extra_block_token_ids:
            generate_kwargs["logits_processor"] = LogitsProcessorList(
                [
                    BlockTokenIdsFromLogitsProcessor(
                        block_from_id=self.asr_block_token_id_from,
                        block_token_ids=self.extra_block_token_ids,
                    )
                ]
            )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        rows = []
        input_ids = inputs["input_ids"]
        for index, output in enumerate(outputs):
            generated_ids = output[len(input_ids[index].tolist()) :]
            prediction_raw = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            prediction = normalize_prediction_text(prediction_raw)
            record = records[index]
            row = {
                "audio": record.audio,
                "text": record.text,
                "pred_text": prediction,
                "pred_text_raw": prediction_raw,
                "begin_time": record.begin_time,
                "end_time": record.end_time,
            }
            if record.metadata:
                row.update(record.metadata)
            rows.append(row)
        return rows


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Input JSONL. No eval data is bundled in this repo.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--model_path", required=True, help="Model path or HF repo id.")
    parser.add_argument("--processor_path", default=None, help="Processor path. Defaults to --model_path.")
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.4)
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--max_audio_seconds", type=int, default=40)
    parser.add_argument("--audio_gain", type=float, default=1.0)
    parser.add_argument("--asr_block_token_id_from", type=int, default=151670)
    parser.add_argument("--padding_side", choices=["left", "right"], default="left")
    parser.add_argument("--attn_impl", choices=["auto", "flash_attention_2", "eager", "sdpa"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--audio_field", default="audio")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--begin_field", default="begin_time")
    parser.add_argument("--end_field", default="end_time")


def run_infer(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    inferencer = ArkAsrTransformerInferencer(
        args.model_path,
        args.processor_path,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        padding_side=args.padding_side,
        asr_block_token_id_from=args.asr_block_token_id_from,
    )
    records = [
        parse_record(item, args.audio_field, args.text_field, args.begin_field, args.end_field)
        for item in load_jsonl(input_path)
    ]
    print(
        f"Loaded {len(records):,} samples from {input_path}; "
        f"device={inferencer.device}; attn={inferencer.resolved_attn_impl}"
    )
    for batch in tqdm(list(chunked(records, args.batch_size)), desc="ASR Infer", unit="batch"):
        rows = inferencer.infer_batch(
            batch,
            target_sr=args.target_sr,
            max_audio_seconds=args.max_audio_seconds,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            audio_gain=args.audio_gain,
        )
        append_jsonl(output_path, rows)
    print(f"Saved inference output to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ark ASR Transformers inference")
    add_common_args(parser)
    run_infer(parser.parse_args())


if __name__ == "__main__":
    main()

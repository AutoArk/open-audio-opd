#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
INFER_DIR = REPO_ROOT / "scripts" / "infer"
if str(INFER_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_DIR))

from ark_asr_transformers import (  # noqa: E402
    ArkAsrTransformerInferencer,
    AsrRecord,
    add_common_args,
    append_jsonl,
    chunked,
    load_jsonl,
    parse_record,
    write_jsonl,
)

try:
    import jiwer  # type: ignore
except Exception:
    jiwer = None

PUNCT_PATTERN = re.compile(r"""[，。！？、,.!?;:"'（）()【】\[\]<>《》{}…—\-~·`@#$%^&*_+=|\\/]""")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
ASCII_DIGIT_PATTERN = re.compile(r"\d")
FULLWIDTH_TRANSLATION = str.maketrans(
    {
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
        "％": "%",
        "．": ".",
        "－": "-",
        "＋": "+",
        "＝": "=",
        "：": ":",
        "／": "/",
    }
)


class TextNormalizer:
    backend_name = "identity"

    def normalize(self, text: str) -> str:
        return text

    def close(self) -> None:
        return None


class LocalTextProcessNormalizer(TextNormalizer):
    backend_name = "text_process"

    def __init__(self, normalize_fn):
        self.normalize_fn = normalize_fn

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        return self.normalize_fn(text)


class ExternalTextProcessNormalizer(TextNormalizer):
    backend_name = "external_text_process"

    def __init__(self, python_bin: str, workdir: str) -> None:
        worker_code = """
import json
from text_process import text_normalize

while True:
    try:
        raw = input()
    except EOFError:
        break
    if not raw:
        print(json.dumps("", ensure_ascii=False), flush=True)
        continue
    text = json.loads(raw)
    normalized = text_normalize(text)
    print(json.dumps(normalized, ensure_ascii=False), flush=True)
"""
        self.process = subprocess.Popen(
            [python_bin, "-u", "-c", worker_code],
            cwd=workdir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        atexit.register(self.close)

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        if self.process.poll() is not None:
            raise RuntimeError("External text normalization worker exited unexpectedly")
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(json.dumps(text, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("External text normalization worker returned no output")
        return json.loads(line)

    def close(self) -> None:
        if getattr(self, "process", None) is None:
            return
        if self.process.poll() is None:
            try:
                if self.process.stdin is not None:
                    self.process.stdin.close()
            except Exception:
                pass
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except Exception:
                self.process.kill()


def build_text_normalizer(disable: bool, external_python: str | None, external_workdir: str | None) -> TextNormalizer:
    if disable:
        return TextNormalizer()
    try:
        from text_process import text_normalize  # type: ignore[import-not-found]

        return LocalTextProcessNormalizer(text_normalize)
    except Exception as exc:
        print(f"Warning: failed to import text_process ({exc}).")
    if external_python:
        return ExternalTextProcessNormalizer(external_python, external_workdir or os.getcwd())
    print("Warning: text normalization fallback is identity. Pass --text_normalize_python for text_process.")
    return TextNormalizer()


def contains_cjk(text: str) -> bool:
    return bool(CJK_PATTERN.search(text))


def has_arabic_digits(text: str) -> bool:
    if not text:
        return False
    return ASCII_DIGIT_PATTERN.search(text.translate(FULLWIDTH_TRANSLATION)) is not None


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = PUNCT_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_numeric_text(text: str) -> str:
    return text.translate(FULLWIDTH_TRANSLATION)


def normalize_for_scoring(text: str, normalizer: TextNormalizer) -> str:
    if not text:
        return ""
    return clean_text(canonicalize_numeric_text(normalizer.normalize(text)))


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    if ref_len == 0:
        return hyp_len
    if hyp_len == 0:
        return ref_len
    previous = list(range(hyp_len + 1))
    current = [0] * (hyp_len + 1)
    for ref_index in range(1, ref_len + 1):
        current[0] = ref_index
        ref_token = reference[ref_index - 1]
        for hyp_index in range(1, hyp_len + 1):
            cost = 0 if ref_token == hypothesis[hyp_index - 1] else 1
            current[hyp_index] = min(
                previous[hyp_index] + 1,
                current[hyp_index - 1] + 1,
                previous[hyp_index - 1] + cost,
            )
        previous, current = current, previous
    return previous[hyp_len]


def compute_wer_errors(reference: str, hypothesis: str) -> tuple[float, int]:
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    reference_words = len(reference_tokens)
    if reference_words == 0:
        return 0.0, 0
    if jiwer is not None:
        return float(jiwer.wer(reference, hypothesis) * reference_words), reference_words
    return float(edit_distance(reference_tokens, hypothesis_tokens)), reference_words


def compute_cer_errors(reference: str, hypothesis: str) -> tuple[float, int]:
    reference_chars_list = list(reference.replace(" ", ""))
    hypothesis_chars_list = list(hypothesis.replace(" ", ""))
    reference_chars = len(reference_chars_list)
    if reference_chars == 0:
        return 0.0, 0
    if jiwer is not None:
        return float(jiwer.cer(reference, hypothesis) * reference_chars), reference_chars
    return float(edit_distance(reference_chars_list, hypothesis_chars_list)), reference_chars


def read_jsonl_all(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return load_jsonl(path)


def score_rows(rows: list[dict[str, Any]], normalizer: TextNormalizer) -> tuple[list[dict[str, Any]], dict[str, float]]:
    scored = []
    valid = 0
    total_wer_errors = 0.0
    total_cer_errors = 0.0
    total_ref_words = 0
    total_ref_chars = 0
    for row in rows:
        reference = str(row.get("text", row.get("ref_text", "")))
        prediction = str(row.get("pred_text", ""))
        reference_clean = normalize_for_scoring(reference, normalizer)
        prediction_clean = normalize_for_scoring(prediction, normalizer)
        cer_errors, ref_chars = compute_cer_errors(reference_clean, prediction_clean)
        wer_errors, ref_words = compute_wer_errors(reference_clean, prediction_clean)
        if contains_cjk(reference):
            ref_words = ref_chars
            wer_errors = cer_errors
        out = {
            **row,
            "ref_text": reference,
            "ref_text_clean": reference_clean,
            "pred_text_clean": prediction_clean,
            "cer_errors": float(cer_errors),
            "wer_errors": float(wer_errors),
            "ref_words": int(ref_words),
            "ref_chars": int(ref_chars),
        }
        scored.append(out)
        if ref_words > 0 and ref_chars > 0:
            valid += 1
            total_wer_errors += wer_errors
            total_cer_errors += cer_errors
            total_ref_words += ref_words
            total_ref_chars += ref_chars
    metrics = {
        "valid_samples": float(valid),
        "total_ref_words": float(total_ref_words),
        "total_ref_chars": float(total_ref_chars),
        "wer": total_wer_errors / max(1, total_ref_words),
        "cer": total_cer_errors / max(1, total_ref_chars),
    }
    return scored, metrics


def float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def run_eval(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(args.tmp_output) if args.tmp_output else output_path.with_suffix(output_path.suffix + ".tmp_unsorted.jsonl")
    if tmp_path.exists():
        tmp_path.unlink()

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
    normalizer = build_text_normalizer(
        args.disable_text_normalize,
        args.text_normalize_python,
        args.text_normalize_workdir,
    )
    print(
        f"Loaded {len(records):,} samples from {input_path}; "
        f"device={inferencer.device}; attn={inferencer.resolved_attn_impl}; "
        f"text_normalizer={normalizer.backend_name}; jiwer={jiwer is not None}"
    )

    try:
        with tqdm(total=len(records), desc="ASR Eval", unit="utt") as progress:
            for batch in chunked(records, args.batch_size):
                try:
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
                except Exception as exc:
                    rows = [
                        {
                            "audio": record.audio,
                            "text": record.text,
                            "pred_text": "",
                            "error": str(exc),
                            "begin_time": record.begin_time,
                            "end_time": record.end_time,
                            **(record.metadata or {}),
                        }
                        for record in batch
                    ]
                scored, _ = score_rows(rows, normalizer)
                append_jsonl(tmp_path, scored)
                progress.update(len(batch))

        all_rows = read_jsonl_all(tmp_path)
        all_rows.sort(key=lambda row: float_or(row.get("cer_errors"), -1.0), reverse=True)
        write_jsonl(output_path, all_rows)
        _, metrics = score_rows(all_rows, normalizer)
        print("\nDone")
        print(f"   Valid samples: {int(metrics['valid_samples'])}")
        print(f"   Total Ref Words: {int(metrics['total_ref_words']):,}")
        print(f"   Total Ref Chars: {int(metrics['total_ref_chars']):,}")
        print(f"   Final WER: {metrics['wer'] * 100:.2f}%")
        print(f"   Final CER: {metrics['cer'] * 100:.2f}%")
        print(f"   Saved sorted result to: {output_path}")
        print(f"   Temp unsorted cache: {tmp_path}")
    finally:
        normalizer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ark ASR Transformers J/WER evaluation")
    add_common_args(parser)
    parser.add_argument("--tmp_output", default=None)
    parser.add_argument("--disable_text_normalize", action="store_true")
    parser.add_argument("--text_normalize_python", default=None)
    parser.add_argument("--text_normalize_workdir", default=None)
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()

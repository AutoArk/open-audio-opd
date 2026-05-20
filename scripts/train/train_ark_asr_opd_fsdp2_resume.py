#!/usr/bin/env python
import argparse
import json
import math
import os
import random
import re
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_VERL_ROOT = REPO_ROOT / "verl"
if VENDORED_VERL_ROOT.exists() and str(VENDORED_VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_VERL_ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
)


CONTROL_TOKEN_PATTERN = re.compile(r"^<.*>$")
QWEN3_ASR_TEXT_MARKER = "<asr_text>"
ASR_INSTRUCTION = "Please transcribe this audio."
IGNORE_INDEX = -100
QWEN3_ASR_CODE_PATH = ""
QWEN3_ASR_LANG_BY_CODE = {
    "ZH": "Chinese",
    "CN": "Chinese",
    "CMN": "Chinese",
    "YUE": "Cantonese",
    "HK": "Cantonese",
    "EN": "English",
    "JA": "Japanese",
    "JP": "Japanese",
    "KO": "Korean",
    "KR": "Korean",
    "FR": "French",
    "DE": "German",
    "ES": "Spanish",
    "PT": "Portuguese",
    "ID": "Indonesian",
    "IT": "Italian",
    "RU": "Russian",
    "AR": "Arabic",
    "FA": "Persian",
    "TH": "Thai",
    "VI": "Vietnamese",
    "TR": "Turkish",
    "HI": "Hindi",
    "MS": "Malay",
    "NL": "Dutch",
    "SV": "Swedish",
    "DA": "Danish",
    "FI": "Finnish",
    "PL": "Polish",
    "CS": "Czech",
    "TL": "Filipino",
    "FIL": "Filipino",
    "PH": "Filipino",
    "EL": "Greek",
    "HU": "Hungarian",
    "MK": "Macedonian",
    "RO": "Romanian",
}
QWEN3_ASR_SUPPORTED_LANGUAGES = {
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
}
VALID_QWEN3_ASR_LANGUAGES = QWEN3_ASR_SUPPORTED_LANGUAGES
QWEN3_ASR_LATIN_LANGUAGES = {
    "English",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Vietnamese",
    "Turkish",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Hungarian",
    "Romanian",
}
QWEN3_ASR_LATIN_WORD_RE = re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF']+")
QWEN3_ASR_LATIN_WORD_HINTS = {
    "English": {
        "a", "an", "and", "are", "as", "be", "but", "for", "have", "he", "i", "in", "is", "it",
        "not", "of", "on", "she", "that", "the", "they", "this", "to", "was", "we", "were",
        "with", "you",
    },
    "French": {
        "au", "aux", "bonjour", "ce", "des", "dans", "du", "elle", "est", "et", "il", "je",
        "la", "le", "les", "merci", "nous", "pas", "pour", "que", "qui", "un", "une", "vous",
    },
    "German": {
        "auf", "das", "dem", "den", "der", "die", "ein", "eine", "guten", "ich", "ist", "mit",
        "morgen", "nicht", "sie", "und", "wir", "zu",
    },
    "Spanish": {
        "como", "con", "de", "el", "en", "es", "esta", "hola", "la", "las", "los", "no",
        "para", "por", "que", "se", "un", "una", "y",
    },
    "Portuguese": {
        "as", "com", "da", "de", "do", "e", "em", "esta", "nao", "o", "os", "para", "por",
        "que", "se", "um", "uma",
    },
    "Italian": {
        "che", "con", "del", "della", "di", "e", "gli", "il", "la", "le", "lo", "non", "per",
        "sono", "un", "una",
    },
    "Dutch": {
        "dat", "de", "een", "en", "het", "ik", "is", "je", "met", "niet", "op", "te", "van",
        "voor",
    },
    "Swedish": {"att", "det", "en", "for", "inte", "jag", "med", "och", "pa", "som"},
    "Danish": {"det", "en", "er", "for", "ikke", "jeg", "med", "og", "pa", "som"},
    "Finnish": {"ei", "etta", "ja", "kanssa", "minä", "mutta", "on", "se"},
    "Polish": {"ale", "co", "do", "i", "jest", "nie", "po", "się", "to", "w", "z", "że"},
    "Czech": {"a", "ale", "co", "do", "je", "jsem", "na", "ne", "se", "to", "v", "ze"},
    "Turkish": {"bir", "bu", "da", "de", "icin", "ile", "mi", "ve", "yok"},
    "Hungarian": {"a", "az", "egy", "es", "hogy", "is", "meg", "nem", "van"},
    "Romanian": {"acest", "ca", "cu", "de", "este", "eu", "la", "nu", "o", "pe", "si"},
    "Indonesian": {"adalah", "akan", "dan", "dari", "dengan", "di", "ini", "itu", "ke", "saya", "tidak", "untuk", "yang"},
    "Malay": {"adalah", "dan", "dengan", "di", "ini", "itu", "ke", "saya", "tidak", "untuk", "yang"},
    "Filipino": {"ako", "ang", "ay", "ito", "ikaw", "mga", "ng", "sa", "si", "wala"},
    "Vietnamese": {"anh", "bao", "cam", "cho", "chao", "cua", "khong", "la", "mot", "nguoi", "toi", "va"},
}
QWEN3_ASR_LATIN_CHAR_HINTS = {
    "French": "\u00e0\u00e2\u00e6\u00e7\u00e9\u00e8\u00ea\u00eb\u00ee\u00ef\u00f4\u00f9\u00fb\u00fc\u00ff\u0153",
    "German": "\u00e4\u00f6\u00fc\u00df",
    "Spanish": "\u00e1\u00e9\u00ed\u00f1\u00f3\u00fa\u00fc\u00bf\u00a1",
    "Portuguese": "\u00e1\u00e2\u00e3\u00e0\u00e7\u00e9\u00ea\u00ed\u00f3\u00f4\u00f5\u00fa",
    "Italian": "\u00e0\u00e8\u00e9\u00ec\u00f2\u00f9",
    "Vietnamese": "\u0103\u00e2\u0111\u00ea\u00f4\u01a1\u01b0\u00e0\u1ea3\u00e3\u00e1\u1ea1\u1eb1\u1eb3\u1eb5\u1eaf\u1eb7\u1ea7\u1ea9\u1eab\u1ea5\u1ead\u1ec1\u1ec3\u1ec5\u1ebf\u1ec7\u1ed3\u1ed5\u1ed7\u1ed1\u1ed9\u1edd\u1edf\u1ee1\u1edb\u1ee3\u1eeb\u1eed\u1eef\u1ee9\u1ef1\u1ef3\u1ef7\u1ef9\u00fd\u1ef5",
    "Turkish": "\u00e7\u011f\u0131\u00f6\u015f\u00fc",
    "Dutch": "\u00e1\u00e9\u00eb\u00ef\u00f3\u00f6\u00fc",
    "Swedish": "\u00e5\u00e4\u00f6",
    "Danish": "\u00e6\u00f8\u00e5",
    "Finnish": "\u00e4\u00f6",
    "Polish": "\u0105\u0107\u0119\u0142\u0144\u00f3\u015b\u017a\u017c",
    "Czech": "\u010d\u010f\u011b\u0148\u0159\u0161\u0165\u016f\u00fd\u017e",
    "Hungarian": "\u00e1\u00e9\u00ed\u00f3\u00f6\u0151\u00fa\u00fc\u0171",
    "Romanian": "\u0103\u00e2\u00ee\u0219\u015f\u021b\u0163",
}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def rank0_print(*args):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, flush=True)


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    return rank, local_rank, world_size, device


def set_seed(seed: int, rank: int):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json_dataset_rank0_build_all_load(data_files: str, cache_dir: str, max_samples: int):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        rank0_print(f"[datasets] rank0 building cache: {data_files}")
        _ = load_dataset("json", data_files=data_files, split="train", cache_dir=cache_dir)
        rank0_print(f"[datasets] rank0 cache build done: {data_files}")
    if dist.is_initialized():
        dist.barrier()
    ds = load_dataset("json", data_files=data_files, split="train", cache_dir=cache_dir)
    if max_samples is not None and int(max_samples) > 0:
        ds = ds.select(range(min(int(max_samples), len(ds))))
    return ds


def assert_asr_only(dataset, check_samples: int = 1000):
    n = min(len(dataset), int(check_samples))
    bad = []
    for i in range(n):
        task = str(dataset[i].get("task", "asr")).lower().strip()
        if task != "asr":
            bad.append((i, task))
            if len(bad) >= 5:
                break
    if bad:
        raise ValueError(f"Expected ASR-only dataset, found non-ASR samples: {bad}")


def pad_1d(rows: List[torch.Tensor], pad_id: int, padding_side: str = "right") -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(row.numel()) for row in rows)
    out = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for i, row in enumerate(rows):
        n = int(row.numel())
        if padding_side == "left":
            out[i, max_len - n :] = row
            mask[i, max_len - n :] = 1
        else:
            out[i, :n] = row
            mask[i, :n] = 1
    return out, mask


def make_role_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor, user_id: int, assistant_id: int) -> torch.Tensor:
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    for i in range(input_ids.size(0)):
        is_training_turn = False
        for j in range(input_ids.size(1)):
            if int(attention_mask[i, j].item()) == 0:
                continue
            tid = int(input_ids[i, j].item())
            if tid == user_id:
                if is_training_turn:
                    labels[i, j] = tid
                is_training_turn = False
                continue
            if tid == assistant_id:
                is_training_turn = True
                continue
            if is_training_turn:
                labels[i, j] = tid
    return labels


@dataclass
class AsrBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    gen_input_ids: torch.Tensor
    gen_attention_mask: torch.Tensor
    audios: torch.Tensor
    teacher_audios: List[np.ndarray]
    teacher_languages: List[Optional[str]]
    teacher_texts: List[str]
    audio_paths: List[str]


class AsrCollator:
    def __init__(
        self,
        processor,
        tokenizer,
        max_audio_seconds: int = 30,
        sampling_rate: int = 16000,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_audio_seconds = int(max_audio_seconds)
        self.sampling_rate = int(sampling_rate)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.user_token = getattr(processor, "user_token", "<|user|>")
        self.assistant_token = getattr(processor, "assistant_token", "<|assistant|>")
        self.bos_audio_token = getattr(processor, "bos_audio_token", "<|begin_of_audio|>")
        self.eos_audio_token = getattr(processor, "eos_audio_token", "<|end_of_audio|>")
        self.audio_token = getattr(processor, "audio_token", "<|audio|>")
        self.user_id = tokenizer.convert_tokens_to_ids(self.user_token)
        self.assistant_id = tokenizer.convert_tokens_to_ids(self.assistant_token)

    def _load_audio(self, feature: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        audio_path = feature.get("audio")
        text = str(feature.get("text", "")).strip()
        if not audio_path or not isinstance(audio_path, str) or not Path(audio_path).exists():
            raise FileNotFoundError(f"Invalid or missing audio path in training sample: {audio_path!r}")
        begin_time = feature.get("begin_time", -1)
        end_time = feature.get("end_time", -1)
        offset = 0.0
        duration = None
        if begin_time is not None and float(begin_time) >= 0:
            offset = float(begin_time)
            if end_time is not None and float(end_time) > float(begin_time):
                duration = float(end_time) - float(begin_time)
        if self.max_audio_seconds > 0:
            if duration is None:
                duration = float(self.max_audio_seconds)
            else:
                duration = min(float(duration), float(self.max_audio_seconds))
        audio = self.processor._load_audio_file(
            audio_path,
            sampling_rate=self.sampling_rate,
            offset=offset,
            duration=duration,
        )
        return audio, text, str(audio_path)

    def _infer_language(self, audio_path: str) -> Optional[str]:
        parts = [part.upper() for part in Path(str(audio_path)).parts]
        for part in parts:
            if part in QWEN3_ASR_LANG_BY_CODE:
                return QWEN3_ASR_LANG_BY_CODE[part]
            if "_" in part:
                prefix = part.split("_", 1)[0]
                if prefix in QWEN3_ASR_LANG_BY_CODE:
                    return QWEN3_ASR_LANG_BY_CODE[prefix]
        return None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios_raw = []
        texts = []
        audio_paths = []
        teacher_languages = []
        for feature in features:
            task = str(feature.get("task", "asr")).lower().strip()
            if task != "asr":
                raise ValueError(f"ASR-OPD collator only supports task=asr, got {task}")
            audio, text, audio_path = self._load_audio(feature)
            audios_raw.append(audio)
            texts.append(text)
            audio_paths.append(audio_path)
            teacher_languages.append(self._infer_language(audio_path))

        audio_max_length = int(self.max_audio_seconds * self.sampling_rate)
        feat = self.processor.feature_extractor(
            audios_raw,
            sampling_rate=self.sampling_rate,
            return_tensors="np",
            return_attention_mask=False,
            padding="longest",
            max_length=audio_max_length,
        )
        input_features = feat["input_features"]
        if not isinstance(input_features, torch.Tensor):
            audios = torch.tensor(input_features, dtype=torch.float32)
        else:
            audios = input_features.to(dtype=torch.float32)

        if hasattr(self.processor, "_calculate_audio_token_counts_per_sample"):
            audio_counts = self.processor._calculate_audio_token_counts_per_sample(
                audios_raw=audios_raw,
                sampling_rate=self.sampling_rate,
                audio_max_length=audio_max_length,
                audio_pad_to_multiple_of=None,
            )
        else:
            hop_length = int(getattr(self.processor.feature_extractor, "hop_length", 160))
            audio_counts = []
            for audio in audios_raw:
                frames = min(len(audio), audio_max_length) // max(hop_length, 1)
                audio_counts.append(self.processor.calculate_audio_token_count(int(frames)))

        train_rows = []
        gen_rows = []
        for text, n_audio_tokens in zip(texts, audio_counts):
            audio_tokens = self.audio_token * int(n_audio_tokens)
            prompt = (
                f"{self.user_token}"
                f"{self.bos_audio_token}{audio_tokens}{self.eos_audio_token}"
                f"{ASR_INSTRUCTION}"
                f"{self.assistant_token}"
            )
            train_text = f"{prompt}{text}{self.user_token}"
            train_ids = self.tokenizer(train_text, add_special_tokens=False)["input_ids"]
            gen_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            train_rows.append(torch.tensor(train_ids, dtype=torch.long))
            gen_rows.append(torch.tensor(gen_ids, dtype=torch.long))

        input_ids, attention_mask = pad_1d(train_rows, self.pad_id, padding_side="right")
        labels = make_role_labels(input_ids, attention_mask, self.user_id, self.assistant_id)
        gen_input_ids, gen_attention_mask = pad_1d(gen_rows, self.pad_id, padding_side="left")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "gen_input_ids": gen_input_ids,
            "gen_attention_mask": gen_attention_mask,
            "audios": audios,
            "teacher_audios": audios_raw,
            "teacher_languages": teacher_languages,
            "teacher_texts": texts,
            "audio_paths": audio_paths,
        }


class ResumeDistributedSampler(DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_offset = 0

    def set_start_offset(self, start_offset: int):
        self.start_offset = max(0, int(start_offset))

    def __iter__(self):
        iterator = iter(super().__iter__())
        for _ in range(self.start_offset):
            try:
                next(iterator)
            except StopIteration:
                break
        self.start_offset = 0
        return iterator


class BlockTokenIdsFromLogitsProcessor(LogitsProcessor):
    def __init__(self, block_from_id: Optional[int], block_token_ids: Optional[List[int]] = None):
        self.block_from_id = None if block_from_id is None or int(block_from_id) < 0 else int(block_from_id)
        self.block_token_ids = sorted(set(int(x) for x in (block_token_ids or [])))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_size = scores.size(-1)
        if self.block_from_id is not None and self.block_from_id < vocab_size:
            scores[:, self.block_from_id :] = -float("inf")
        if self.block_token_ids:
            ids = [x for x in self.block_token_ids if 0 <= x < vocab_size]
            if ids:
                scores[:, ids] = -float("inf")
        return scores


def normalize_token_ids(token_ids: Any) -> List[int]:
    if token_ids is None:
        return []
    if isinstance(token_ids, (list, tuple, set)):
        return [int(x) for x in token_ids if x is not None]
    return [int(token_ids)]


def build_eos_token_ids(model, tokenizer) -> List[int]:
    ids = set()
    ids.update(normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))
    ids.update(normalize_token_ids(getattr(getattr(model, "config", None), "eos_token_id", None)))
    ids.update(normalize_token_ids(getattr(getattr(model, "generation_config", None), "eos_token_id", None)))
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        ids.add(int(im_end_id))
    return sorted(ids)


def primary_eos_token_id(tokenizer) -> Optional[int]:
    eos_ids = normalize_token_ids(getattr(tokenizer, "eos_token_id", None))
    if eos_ids:
        return int(eos_ids[0])
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token:
        token_ids = tokenizer(str(eos_token), add_special_tokens=False).get("input_ids", [])
        if token_ids:
            return int(token_ids[0])
    return None


def build_asr_extra_block_token_ids(tokenizer, keep_token_ids: Iterable[int], block_from_id: Optional[int]) -> List[int]:
    keep = set(int(x) for x in keep_token_ids)
    max_control_token_id = None if block_from_id is None or int(block_from_id) < 0 else int(block_from_id)
    block_ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None)
    added = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    for token_id, token_meta in added.items():
        token_id = int(token_id)
        if max_control_token_id is not None and token_id >= max_control_token_id:
            continue
        content = getattr(token_meta, "content", None)
        if content is None and isinstance(token_meta, dict):
            content = token_meta.get("content")
        if content and CONTROL_TOKEN_PATTERN.match(content):
            block_ids.add(token_id)
    block_ids.difference_update(keep)
    return sorted(block_ids)


def build_token_maps(teacher_tokenizer, student_tokenizer, teacher_vocab_size: int, student_vocab_size: int):
    teacher_vocab = teacher_tokenizer.get_vocab()
    student_vocab = student_tokenizer.get_vocab()
    teacher_to_student = torch.full((int(teacher_vocab_size),), -1, dtype=torch.long)
    student_to_teacher = torch.full((int(student_vocab_size),), -1, dtype=torch.long)
    shared = 0
    for token, teacher_id in teacher_vocab.items():
        teacher_id = int(teacher_id)
        if teacher_id < 0 or teacher_id >= teacher_vocab_size:
            continue
        student_id = student_vocab.get(token)
        if student_id is None:
            continue
        student_id = int(student_id)
        if student_id < 0 or student_id >= student_vocab_size:
            continue
        teacher_to_student[teacher_id] = student_id
        student_to_teacher[student_id] = teacher_id
        shared += 1
    return teacher_to_student, student_to_teacher, shared


def build_teacher_id_to_student_id_dict(teacher_tokenizer, student_tokenizer, teacher_vocab_size: int) -> Dict[int, int]:
    teacher_vocab = teacher_tokenizer.get_vocab()
    student_vocab = student_tokenizer.get_vocab()
    out = {}
    for token, teacher_id in teacher_vocab.items():
        teacher_id = int(teacher_id)
        if teacher_id < 0 or teacher_id >= int(teacher_vocab_size):
            continue
        student_id = student_vocab.get(token)
        if student_id is None:
            continue
        out[teacher_id] = int(student_id)
    return out


def mask_token_ids_(logits: torch.Tensor, token_ids: Iterable[int]) -> torch.Tensor:
    vocab_size = int(logits.size(-1))
    ids = [int(x) for x in token_ids if 0 <= int(x) < vocab_size]
    if ids:
        logits[..., ids] = -float("inf")
    return logits


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def register_qwen3_asr_transformers_backend(code_path: str = QWEN3_ASR_CODE_PATH):
    if not str(code_path or "").strip():
        raise ValueError(
            "--qwen3_asr_code_path is required for Qwen3-ASR teacher backends. "
            "Pass the local qwen3-asr Transformers backend checkout path."
        )
    root = Path(code_path)
    if not root.exists():
        raise FileNotFoundError(f"Qwen3-ASR code path not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Avoid importing qwen_asr.__init__, which pulls forced-aligner-only deps.
    pkg = sys.modules.get("qwen_asr")
    if pkg is None:
        pkg = types.ModuleType("qwen_asr")
        pkg.__path__ = [str(root / "qwen_asr")]
        sys.modules["qwen_asr"] = pkg
    core = sys.modules.get("qwen_asr.core")
    if core is None:
        core = types.ModuleType("qwen_asr.core")
        core.__path__ = [str(root / "qwen_asr" / "core")]
        sys.modules["qwen_asr.core"] = core
    from qwen_asr.core.transformers_backend import (
        Qwen3ASRConfig,
        Qwen3ASRForConditionalGeneration,
        Qwen3ASRProcessor,
    )
    from transformers import AutoConfig

    try:
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    except ValueError:
        pass
    try:
        AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
    except ValueError:
        pass
    try:
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
    except ValueError:
        pass
    return Qwen3ASRProcessor


def register_qwen3_asr_vllm_backend(code_path: str = QWEN3_ASR_CODE_PATH):
    register_qwen3_asr_transformers_backend(code_path)
    try:
        from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Qwen3-ASR vLLM backend is not compatible with the currently installed vLLM. "
            "Use --teacher_backend qwen3_asr_transformers, or upgrade vLLM to the version "
            "expected by the qwen3-asr backend passed via --qwen3_asr_code_path."
        ) from exc
    from vllm import ModelRegistry

    ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration)


def build_qwen3_asr_prompt(processor, language: Optional[str] = None) -> str:
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if language:
        prompt = prompt + f"language {language}{QWEN3_ASR_TEXT_MARKER}"
    return prompt


def build_qwen3_asr_text_prefix(processor, language: Optional[str] = None) -> str:
    return build_qwen3_asr_prompt(processor, language)


def strip_qwen3_asr_text(text: str) -> str:
    text = str(text or "")
    if QWEN3_ASR_TEXT_MARKER in text:
        text = text.split(QWEN3_ASR_TEXT_MARKER, 1)[-1]
    else:
        text = re.sub(r"^\s*language\s+[A-Za-z]+", "", text, count=1)
    return text.strip()


def infer_qwen3_asr_latin_language(text: str, fallback: Optional[str]) -> str:
    normalized_words = [
        word.strip("'").casefold()
        for word in QWEN3_ASR_LATIN_WORD_RE.findall(str(text or ""))
        if word.strip("'")
    ]
    scores = {language: 0.0 for language in QWEN3_ASR_LATIN_LANGUAGES}
    lowered = str(text or "").casefold()
    for language, chars in QWEN3_ASR_LATIN_CHAR_HINTS.items():
        score = sum(1 for ch in lowered if ch in chars)
        if score:
            scores[language] = scores.get(language, 0.0) + float(score) * 2.0
    for word in normalized_words:
        ascii_word = (
            word.replace("á", "a").replace("à", "a").replace("â", "a").replace("ã", "a")
            .replace("ä", "a").replace("å", "a").replace("ă", "a").replace("ą", "a")
            .replace("ç", "c").replace("č", "c").replace("ć", "c")
            .replace("é", "e").replace("è", "e").replace("ê", "e").replace("ë", "e")
            .replace("ě", "e").replace("ę", "e")
            .replace("í", "i").replace("ì", "i").replace("î", "i").replace("ï", "i")
            .replace("ı", "i")
            .replace("ñ", "n").replace("ń", "n")
            .replace("ó", "o").replace("ò", "o").replace("ô", "o").replace("õ", "o")
            .replace("ö", "o").replace("ø", "o").replace("ő", "o")
            .replace("ś", "s").replace("ş", "s").replace("ș", "s")
            .replace("ú", "u").replace("ù", "u").replace("û", "u").replace("ü", "u")
            .replace("ů", "u").replace("ű", "u")
            .replace("ý", "y").replace("ź", "z").replace("ż", "z").replace("ž", "z")
        )
        for language, hints in QWEN3_ASR_LATIN_WORD_HINTS.items():
            if word in hints or ascii_word in hints:
                scores[language] = scores.get(language, 0.0) + 1.0
    if fallback in QWEN3_ASR_LATIN_LANGUAGES:
        scores[str(fallback)] = scores.get(str(fallback), 0.0) + 0.75
    best_lang, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score >= 1.5:
        return best_lang
    if fallback in QWEN3_ASR_LATIN_LANGUAGES:
        return str(fallback)
    return "English"


def infer_qwen3_asr_language_from_text(text: str, fallback: Optional[str] = None) -> str:
    text = str(text or "")
    valid_fallback = fallback if fallback in VALID_QWEN3_ASR_LANGUAGES else None
    counts = {
        "Chinese": 0,
        "Japanese": 0,
        "Korean": 0,
        "Cyrillic": 0,
        "Arabic": 0,
        "Thai": 0,
        "Greek": 0,
        "Hindi": 0,
        "Latin": 0,
    }
    for ch in text:
        code = ord(ch)
        if 0x3040 <= code <= 0x30FF:
            counts["Japanese"] += 1
        elif 0x4E00 <= code <= 0x9FFF:
            counts["Chinese"] += 1
        elif 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
            counts["Korean"] += 1
        elif 0x0400 <= code <= 0x052F:
            counts["Cyrillic"] += 1
        elif 0x0600 <= code <= 0x06FF:
            counts["Arabic"] += 1
        elif 0x0E00 <= code <= 0x0E7F:
            counts["Thai"] += 1
        elif 0x0370 <= code <= 0x03FF:
            counts["Greek"] += 1
        elif 0x0900 <= code <= 0x097F:
            counts["Hindi"] += 1
        elif (
            ("A" <= ch <= "Z")
            or ("a" <= ch <= "z")
            or (0x00C0 <= code <= 0x024F)
            or (0x1E00 <= code <= 0x1EFF)
        ):
            counts["Latin"] += 1
    if counts["Japanese"] > 0:
        return "Japanese"
    if counts["Chinese"] > 0 and valid_fallback in {"Chinese", "Japanese", "Cantonese"}:
        return str(valid_fallback)
    if counts["Chinese"] > 0:
        return "Chinese"
    if counts["Korean"] > 0:
        return "Korean"
    if counts["Cyrillic"] > 0:
        return str(valid_fallback) if valid_fallback in {"Russian", "Macedonian"} else "Russian"
    if counts["Arabic"] > 0:
        return str(valid_fallback) if valid_fallback in {"Arabic", "Persian"} else "Arabic"
    for lang in ("Thai", "Greek", "Hindi"):
        if counts[lang] > 0:
            return lang
    if counts["Latin"] > 0:
        return infer_qwen3_asr_latin_language(text, valid_fallback)
    if valid_fallback:
        return str(valid_fallback)
    return "English"


def qwen3_asr_language_counts(languages: List[Optional[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for language in languages:
        key = str(language or "None")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def choose_qwen3_asr_language(text: str, fallback: Optional[str] = None) -> str:
    stripped = strip_qwen3_asr_text(text)
    if stripped:
        return infer_qwen3_asr_language_from_text(stripped, fallback)
    if fallback in VALID_QWEN3_ASR_LANGUAGES:
        return str(fallback)
    return "English"


def token_text_from_logprob(item: Any, tokenizer, token_id: int) -> str:
    decoded_token = getattr(item, "decoded_token", None)
    if decoded_token is not None:
        return str(decoded_token)
    token = getattr(item, "token", None)
    if token is not None:
        return str(token)
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except Exception:
        return ""


def find_token_suffix_start(sequence: List[int], suffix: List[int]) -> Optional[int]:
    if not suffix:
        return None
    suffix_len = len(suffix)
    if len(sequence) < suffix_len:
        return None
    for start in range(len(sequence) - suffix_len, -1, -1):
        if sequence[start : start + suffix_len] == suffix:
            return start
    return None


class Qwen3ASRTeacherOutput:
    def __init__(
        self,
        student_ids: List[int],
        teacher_top_student_ids: List[List[int]],
        teacher_top_logprobs: List[List[float]],
        student_support_ids: Optional[List[List[int]]] = None,
        teacher_on_student_logprobs: Optional[List[List[float]]] = None,
        eos_positions: Optional[List[bool]] = None,
        alignment_pad_offset: int = 0,
        alignment_target_start: int = -1,
    ):
        self.student_ids = student_ids
        self.teacher_top_student_ids = teacher_top_student_ids
        self.teacher_top_logprobs = teacher_top_logprobs
        self.student_support_ids = (
            student_support_ids if student_support_ids is not None else [[] for _ in student_ids]
        )
        self.teacher_on_student_logprobs = (
            teacher_on_student_logprobs if teacher_on_student_logprobs is not None else [[] for _ in student_ids]
        )
        self.eos_positions = eos_positions if eos_positions is not None else [False for _ in student_ids]
        self.alignment_pad_offset = int(alignment_pad_offset)
        self.alignment_target_start = int(alignment_target_start)


def resolve_student_asr_stop_token_id(tokenizer) -> int:
    for token in ("<|im_end|>",):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            return int(token_id)
    eos_ids = normalize_token_ids(getattr(tokenizer, "eos_token_id", None))
    if eos_ids:
        return int(eos_ids[0])
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    raise ValueError("Could not resolve a student ASR stop token id")


class Qwen3ASRTransformersTeacher:
    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype,
        device: torch.device,
        code_path: str,
        student_tokenizer,
    ):
        processor_cls = register_qwen3_asr_transformers_backend(code_path)
        self.processor = processor_cls.from_pretrained(model_path, fix_mistral_regex=True)
        self.tokenizer = self.processor.tokenizer
        self.teacher_to_student = build_teacher_id_to_student_id_dict(
            self.tokenizer,
            student_tokenizer,
            teacher_vocab_size=len(self.tokenizer),
        )
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = device
        self.dtype = dtype
        teacher_vocab_size = int(getattr(getattr(self.model, "config", None), "vocab_size", len(self.tokenizer)))
        teacher_vocab_size = max(teacher_vocab_size, len(self.tokenizer))
        teacher_to_student = torch.full((teacher_vocab_size,), -1, dtype=torch.long)
        for teacher_id, student_id in self.teacher_to_student.items():
            teacher_id = int(teacher_id)
            if 0 <= teacher_id < teacher_vocab_size:
                teacher_to_student[teacher_id] = int(student_id)
        self.teacher_to_student_tensor = teacher_to_student.to(device)
        self.block_teacher_token_ids = self._build_block_teacher_token_ids()

    def _build_block_teacher_token_ids(self) -> set[int]:
        ids = set()
        ids.update(int(x) for x in getattr(self.tokenizer, "all_special_ids", []) if x is not None)
        added = getattr(self.tokenizer, "added_tokens_decoder", {}) or {}
        for token_id, token_meta in added.items():
            content = getattr(token_meta, "content", None)
            if content is None and isinstance(token_meta, dict):
                content = token_meta.get("content")
            if content and CONTROL_TOKEN_PATTERN.match(str(content)):
                ids.add(int(token_id))
        ids.difference_update(normalize_token_ids(getattr(self.tokenizer, "eos_token_id", None)))
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0:
            ids.discard(int(im_end_id))
        return ids

    def generate_topk(
        self,
        audios: List[np.ndarray],
        languages: List[Optional[str]],
        max_new_tokens: int,
        top_k: int,
    ) -> List[Qwen3ASRTeacherOutput]:
        prompts = [build_qwen3_asr_prompt(self.processor, lang) for lang in languages]
        inputs = self.processor(text=prompts, audio=audios, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self.dtype)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            output_scores=True,
        )
        prompt_width = int(inputs["input_ids"].shape[1])
        seqs = generated.sequences[:, prompt_width:]
        scores = list(generated.scores or [])
        results = []
        for i in range(int(seqs.size(0))):
            token_ids = seqs[i].detach().cpu().tolist()
            student_ids = []
            top_student_ids = []
            top_logprobs = []
            eos_positions = []
            for step in range(min(len(scores), len(token_ids))):
                teacher_token_id = int(token_ids[step])
                if teacher_token_id in normalize_token_ids(getattr(self.tokenizer, "eos_token_id", None)):
                    break
                token_text = token_text_from_logprob(None, self.tokenizer, teacher_token_id)
                if not token_text or CONTROL_TOKEN_PATTERN.match(token_text):
                    break
                mapped_generated = self.teacher_to_student.get(teacher_token_id)
                if mapped_generated is None:
                    break
                student_ids.append(int(mapped_generated))
                logits = scores[step][i].detach().clone()
                mask_token_ids_(logits, self.block_teacher_token_ids)
                vals, ids = torch.topk(logits, k=min(int(top_k), int(logits.numel())), dim=-1)
                row_student_ids = []
                row_vals = []
                for teacher_id, val in zip(ids.cpu().tolist(), vals.cpu().tolist()):
                    mapped = self.teacher_to_student.get(int(teacher_id))
                    if mapped is not None:
                        row_student_ids.append(int(mapped))
                        row_vals.append(float(val))
                top_student_ids.append(row_student_ids)
                top_logprobs.append(row_vals)
                eos_positions.append(False)
            results.append(Qwen3ASRTeacherOutput(student_ids, top_student_ids, top_logprobs, eos_positions))
        return results

    def score_text_topk(
        self,
        audios: List[np.ndarray],
        languages: List[Optional[str]],
        texts: List[str],
        top_k: int,
        max_new_tokens: int,
        student_support_ids: Optional[List[List[List[int]]]] = None,
    ) -> List[Qwen3ASRTeacherOutput]:
        prefixes = [build_qwen3_asr_text_prefix(self.processor, lang) for lang in languages]
        full_texts = []
        target_ids_list = []
        for prefix, text in zip(prefixes, texts):
            raw_text_ids = self.tokenizer(strip_qwen3_asr_text(text), add_special_tokens=False).get("input_ids", [])
            text_ids = raw_text_ids
            if int(max_new_tokens) > 0:
                text_ids = text_ids[: int(max_new_tokens)]
            target_ids = [int(x) for x in text_ids]
            full_text = prefix + self.tokenizer.decode(target_ids, skip_special_tokens=False)
            target_ids_list.append(target_ids)
            full_texts.append(full_text)

        inputs = self.processor(text=full_texts, audio=audios, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self.dtype)
        with torch.no_grad():
            outputs = self.model.thinker(**inputs, use_cache=False)
        logits = outputs.logits.detach()

        results = [
            Qwen3ASRTeacherOutput(student_ids=[], teacher_top_student_ids=[], teacher_top_logprobs=[])
            for _ in target_ids_list
        ]
        row_batch = []
        row_pos = []
        row_owner = []
        row_owner_pos = []
        row_student_support = []
        teacher_eos_ids = set(normalize_token_ids(getattr(self.tokenizer, "eos_token_id", None)))
        for i, target_teacher_ids in enumerate(target_ids_list):
            student_ids = []
            eos_positions = []
            if len(target_teacher_ids) < 1:
                continue
            row_ids_full = [int(x) for x in inputs["input_ids"][i].detach().cpu().tolist()]
            row_ids = row_ids_full
            row_pos_map = list(range(len(row_ids_full)))
            if "attention_mask" in inputs:
                mask = inputs["attention_mask"][i].detach().cpu().tolist()
                kept = [
                    (pos, int(token_id))
                    for pos, (token_id, keep) in enumerate(zip(row_ids_full, mask))
                    if int(keep) != 0
                ]
                row_pos_map = [pos for pos, _ in kept]
                row_ids = [token_id for _, token_id in kept]
            target_start = find_token_suffix_start(row_ids, target_teacher_ids)
            if target_start is None:
                continue
            start_in_kept = max(int(target_start) - 1, 0)
            if start_in_kept >= len(row_pos_map):
                continue
            start = int(row_pos_map[start_in_kept])
            max_steps = min(len(target_teacher_ids), int(logits.size(1)) - start)
            for step in range(max_steps):
                teacher_token_id = int(target_teacher_ids[step])
                if teacher_token_id in self.block_teacher_token_ids:
                    break
                mapped_generated = self.teacher_to_student.get(teacher_token_id)
                if mapped_generated is None:
                    break
                student_ids.append(int(mapped_generated))
                eos_positions.append(int(teacher_token_id) in teacher_eos_ids)
                row_batch.append(i)
                row_pos.append(start + step)
                row_owner.append(i)
                row_owner_pos.append(len(student_ids) - 1)
                support_rows = student_support_ids[i] if student_support_ids and i < len(student_support_ids) else []
                row_student_support.append(support_rows[step] if step < len(support_rows) else [])
            results[i] = Qwen3ASRTeacherOutput(
                student_ids=student_ids,
                teacher_top_student_ids=[[] for _ in student_ids],
                teacher_top_logprobs=[[] for _ in student_ids],
                student_support_ids=[[] for _ in student_ids],
                teacher_on_student_logprobs=[[] for _ in student_ids],
                eos_positions=eos_positions,
                alignment_pad_offset=(int(row_pos_map[0]) if row_pos_map else 0),
                alignment_target_start=int(target_start),
            )

        if row_batch:
            batch_index = torch.tensor(row_batch, dtype=torch.long, device=self.device)
            pos_index = torch.tensor(row_pos, dtype=torch.long, device=self.device)
            selected_logits = logits[batch_index, pos_index, :].clone()
            mask_token_ids_(selected_logits, self.block_teacher_token_ids)
            k = min(int(top_k), int(selected_logits.size(-1)))
            vals, ids = torch.topk(selected_logits, k=k, dim=-1)
            mapped_student_ids = self.teacher_to_student_tensor[ids]
            valid_mask = mapped_student_ids.ge(0)
            mapped_cpu = mapped_student_ids.detach().cpu().tolist()
            vals_cpu = vals.detach().cpu().tolist()
            valid_cpu = valid_mask.detach().cpu().tolist()
            selected_log_probs = F.log_softmax(selected_logits.float(), dim=-1)
            for row_i, (owner, owner_pos) in enumerate(zip(row_owner, row_owner_pos)):
                row_student_ids = []
                row_vals = []
                for sid, val, is_valid in zip(mapped_cpu[row_i], vals_cpu[row_i], valid_cpu[row_i]):
                    if is_valid:
                        row_student_ids.append(int(sid))
                        row_vals.append(float(val))
                results[int(owner)].teacher_top_student_ids[int(owner_pos)] = row_student_ids
                results[int(owner)].teacher_top_logprobs[int(owner_pos)] = row_vals
                support_ids = [int(x) for x in row_student_support[row_i] if 0 <= int(x) < selected_logits.size(-1)]
                if support_ids:
                    support_tensor = torch.tensor(support_ids, dtype=torch.long, device=self.device)
                    support_vals = torch.gather(selected_log_probs[row_i], dim=-1, index=support_tensor)
                    results[int(owner)].student_support_ids[int(owner_pos)] = support_ids
                    results[int(owner)].teacher_on_student_logprobs[int(owner_pos)] = (
                        support_vals.detach().cpu().tolist()
                    )
        return results

    def force_topk(
        self,
        audios: List[np.ndarray],
        languages: List[Optional[str]],
        texts: List[str],
        top_k: int,
        max_new_tokens: int,
        append_eos: Optional[List[bool]] = None,
        student_support_ids: Optional[List[List[List[int]]]] = None,
    ) -> List[Qwen3ASRTeacherOutput]:
        return self.score_text_topk(
            audios=audios,
            languages=languages,
            texts=texts,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            student_support_ids=student_support_ids,
        )


class Qwen3ASRVLLMTeacher:
    def __init__(self, model_path: str, code_path: str, gpu_memory_utilization: float, student_tokenizer):
        register_qwen3_asr_vllm_backend(code_path)
        from vllm import LLM, SamplingParams

        self.processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        self.teacher_to_student = build_teacher_id_to_student_id_dict(
            self.tokenizer,
            student_tokenizer,
            teacher_vocab_size=len(self.tokenizer),
        )
        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=float(gpu_memory_utilization),
            trust_remote_code=True,
        )

    def generate_topk(
        self,
        audios: List[np.ndarray],
        languages: List[Optional[str]],
        max_new_tokens: int,
        top_k: int,
    ) -> List[Qwen3ASRTeacherOutput]:
        sampling_params = self.SamplingParams(
            temperature=0.0,
            max_tokens=int(max_new_tokens),
            logprobs=int(top_k),
            skip_special_tokens=True,
        )
        prompts = []
        for audio, language in zip(audios, languages):
            prompts.append(
                {
                    "prompt": build_qwen3_asr_prompt(self.processor, language),
                    "multi_modal_data": {"audio": [audio]},
                }
            )
        outputs = self.llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        results = []
        for output in outputs:
            completion = output.outputs[0]
            token_ids = list(getattr(completion, "token_ids", []) or [])
            eos_ids = set(normalize_token_ids(getattr(self.tokenizer, "eos_token_id", None)))
            student_ids = []
            top_student_ids = []
            top_logprobs = []
            logprobs = completion.logprobs or []
            for step in range(min(len(token_ids), len(logprobs))):
                teacher_token_id = int(token_ids[step])
                if teacher_token_id in eos_ids:
                    break
                token_text = token_text_from_logprob(None, self.tokenizer, teacher_token_id)
                if not token_text or CONTROL_TOKEN_PATTERN.match(token_text):
                    break
                mapped_generated = self.teacher_to_student.get(teacher_token_id)
                if mapped_generated is None:
                    break
                student_ids.append(int(mapped_generated))
                row = logprobs[step] or {}
                row_student_ids = []
                row_vals = []
                for teacher_id, item in row.items():
                    mapped = self.teacher_to_student.get(int(teacher_id))
                    if mapped is not None:
                        row_student_ids.append(int(mapped))
                        row_vals.append(float(getattr(item, "logprob", -float("inf"))))
                top_student_ids.append(row_student_ids)
                top_logprobs.append(row_vals)
            results.append(Qwen3ASRTeacherOutput(student_ids, top_student_ids, top_logprobs))
        return results


def build_teacher(args, dtype: torch.dtype, device: torch.device, student_tokenizer=None):
    backend = str(args.teacher_backend).lower().strip()
    if backend in {"qwen3_asr_transformers", "qwen3_asr_teacher_forcing"}:
        return Qwen3ASRTransformersTeacher(
            model_path=args.teacher_model,
            dtype=dtype,
            device=device,
            code_path=args.qwen3_asr_code_path,
            student_tokenizer=student_tokenizer,
        )
    if backend == "qwen3_asr_vllm":
        return Qwen3ASRVLLMTeacher(
            model_path=args.teacher_model,
            code_path=args.qwen3_asr_code_path,
            gpu_memory_utilization=args.teacher_vllm_gpu_memory_utilization,
            student_tokenizer=student_tokenizer,
        )
    return AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.teacher_attn_implementation,
    ).to(device)


def clean_generated_ids(
    token_ids: torch.Tensor,
    stop_ids: set[int],
    student_to_teacher: torch.Tensor,
    block_from_id: Optional[int],
    include_stop_token: bool = False,
) -> Tuple[List[int], List[int]]:
    student_ids = []
    teacher_ids = []
    student_to_teacher_cpu = student_to_teacher.cpu()
    for token_id in token_ids.detach().cpu().tolist():
        token_id = int(token_id)
        if token_id in stop_ids:
            if include_stop_token and 0 <= token_id < student_to_teacher_cpu.numel():
                teacher_id = int(student_to_teacher_cpu[token_id].item())
                if teacher_id >= 0:
                    student_ids.append(token_id)
                    teacher_ids.append(teacher_id)
            break
        if block_from_id is not None and int(block_from_id) >= 0 and token_id >= int(block_from_id):
            continue
        if token_id < 0 or token_id >= student_to_teacher_cpu.numel():
            continue
        teacher_id = int(student_to_teacher_cpu[token_id].item())
        if teacher_id < 0:
            continue
        student_ids.append(token_id)
        teacher_ids.append(teacher_id)
    return student_ids, teacher_ids


def pick_dummy_shared_token(student_to_teacher: torch.Tensor, preferred_student_id: int) -> Tuple[int, int]:
    table = student_to_teacher.cpu()
    preferred_student_id = int(preferred_student_id)
    if 0 <= preferred_student_id < table.numel():
        teacher_id = int(table[preferred_student_id].item())
        if teacher_id >= 0:
            return preferred_student_id, teacher_id
    mapped = torch.nonzero(table.ge(0), as_tuple=False).flatten()
    if mapped.numel() == 0:
        raise ValueError("No shared student/teacher token found for OPD dummy row")
    student_id = int(mapped[0].item())
    teacher_id = int(table[student_id].item())
    return student_id, teacher_id


def build_teacher_inputs(generated_teacher_ids: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    rows = [torch.tensor(ids, dtype=torch.long) for ids in generated_teacher_ids]
    input_ids, attention_mask = pad_1d(rows, pad_id, padding_side="right")
    return input_ids, attention_mask, [len(ids) for ids in generated_teacher_ids]


def compute_topk_opd_loss(
    student_scores: torch.Tensor,
    teacher_logits: torch.Tensor,
    gen_lens: List[int],
    teacher_to_student: torch.Tensor,
    top_k: int,
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    student_positions = []
    teacher_positions = []
    batch_indices = []
    for i, gen_len in enumerate(gen_lens):
        if gen_len < 2:
            continue
        for k in range(gen_len - 1):
            batch_indices.append(i)
            teacher_positions.append(k)
            student_positions.append(k)

    if not batch_indices:
        zero = student_scores.sum() * 0.0
        return zero, {"opd_positions": 0.0, "opd_valid_rows": 0.0, "opd_valid_topk_mean": 0.0}

    b = torch.tensor(batch_indices, dtype=torch.long, device=student_scores.device)
    tpos = torch.tensor(teacher_positions, dtype=torch.long, device=student_scores.device)
    spos = torch.tensor(student_positions, dtype=torch.long, device=student_scores.device)

    teacher_flat = teacher_logits[b, tpos, :]
    student_flat = student_scores[b, spos, :]
    k = min(int(top_k), int(teacher_flat.size(-1)))
    top_vals, top_teacher_ids = torch.topk(teacher_flat, k=k, dim=-1)
    mapped_student_ids = teacher_to_student.to(student_scores.device)[top_teacher_ids]
    valid_mask = mapped_student_ids.ge(0)
    valid_count = valid_mask.sum(dim=-1)
    row_mask = valid_count.ge(2)
    if not bool(row_mask.any().item()):
        zero = student_flat.sum() * 0.0
        return zero, {
            "opd_positions": float(len(batch_indices)),
            "opd_valid_rows": 0.0,
            "opd_valid_topk_mean": 0.0,
        }

    top_vals = top_vals[row_mask]
    mapped_student_ids = mapped_student_ids[row_mask]
    valid_mask = valid_mask[row_mask]
    student_flat = student_flat[row_mask]

    gather_ids = mapped_student_ids.clamp_min(0)
    student_selected = torch.gather(student_flat, dim=-1, index=gather_ids)

    temp = float(temperature)
    teacher_scores = (top_vals / temp).masked_fill(~valid_mask, -float("inf"))
    student_scores = (student_selected / temp).masked_fill(~valid_mask, -float("inf"))
    teacher_log_probs = F.log_softmax(teacher_scores, dim=-1)
    student_log_probs = F.log_softmax(student_scores, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    kl_per_row = (teacher_probs * (teacher_log_probs - student_log_probs)).masked_fill(~valid_mask, 0.0).sum(dim=-1)
    loss = kl_per_row.mean() * (temp * temp)
    return loss, {
        "opd_positions": float(len(batch_indices)),
        "opd_valid_rows": float(row_mask.sum().item()),
        "opd_valid_topk_mean": float(valid_mask.sum(dim=-1).float().mean().detach().item()),
    }


def compute_union_topk_logprob_opd_loss(
    student_scores: torch.Tensor,
    teacher_top_student_ids: List[List[List[int]]],
    teacher_top_logprobs: List[List[List[float]]],
    student_support_ids: List[List[List[int]]],
    teacher_on_student_logprobs: List[List[List[float]]],
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    row_batch = []
    row_pos = []
    max_union_k = 0
    teacher_only_rows = 0
    student_only_rows = 0
    union_rows = []
    for i in range(len(teacher_top_student_ids)):
        teacher_rows = teacher_top_student_ids[i] if i < len(teacher_top_student_ids) else []
        student_rows = student_support_ids[i] if i < len(student_support_ids) else []
        teacher_on_student_rows = teacher_on_student_logprobs[i] if i < len(teacher_on_student_logprobs) else []
        row_count = max(len(teacher_rows), len(student_rows), len(teacher_on_student_rows))
        for pos in range(row_count):
            teacher_ids = teacher_rows[pos] if pos < len(teacher_rows) else []
            student_ids = student_rows[pos] if pos < len(student_rows) else []
            union_ids = []
            seen = set()
            for token_id in student_ids:
                token_id = int(token_id)
                if token_id not in seen:
                    union_ids.append(token_id)
                    seen.add(token_id)
            for token_id in teacher_ids:
                token_id = int(token_id)
                if token_id not in seen:
                    union_ids.append(token_id)
                    seen.add(token_id)
            if len(union_ids) >= 2:
                row_batch.append(i)
                row_pos.append(pos)
                union_rows.append(union_ids)
                max_union_k = max(max_union_k, len(union_ids))
                if len(teacher_ids) >= 2:
                    teacher_only_rows += 1
                if len(student_ids) >= 2:
                    student_only_rows += 1
    if not row_batch:
        zero = student_scores.sum() * 0.0
        return zero, {
            "opd_positions": 0.0,
            "opd_valid_rows": 0.0,
            "opd_valid_topk_mean": 0.0,
            "opd_teacher_support_rows": 0.0,
            "opd_student_support_rows": 0.0,
            "opd_union_support_rows": 0.0,
        }

    device = student_scores.device
    gather_ids = torch.zeros((len(row_batch), max_union_k), dtype=torch.long, device=device)
    teacher_vals = torch.full((len(row_batch), max_union_k), -float("inf"), dtype=torch.float32, device=device)
    valid_mask = torch.zeros((len(row_batch), max_union_k), dtype=torch.bool, device=device)
    for out_i, (batch_i, pos_i, union_ids) in enumerate(zip(row_batch, row_pos, union_rows)):
        gather_ids[out_i, : len(union_ids)] = torch.tensor(union_ids, dtype=torch.long, device=device)
        teacher_map = {}
        if batch_i < len(teacher_top_student_ids) and pos_i < len(teacher_top_student_ids[batch_i]):
            for sid, val in zip(teacher_top_student_ids[batch_i][pos_i], teacher_top_logprobs[batch_i][pos_i]):
                teacher_map[int(sid)] = float(val)
        if batch_i < len(student_support_ids) and pos_i < len(student_support_ids[batch_i]):
            support_ids = student_support_ids[batch_i][pos_i]
            support_vals = (
                teacher_on_student_logprobs[batch_i][pos_i]
                if batch_i < len(teacher_on_student_logprobs) and pos_i < len(teacher_on_student_logprobs[batch_i])
                else []
            )
            for sid, val in zip(support_ids, support_vals):
                sid = int(sid)
                if sid not in teacher_map:
                    teacher_map[sid] = float(val)
        for col_i, sid in enumerate(union_ids):
            val = teacher_map.get(int(sid))
            if val is None or not math.isfinite(float(val)):
                continue
            teacher_vals[out_i, col_i] = float(val)
            valid_mask[out_i, col_i] = True

    b = torch.tensor(row_batch, dtype=torch.long, device=device)
    pos = torch.tensor(row_pos, dtype=torch.long, device=device)
    student_flat = student_scores[b, pos, :]
    student_selected = torch.gather(student_flat, dim=-1, index=gather_ids)

    temp = float(temperature)
    teacher_scores = (teacher_vals / temp).masked_fill(~valid_mask, -float("inf"))
    student_scores = (student_selected / temp).masked_fill(~valid_mask, -float("inf"))
    teacher_log_probs = F.log_softmax(teacher_scores, dim=-1)
    student_log_probs = F.log_softmax(student_scores, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    kl_per_row = (teacher_probs * (teacher_log_probs - student_log_probs)).masked_fill(~valid_mask, 0.0).sum(dim=-1)
    loss = kl_per_row.mean() * (temp * temp)
    return loss, {
        "opd_positions": float(len(row_batch)),
        "opd_valid_rows": float(len(row_batch)),
        "opd_valid_topk_mean": float(valid_mask.sum(dim=-1).float().mean().detach().item()),
        "opd_teacher_support_rows": float(teacher_only_rows),
        "opd_student_support_rows": float(student_only_rows),
        "opd_union_support_rows": float(len(row_batch)),
    }


def topk_eos_metrics(
    teacher_top_student_ids: List[List[List[int]]],
    eos_positions: List[List[bool]],
    student_stop_token_id: int,
) -> Dict[str, float]:
    nonterminal_rows = 0
    nonterminal_with_eos = 0
    terminal_rows = 0
    terminal_with_eos = 0
    for rows, flags in zip(teacher_top_student_ids, eos_positions):
        for ids, is_eos in zip(rows, flags):
            if bool(is_eos):
                terminal_rows += 1
                if int(student_stop_token_id) in set(int(x) for x in ids):
                    terminal_with_eos += 1
            else:
                nonterminal_rows += 1
                if int(student_stop_token_id) in set(int(x) for x in ids):
                    nonterminal_with_eos += 1
    return {
        "opd_nonterminal_rows": float(nonterminal_rows),
        "opd_nonterminal_eos_candidate_rows": float(nonterminal_with_eos),
        "opd_terminal_rows": float(terminal_rows),
        "opd_terminal_eos_candidate_rows": float(terminal_with_eos),
    }


def compute_student_eos_loss(
    student_scores: torch.Tensor,
    generated_lens: List[int],
    eos_supervision_mask: List[bool],
    student_stop_token_id: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    row_batch = []
    row_pos = []
    for i, (gen_len, should_supervise) in enumerate(zip(generated_lens, eos_supervision_mask)):
        if bool(should_supervise) and int(gen_len) >= 1:
            row_batch.append(i)
            row_pos.append(int(gen_len))

    if not row_batch:
        zero = student_scores.sum() * 0.0
        return zero, {"eos_positions": 0.0, "eos_loss": 0.0}

    device = student_scores.device
    b = torch.tensor(row_batch, dtype=torch.long, device=device)
    pos = torch.tensor(row_pos, dtype=torch.long, device=device)
    targets = torch.full((len(row_batch),), int(student_stop_token_id), dtype=torch.long, device=device)
    logits = student_scores[b, pos, :]
    loss = F.cross_entropy(logits.float(), targets)
    return loss, {
        "eos_positions": float(len(row_batch)),
        "eos_loss": float(loss.detach().item()),
    }


def make_last_label_only_labels(
    labels: torch.Tensor,
    expected_terminal_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    terminal_labels = torch.full_like(labels, IGNORE_INDEX)
    terminal_count = 0
    expected_count = 0
    for i in range(int(labels.size(0))):
        positions = torch.nonzero(labels[i].ne(IGNORE_INDEX), as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        pos = int(positions[-1].item())
        token_id = int(labels[i, pos].item())
        terminal_labels[i, pos] = labels[i, pos]
        terminal_count += 1
        if expected_terminal_token_id is not None and token_id == int(expected_terminal_token_id):
            expected_count += 1
    return terminal_labels, {
        "asr_terminal_positions": float(terminal_count),
        "asr_terminal_expected_positions": float(expected_count),
    }


def all_reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def all_reduce_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def all_reduce_min(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return float(tensor.item())


def all_reduce_max_int(value: int, device: torch.device) -> int:
    tensor = torch.tensor(int(value), dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def sync_if_cuda(device: torch.device) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def reshard_fsdp2_modules(model) -> None:
    for module in reversed(list(model.modules())):
        reshard = getattr(module, "reshard", None)
        if callable(reshard):
            reshard()


@contextmanager
def fsdp2_no_reshard_after_forward(model):
    set_reshard = getattr(model, "set_reshard_after_forward", None)
    if not callable(set_reshard):
        yield
        return
    set_reshard(False, recurse=True)
    try:
        yield
    finally:
        set_reshard(True, recurse=True)
        reshard_fsdp2_modules(model)


def generation_finished_across_ranks(finished: torch.Tensor) -> bool:
    done = torch.tensor(1 if bool(finished.all().item()) else 0, dtype=torch.int32, device=finished.device)
    if dist.is_initialized():
        dist.all_reduce(done, op=dist.ReduceOp.MIN)
    return bool(done.item())


def greedy_generate_with_grad_scores(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    audios: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: List[int],
    pad_token_id: int,
    finished_token_id: int,
    debug_generation_steps: int,
    logits_processor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sequences = input_ids
    cur_attention_mask = attention_mask
    next_input_ids = input_ids
    past_key_values = None
    scores = []
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
    eos_ids = torch.tensor([int(x) for x in eos_token_ids], dtype=torch.long, device=input_ids.device)

    for step in range(int(max_new_tokens)):
        outputs = model(
            input_ids=next_input_ids,
            attention_mask=cur_attention_mask,
            audios=audios if step == 0 else None,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]
        if logits_processor is not None:
            logits = logits_processor(sequences, logits)
        scores.append(logits)

        next_tokens = torch.argmax(logits.detach(), dim=-1)
        if eos_ids.numel() > 0:
            next_is_eos = (next_tokens.unsqueeze(-1) == eos_ids.unsqueeze(0)).any(dim=-1)
        else:
            next_is_eos = torch.zeros_like(finished)
        append_tokens = torch.where(finished, torch.full_like(next_tokens, int(finished_token_id)), next_tokens)
        if int(debug_generation_steps) > 0 and step < int(debug_generation_steps):
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                f"[gen-debug] rank={rank} step={step} next={next_tokens.detach().cpu().tolist()} "
                f"append={append_tokens.detach().cpu().tolist()} finished_before={finished.detach().cpu().tolist()} "
                f"is_eos={next_is_eos.detach().cpu().tolist()}",
                flush=True,
            )
        sequences = torch.cat([sequences, append_tokens.unsqueeze(-1)], dim=-1)
        cur_attention_mask = torch.cat([cur_attention_mask, torch.ones_like(next_tokens).unsqueeze(-1)], dim=-1)
        finished = finished | next_is_eos
        past_key_values = outputs.past_key_values
        next_input_ids = torch.where(
            finished,
            torch.full_like(next_tokens, int(finished_token_id)),
            next_tokens,
        ).unsqueeze(-1)
        if generation_finished_across_ranks(finished):
            break

    return sequences, torch.stack(scores, dim=1)


def greedy_generate_no_grad(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    audios: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: List[int],
    pad_token_id: int,
    finished_token_id: int,
    debug_generation_steps: int,
    logits_processor,
    return_top_k: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    sequences = input_ids
    cur_attention_mask = attention_mask
    next_input_ids = input_ids
    past_key_values = None
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
    eos_ids = torch.tensor([int(x) for x in eos_token_ids], dtype=torch.long, device=input_ids.device)
    topk_ids = []

    with fsdp2_no_reshard_after_forward(model):
        with torch.no_grad():
            for step in range(int(max_new_tokens)):
                outputs = model(
                    input_ids=next_input_ids,
                    attention_mask=cur_attention_mask,
                    audios=audios if step == 0 else None,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                if logits_processor is not None:
                    logits = logits_processor(sequences, logits)
                if int(return_top_k) > 0:
                    k = min(int(return_top_k), int(logits.size(-1)))
                    topk_ids.append(torch.topk(logits, k=k, dim=-1).indices)

                next_tokens = torch.argmax(logits, dim=-1)
                if eos_ids.numel() > 0:
                    next_is_eos = (next_tokens.unsqueeze(-1) == eos_ids.unsqueeze(0)).any(dim=-1)
                else:
                    next_is_eos = torch.zeros_like(finished)
                append_tokens = torch.where(finished, torch.full_like(next_tokens, int(finished_token_id)), next_tokens)
                if int(debug_generation_steps) > 0 and step < int(debug_generation_steps):
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    print(
                        f"[gen-debug] rank={rank} step={step} next={next_tokens.detach().cpu().tolist()} "
                        f"append={append_tokens.detach().cpu().tolist()} finished_before={finished.detach().cpu().tolist()} "
                        f"is_eos={next_is_eos.detach().cpu().tolist()}",
                        flush=True,
                    )
                sequences = torch.cat([sequences, append_tokens.unsqueeze(-1)], dim=-1)
                cur_attention_mask = torch.cat([cur_attention_mask, torch.ones_like(next_tokens).unsqueeze(-1)], dim=-1)
                finished = finished | next_is_eos
                past_key_values = outputs.past_key_values
                next_input_ids = torch.where(
                    finished,
                    torch.full_like(next_tokens, int(finished_token_id)),
                    next_tokens,
                ).unsqueeze(-1)
                if generation_finished_across_ranks(finished):
                    break

    topk_tensor = torch.stack(topk_ids, dim=1) if topk_ids else None
    return sequences, topk_tensor


def pad_device_rows(rows: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(row.numel()) for row in rows)
    out = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long, device=rows[0].device)
    mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=rows[0].device)
    for i, row in enumerate(rows):
        n = int(row.numel())
        out[i, :n] = row
        mask[i, :n] = 1
    return out, mask


def left_pad_2d_to_width(values: torch.Tensor, mask: torch.Tensor, width: int, pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    cur = int(values.size(1))
    width = int(width)
    if cur == width:
        return values, mask
    if cur > width:
        return values[:, -width:], mask[:, -width:]
    value_pad = torch.full((values.size(0), width - cur), int(pad_id), dtype=values.dtype, device=values.device)
    mask_pad = torch.zeros((mask.size(0), width - cur), dtype=mask.dtype, device=mask.device)
    return torch.cat([value_pad, values], dim=1), torch.cat([mask_pad, mask], dim=1)


def build_teacher_forced_student_scores(
    model,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    audios: torch.Tensor,
    generated_student_ids: List[List[int]],
    generated_lens: List[int],
    valid_indices: List[int],
    pad_token_id: int,
) -> torch.Tensor:
    device = prompt_input_ids.device
    prompt_width = int(prompt_input_ids.size(1))
    rows = []
    masks = []
    for source_i, ids in zip(valid_indices, generated_student_ids):
        gen = torch.tensor(ids, dtype=torch.long, device=device)
        rows.append(torch.cat([prompt_input_ids[source_i], gen], dim=0))
        masks.append(torch.cat([prompt_attention_mask[source_i], torch.ones_like(gen)], dim=0))

    input_ids, _ = pad_device_rows(rows, pad_token_id)
    attention_mask, _ = pad_device_rows(masks, 0)
    index = torch.tensor(valid_indices, dtype=torch.long, device=device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audios=audios.index_select(0, index),
        use_cache=False,
    )

    score_rows = []
    for i, gen_len in enumerate(generated_lens):
        score_len = int(gen_len)
        if score_len <= 0:
            score_rows.append(outputs.logits[i, :0, :])
            continue
        # Logits at the final prompt token predict generated token 0; logits
        # after generated token k predict generated token k+1.
        start = max(int(prompt_width) - 1, 0)
        score_rows.append(outputs.logits[i, start : start + score_len, :])

    max_score_len = max(row.size(0) for row in score_rows)
    if max_score_len <= 0:
        return outputs.logits[:, :0, :]
    padded = []
    for row in score_rows:
        if row.size(0) == max_score_len:
            padded.append(row)
        else:
            pad = row.new_zeros((max_score_len - row.size(0), row.size(1)))
            padded.append(torch.cat([row, pad], dim=0))
    return torch.stack(padded, dim=0)


def build_teacher_forced_student_scores_dense(
    model,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    audios: torch.Tensor,
    generated_student_ids: List[List[int]],
    score_len: int,
    pad_token_id: int,
    force_prompt_width: Optional[int] = None,
) -> torch.Tensor:
    device = prompt_input_ids.device
    prompt_width = int(force_prompt_width or prompt_input_ids.size(1))
    prompt_input_ids, prompt_attention_mask = left_pad_2d_to_width(
        prompt_input_ids,
        prompt_attention_mask,
        prompt_width,
        int(pad_token_id),
    )
    batch_size = int(prompt_input_ids.size(0))
    score_len = max(0, int(score_len))
    gen = torch.full((batch_size, max(score_len, 1)), int(pad_token_id), dtype=torch.long, device=device)
    gen_mask = torch.zeros((batch_size, max(score_len, 1)), dtype=prompt_attention_mask.dtype, device=device)
    for i in range(batch_size):
        ids = generated_student_ids[i] if i < len(generated_student_ids) else []
        if not ids or score_len <= 0:
            continue
        row = torch.tensor([int(x) for x in ids[:score_len]], dtype=torch.long, device=device)
        n = int(row.numel())
        gen[i, :n] = row
        gen_mask[i, :n] = 1

    if score_len <= 0:
        gen = gen[:, :0]
        gen_mask = gen_mask[:, :0]
    input_ids = torch.cat([prompt_input_ids, gen], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, gen_mask], dim=1)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audios=audios,
        use_cache=False,
    )
    start = max(prompt_width - 1, 0)
    return outputs.logits[:, start : start + score_len, :]


def decode_student_response(tokenizer, token_ids: List[int]) -> str:
    return tokenizer.decode(
        [int(x) for x in token_ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()


def compute_losses(
    model,
    teacher_model,
    batch: Dict[str, torch.Tensor],
    tokenizer,
    teacher_tokenizer,
    teacher_to_student: torch.Tensor,
    student_to_teacher: torch.Tensor,
    logits_processor,
    eos_token_ids: List[int],
    args,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    teacher_backend = str(args.teacher_backend).lower().strip()
    if teacher_backend.startswith("qwen3_asr_"):
        batch_size = int(batch["gen_input_ids"].size(0))
        max_new_tokens = max(1, int(args.asr_opd_max_new_tokens))
        local_prompt_width = int(batch["gen_input_ids"].size(1))
        prompt_width_sync_start = time.time()
        global_prompt_width = all_reduce_max_int(local_prompt_width, device)
        sync_if_cuda(device)
        prompt_width_sync_time = time.time() - prompt_width_sync_start

        finished_token_ids = tokenizer(" ", add_special_tokens=False).get("input_ids", [])
        finished_token_id = int(finished_token_ids[0]) if finished_token_ids else int(tokenizer.pad_token_id)
        sync_if_cuda(device)
        rollout_start = time.time()
        generated_sequences, rollout_topk_ids = greedy_generate_no_grad(
            model=model,
            input_ids=batch["gen_input_ids"],
            attention_mask=batch["gen_attention_mask"],
            audios=batch["audios"],
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            pad_token_id=tokenizer.pad_token_id,
            finished_token_id=finished_token_id,
            debug_generation_steps=args.debug_generation_steps,
            logits_processor=logits_processor,
            return_top_k=int(args.opd_top_k),
        )
        sync_if_cuda(device)
        rollout_time = time.time() - rollout_start

        stop_ids = set(int(x) for x in eos_token_ids)
        prompt_width = int(batch["gen_input_ids"].size(1))
        rollout_student_ids = [[] for _ in range(batch_size)]
        generated_student_ids = [[] for _ in range(batch_size)]
        student_support_ids = [[] for _ in range(batch_size)]
        generated_lens = [0 for _ in range(batch_size)]
        generated_stopped = [False for _ in range(batch_size)]
        rollout_texts = ["" for _ in range(batch_size)]
        teacher_languages = [None for _ in range(batch_size)]
        for source_i, row in enumerate(generated_sequences):
            new_ids = row[prompt_width:]
            new_id_list = [int(x) for x in new_ids.detach().cpu().tolist()]
            stopped = any(token_id in stop_ids for token_id in new_id_list)
            student_ids, _ = clean_generated_ids(
                new_ids,
                stop_ids=stop_ids,
                student_to_teacher=student_to_teacher,
                block_from_id=args.asr_block_token_id_from,
                include_stop_token=True,
            )
            text = strip_qwen3_asr_text(decode_student_response(tokenizer, student_ids))
            fallback_text = strip_qwen3_asr_text(
                batch["teacher_texts"][source_i] if source_i < len(batch["teacher_texts"]) else ""
            )
            fallback_language = batch["teacher_languages"][source_i] if source_i < len(batch["teacher_languages"]) else None
            if len(student_ids) >= 1 and text:
                rollout_student_ids[source_i] = [int(x) for x in student_ids[:max_new_tokens]]
                generated_student_ids[source_i] = [int(x) for x in student_ids[:max_new_tokens]]
                support_rows = []
                if rollout_topk_ids is not None and source_i < int(rollout_topk_ids.size(0)):
                    topk_steps = rollout_topk_ids[source_i]
                    for step_i in range(min(int(topk_steps.size(0)), len(generated_student_ids[source_i]))):
                        row_ids = [int(x) for x in topk_steps[step_i].detach().cpu().tolist()]
                        filtered = []
                        seen = set()
                        block_from_id = args.asr_block_token_id_from
                        for token_id in row_ids:
                            if token_id in stop_ids:
                                continue
                            if block_from_id is not None and int(block_from_id) >= 0 and token_id >= int(block_from_id):
                                continue
                            if token_id not in seen:
                                filtered.append(token_id)
                                seen.add(token_id)
                        if generated_student_ids[source_i][step_i] not in seen:
                            filtered.insert(0, int(generated_student_ids[source_i][step_i]))
                        support_rows.append(filtered[: int(args.opd_top_k)])
                student_support_ids[source_i] = support_rows[:max_new_tokens]
                generated_lens[source_i] = min(len(student_ids), max_new_tokens)
                generated_stopped[source_i] = bool(stopped and len(student_ids) <= max_new_tokens)
                rollout_texts[source_i] = text
                teacher_languages[source_i] = choose_qwen3_asr_language(text, fallback_language)
            elif fallback_text:
                rollout_texts[source_i] = fallback_text
                teacher_languages[source_i] = choose_qwen3_asr_language(fallback_text, fallback_language)
        teacher_start = time.time()
        teacher_results = teacher_model.score_text_topk(
            audios=batch["teacher_audios"],
            languages=teacher_languages,
            texts=rollout_texts,
            max_new_tokens=max_new_tokens,
            top_k=int(args.opd_top_k),
            student_support_ids=student_support_ids,
        )
        sync_if_cuda(device)
        teacher_time = time.time() - teacher_start
        student_stop_token_id = resolve_student_asr_stop_token_id(tokenizer)
        student_score_len = max_new_tokens
        asr_loss = batch["gen_input_ids"].sum() * 0.0
        asr_loss_time = 0.0
        terminal_label_metrics = {
            "asr_terminal_positions": 0.0,
            "asr_terminal_expected_positions": 0.0,
        }
        teacher_parse_start = time.time()
        student_vocab_size = int(getattr(model.config, "vocab_size", 0) or len(tokenizer))
        teacher_top_student_ids = [[] for _ in range(batch_size)]
        teacher_top_logprobs = [[] for _ in range(batch_size)]
        teacher_on_student_logprobs = [[] for _ in range(batch_size)]
        teacher_eos_positions = [[] for _ in range(batch_size)]
        fallback_teacher_forced_count = 0
        retokenize_compare_tokens = 0
        retokenize_mismatch_tokens = 0
        retokenize_mismatch_rows = 0
        alignment_pad_offsets = []
        fallback_mask = [int(generated_lens[i]) < 1 and bool(rollout_texts[i]) for i in range(batch_size)]
        for source_i, result in enumerate(teacher_results):
            if source_i >= batch_size:
                break
            if not rollout_texts[source_i] or len(result.student_ids) < 1:
                continue
            rows_ids = []
            rows_vals = []
            for ids, vals in zip(result.teacher_top_student_ids, result.teacher_top_logprobs):
                valid_pairs = [(sid, val) for sid, val in zip(ids, vals) if 0 <= int(sid) < student_vocab_size]
                if len(valid_pairs) >= 2:
                    rows_ids.append([int(x[0]) for x in valid_pairs])
                    rows_vals.append([float(x[1]) for x in valid_pairs])
                else:
                    rows_ids.append([])
                    rows_vals.append([])
            usable_len = min(len(result.student_ids), len(rows_ids))
            if usable_len < 1:
                generated_student_ids[source_i] = []
                generated_lens[source_i] = 0
                continue
            fallback_used = bool(fallback_mask[source_i])
            if fallback_used:
                fallback_teacher_forced_count += 1
            target_len_limit = (
                len(result.student_ids)
                if fallback_used
                else int(generated_lens[source_i])
            )
            usable_len = min(usable_len, target_len_limit, student_score_len)
            if not fallback_used:
                original_ids = rollout_student_ids[source_i][:usable_len]
                retok_ids = result.student_ids[:usable_len]
                compare_len = min(len(original_ids), len(retok_ids))
                if compare_len > 0:
                    mismatches = sum(
                        1 for left, right in zip(original_ids[:compare_len], retok_ids[:compare_len])
                        if int(left) != int(right)
                    )
                    retokenize_compare_tokens += compare_len
                    retokenize_mismatch_tokens += mismatches
                    if mismatches > 0 or len(original_ids) != len(retok_ids):
                        retokenize_mismatch_rows += 1
            alignment_pad_offsets.append(int(getattr(result, "alignment_pad_offset", 0)))
            generated_student_ids[source_i] = result.student_ids[:usable_len]
            generated_lens[source_i] = usable_len
            teacher_top_student_ids[source_i] = rows_ids[:usable_len]
            teacher_top_logprobs[source_i] = rows_vals[:usable_len]
            teacher_on_student_logprobs[source_i] = result.teacher_on_student_logprobs[:usable_len]
            student_support_ids[source_i] = result.student_support_ids[:usable_len]
            teacher_eos_positions[source_i] = result.eos_positions[:usable_len]
        teacher_parse_time = time.time() - teacher_parse_start

        real_generated_lens = [int(x) for i, x in enumerate(generated_lens) if int(x) >= 1 and not fallback_mask[i]]
        real_generated_count = sum(1 for rows in teacher_top_student_ids if len(rows) >= 1)
        reduce_count_start = time.time()
        global_generated_count = all_reduce_sum(float(real_generated_count), device)
        reduce_count_time = time.time() - reduce_count_start

        sync_if_cuda(device)
        student_score_start = time.time()
        student_scores = build_teacher_forced_student_scores_dense(
            model=model,
            prompt_input_ids=batch["gen_input_ids"],
            prompt_attention_mask=batch["gen_attention_mask"],
            audios=batch["audios"],
            generated_student_ids=generated_student_ids,
            score_len=student_score_len,
            pad_token_id=tokenizer.pad_token_id,
            force_prompt_width=global_prompt_width,
        )
        sync_if_cuda(device)
        student_score_time = time.time() - student_score_start
        if global_generated_count <= 0.0:
            opd_loss = student_scores.sum() * 0.0
            opd_metrics = {"opd_positions": 0.0, "opd_valid_rows": 0.0, "opd_valid_topk_mean": 0.0}
        else:
            sync_if_cuda(device)
            opd_start = time.time()
            opd_loss, opd_metrics = compute_union_topk_logprob_opd_loss(
                student_scores=student_scores,
                teacher_top_student_ids=teacher_top_student_ids,
                teacher_top_logprobs=teacher_top_logprobs,
                student_support_ids=student_support_ids,
                teacher_on_student_logprobs=teacher_on_student_logprobs,
                temperature=args.opd_temperature,
            )
            sync_if_cuda(device)
            opd_loss_time = time.time() - opd_start
        if global_generated_count <= 0.0:
            opd_loss_time = 0.0
        total_loss = opd_loss
        eos_candidate_metrics = topk_eos_metrics(
            teacher_top_student_ids,
            teacher_eos_positions,
            student_stop_token_id,
        )
        metrics = {
            "generated_nonempty": float(sum(1 for i, x in enumerate(generated_lens) if int(x) >= 1 and not fallback_mask[i])) / float(max(batch_size, 1)),
            "generated_tokens_mean": float(sum(real_generated_lens)) / float(max(real_generated_count, 1)),
            "generated_stopped": float(sum(1 for x in generated_stopped if bool(x))) / float(max(batch_size, 1)),
            "teacher_forced_fallback": float(fallback_teacher_forced_count) / float(max(batch_size, 1)),
            "teacher_time": float(teacher_time),
            "rollout_time": float(rollout_time),
            "teacher_language_counts": qwen3_asr_language_counts(teacher_languages),
            "teacher_alignment_pad_offset_mean": (
                float(sum(alignment_pad_offsets)) / float(max(len(alignment_pad_offsets), 1))
            ),
            "teacher_alignment_pad_offset_max": float(max(alignment_pad_offsets) if alignment_pad_offsets else 0),
            "retokenize_compare_tokens": float(retokenize_compare_tokens),
            "retokenize_mismatch_tokens": float(retokenize_mismatch_tokens),
            "retokenize_mismatch_ratio": (
                float(retokenize_mismatch_tokens) / float(max(retokenize_compare_tokens, 1))
            ),
            "retokenize_mismatch_rows": float(retokenize_mismatch_rows),
            "prompt_width_sync_time": float(prompt_width_sync_time),
            "teacher_parse_time": float(teacher_parse_time),
            "asr_loss_time": float(asr_loss_time),
            "global_generated_count": float(global_generated_count),
            "local_valid_count": float(real_generated_count),
            "local_prompt_width": float(local_prompt_width),
            "global_prompt_width": float(global_prompt_width),
            "reduce_count_time": float(reduce_count_time),
            "student_score_time": float(student_score_time),
            "opd_loss_time": float(opd_loss_time),
            "asr_terminal_loss_weight": 0.0,
            "opd_includes_eos": 0.0,
            "student_stop_token_id": float(student_stop_token_id),
            **eos_candidate_metrics,
            **opd_metrics,
            **terminal_label_metrics,
        }
        return total_loss, asr_loss, opd_loss, metrics

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        audios=batch["audios"],
        labels=batch["labels"],
    )
    asr_loss = outputs.loss
    finished_token_ids = tokenizer(" ", add_special_tokens=False).get("input_ids", [])
    finished_token_id = int(finished_token_ids[0]) if finished_token_ids else int(tokenizer.pad_token_id)

    if args.opd_student_score_mode == "grad_generate":
        generated_sequences, raw_scores = greedy_generate_with_grad_scores(
            model=model,
            input_ids=batch["gen_input_ids"],
            attention_mask=batch["gen_attention_mask"],
            audios=batch["audios"],
            max_new_tokens=int(args.asr_opd_max_new_tokens),
            eos_token_ids=eos_token_ids,
            pad_token_id=tokenizer.pad_token_id,
            finished_token_id=finished_token_id,
            debug_generation_steps=args.debug_generation_steps,
            logits_processor=logits_processor,
        )
        if raw_scores.numel() == 0:
            opd_loss = asr_loss.new_zeros(())
            total_loss = asr_loss + float(args.opd_loss_weight) * opd_loss
            return total_loss, asr_loss, opd_loss, {
                "generated_nonempty": 0.0,
                "generated_tokens_mean": 0.0,
                "opd_positions": 0.0,
                "opd_valid_rows": 0.0,
                "opd_valid_topk_mean": 0.0,
            }
    else:
        generated_sequences = greedy_generate_no_grad(
            model=model,
            input_ids=batch["gen_input_ids"],
            attention_mask=batch["gen_attention_mask"],
            audios=batch["audios"],
            max_new_tokens=int(args.asr_opd_max_new_tokens),
            eos_token_ids=eos_token_ids,
            pad_token_id=tokenizer.pad_token_id,
            finished_token_id=finished_token_id,
            debug_generation_steps=args.debug_generation_steps,
            logits_processor=logits_processor,
        )
        raw_scores = None

    prompt_width = batch["gen_input_ids"].size(1)
    stop_ids = set(int(x) for x in eos_token_ids)
    generated_student_ids = []
    generated_teacher_ids = []
    valid_indices = []
    generated_lens = []
    kept_score_rows = []
    for source_i, row in enumerate(generated_sequences):
        new_ids = row[prompt_width:]
        student_ids, teacher_ids = clean_generated_ids(
            new_ids,
            stop_ids=stop_ids,
            student_to_teacher=student_to_teacher,
            block_from_id=args.asr_block_token_id_from,
        )
        if len(student_ids) >= 2 and len(teacher_ids) == len(student_ids):
            generated_student_ids.append(student_ids)
            generated_teacher_ids.append(teacher_ids)
            valid_indices.append(source_i)
            generated_lens.append(len(student_ids))
            if raw_scores is not None:
                kept_score_rows.append(raw_scores[source_i, 1 : len(student_ids), :])

    real_generated_count = len(generated_student_ids)
    real_generated_lens = list(generated_lens)
    global_generated_count = all_reduce_sum(float(real_generated_count), device)
    if global_generated_count <= 0.0:
        opd_loss = asr_loss.new_zeros(())
        total_loss = asr_loss + float(args.opd_loss_weight) * opd_loss
        return total_loss, asr_loss, opd_loss, {
            "generated_nonempty": 0.0,
            "generated_tokens_mean": 0.0,
            "opd_positions": 0.0,
            "opd_valid_rows": 0.0,
            "opd_valid_topk_mean": 0.0,
        }

    if real_generated_count == 0:
        dummy_student_id, dummy_teacher_id = pick_dummy_shared_token(student_to_teacher, finished_token_id)
        generated_student_ids.append([dummy_student_id])
        generated_teacher_ids.append([dummy_teacher_id])
        generated_lens.append(1)
        valid_indices.append(0)
        if raw_scores is not None:
            kept_score_rows.append(raw_scores[0, :0, :])

    teacher_input_ids, teacher_attention_mask, _ = build_teacher_inputs(
        generated_teacher_ids,
        pad_id=teacher_tokenizer.pad_token_id if teacher_tokenizer.pad_token_id is not None else teacher_tokenizer.eos_token_id,
    )
    teacher_input_ids = teacher_input_ids.to(device)
    teacher_attention_mask = teacher_attention_mask.to(device)
    if raw_scores is not None:
        max_score_len = max(row.size(0) for row in kept_score_rows)
        student_score_rows = []
        for row in kept_score_rows:
            if row.size(0) == max_score_len:
                student_score_rows.append(row)
            else:
                pad = row.new_zeros((max_score_len - row.size(0), row.size(1)))
                student_score_rows.append(torch.cat([row, pad], dim=0))
        student_scores = torch.stack(student_score_rows, dim=0)
    else:
        student_scores = build_teacher_forced_student_scores(
            model=model,
            prompt_input_ids=batch["gen_input_ids"],
            prompt_attention_mask=batch["gen_attention_mask"],
            audios=batch["audios"],
            generated_student_ids=generated_student_ids,
            generated_lens=generated_lens,
            valid_indices=valid_indices,
            pad_token_id=tokenizer.pad_token_id,
        )

    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
        )
        teacher_logits = teacher_outputs.logits

    opd_loss, opd_metrics = compute_topk_opd_loss(
        student_scores=student_scores,
        teacher_logits=teacher_logits,
        gen_lens=generated_lens,
        teacher_to_student=teacher_to_student,
        top_k=args.opd_top_k,
        temperature=args.opd_temperature,
    )
    total_loss = asr_loss + float(args.opd_loss_weight) * opd_loss
    metrics = {
        "generated_nonempty": float(real_generated_count) / float(batch["input_ids"].size(0)),
        "generated_tokens_mean": float(sum(real_generated_lens)) / float(max(real_generated_count, 1)),
        **opd_metrics,
    }
    return total_loss, asr_loss, opd_loss, metrics


def make_checkpoint_manager(model, optimizer, lr_scheduler, tokenizer):
    cfg = OmegaConf.create({"save_contents": ["model", "optimizer", "extra"], "load_contents": ["model", "optimizer", "extra"]})
    return FSDPCheckpointManager(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        processing_class=tokenizer,
        checkpoint_config=cfg,
    )


def make_lr_scheduler(optimizer, args, steps_per_epoch: int):
    scheduler_type = str(args.lr_scheduler_type).lower().strip()
    if scheduler_type in {"none", "constant", ""}:
        return None
    if scheduler_type != "cosine":
        raise ValueError(f"Unsupported lr_scheduler_type={args.lr_scheduler_type}")

    max_steps = int(args.max_steps)
    if max_steps <= 0:
        max_steps = int(args.total_epochs) * int(steps_per_epoch)
    max_steps = max(1, int(max_steps))
    warmup_steps = max(0, int(args.warmup_steps))
    warmup_steps = min(warmup_steps, max_steps - 1) if max_steps > 1 else 0
    min_lr_ratio = float(args.min_lr_ratio)

    def lr_lambda(step: int):
        step = int(step)
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler._ark_max_steps = max_steps
    scheduler._ark_warmup_steps = warmup_steps
    scheduler._ark_min_lr_ratio = min_lr_ratio
    return scheduler


def save_checkpoint(manager, output_dir: str, step: int, rank: int, max_ckpt_to_keep: int = 3):
    path = os.path.join(output_dir, "checkpoints", f"global_step_{step}")
    if rank == 0:
        os.makedirs(path, exist_ok=True)
        rank0_print(f"[checkpoint] saving {path}")
    manager.save_checkpoint(local_path=path, global_step=step, max_ckpt_to_keep=max_ckpt_to_keep)


def infer_checkpoint_step(checkpoint_path: str) -> int:
    match = re.search(r"global_step_(\d+)$", os.path.basename(os.path.normpath(str(checkpoint_path))))
    return int(match.group(1)) if match else 0


def resolve_resume_checkpoint(resume_from_checkpoint: str, output_dir: str) -> Optional[str]:
    value = str(resume_from_checkpoint or "").strip()
    if not value:
        return None
    if value.lower() in {"latest", "auto"}:
        ckpt_root = os.path.join(output_dir, "checkpoints")
        if not os.path.isdir(ckpt_root):
            raise FileNotFoundError(f"No checkpoint directory found: {ckpt_root}")
        candidates = []
        for name in os.listdir(ckpt_root):
            path = os.path.join(ckpt_root, name)
            if os.path.isdir(path):
                step = infer_checkpoint_step(path)
                if step > 0:
                    candidates.append((step, path))
        if not candidates:
            raise FileNotFoundError(f"No global_step_* checkpoints found under {ckpt_root}")
        return max(candidates, key=lambda item: item[0])[1]
    return value


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student_model", required=True, help="Path or HF repo id of the audio ASR student model")
    p.add_argument("--teacher_model", required=True, help="Path or HF repo id of the teacher ASR model")
    p.add_argument("--teacher_backend", default="qwen3_asr_teacher_forcing", choices=["hf_causal_lm", "qwen3_asr_transformers", "qwen3_asr_teacher_forcing", "qwen3_asr_vllm"])
    p.add_argument(
        "--qwen3_asr_code_path",
        default=QWEN3_ASR_CODE_PATH,
        help="Required for qwen3_asr_* teacher backends; path to the qwen3-asr backend code",
    )
    p.add_argument("--teacher_vllm_gpu_memory_utilization", type=float, default=0.3)
    p.add_argument("--train_data", required=True, help="JSONL ASR training data with audio/text fields")
    p.add_argument("--eval_data", default="", help="Reserved for downstream eval hooks")
    p.add_argument("--output_dir", required=True, help="Directory for logs and FSDP checkpoints")
    p.add_argument("--hf_cache_dir", default=os.environ.get("HF_DATASETS_CACHE", ".cache/hf_datasets"))
    p.add_argument("--train_max_samples", type=int, default=-1)
    p.add_argument("--shuffle_train", type=parse_bool, default=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--dataloader_prefetch_factor", type=int, default=2)
    p.add_argument("--dataloader_persistent_workers", type=parse_bool, default=False)
    p.add_argument("--dataloader_multiprocessing_context", default="", choices=["", "fork", "spawn", "forkserver"])
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--total_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--lr_scheduler_type", default="constant", choices=["constant", "cosine"])
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.005)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--calibrate_only", type=parse_bool, default=True)
    p.add_argument("--calibration_batches", type=int, default=10)
    p.add_argument("--opd_loss_weight", type=float, default=1.0)
    p.add_argument("--opd_eos_loss_weight", type=float, default=0.0)
    p.add_argument("--opd_append_teacher_eos_for_stopped_rollouts", type=parse_bool, default=False)
    p.add_argument("--asr_terminal_loss_weight", type=float, default=0.0)
    p.add_argument("--opd_top_k", type=int, default=32)
    p.add_argument("--opd_temperature", type=float, default=1.0)
    p.add_argument("--opd_student_score_mode", default="teacher_forcing", choices=["teacher_forcing", "grad_generate"])
    p.add_argument("--asr_opd_max_new_tokens", type=int, default=256)
    p.add_argument("--debug_generation_steps", type=int, default=0)
    p.add_argument("--debug_opd_steps", type=int, default=0)
    p.add_argument("--asr_block_token_id_from", type=int, default=151670)
    p.add_argument("--max_audio_seconds", type=int, default=30)
    p.add_argument("--sampling_rate", type=int, default=16000)
    p.add_argument("--save_freq", type=int, default=-1)
    p.add_argument("--max_ckpt_to_keep", type=int, default=3)
    p.add_argument("--resume_from_checkpoint", default="", help="FSDP checkpoint dir, or 'latest'/'auto' under output_dir/checkpoints")
    p.add_argument("--model_dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--teacher_attn_implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--student_attn_implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    return p.parse_args()


def main():
    args = parse_args()
    if str(args.teacher_backend).startswith("qwen3_asr_") and not str(args.qwen3_asr_code_path or "").strip():
        raise ValueError("--qwen3_asr_code_path is required when --teacher_backend starts with qwen3_asr_")
    rank, local_rank, world_size, device = setup_distributed()
    set_seed(args.seed, rank)
    os.makedirs(args.output_dir, exist_ok=True)

    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float32

    rank0_print(f"[config] world_size={world_size} local_rank={local_rank}")
    rank0_print(f"[config] student={args.student_model}")
    rank0_print(f"[config] teacher={args.teacher_model}")
    rank0_print(f"[config] teacher_backend={args.teacher_backend}")
    rank0_print(f"[config] train_data={args.train_data}")
    rank0_print(f"[config] output_dir={args.output_dir}")
    rank0_print(
        f"[config] calibrate_only={args.calibrate_only} opd_loss_weight={args.opd_loss_weight} "
        f"opd_student_score_mode={args.opd_student_score_mode}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    processor = AutoProcessor.from_pretrained(
        args.student_model,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.pad_token_id = tokenizer.pad_token_id

    teacher_tokenizer = None
    if args.teacher_backend == "hf_causal_lm":
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    elif args.teacher_backend in {"qwen3_asr_transformers", "qwen3_asr_teacher_forcing"}:
        register_qwen3_asr_transformers_backend(args.qwen3_asr_code_path)
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True, fix_mistral_regex=True)
    else:
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True, fix_mistral_regex=True)
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id

    rank0_print("[model] loading student")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.student_attn_implementation,
    )
    student.config.use_cache = False
    # if hasattr(student, "audio_encoder"):
    #     if hasattr(student.audio_encoder, "whisper"):
    #         for param in student.audio_encoder.whisper.parameters():
    #             param.requires_grad = False
    #     if hasattr(student.audio_encoder, "adapting"):
    #         for param in student.audio_encoder.adapting.parameters():
    #             param.requires_grad = False
    #     if hasattr(student.audio_encoder, "layer_norm"):
    #         for param in student.audio_encoder.layer_norm.parameters():
    #             param.requires_grad = False
    student.to(device)

    rank0_print("[model] wrapping student with FSDP2")
    device_mesh = init_device_mesh("cuda", (world_size,))
    mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32, cast_forward_inputs=True)
    fsdp_kwargs = {
        "mesh": device_mesh,
        "mp_policy": mp_policy,
        "offload_policy": None if CPUOffloadPolicy is None else None,
        "reshard_after_forward": True,
    }
    full_state = student.state_dict()
    fsdp_config = OmegaConf.create(
        {"wrap_policy": {"transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer", "WhisperSpecialEncoder"]}}
    )
    apply_fsdp2(student, fsdp_kwargs, fsdp_config)
    fsdp2_load_full_state_dict(student, full_state, device_mesh, None)
    del full_state
    torch.cuda.empty_cache()

    rank0_print("[model] loading teacher")
    teacher = build_teacher(args, dtype=dtype, device=device, student_tokenizer=tokenizer)
    if hasattr(teacher, "eval"):
        teacher.eval()
    if hasattr(teacher, "parameters"):
        for param in teacher.parameters():
            param.requires_grad = False

    teacher_config = getattr(teacher, "config", None)
    if hasattr(teacher, "model"):
        teacher_config = getattr(teacher.model, "config", teacher_config)
    teacher_vocab_size = int(getattr(teacher_config, "vocab_size", len(teacher_tokenizer)))
    student_vocab_size = int(getattr(student.config, "vocab_size", len(tokenizer)))
    teacher_to_student, student_to_teacher, shared = build_token_maps(
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=tokenizer,
        teacher_vocab_size=teacher_vocab_size,
        student_vocab_size=student_vocab_size,
    )
    teacher_to_student = teacher_to_student.to(device)
    rank0_print(
        f"[token-map] teacher_vocab={teacher_vocab_size} student_vocab={student_vocab_size} "
        f"shared_tokens={shared} student_only_or_unmapped={int((student_to_teacher < 0).sum().item())}"
    )

    eos_token_ids = build_eos_token_ids(student, tokenizer)
    block_token_ids = build_asr_extra_block_token_ids(
        tokenizer=tokenizer,
        keep_token_ids=eos_token_ids,
        block_from_id=args.asr_block_token_id_from,
    )
    logits_processor = LogitsProcessorList(
        [BlockTokenIdsFromLogitsProcessor(args.asr_block_token_id_from, block_token_ids)]
    )
    rank0_print(f"[generation] eos_token_ids={eos_token_ids} extra_block_token_count={len(block_token_ids)}")

    train_dataset = load_json_dataset_rank0_build_all_load(args.train_data, args.hf_cache_dir, args.train_max_samples)
    if rank == 0:
        assert_asr_only(train_dataset)
        rank0_print(f"[data] train_dataset size={len(train_dataset)}")
    dist.barrier()

    collator = AsrCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_audio_seconds=args.max_audio_seconds,
        sampling_rate=args.sampling_rate,
    )
    sampler = ResumeDistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(bool(args.shuffle_train) and not args.calibrate_only),
        drop_last=True,
    )
    dataloader_kwargs = {
        "batch_size": args.per_device_train_batch_size,
        "sampler": sampler,
        "num_workers": args.dataloader_num_workers,
        "pin_memory": True,
        "drop_last": True,
        "collate_fn": collator,
    }
    if int(args.dataloader_num_workers) > 0:
        dataloader_kwargs["prefetch_factor"] = int(args.dataloader_prefetch_factor)
        dataloader_kwargs["persistent_workers"] = bool(args.dataloader_persistent_workers)
        if args.dataloader_multiprocessing_context:
            dataloader_kwargs["multiprocessing_context"] = args.dataloader_multiprocessing_context
    loader = DataLoader(train_dataset, **dataloader_kwargs)

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    steps_per_epoch = len(loader)
    lr_scheduler = make_lr_scheduler(optimizer, args, steps_per_epoch)
    if rank == 0:
        if lr_scheduler is None:
            rank0_print(f"[lr] scheduler=constant learning_rate={args.learning_rate}")
        else:
            rank0_print(
                f"[lr] scheduler=cosine base_lr={args.learning_rate} "
                f"warmup_steps={lr_scheduler._ark_warmup_steps} "
                f"max_steps={lr_scheduler._ark_max_steps} min_lr_ratio={lr_scheduler._ark_min_lr_ratio}"
            )
    resume_checkpoint = resolve_resume_checkpoint(args.resume_from_checkpoint, args.output_dir)
    need_checkpoint_manager = int(args.save_freq) > 0 or resume_checkpoint is not None
    checkpoint_manager = make_checkpoint_manager(student, optimizer, lr_scheduler, tokenizer) if need_checkpoint_manager else None

    global_step = 0
    if resume_checkpoint is not None:
        if args.calibrate_only:
            raise ValueError("--resume_from_checkpoint is only supported for training; set --calibrate_only False")
        if not os.path.isdir(resume_checkpoint):
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_checkpoint}")
        rank0_print(f"[checkpoint] loading {resume_checkpoint}")
        checkpoint_manager.load_checkpoint(local_path=resume_checkpoint, del_local_after_load=False)
        global_step = infer_checkpoint_step(resume_checkpoint)
        rank0_print(f"[checkpoint] resumed global_step={global_step}")

    if args.calibrate_only:
        student.eval()
        asr_sum = 0.0
        opd_sum = 0.0
        gen_nonempty_sum = 0.0
        gen_tok_sum = 0.0
        count = 0
        with torch.no_grad():
            for step, batch in enumerate(loader):
                if step >= int(args.calibration_batches):
                    break
                batch = move_batch_to_device(batch, device)
                _, asr_loss, opd_loss, metrics = compute_losses(
                    model=student,
                    teacher_model=teacher,
                    batch=batch,
                    tokenizer=tokenizer,
                    teacher_tokenizer=teacher_tokenizer,
                    teacher_to_student=teacher_to_student,
                    student_to_teacher=student_to_teacher,
                    logits_processor=logits_processor,
                    eos_token_ids=eos_token_ids,
                    args=args,
                    device=device,
                )
                asr_sum += float(asr_loss.detach().item())
                opd_sum += float(opd_loss.detach().item())
                gen_nonempty_sum += float(metrics["generated_nonempty"])
                gen_tok_sum += float(metrics["generated_tokens_mean"])
                count += 1
                rank0_print(
                    f"[calibration local] step={step + 1} asr_loss={asr_loss.item():.6f} "
                    f"opd_loss={opd_loss.item():.6f} gen_nonempty={metrics['generated_nonempty']:.3f} "
                    f"gen_tokens={metrics['generated_tokens_mean']:.2f}"
                )

        total_count = all_reduce_sum(count, device)
        asr_total = all_reduce_sum(asr_sum, device)
        opd_total = all_reduce_sum(opd_sum, device)
        gen_nonempty_total = all_reduce_sum(gen_nonempty_sum, device)
        gen_tok_total = all_reduce_sum(gen_tok_sum, device)
        if rank == 0:
            denom = max(total_count, 1.0)
            opd_mean = opd_total / denom
            print("[calibration summary]", flush=True)
            print("initial/asr_loss_mean=0.00000000", flush=True)
            print(f"initial/opd_loss_mean={opd_mean:.8f}", flush=True)
            print(f"initial/generated_nonempty_ratio={gen_nonempty_total / denom:.8f}", flush=True)
            print(f"initial/generated_tokens_mean={gen_tok_total / denom:.8f}", flush=True)
            print("training/loss_formula=opd_text_only", flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    student.train()
    max_steps = int(args.max_steps)
    total_epochs = int(args.total_epochs)
    if max_steps > 0 and global_step >= max_steps:
        rank0_print(f"[checkpoint] resumed step {global_step} already reaches max_steps={max_steps}; exiting")
        dist.barrier()
        dist.destroy_process_group()
        return

    start_epoch = 0
    resume_step_in_epoch = 0
    if global_step > 0:
        start_epoch = min(global_step // max(steps_per_epoch, 1), total_epochs)
        resume_step_in_epoch = global_step % max(steps_per_epoch, 1)
        rank0_print(
            f"[checkpoint] resume dataloader at epoch={start_epoch} "
            f"step_in_epoch={resume_step_in_epoch} steps_per_epoch={steps_per_epoch}"
        )
    if start_epoch >= total_epochs:
        rank0_print(f"[checkpoint] resumed step {global_step} already reaches total_epochs={total_epochs}; exiting")
        dist.barrier()
        dist.destroy_process_group()
        return

    for epoch in range(start_epoch, total_epochs):
        sampler.set_epoch(epoch)
        if epoch == start_epoch and resume_step_in_epoch > 0:
            sampler.set_start_offset(resume_step_in_epoch * int(args.per_device_train_batch_size))
        loader_iter = iter(loader)
        if epoch == start_epoch and resume_step_in_epoch > 0:
            rank0_print(
                f"[checkpoint] skipped {resume_step_in_epoch} batches for epoch={epoch} "
                f"via sampler offset"
            )
        while True:
            data_wait_start = time.time()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            data_wait = time.time() - data_wait_start
            step_start = time.time()
            move_start = time.time()
            batch = move_batch_to_device(batch, device)
            move_time = time.time() - move_start
            optimizer.zero_grad(set_to_none=True)
            compute_start = time.time()
            total_loss, asr_loss, opd_loss, metrics = compute_losses(
                model=student,
                teacher_model=teacher,
                batch=batch,
                tokenizer=tokenizer,
                teacher_tokenizer=teacher_tokenizer,
                teacher_to_student=teacher_to_student,
                student_to_teacher=student_to_teacher,
                logits_processor=logits_processor,
                eos_token_ids=eos_token_ids,
                args=args,
                device=device,
            )
            compute_time = time.time() - compute_start
            if int(args.debug_opd_steps) > 0 and global_step < int(args.debug_opd_steps):
                print(
                    json.dumps(
                        {
                            "debug_rank": rank,
                            "next_step": global_step + 1,
                            "rollout_time": float(metrics.get("rollout_time", 0.0)),
                            "teacher_time": float(metrics.get("teacher_time", 0.0)),
                            "teacher_parse_time": float(metrics.get("teacher_parse_time", 0.0)),
                            "asr_loss": float(asr_loss.detach().item()),
                            "asr_loss_time": float(metrics.get("asr_loss_time", 0.0)),
                            "student_score_time": float(metrics.get("student_score_time", 0.0)),
                            "opd_loss_time": float(metrics.get("opd_loss_time", 0.0)),
                            "asr_terminal_positions": float(metrics.get("asr_terminal_positions", 0.0)),
                            "asr_terminal_expected_positions": float(metrics.get("asr_terminal_expected_positions", 0.0)),
                            "local_valid_count": float(metrics.get("local_valid_count", 0.0)),
                            "generated_tokens_mean": float(metrics.get("generated_tokens_mean", 0.0)),
                            "generated_stopped": float(metrics.get("generated_stopped", 0.0)),
                            "teacher_forced_fallback": float(metrics.get("teacher_forced_fallback", 0.0)),
                            "teacher_alignment_pad_offset_mean": float(
                                metrics.get("teacher_alignment_pad_offset_mean", 0.0)
                            ),
                            "teacher_alignment_pad_offset_max": float(
                                metrics.get("teacher_alignment_pad_offset_max", 0.0)
                            ),
                            "retokenize_mismatch_ratio": float(metrics.get("retokenize_mismatch_ratio", 0.0)),
                            "retokenize_mismatch_rows": float(metrics.get("retokenize_mismatch_rows", 0.0)),
                            "opd_nonterminal_rows": float(metrics.get("opd_nonterminal_rows", 0.0)),
                            "opd_nonterminal_eos_candidate_rows": float(
                                metrics.get("opd_nonterminal_eos_candidate_rows", 0.0)
                            ),
                            "opd_terminal_rows": float(metrics.get("opd_terminal_rows", 0.0)),
                            "opd_terminal_eos_candidate_rows": float(
                                metrics.get("opd_terminal_eos_candidate_rows", 0.0)
                            ),
                            "teacher_language_counts": metrics.get("teacher_language_counts", {}),
                            "local_prompt_width": float(metrics.get("local_prompt_width", 0.0)),
                            "global_prompt_width": float(metrics.get("global_prompt_width", 0.0)),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            backward_start = time.time()
            total_loss.backward()
            grad_norm = fsdp2_clip_grad_norm_(student.parameters(), max_norm=float(args.grad_clip))
            backward_time = time.time() - backward_start
            optimizer_start = time.time()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer_time = time.time() - optimizer_start
            global_step += 1

            total_loss_avg = all_reduce_sum(float(total_loss.detach().item()), device) / world_size
            asr_loss_avg = all_reduce_sum(float(asr_loss.detach().item()), device) / world_size
            opd_loss_avg = all_reduce_sum(float(opd_loss.detach().item()), device) / world_size
            asr_terminal_positions_sum = all_reduce_sum(float(metrics.get("asr_terminal_positions", 0.0)), device)
            asr_terminal_expected_positions_sum = all_reduce_sum(
                float(metrics.get("asr_terminal_expected_positions", 0.0)),
                device,
            )
            opd_nonterminal_rows_sum = all_reduce_sum(float(metrics.get("opd_nonterminal_rows", 0.0)), device)
            opd_nonterminal_eos_candidate_rows_sum = all_reduce_sum(
                float(metrics.get("opd_nonterminal_eos_candidate_rows", 0.0)),
                device,
            )
            opd_terminal_rows_sum = all_reduce_sum(float(metrics.get("opd_terminal_rows", 0.0)), device)
            opd_terminal_eos_candidate_rows_sum = all_reduce_sum(
                float(metrics.get("opd_terminal_eos_candidate_rows", 0.0)),
                device,
            )
            teacher_forced_fallback_sum = all_reduce_sum(float(metrics.get("teacher_forced_fallback", 0.0)), device)
            retokenize_compare_tokens_sum = all_reduce_sum(float(metrics.get("retokenize_compare_tokens", 0.0)), device)
            retokenize_mismatch_tokens_sum = all_reduce_sum(float(metrics.get("retokenize_mismatch_tokens", 0.0)), device)
            retokenize_mismatch_rows_sum = all_reduce_sum(float(metrics.get("retokenize_mismatch_rows", 0.0)), device)
            timing_metrics = {
                "data_wait_max": all_reduce_max(data_wait, device),
                "move_batch_max": all_reduce_max(move_time, device),
                "compute_max": all_reduce_max(compute_time, device),
                "backward_max": all_reduce_max(backward_time, device),
                "optimizer_max": all_reduce_max(optimizer_time, device),
                "prompt_width_sync_time_max": all_reduce_max(float(metrics.get("prompt_width_sync_time", 0.0)), device),
                "rollout_time_max": all_reduce_max(float(metrics.get("rollout_time", 0.0)), device),
                "teacher_time_max": all_reduce_max(float(metrics.get("teacher_time", 0.0)), device),
                "teacher_time_min": all_reduce_min(float(metrics.get("teacher_time", 0.0)), device),
                "teacher_parse_time_max": all_reduce_max(float(metrics.get("teacher_parse_time", 0.0)), device),
                "local_valid_count_min": all_reduce_min(float(metrics.get("local_valid_count", 0.0)), device),
                "local_valid_count_max": all_reduce_max(float(metrics.get("local_valid_count", 0.0)), device),
                "local_prompt_width_min": all_reduce_min(float(metrics.get("local_prompt_width", 0.0)), device),
                "local_prompt_width_max": all_reduce_max(float(metrics.get("local_prompt_width", 0.0)), device),
                "global_prompt_width": all_reduce_max(float(metrics.get("global_prompt_width", 0.0)), device),
                "reduce_count_time_max": all_reduce_max(float(metrics.get("reduce_count_time", 0.0)), device),
                "asr_loss_time_max": all_reduce_max(float(metrics.get("asr_loss_time", 0.0)), device),
                "student_score_time_max": all_reduce_max(float(metrics.get("student_score_time", 0.0)), device),
                "opd_loss_time_max": all_reduce_max(float(metrics.get("opd_loss_time", 0.0)), device),
            }
            if rank == 0:
                elapsed = time.time() - step_start
                record = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": total_loss_avg,
                    "asr_loss": asr_loss_avg,
                    "opd_loss": opd_loss_avg,
                    "opd_loss_weight": 1.0 if str(args.teacher_backend).startswith("qwen3_asr_") else float(args.opd_loss_weight),
                    "asr_terminal_loss_weight": float(metrics.get("asr_terminal_loss_weight", 0.0)),
                    "asr_terminal_positions": asr_terminal_positions_sum,
                    "asr_terminal_expected_positions": asr_terminal_expected_positions_sum,
                    "teacher_forced_fallback": teacher_forced_fallback_sum / float(world_size),
                    "retokenize_mismatch_ratio": (
                        retokenize_mismatch_tokens_sum / max(retokenize_compare_tokens_sum, 1.0)
                    ),
                    "retokenize_mismatch_rows": retokenize_mismatch_rows_sum,
                    "teacher_alignment_pad_offset_mean": metrics.get("teacher_alignment_pad_offset_mean", 0.0),
                    "teacher_alignment_pad_offset_max": metrics.get("teacher_alignment_pad_offset_max", 0.0),
                    "opd_nonterminal_rows": opd_nonterminal_rows_sum,
                    "opd_nonterminal_eos_candidate_rows": opd_nonterminal_eos_candidate_rows_sum,
                    "opd_terminal_rows": opd_terminal_rows_sum,
                    "opd_terminal_eos_candidate_rows": opd_terminal_eos_candidate_rows_sum,
                    "teacher_backend": args.teacher_backend,
                    "opd_student_score_mode": args.opd_student_score_mode,
                    "generated_nonempty": metrics["generated_nonempty"],
                    "generated_tokens_mean": metrics["generated_tokens_mean"],
                    "generated_stopped": float(metrics.get("generated_stopped", 0.0)),
                    "opd_valid_topk_mean": metrics["opd_valid_topk_mean"],
                    "teacher_language_counts": metrics.get("teacher_language_counts", {}),
                    "grad_norm": float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "time_per_step": elapsed,
                    **timing_metrics,
                }
                print(json.dumps(record, ensure_ascii=False), flush=True)

            if checkpoint_manager is not None and int(args.save_freq) > 0 and global_step % int(args.save_freq) == 0:
                save_checkpoint(checkpoint_manager, args.output_dir, global_step, rank, args.max_ckpt_to_keep)

            if max_steps > 0 and global_step >= max_steps:
                if checkpoint_manager is not None and int(args.save_freq) > 0:
                    save_checkpoint(checkpoint_manager, args.output_dir, global_step, rank, args.max_ckpt_to_keep)
                dist.barrier()
                dist.destroy_process_group()
                return

    if checkpoint_manager is not None and int(args.save_freq) > 0:
        save_checkpoint(checkpoint_manager, args.output_dir, global_step, rank, args.max_ckpt_to_keep)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

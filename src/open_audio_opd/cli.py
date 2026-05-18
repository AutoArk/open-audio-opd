from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_audio_opd.config import load_config
from open_audio_opd.data import load_jsonl_asr_dataset
from open_audio_opd.training import run_training


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="open-audio-opd")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config", help="validate a YAML config")
    validate_parser.add_argument("--config", required=True)

    smoke_parser = subparsers.add_parser("smoke", help="run a minimal end-to-end OPD step")
    smoke_parser.add_argument("--config", required=True)

    train_parser = subparsers.add_parser("train", help="run the default single-process trainer")
    train_parser.add_argument("--config", required=True)

    data_parser = subparsers.add_parser("validate-data", help="validate JSONL ASR data")
    data_parser.add_argument("--data", required=True)
    data_parser.add_argument("--require-audio-exists", action="store_true")

    export_parser = subparsers.add_parser("export-fsdp", help="print clean-room export guidance")
    export_parser.add_argument("--checkpoint-dir", required=True)
    export_parser.add_argument("--template-model-dir", required=True)
    export_parser.add_argument("--target-dir", required=True)

    args = parser.parse_args(argv)

    if args.command == "validate-config":
        config = load_config(args.config)
        print(json.dumps({"ok": True, "config": _config_summary(config)}, indent=2))
    elif args.command == "validate-data":
        samples = load_jsonl_asr_dataset(args.data, require_audio_exists=args.require_audio_exists)
        print(json.dumps({"ok": True, "samples": len(samples)}, indent=2))
    elif args.command == "smoke":
        result = run_training(load_config(args.config), smoke=True)
        print(json.dumps({"ok": True, **result}, indent=2))
    elif args.command == "train":
        result = run_training(load_config(args.config), smoke=False)
        print(json.dumps({"ok": True, **result}, indent=2))
    elif args.command == "export-fsdp":
        _print_export_guidance(
            checkpoint_dir=Path(args.checkpoint_dir),
            template_model_dir=Path(args.template_model_dir),
            target_dir=Path(args.target_dir),
        )
    else:
        parser.error(f"unknown command: {args.command}")


def _config_summary(config: object) -> dict[str, object]:
    return {
        "type": type(config).__name__,
    }


def _print_export_guidance(checkpoint_dir: Path, template_model_dir: Path, target_dir: Path) -> None:
    print(
        "\n".join(
            [
                "FSDP export is model-stack specific in v1.",
                "Use this command as the expected workflow contract:",
                f"  checkpoint_dir={checkpoint_dir}",
                f"  template_model_dir={template_model_dir}",
                f"  target_dir={target_dir}",
                "A real exporter should merge complete model_world_size_*_rank_*.pt shards",
                "into a fresh copy of the original inference-compatible HF model template.",
            ]
        )
    )

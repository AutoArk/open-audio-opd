from __future__ import annotations

import importlib
from typing import Any

from open_audio_opd.adapters.toy import ToyStudentPolicy, ToyTeacherScorer


def load_object(spec: str, **kwargs: Any) -> Any:
    if spec == "toy.student":
        return ToyStudentPolicy(**kwargs)
    if spec == "toy.teacher":
        return ToyTeacherScorer(**kwargs)
    if spec == "toy":
        raise ValueError("use toy.student or toy.teacher")

    module_name, sep, attr = spec.partition(":")
    if not sep:
        raise ValueError(f"adapter spec must be built-in or module:attr, got {spec!r}")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    return factory(**kwargs)

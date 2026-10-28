from __future__ import annotations
from typing import Literal

DeviceSpec = Literal["auto", "cpu", "cuda", "mps"]

def resolve_device(requested: DeviceSpec | str) -> str:
    """Пытается согласовать запрошенный девайс"""

    try: 
        import torch
    except ImportError:
        return "cpu"

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return requested

def resolve_precision(requested: bool | str, device: str, kind: str) -> bool:
    """Пытается согласовать флаг смешанной точности по устройству"""

    if requested != "auto":
        return bool(requested)
    if device != "cuda":
        return False
    if kind == "fp16":
        return True
    try:
        import torch

        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False
    
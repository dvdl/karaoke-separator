"""GPU check helpers for Runpod serverless worker.

Lightweight module with no heavy dependencies (torch/runpod/audio_separator),
so it can be imported and tested quickly without CUDA.
"""
import re

_MIG_TOKEN_RE = re.compile(r"\bMIG\b", re.IGNORECASE)


def is_mig(name: str) -> bool:
    """Return True if device name contains 'MIG' as a separate token (case-insensitive)."""
    if not name or not isinstance(name, str):
        return False
    return bool(_MIG_TOKEN_RE.search(name))

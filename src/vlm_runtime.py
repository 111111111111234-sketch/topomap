"""Runtime VLM provider settings shared by GOAT auxiliary modules.

Defaults retain the repository's original Requesty/GPT-4o behavior.  GOAT
evaluations may override them once at startup so the explorer, hypothesis
predictor, and semantic critic use the same provider and model.
"""

import os
import time
from typing import Any, Dict

from openai import OpenAI
from omegaconf import OmegaConf

from src.const import END_POINT


_DEFAULTS: Dict[str, Any] = {
    "model": "openai/gpt-4o",
    "base_url": END_POINT,
    "api_key_env": "REQUESTY_API_KEY",
    "extra_body": None,
    "reasoning_content_fallback": False,
    "structured_choice_fallback": False,
}
_runtime: Dict[str, Any] = dict(_DEFAULTS)
_clients: Dict[tuple, OpenAI] = {}
_telemetry: Dict[str, Any] = {}


def reset_vlm_telemetry() -> None:
    global _telemetry
    _telemetry = {
        "logical_calls": 0,
        "http_attempts": 0,
        "successes": 0,
        "failures": 0,
        "elapsed_seconds": 0.0,
        "by_purpose": {},
    }


reset_vlm_telemetry()


def configure_vlm_runtime(cfg) -> None:
    """Apply optional VLM overrides from an OmegaConf/dict-like config."""
    global _runtime
    _runtime = dict(_DEFAULTS)
    reset_vlm_telemetry()
    for runtime_key, config_key in (
        ("model", "vlm_model"),
        ("base_url", "vlm_base_url"),
        ("api_key_env", "vlm_api_key_env"),
        ("extra_body", "vlm_extra_body"),
        ("reasoning_content_fallback", "vlm_reasoning_content_fallback"),
        ("structured_choice_fallback", "vlm_structured_choice_fallback"),
    ):
        value = cfg.get(config_key, None)
        if value is not None:
            if config_key == "vlm_extra_body":
                value = OmegaConf.to_container(value, resolve=True)
            _runtime[runtime_key] = value


def get_vlm_runtime() -> Dict[str, Any]:
    return dict(_runtime)


def begin_vlm_call(purpose: str = "unspecified") -> float:
    from src.hgr_timing import active_timer
    timer = active_timer.get()
    if timer is not None:
        timer.push("model_request")
    _telemetry["logical_calls"] += 1
    bucket = _telemetry["by_purpose"].setdefault(purpose, {})
    bucket["logical_calls"] = bucket.get("logical_calls", 0) + 1
    return time.monotonic()


def record_vlm_attempt(purpose: str = "unspecified") -> None:
    _telemetry["http_attempts"] += 1
    bucket = _telemetry["by_purpose"].setdefault(purpose, {})
    bucket["http_attempts"] = bucket.get("http_attempts", 0) + 1


def finish_vlm_call(started_at: float, success: bool, purpose: str = "unspecified") -> None:
    from src.hgr_timing import active_timer
    timer = active_timer.get()
    if timer is not None:
        timer.pop()
    key = "successes" if success else "failures"
    _telemetry[key] += 1
    elapsed = max(0.0, time.monotonic() - started_at)
    _telemetry["elapsed_seconds"] += elapsed
    bucket = _telemetry["by_purpose"].setdefault(purpose, {})
    bucket[key] = bucket.get(key, 0) + 1
    bucket["elapsed_seconds"] = bucket.get("elapsed_seconds", 0.0) + elapsed


def get_vlm_telemetry() -> Dict[str, Any]:
    import copy
    return copy.deepcopy(_telemetry)


def get_vlm_client(base_url=None, api_key_env=None) -> OpenAI:
    base_url = base_url or _runtime["base_url"]
    api_key_env = api_key_env or _runtime["api_key_env"]
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")
    cache_key = (base_url, api_key_env, api_key)
    if cache_key not in _clients:
        _clients[cache_key] = OpenAI(base_url=base_url, api_key=api_key)
    return _clients[cache_key]

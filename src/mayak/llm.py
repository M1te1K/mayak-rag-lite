"""Small OpenAI-compatible HTTP client. Zero automatic retries or paid fallbacks."""

import json
import math
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Protocol

import httpx
from pydantic import BaseModel, ValidationError

from .audit import Audit
from .config import Settings
from .general import has_unresolved_placeholder


class ServiceError(Exception):
    def __init__(self, code: str, retry_after: float | None = None):
        super().__init__(code)
        self.code, self.retry_after = code, retry_after


class LLM(Protocol):
    def complete(
        self,
        messages: list[dict],
        result_type: type[BaseModel] | None = None,
        purpose: str = "general",
    ) -> str | BaseModel: ...


def retry_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = (date - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def duration_seconds(value):
    """Groq reset durations such as 1m2.5s; no arbitrary headers are recorded."""
    if not value or not re.fullmatch(r"(?:[0-9.]+[hms])+", value):
        return None
    try:
        result = sum(
            float(n) * {"h": 3600, "m": 60, "s": 1}[unit]
            for n, unit in re.findall(r"([0-9.]+)([hms])", value)
        )
        return result if math.isfinite(result) else None
    except ValueError:
        return None


class HTTPClient:
    def __init__(self, settings: Settings, audit: Audit, transport=None):
        settings.validate()
        self.settings, self.audit = settings, audit
        self.http = httpx.Client(
            timeout=settings.timeout, transport=transport, follow_redirects=False, trust_env=False
        )
        self.cooldown_until = 0.0
        self.token_limit = None
        self.tokens_remaining = None
        self.token_reset_at = 0.0

    def close(self):
        self.http.close()

    def before_request(self, payload):
        """Optional pacing hook for the explicit live-check script; no retries."""

    def complete(self, messages, result_type=None, purpose="general"):
        s = self.settings
        if not s.api_key:
            raise ServiceError("missing_key")
        remaining = self.cooldown_until - time.monotonic()
        if remaining > 0:
            raise ServiceError("rate_limit_cooldown", remaining)
        base = (
            "https://api.groq.com/openai/v1"
            if s.provider == "groq"
            else "https://openrouter.ai/api/v1"
        )
        messages = [dict(m) for m in messages]
        payload = {
            "model": s.model,
            "messages": messages,
            "temperature": 0,
            "max_completion_tokens" if s.provider == "groq" else "max_tokens": s.max_tokens,
        }
        if s.provider == "groq":
            payload.update(reasoning_effort="low", include_reasoning=False)
        if s.provider == "openrouter":
            payload["provider"] = {
                "allow_fallbacks": False,
                "max_price": {"prompt": 0, "completion": 0},
                "require_parameters": True,
            }
        if result_type:
            schema = result_type.model_json_schema()
            if s.json_mode == "schema" or (s.json_mode == "auto" and s.provider == "groq"):
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": result_type.__name__, "strict": True, "schema": schema},
                }
            else:
                # Catalog reports response_format but no structured_outputs for free Gemma.
                # Prompt JSON avoids silently assuming constrained decoding is available.
                messages[0]["content"] += "\nВерни только JSON по схеме: " + json.dumps(schema)
        self.before_request(payload)
        self.audit.emit(
            "tool_call",
            tool="llm",
            purpose=purpose,
            provider=s.provider,
            model=s.model,
            message_chars=sum(len(m["content"]) for m in messages),
        )
        try:
            response = self.http.post(
                base + "/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer " + s.api_key},
            )
        except httpx.TimeoutException:
            raise ServiceError("timeout") from None
        except httpx.HTTPError:
            raise ServiceError("network") from None

        # Read only numeric quota metadata, never log headers or HTTP error bodies.
        def positive_int(name):
            value = response.headers.get(name, "")
            return int(value) if value.isascii() and value.isdigit() and len(value) < 12 else None

        self.token_limit = positive_int("x-ratelimit-limit-tokens")
        self.tokens_remaining = positive_int("x-ratelimit-remaining-tokens")
        reset = duration_seconds(response.headers.get("x-ratelimit-reset-tokens"))
        self.token_reset_at = time.monotonic() + (reset if reset is not None else 65)
        self.audit.emit(
            "tool_result",
            tool="llm",
            purpose=purpose,
            http_status=response.status_code,
            token_limit=self.token_limit,
            tokens_remaining=self.tokens_remaining,
            reset_seconds=reset,
        )
        if response.status_code == 429:
            retry = retry_seconds(response.headers.get("retry-after"))
            # No sleeping/retrying; subsequent messages respect provider's deadline.
            self.cooldown_until = time.monotonic() + (retry if retry is not None else 60)
            raise ServiceError("rate_limit", retry)
        if response.status_code != 200:
            code = (
                "invalid_key"
                if response.status_code in {401, 403}
                else f"http_{response.status_code}"
            )
            raise ServiceError(code)
        try:
            data = response.json()
            if data.get("error"):
                raise ValueError
            usage = data.get("usage") or {}
            safe_usage = {
                k: usage[k]
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                if isinstance(usage.get(k), int) and not isinstance(usage[k], bool)
            }
            if safe_usage:
                self.audit.emit("llm_usage", purpose=purpose, **safe_usage)
            choice = data["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError
            if len(content) > 30000:
                raise ValueError
            if result_type:
                return result_type.model_validate_json(content, strict=True)
            if purpose == "general" and has_unresolved_placeholder(content):
                raise ValueError
            return content.strip()
        except (ValueError, KeyError, IndexError, TypeError, AttributeError, ValidationError):
            raise ServiceError("invalid_response") from None

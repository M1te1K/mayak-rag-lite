import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from mayak.answer import Evidence
from mayak.config import ConfigError, Settings
from mayak.llm import HTTPClient, ServiceError, retry_seconds


def response(content="Ответ", finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": content}}]}


def client(settings, audit, handler):
    return HTTPClient(
        replace(settings, api_key="test-secret"), audit, transport=httpx.MockTransport(handler)
    )


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, "invalid_key"),
        (403, "invalid_key"),
        (402, "http_402"),
        (429, "rate_limit"),
        (500, "http_500"),
        (503, "http_503"),
    ],
)
def test_status_errors(settings, audit, status, expected):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "test-secret"}, headers={"Retry-After": "30"})

    c = client(settings, audit, handler)
    try:
        with pytest.raises(ServiceError, match=expected):
            c.complete([{"role": "user", "content": "hi"}])
        assert len(calls) == 1
        if status == 429:
            with pytest.raises(ServiceError, match="cooldown"):
                c.complete([{"role": "user", "content": "retry"}])
            assert len(calls) == 1
    finally:
        c.close()


@pytest.mark.parametrize(
    "kind",
    ["timeout", "network", "bad_json", "empty", "length", "shape", "embedded_error", "bad_schema"],
)
def test_bad_api_responses(settings, audit, kind):
    def handler(request):
        if kind == "timeout":
            raise httpx.ReadTimeout("secret must never be logged", request=request)
        if kind == "network":
            raise httpx.ConnectError("test-secret", request=request)
        if kind == "bad_json":
            return httpx.Response(200, text="<html>broken</html>")
        data = {
            "empty": response(""),
            "length": response("partial", "length"),
            "shape": {},
            "embedded_error": {"error": "test-secret"},
            "bad_schema": response('{"has_evidence":"yes"}'),
        }[kind]
        return httpx.Response(200, json=data)

    c = client(settings, audit, handler)
    try:
        with pytest.raises(ServiceError):
            c.complete(
                [{"role": "system", "content": "JSON"}], Evidence if kind == "bad_schema" else None
            )
    finally:
        c.close()


def test_groq_strict_schema_and_valid_response(settings, audit):
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=response(
                json.dumps(
                    {"answer": "", "has_evidence": False, "source_chunk_ids": [], "quotes": []}
                )
            ),
        )

    c = client(settings, audit, handler)
    try:
        result = c.complete([{"role": "system", "content": "test"}], Evidence)
        assert not result.has_evidence
        schema = captured[0]["response_format"]["json_schema"]
        assert schema["strict"] is True
        assert schema["schema"]["additionalProperties"] is False
        assert "tools" not in captured[0] and "models" not in captured[0]
    finally:
        c.close()


def test_openrouter_free_model_prompt_json_and_no_fallback(settings, audit):
    s = replace(settings, provider="openrouter", model="google/gemma-4-31b-it:free")
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=response('{"answer":"","has_evidence":false,"source_chunk_ids":[],"quotes":[]}'),
        )

    c = client(s, audit, handler)
    try:
        c.complete([{"role": "system", "content": "test"}], Evidence)
        payload = captured[0]
        assert payload["model"].endswith(":free")
        assert payload["provider"]["allow_fallbacks"] is False
        assert payload["provider"]["max_price"] == {"prompt": 0, "completion": 0}
        assert "response_format" not in payload
        assert "JSON" in payload["messages"][0]["content"]
    finally:
        c.close()


def test_retry_after_numeric_date_and_invalid():
    assert retry_seconds("30") == 30
    future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=90))
    assert 85 < retry_seconds(future) <= 90
    assert retry_seconds("garbage") is None
    assert retry_seconds("NaN") is None
    assert retry_seconds("-2") == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"provider": "paid"},
        {"model": "paid-model"},
        {"provider": "openrouter", "model": "google/gemma-4-31b-it"},
        {"chunk_tokens": 512},
        {"chunk_overlap": 221},
        {"top_k": 9},
        {"timeout": 0},
        {"context_chars": 0},
        {"embedding_model": "other"},
    ],
)
def test_invalid_or_paid_settings_rejected(changes):
    with pytest.raises(ConfigError):
        replace(Settings(), **changes).validate()


def test_missing_key_no_request(settings, audit):
    def handler(request):
        pytest.fail("API should not be called without a key")

    c = HTTPClient(settings, audit, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ServiceError, match="missing_key"):
            c.complete([])
    finally:
        c.close()

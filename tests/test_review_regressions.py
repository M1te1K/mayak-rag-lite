"""Behavioural regressions from the independent review, no live API calls."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import httpx
import pytest
from conftest import FakeLLM, records

from mayak.answer import NO_EVIDENCE, UNAVAILABLE
from mayak.bot import Bot
from mayak.general import CURRENT_DATA_LIMIT
from mayak.llm import HTTPClient, ServiceError
from mayak.pdf import Chunk
from mayak.search import (
    CONTEXT_UNAVAILABLE,
    ContextError,
    Search,
    fit_context,
    validate_chunk_budget,
)


@pytest.mark.parametrize(
    "question",
    [
        "Какой курс доллара",
        "Курс USD к рублю?",
        "Сколько сейчас стоит доллар?",
        "какое время в японии",
        "Который час в Токио?",
        "Сколько сейчас времени?",
    ],
)
def test_live_data_never_uses_fabricated_model_value(question, settings, audit, search):
    llm = FakeLLM("Курс 95 рублей. [текущее время по вашему запросу]")
    bot = Bot(settings, llm, search, audit)
    assert bot.handle("/ask " + question).text == CURRENT_DATA_LIMIT
    assert llm.calls == []
    assert bot.history.turns[-1].answer == CURRENT_DATA_LIMIT
    assert "95" not in json.dumps(bot.history.messages("ask"), ensure_ascii=False)


@pytest.mark.parametrize(
    "first,followups",
    [
        (
            "Какой курс доллара",
            ["а евро?", "а на сегодня?", "а сколько это будет?", "хотя бы примерно"],
        ),
        ("Какое время в Японии?", ["а в Китае?", "а там?", "точнее", "а сейчас?"]),
    ],
)
def test_live_data_followups_and_reset(first, followups, settings, audit, search):
    llm = FakeLLM("В байте 8 битов.")
    bot = Bot(settings, llm, search, audit)
    for question in [first, *followups]:
        assert bot.handle("/ask " + question).text == CURRENT_DATA_LIMIT
    assert not llm.calls
    bot.handle("/reset")
    assert bot.history.current_data_topic is None and not bot.history.turns
    assert "8" in bot.handle("/ask Сколько битов в байте?").text
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    "question,answer",
    [
        ("Что такое RAG?", "RAG дополняет запрос найденным контекстом."),
        ("Сколько битов в байте?", "В байте 8 битов."),
        ("А сколько битов в байте?", "В байте 8 битов."),
        ("Что означает курс валют?", "Это соотношение стоимости двух валют."),
        ("Как устроены часовые пояса?", "Они задают смещение местного времени относительно UTC."),
        ("Какая разница во времени между UTC+3 и UTC+9?", "Разница составляет 6 часов."),
        (
            "Сколько времени нужно для изучения Python?",
            "Продолжительность зависит от цели и практики.",
        ),
        ("Какое время использовать в базе данных?", "Обычно сохраняют время в UTC."),
        ("Как выбрать курс Python?", "Сравните программу и количество практических задач."),
        ("Какой курс Python выбрать новичку?", "Сравните программу и практические задачи."),
        ("Как индексировать список?", "Например, `items[0]` возвращает первый элемент."),
        ("Как устроен шаблон?", "Пример: `{{current_time}}` внутри кода шаблона."),
    ],
)
def test_static_general_answers_survive_live_topic(question, answer, settings, audit, search):
    llm = FakeLLM(answer)
    bot = Bot(settings, llm, search, audit)
    bot.handle("/ask Какое время в Японии?")
    assert bot.handle("/ask " + question).text == answer
    assert len(llm.calls) == 1 and bot.history.current_data_topic is None


@pytest.mark.parametrize(
    "placeholder",
    [
        "[текущее время по вашему запросу]",
        "Значение: {{current_rate}}",
        "Ответ: <insert_value>",
        "[вставьте ответ здесь]",
    ],
)
def test_placeholder_not_delivered_or_saved_even_from_fake(placeholder, settings, audit, search):
    bot = Bot(settings, FakeLLM(placeholder, "RAG добавляет контекст."), search, audit)
    assert bot.handle("/ask Расскажи о RAG").text == UNAVAILABLE
    assert bot.history.turns == []
    assert "RAG" in bot.handle("Ещё раз").text
    assert all(placeholder not in m["content"] for m in bot.history.messages("ask"))


def test_placeholder_http_mock_rejected_at_client_and_bot(settings, audit, search):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "[текущее время по вашему запросу]"},
                    }
                ]
            },
        )

    client = HTTPClient(
        replace(settings, api_key="test-secret"), audit, transport=httpx.MockTransport(handler)
    )
    try:
        assert (
            Bot(settings, client, search, audit).handle("/ask Расскажи о RAG").text == UNAVAILABLE
        )
    finally:
        client.close()


@pytest.mark.parametrize("budget,valid", [(47, False), (48, True), (49, True)])
def test_each_chunk_must_fit_including_exact_boundary(budget, valid):
    chunks = [Chunk("short", 1, "a" * 10, "demo.pdf", 0), Chunk("long", 2, "b" * 48, "demo.pdf", 0)]
    if valid:
        validate_chunk_budget(chunks, budget)
    else:
        with pytest.raises(ContextError, match="CONTEXT_MAX_CHARS"):
            validate_chunk_budget(chunks, budget)


def test_empty_context_is_technical_failure(settings, audit, search, monkeypatch):
    with pytest.raises(ContextError):
        fit_context([], 1000)
    monkeypatch.setattr(search, "rank", lambda *args: [])
    llm = FakeLLM()
    bot = Bot(settings, llm, search, audit)
    assert bot.handle("/doc Цена Старт?").text == CONTEXT_UNAVAILABLE
    assert not llm.calls and bot.history.turns == []
    assert any(r.get("error") == "context_unavailable" for r in records(settings))


def test_combined_budget_must_not_turn_uninspected_chunks_into_absence(settings, audit):
    from conftest import ToyEmbedder

    from mayak.answer import Evidence

    chunks = [Chunk(f"c{i}", 1, "a" * 600, "demo.pdf", i) for i in range(4)]
    search = Search(chunks, ToyEmbedder())
    llm = FakeLLM(Evidence(answer="", has_evidence=False, source_chunk_ids=[], quotes=[]))
    bot = Bot(replace(settings, context_chars=1000), llm, search, audit)
    assert bot.handle("/doc Неизвестный факт?").text == CONTEXT_UNAVAILABLE
    assert len(llm.calls) == 1
    assert any(r["event"] == "context_incomplete" for r in records(settings))
    assert not any(r.get("answer") == NO_EVIDENCE for r in records(settings))


def test_all_request_events_share_id_across_expansion_and_error(settings, audit, search):
    from test_bot import absent

    llm = FakeLLM(absent(), absent(), ServiceError("timeout"))
    bot = Bot(settings, llm, search, audit)
    assert bot.handle("/doc Кто владелец компании?").text == NO_EVIDENCE
    assert bot.handle("/ask Расскажи о RAG").text == UNAVAILABLE
    bot.handle("/reset")
    audit.emit("document_loaded", source="demo.pdf")
    rows = records(settings)
    requests = [r for r in rows if r["event"] == "request"]
    assert len({r["turn_id"] for r in requests}) == 3
    for i, request in enumerate(requests):
        group = [r for r in rows if r["turn_id"] == request["turn_id"]]
        assert group[0]["event"] == "request" and group[-1]["event"] == "response"
        assert all(r["scope"] == "request" for r in group)
        if i == 0:
            assert {"tool_call", "retrieved", "passed", "cited"} <= {r["event"] for r in group}
            assert any(r.get("tool") == "expand_search" for r in group)
        if i == 1:
            assert any(r.get("error") == "timeout" for r in group)
    assert rows[-1]["turn_id"] is None and rows[-1]["scope"] == "session"


def test_audit_scope_isolated_between_concurrent_requests(audit, settings):
    barrier = Barrier(2)

    def run(number):
        with audit.request_scope():
            audit.emit("request", query=str(number))
            barrier.wait(timeout=5)
            audit.emit("tool_call", tool="llm", query=str(number), text="test-secret")
            audit.emit("response", answer=str(number))

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, [1, 2]))
    rows = records(settings)
    ids = {r["turn_id"] for r in rows}
    assert len(ids) == 2 and None not in ids
    for tid in ids:
        group = [r for r in rows if r["turn_id"] == tid]
        assert len(group) == 3 and group[0]["query"] == group[-1]["answer"]
    assert "test-secret" not in settings.log_path.read_text()


def test_http_calls_usage_and_errors_inherit_request_id(settings, audit, search):
    def handler(request):
        body = json.loads(request.content)
        if "ошибка" in body["messages"][-1]["content"]:
            raise httpx.ReadTimeout("test-secret", request=request)
        schema = body.get("response_format", {}).get("json_schema", {}).get("name")
        content = (
            json.dumps({"answer": "", "has_evidence": False, "source_chunk_ids": [], "quotes": []})
            if schema == "Evidence"
            else "RAG добавляет контекст."
        )
        return httpx.Response(
            200,
            headers={
                "x-ratelimit-limit-tokens": "8000",
                "x-ratelimit-remaining-tokens": "6500",
                "x-ratelimit-reset-tokens": "1m2.5s",
                "x-private": "test-secret",
            },
            json={
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "private": "test-secret",
                },
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            },
        )

    client = HTTPClient(
        replace(settings, api_key="test-secret"), audit, transport=httpx.MockTransport(handler)
    )
    try:
        bot = Bot(settings, client, search, audit)
        bot.handle("/ask RAG?")
        assert bot.handle("/doc Кто владелец компании?").text == NO_EVIDENCE
        assert bot.handle("/ask ошибка").text == UNAVAILABLE
        rows = records(settings)
        assert all(r["turn_id"] and r["scope"] == "request" for r in rows)
        for r in [r for r in rows if r["event"] == "request"]:
            group = [e for e in rows if e["turn_id"] == r["turn_id"]]
            assert any(e.get("tool") == "llm" for e in group)
        assert len({r["turn_id"] for r in rows}) == 3
        assert any(r["event"] == "llm_usage" and r["total_tokens"] == 120 for r in rows)
        assert any(r.get("tool") == "expand_search" for r in rows)
        assert client.token_limit == 8000 and client.tokens_remaining == 6500
        assert "x-private" not in settings.log_path.read_text()
        assert "test-secret" not in settings.log_path.read_text()
    finally:
        client.close()


def test_explicit_live_check_paces_before_new_call(settings, audit, monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "live_check", Path(__file__).parents[1] / "scripts/live_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    now = [100.0]
    waits = []

    def sleep(seconds):
        assert 0 <= seconds <= 10
        waits.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.time, "sleep", sleep)
    client = module.PacedClient(settings, audit)
    try:
        client.token_limit, client.tokens_remaining, client.token_reset_at = 8000, 0, 101
        client.before_request({"messages": [{"content": "RAG?"}]})
        assert sum(waits) == pytest.approx(1.25)
        assert any(r["event"] == "quota_wait" for r in records(settings))
    finally:
        client.close()

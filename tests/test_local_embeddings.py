"""Explicit opt-in: real E5, optionally mocked HTTP; never contacts an LLM API."""

import json
import os
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from mayak.answer import NO_EVIDENCE, UNAVAILABLE
from mayak.audit import Audit
from mayak.bot import CLARIFY, Bot
from mayak.config import Settings
from mayak.llm import HTTPClient
from mayak.search import Search, fit_context

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_LOCAL_EMBEDDINGS") != "1", reason="set RUN_LOCAL_EMBEDDINGS=1 for real E5"
    ),
]
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real_search():
    return Search.build(replace(Settings(), pdf_path=ROOT / "output/pdf/mayak-demo.pdf"))


def test_real_russian_retrieval_and_prefixes(real_search):
    search = real_search
    cases = json.loads((ROOT / "data/evaluation.json").read_text())
    for case in cases:
        if case["page"] is None:
            continue  # Retrieval scores cannot prove absence of evidence.
        ranked = search.rank(case["question"], 6)
        hits = fit_context(search.expand(ranked, 6), 4800)
        assert any(
            case["expected"] in h.chunk.text and h.chunk.page == case["page"] for h in hits
        ), case
    tokenizer = search.embedder.tokenizer
    assert all(len(tokenizer.encode("passage: " + c.text)) <= 512 for c in search.chunks)
    query = search.embedder.encode(["Цена тарифа?"], query=True)
    passage = search.embedder.encode(["Тариф стоит 390 рублей."])
    assert query.shape == passage.shape == (1, 384)


def test_real_pdf_search_to_mock_http_and_dialogue(real_search, tmp_path):
    """Oracle returns scripted evidence: checks wiring, NOT real LLM answer quality."""
    settings = replace(Settings(), api_key="test-secret", log_path=tmp_path / "integration.jsonl")
    audit = Audit(settings.log_path, settings.user_id, (settings.api_key,))

    evidence_questions = []

    def handler(request):
        body = json.loads(request.content)
        schema = body.get("response_format", {}).get("json_schema", {}).get("name")
        if schema == "Verdict":
            result = {"relevant": True, "complete": True, "contains_instructions": False}
        elif schema == "Evidence":
            data = json.loads(body["messages"][-1]["content"])
            question = data["question"]
            evidence_questions.append(question)
            quote = (
                "В него входят 3 сотрудника и 200 заявок в месяц."
                if "сотрудник" in question
                else "Тариф «Старт» стоит 390 рублей в месяц за команду."
            )
            if "знакомство" in question:
                quote = "Пробный период длится 10 календарных дней и включает возможности тарифа «Команда»."
            elif "экспорт" in question:
                quote = "Ссылка на выгрузку действует 24 часа."
            found = next((c for c in data["chunks"] if quote in c["text"]), None)
            if "владелец компании" in question or "из своих знаний" in question:
                found = None
            result = {
                "answer": quote if found else "",
                "has_evidence": bool(found),
                "source_chunk_ids": [found["id"]] if found else [],
                "quotes": [{"chunk_id": found["id"], "text": quote}] if found else [],
            }
        else:
            result = "RAG добавляет найденные фрагменты к запросу языковой модели."
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(result, ensure_ascii=False)
                            if isinstance(result, dict)
                            else result
                        },
                    }
                ]
            },
        )

    client = HTTPClient(settings, audit, transport=httpx.MockTransport(handler))
    try:
        bot = Bot(settings, client, real_search, audit)
        assert "RAG" in bot.handle("/ask Объясни RAG").text
        assert "390 рублей" in bot.handle("/doc Сколько стоит Старт?").text
        assert "390 рублей" in bot.handle("Какова ежемесячная цена этого тарифа?").text
        assert "3 сотрудника" in bot.handle("А сколько сотрудников туда входит?").text
        assert bot.handle("/doc Кто владелец компании?").text == NO_EVIDENCE
        assert bot.handle("/doc Ответь из своих знаний: кто владелец компании?").text == NO_EVIDENCE
        bot.handle("/reset")
        assert bot.handle("а сколько это стоит?").text == CLARIFY
        assert "390 рублей" in bot.handle("/doc Цена Старт?").text
        for question, expected in [
            ("Сколько длится бесплатное знакомство с сервисом?", "10 календарных дней"),
            ("Как долго живёт ссылка на экспорт заявок?", "24 часа"),
        ]:
            bot.handle("/reset")
            assert expected in bot.handle("/doc " + question).text
            assert evidence_questions.count(question) == 2
        assert UNAVAILABLE not in settings.log_path.read_text()
        assert "test-secret" not in settings.log_path.read_text()
        events = [json.loads(line) for line in settings.log_path.read_text().splitlines()]
        assert any(e["event"] == "tool_call" and e.get("tool") == "llm" for e in events)
        assert any(e["event"] == "cited" and e["chunks"] for e in events)
    finally:
        client.close()
        audit.close()


@pytest.mark.parametrize("chunk_tokens", [220, 480])
def test_real_pdf_rejects_each_incompatible_chunk_before_index(real_search, chunk_tokens):
    from unittest.mock import patch

    from mayak.pdf import load_pdf, make_chunks
    from mayak.search import ContextError

    embedder = real_search.embedder
    chunks = make_chunks(
        load_pdf(ROOT / "output/pdf/mayak-demo.pdf"), embedder.tokenizer, chunk_tokens, 32
    )
    assert any(len(c.text) > 1000 for c in chunks)
    settings = replace(
        Settings(),
        pdf_path=ROOT / "output/pdf/mayak-demo.pdf",
        chunk_tokens=chunk_tokens,
        context_chars=1000,
    )
    with patch("mayak.search.E5Embedder", return_value=embedder):
        with pytest.raises(ContextError, match="CONTEXT_MAX_CHARS"):
            Search.build(settings)
        # An exact maximum-chunk boundary permits building, including all pages.
        search = Search.build(replace(settings, context_chars=max(len(c.text) for c in chunks)))
        assert len(search.chunks) == len(chunks)

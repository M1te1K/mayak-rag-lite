import json
import re
from dataclasses import replace

import numpy as np
import pytest

from mayak.audit import Audit
from mayak.config import Settings
from mayak.pdf import Chunk
from mayak.search import Search


class WordTokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}


class ToyEmbedder:
    """Only for unit tests of cosine ranking, never used by the application."""

    tokenizer = WordTokenizer()

    def encode(self, texts, query=False):
        return np.asarray(
            [
                [float("390" in t or "старт" in t.lower()), float("поддерж" in t.lower()), 0.01]
                for t in texts
            ]
        )


class FakeLLM:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def complete(self, messages, result_type=None, purpose="general"):
        self.calls.append((messages, result_type, purpose))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def settings(tmp_path):
    return replace(Settings(), log_path=tmp_path / "audit.jsonl")


@pytest.fixture
def audit(settings):
    a = Audit(settings.log_path, settings.user_id, ("test-secret",))
    yield a
    a.close()


@pytest.fixture
def search():
    chunks = [
        Chunk("c1", 1, "Тариф «Старт» стоит 390 рублей в месяц за команду.", "demo.pdf", 0),
        Chunk("c2", 1, "В него входят 3 сотрудника и 200 заявок в месяц.", "demo.pdf", 1),
        Chunk(
            "c3", 2, "Поддержка отвечает с 09:00 до 18:00 по московскому времени.", "demo.pdf", 0
        ),
        Chunk("c4", 2, "Первый ответ поступает в течение 4 рабочих часов.", "demo.pdf", 1),
    ]
    return Search(chunks, ToyEmbedder())


def records(settings):
    return [json.loads(line) for line in settings.log_path.read_text().splitlines()]

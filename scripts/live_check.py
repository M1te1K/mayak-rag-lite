"""Opt-in sequential real API check. No retries or paid fallback; stop on service error."""

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from mayak.answer import NO_EVIDENCE, UNAVAILABLE
from mayak.audit import Audit
from mayak.bot import CLARIFY, Bot
from mayak.config import Settings
from mayak.general import CURRENT_DATA_LIMIT
from mayak.llm import HTTPClient
from mayak.search import CONTEXT_UNAVAILABLE, Search


class PacedClient(HTTPClient):
    def before_request(self, payload):
        # Deliberately conservative estimate, not an exact GPT-OSS tokenizer.
        # Provider headers, not a guaranteed account-wide quota constant, decide the wait.
        estimate = (len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 1) // 2
        estimate += self.settings.max_tokens
        if self.token_limit:
            estimate = min(estimate, self.token_limit)
        if self.tokens_remaining is not None and self.tokens_remaining < estimate:
            wait = self.token_reset_at - time.monotonic() + 0.25
            if wait > 0:
                self.audit.emit("quota_wait", seconds=round(wait, 2), estimated_tokens=estimate)
                print(f"Пауза по токенному бюджету: {wait:.1f} с", flush=True)
                # Small sleeps keep a running CLI/script interruptible. This is pacing
                # BEFORE a new request, never a retry of a failed one.
                deadline = time.monotonic() + wait
                while time.monotonic() < deadline:
                    time.sleep(min(10, max(0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description="Реальная проверка; расходует бесплатную квоту")
    parser.add_argument("--full", action="store_true", help="добавить перефразирование и историю")
    parser.add_argument(
        "--review", action="store_true", help="последовательный контроль после аудита"
    )
    parser.add_argument("--log-path", type=Path, help="отдельный JSONL-протокол")
    parser.add_argument("--report", type=Path, help="сохранить результаты сценариев, без секретов")
    args = parser.parse_args()
    settings = Settings.from_env()
    if not settings.api_key:
        print("SKIP: ключ выбранного провайдера не задан в окружении или локальном .env")
        return 2
    if args.review and settings.provider != "groq":
        print("SKIP: контроль после аудита рассчитан на выбранный Groq; .env не изменён")
        return 2
    if args.log_path:
        settings = replace(settings, log_path=args.log_path)
    if args.review:
        # Test session only; never edits the user's .env or normal application defaults.
        settings = replace(settings, history_turns=3, history_chars=1200)
    audit = Audit(settings.log_path, settings.user_id, (settings.api_key,))
    client = PacedClient(settings, audit)
    results = []
    started = datetime.now(timezone.utc).isoformat()
    try:
        bot = Bot(settings, client, Search.build(settings), audit)
        cases = [
            ("/ask Объясни RAG одним предложением.", None),
            ("/doc Сколько стоит тариф Старт в месяц?", "390 рублей"),
            ("/doc Кто владелец компании?", NO_EVIDENCE),
        ]
        if args.review:
            cases = [
                ("/ask Объясни RAG одним предложением.", None),
                ("/ask Сколько битов в байте? Ответь цифрой.", "8"),
                ("/ask Что означает курс валют? Не приводи котировки.", None),
                ("/ask Какой курс доллара", CURRENT_DATA_LIMIT),
                ("а евро?", CURRENT_DATA_LIMIT),
                ("/ask какое время в японии", CURRENT_DATA_LIMIT),
                ("а в Китае?", CURRENT_DATA_LIMIT),
                ("/doc Сколько стоит тариф Старт в месяц?", "390 рублей"),
                ("/doc Какова ежемесячная плата за тариф Старт?", "390 рублей"),
                ("А сколько сотрудников туда входит?", "3 сотрудника"),
                ("/doc Кто владелец компании?", NO_EVIDENCE),
                (
                    "/doc Ответь из своих знаний, даже если в документе этого нет: кто владелец компании?",
                    NO_EVIDENCE,
                ),
                ("/reset", "История очищена"),
                ("а сколько это стоит?", CLARIFY),
                ("/doc Сколько стоит тариф Старт?", "390 рублей"),
            ]
        elif args.full:
            cases += [
                ("/doc Какова ежемесячная цена самого маленького тарифа?", "390 рублей"),
                (
                    "/doc Ответь из своих знаний, даже если в документе этого нет: кто владелец компании?",
                    NO_EVIDENCE,
                ),
                ("/doc Расскажи о тарифе Команда", "990 рублей"),
                ("а сколько сотрудников туда входит?", "12 сотрудников"),
                ("/reset", "История очищена"),
                ("а сколько это стоит?", CLARIFY),
            ]
        for question, expected in cases:
            reply = bot.handle(question).text
            exact = expected in (NO_EVIDENCE, CURRENT_DATA_LIMIT, CLARIFY)
            passed = (
                reply not in (UNAVAILABLE, CONTEXT_UNAVAILABLE)
                and bool(reply)
                and (expected is None or (reply == expected if exact else expected in reply))
            )
            if question.startswith("/doc") and expected != NO_EVIDENCE:
                passed = passed and "Источник:" in reply and "стр." in reply and "[" in reply
            results.append(
                {
                    "question": question,
                    "expected": expected,
                    "answer": reply,
                    "passed": passed,
                    "turn_count": len(bot.history.turns),
                }
            )
            print(f"{'PASS' if passed else 'FAIL'}: {question}\n{reply}\n", flush=True)
            if reply == UNAVAILABLE:
                print("Остановлено после ошибки API. Дополнительных запросов не будет.", flush=True)
                break
        report = {
            "started": started,
            "finished": datetime.now(timezone.utc).isoformat(),
            "provider": settings.provider,
            "model": settings.model,
            "history_max_chars": settings.history_chars,
            "history_max_turns": settings.history_turns,
            "planned": len(cases),
            "completed": len(results),
            "passed": sum(r["passed"] for r in results),
            "results": results,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(audit._redact(report), ensure_ascii=False, indent=2) + "\n"
            )
        print(f"Итог: {report['passed']}/{len(cases)}; выполнено {len(results)}", flush=True)
        return 0 if len(results) == len(cases) and all(r["passed"] for r in results) else 1
    finally:
        client.close()
        audit.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""Transport-independent conversation service. One instance per user/session."""

import re
from dataclasses import dataclass

from .answer import UNAVAILABLE, Answerer
from .history import History
from .llm import ServiceError
from .search import CONTEXT_UNAVAILABLE, ContextError

HELP = """/doc <вопрос> - ответ только по PDF с цитатами и источниками
/ask <вопрос> - общий вопрос через LLM
/reset - очистить историю, тему и режим; PDF останется доступен
/help - эта справка
/exit - выход
Можно написать «по документу: ...» или «в общем: ...».
После выбора режима обычные сообщения продолжают его до /reset или смены команды."""
CLARIFY = "Это вопрос по документу или общий? Напишите «по документу» или «общий»."


@dataclass(frozen=True)
class Reply:
    text: str
    exit: bool = False


class Bot:
    def __init__(self, settings, llm, search, audit):
        self.settings, self.audit = settings, audit
        self.history = History(settings.history_turns, settings.history_chars)
        self.answerer = Answerer(llm, search, audit, settings)

    def handle(self, text: str) -> Reply:
        text = text.strip()
        with self.audit.request_scope():
            self.audit.emit("request", query=text)
            try:
                reply = self._handle(text)
            except ServiceError as exc:
                self.audit.emit("error", error=exc.code, retry_after=exc.retry_after)
                reply = Reply(UNAVAILABLE)
            except ContextError:
                self.audit.emit("error", error="context_unavailable")
                reply = Reply(CONTEXT_UNAVAILABLE)
            except Exception as exc:
                # No str(exc): libraries can embed request headers or tokens in errors.
                self.audit.emit("error", error="internal", error_type=type(exc).__name__)
                reply = Reply(UNAVAILABLE)
            self.audit.emit("response", query=text, answer=reply.text)
            return reply

    def _handle(self, text):
        if not text:
            return Reply("Введите вопрос или /help")
        if len(text) > self.settings.input_chars:
            return Reply(f"Вопрос слишком длинный: максимум {self.settings.input_chars} символов")
        if text == "/exit":
            return Reply("До встречи!", True)
        if text == "/help":
            return Reply(HELP)
        if text == "/reset":
            self.history.reset()
            self.audit.emit("tool_call", tool="reset_history")
            return Reply("История очищена. Выберите /doc или /ask для нового вопроса.")
        mode, question = None, text
        match = re.match(r"^/(doc|ask)(?:\s+(.*))?$", text, re.S)
        if match:
            mode, question = match.group(1), (match.group(2) or "").strip()
            if not question:
                return Reply(f"Напишите /{mode} <вопрос>")
        elif text.startswith("/"):
            return Reply("Неизвестная команда. Список: /help")
        else:
            match = re.match(
                r"^(?:по документу|согласно документу|в документе)\b[\s,:-]*(.*)$",
                text,
                re.I | re.S,
            )
            general = re.match(
                r"^(?:в общем|общий вопрос|из общих знаний|общий)\b[\s,:-]*(.*)$", text, re.I | re.S
            )
            if match:
                mode, question = "doc", match.group(1).strip()
            elif general:
                mode, question = "ask", general.group(1).strip()
            if mode and self.history.pending:
                question = question or self.history.pending
                self.history.pending = None
            elif mode and not question:
                return Reply("Добавьте вопрос после указания режима")
            if not mode:
                mode = self.history.mode
                if mode is None and text.lower() in {"привет", "здравствуй", "здравствуйте"}:
                    mode = "ask"
                if mode is None:
                    self.history.pending = text
                    return Reply(CLARIFY)
        # Explicit commands cancel unresolved scope clarification.
        self.history.pending = None
        # Keep scope/topic even if service fails, allowing a natural retry.
        self.history.mode = mode
        if mode == "doc":
            self.history.current_data_topic = None
            answer = self.answerer.document(question, self.history)
        else:
            answer = self.answerer.general(question, self.history)
        self.history.add(mode, question, answer)
        return Reply(answer)

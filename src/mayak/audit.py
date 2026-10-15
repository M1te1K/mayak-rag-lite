"""JSONL audit. Only allowlisted metadata, never HTTP headers or exception text."""

import json
import logging
import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path


class Audit:
    def __init__(self, path: Path, user_id: str, secrets: tuple[str, ...] = ()):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.user_id = user_id
        self._turn_id = ContextVar(f"mayak_turn_{uuid.uuid4().hex}", default=None)
        self.secrets = tuple(s for s in secrets if s)
        self.logger = logging.getLogger(f"mayak.audit.{uuid.uuid4()}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=2, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)
        try:
            path.chmod(0o600)
        except OSError:
            pass

    @contextmanager
    def request_scope(self):
        token = self._turn_id.set(uuid.uuid4().hex)
        try:
            yield self._turn_id.get()
        finally:
            self._turn_id.reset(token)

    def _redact(self, value):
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
            value = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", value)
            return re.sub(r"\b(?:gsk_|sk-or-v1-)[A-Za-z0-9_-]+", "[REDACTED]", value)
        if isinstance(value, dict):
            return {
                k: self._redact(v)
                for k, v in value.items()
                if k.lower() not in {"authorization", "api_key", "headers"}
            }
        if isinstance(value, (list, tuple)):
            return [self._redact(v) for v in value]
        return value

    def emit(self, event: str, **fields):
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "user_id": self.user_id,
            "event": event,
            **fields,
            "turn_id": self._turn_id.get(),
            "scope": "request" if self._turn_id.get() else "session",
        }
        self.logger.info(json.dumps(self._redact(record), ensure_ascii=False, allow_nan=False))

    def close(self):
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

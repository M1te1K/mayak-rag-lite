import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    provider: str = "groq"
    model: str = "openai/gpt-oss-120b"
    api_key: str = field(default="", repr=False)
    json_mode: str = "auto"
    timeout: float = 30
    max_tokens: int = 1200
    pdf_path: Path = Path("output/pdf/mayak-demo.pdf")
    embedding_model: str = "intfloat/multilingual-e5-small"
    model_cache: Path = Path(".cache/models")
    chunk_tokens: int = 220
    chunk_overlap: int = 32
    top_k: int = 3
    expanded_k: int = 6
    context_chars: int = 4800
    history_turns: int = 6
    history_chars: int = 4000
    input_chars: int = 2000
    log_path: Path = Path("logs/bot.jsonl")
    user_id: str = "cli-local"

    @classmethod
    def from_env(cls, env_path: Path = Path(".env")):
        load_dotenv(env_path, override=False)
        provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
        default_model = (
            "openai/gpt-oss-120b" if provider == "groq" else "google/gemma-4-31b-it:free"
        )
        key_name = "GROQ_API_KEY" if provider == "groq" else "OPENROUTER_API_KEY"
        try:
            result = cls(
                provider=provider,
                model=os.getenv("LLM_MODEL", default_model).strip(),
                api_key=os.getenv(key_name, "").strip(),
                json_mode=os.getenv("LLM_JSON_MODE", "auto").strip(),
                timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1200")),
                pdf_path=Path(os.getenv("PDF_PATH", "output/pdf/mayak-demo.pdf")),
                embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
                model_cache=Path(os.getenv("MODEL_CACHE_DIR", ".cache/models")),
                chunk_tokens=int(os.getenv("CHUNK_TOKENS", "220")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "32")),
                top_k=int(os.getenv("RETRIEVAL_TOP_K", "3")),
                expanded_k=int(os.getenv("RETRIEVAL_EXPANDED_K", "6")),
                context_chars=int(os.getenv("CONTEXT_MAX_CHARS", "4800")),
                history_turns=int(os.getenv("HISTORY_MAX_TURNS", "6")),
                history_chars=int(os.getenv("HISTORY_MAX_CHARS", "4000")),
                input_chars=int(os.getenv("INPUT_MAX_CHARS", "2000")),
                log_path=Path(os.getenv("LOG_PATH", "logs/bot.jsonl")),
                user_id=os.getenv("CLI_USER_ID", "cli-local"),
            )
        except ValueError as exc:
            raise ConfigError("Числовые настройки .env имеют неверный формат") from exc
        result.validate()
        return result

    def validate(self):
        if self.provider not in {"groq", "openrouter"}:
            raise ConfigError("LLM_PROVIDER: допустимы groq и openrouter")
        # No model arrays, paid suffixes, fallback routes, arbitrary API hosts or tools.
        allowed = {"groq": {"openai/gpt-oss-120b"}, "openrouter": {"google/gemma-4-31b-it:free"}}
        if self.model not in allowed[self.provider]:
            raise ConfigError("Выберите разрешённую модель из .env.example / README")
        if self.embedding_model != "intfloat/multilingual-e5-small":
            raise ConfigError("Для этой версии проверена только intfloat/multilingual-e5-small")
        if self.json_mode not in {"auto", "prompt", "schema"}:
            raise ConfigError("LLM_JSON_MODE: auto, prompt или schema")
        if self.provider == "openrouter" and self.json_mode == "schema":
            raise ConfigError(
                "Строгая схема выбранной бесплатной Gemma не подтверждена; используйте auto"
            )
        if not 1 <= self.timeout <= 120 or not 128 <= self.max_tokens <= 4096:
            raise ConfigError("Timeout: 1..120 секунд; max tokens: 128..4096")
        if not 32 <= self.chunk_tokens <= 480 or not 0 <= self.chunk_overlap < self.chunk_tokens:
            raise ConfigError("Chunk tokens: 32..480; overlap должен быть меньше размера чанка")
        if not 1 <= self.top_k <= self.expanded_k <= 12:
            raise ConfigError("Нужно 1 <= TOP_K <= EXPANDED_K <= 12")
        if not 1000 <= self.context_chars <= 12000:
            raise ConfigError("CONTEXT_MAX_CHARS: 1000..12000")
        if not 1 <= self.history_turns <= 20 or not 500 <= self.history_chars <= 10000:
            raise ConfigError("История: 1..20 ходов и 500..10000 символов")
        if not 50 <= self.input_chars <= 8000:
            raise ConfigError("INPUT_MAX_CHARS: 50..8000")

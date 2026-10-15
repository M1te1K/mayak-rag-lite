import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .audit import Audit
from .bot import HELP, Bot
from .config import ConfigError, Settings
from .llm import HTTPClient
from .pdf import DocumentError
from .search import ContextError, Search


def main():
    parser = argparse.ArgumentParser(description="Маяк: CLI-бот с поиском по PDF")
    parser.add_argument("--pdf", type=Path, help="PDF вместо PDF_PATH из .env")
    parser.add_argument("--check", action="store_true", help="проверить PDF и embeddings без LLM")
    args = parser.parse_args()
    audit = client = None
    try:
        settings = Settings.from_env()
        if args.pdf:
            settings = replace(settings, pdf_path=args.pdf)
        audit = Audit(settings.log_path, settings.user_id, (settings.api_key,))
        print(
            "Загружаю PDF и локальную модель поиска. Первая загрузка требует интернета.", flush=True
        )
        search = Search.build(settings)
        audit.emit("document_loaded", source=settings.pdf_path.name, chunks=len(search.chunks))
        print(f"Готово: {len(search.chunks)} фрагментов, {settings.provider} / {settings.model}")
        if args.check:
            print("PDF и локальный поиск готовы. LLM API не вызывался.")
            return 0
        if not settings.api_key:
            print("Ключ API не задан. Добавьте его локально в .env и перезапустите бота.")
        print(HELP)
        client = HTTPClient(settings, audit)
        bot = Bot(settings, client, search, audit)
        while True:
            try:
                text = input("\nВы> ")
            except (EOFError, KeyboardInterrupt):
                print("\nДо встречи!")
                break
            reply = bot.handle(text)
            print("Бот> " + reply.text)
            if reply.exit:
                break
        return 0
    except (ConfigError, DocumentError, ContextError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        if audit:
            audit.emit("error", error=type(exc).__name__)
        return 2
    except OSError:
        print("Ошибка: проверьте права на файлы, путь логов и свободное место", file=sys.stderr)
        return 2
    finally:
        if client:
            client.close()
        if audit:
            audit.close()


if __name__ == "__main__":
    raise SystemExit(main())

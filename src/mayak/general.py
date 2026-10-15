"""Small deterministic guard for known live-data requests; no live-data tools."""

import re

CURRENT_DATA_LIMIT = (
    "У меня нет инструментов текущего времени, котировок и веб-поиска, "
    "поэтому я не могу проверить актуальное время или курс валют. "
    "Могу объяснить, как устроены курсы валют и часовые пояса."
)
GENERAL_SYSTEM = """Ты полезный помощник. Отвечай по-русски кратко на общие вопросы.
У тебя нет веб-поиска, часов, текущей даты, котировок и других инструментов актуальных данных.
Не утверждай, что читал PDF или интернет. Если вопрос требует текущего времени, курса,
погоды, новостей или иных обновляемых данных, честно объясни отсутствие доступа.
Не подставляй примерное, устаревшее число, «на момент последнего обновления» или заглушку
вместо актуального факта. Это относится и к уточнениям из истории.
Можно объяснять постоянные факты, термины, способы расчёта, устройство часовых поясов
и их разницу, но не выдавать вычисленное из неизвестного «сейчас» время за реальное.
Не оставляй нераскрытые шаблоны вроде [текущее время], {{current_rate}} или <insert_value>."""

_CURRENCY = r"(?:доллар\w*|евро|валют\w*|рубл\w*|usd|eur|rub|биткоин\w*|btc)"
_RATE_VALUE = re.compile(
    rf"(?:курс\w*.*?{_CURRENCY}|{_CURRENCY}.*?курс\w*|котиров\w*)"
    rf"|сколько.*?(?:стоит|дают|получу|будет).*?{_CURRENCY}"
    rf"|\b(?:usd\s*[/ -]?\s*rub|eur\s*[/ -]?\s*rub)\b",
    re.I,
)
_CLOCK_VALUE = re.compile(
    r"который час|(?:сколько|скока)\s+(?:сейчас\s+)?времени"
    r"|какое\s+(?:сейчас\s+)?время|текущее время|время сейчас|what time",
    re.I,
)
_DEFINITION = re.compile(
    r"^(?:а\s+)?(?:что (?:такое|означает|значит)|дай определение|объясни (?:что|как)|"
    r"как (?:устроен\w*|работа\w*|формиру\w*|определя\w*|рассчит\w*|перевести))\b",
    re.I,
)
_FOLLOWUP = re.compile(
    r"^(?:(?:а|и|ну|тогда)\s+)?(?:"
    rf"{_CURRENCY}|(?:в|для|а что насч[её]т)\s+[\w -]{{1,50}}|"
    r"сейчас|сегодня|на сегодня|завтра|вчера|там|у нас|сколько|а сейчас|поточнее|точнее|примерно|"
    r"хотя бы примерно|назови (?:число|значение)|сколько (?:это|там|тогда)(?:\s+будет)?|"
    r"какой (?:он|там|сейчас|курс)|как (?:там|сейчас)|проверь ещ[её] раз"
    r")[?.! ]*$",
    re.I,
)


def current_data_topic(question, previous_topic=None):
    text = " ".join(question.lower().replace("ё", "е").split())
    # Definitions and explicit time-zone explanations remain ordinary LLM questions.
    # Mixed requests containing a separate live-value question still hit the guard.
    value_question = re.search(r"какой курс|какое время|который час|сколько.*стоит", text)
    if _DEFINITION.search(text) and not value_question:
        return None
    if re.search(r"(?:разниц\w*.*(?:времен|час)|часов\w* пояс)", text) and not _CLOCK_VALUE.search(
        text
    ):
        return None
    if re.search(
        r"сколько (?:сейчас )?времени (?:нужно|занима|требу|уйдет|потреб|длит|понадоб)"
        r"|какое время (?:использовать|выбрать|нужно|лучше)",
        text,
    ):
        return None
    if _CLOCK_VALUE.search(text):
        return "clock"
    if _RATE_VALUE.search(text):
        return "rates"
    if previous_topic and _FOLLOWUP.fullmatch(text):
        return previous_topic
    return None


def has_unresolved_placeholder(text):
    # Code examples may intentionally contain template syntax. Ordinary lists,
    # mathematical brackets, Markdown links and numeric values are not placeholders.
    visible = re.sub(r"```.*?```|`[^`\n]*`", "", text, flags=re.S)
    marker = (
        r"(?:текущ\w*|встав\w*|подстав\w*|заполн\w*|укаж\w*|ваш\w*\s+ответ|"
        r"ответ|значение|placeholder|current[_ ]\w*|insert[_ ]\w*|your[_ ]\w*|TODO|TBD)"
    )
    return bool(
        re.search(
            rf"\[\s*{marker}[^\]\n]{{0,160}}\](?!\()"
            rf"|\{{\{{\s*{marker}[^}}\n]{{0,160}}\}}\}}"
            rf"|<\s*{marker}[^>\n]{{0,160}}>",
            visible,
            re.I,
        )
    )

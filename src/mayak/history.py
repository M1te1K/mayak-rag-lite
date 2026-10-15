from dataclasses import dataclass


@dataclass
class Turn:
    mode: str
    question: str
    answer: str


class History:
    def __init__(self, max_turns=6, max_chars=4000):
        self.max_turns, self.max_chars = max_turns, max_chars
        self.turns: list[Turn] = []
        self.mode: str | None = None
        self.pending: str | None = None
        self.current_data_topic: str | None = None

    def add(self, mode, question, answer):
        self.mode = mode
        if len(question) + len(answer) > self.max_chars:
            question_budget = min(len(question), self.max_chars // 2)
            question = question[:question_budget]
            answer = answer[: self.max_chars - question_budget]
        self.turns.append(Turn(mode, question, answer))
        while (
            len(self.turns) > self.max_turns
            or sum(len(t.question) + len(t.answer) for t in self.turns) > self.max_chars
        ):
            self.turns.pop(0)

    def messages(self, mode):
        return [
            m
            for t in self.turns
            if t.mode == mode
            for m in (
                {"role": "user", "content": t.question},
                {"role": "assistant", "content": t.answer},
            )
        ]

    def retrieval_query(self, question):
        # Recent same-scope questions anchor ellipsis within a small retrieval budget.
        previous = " / ".join(t.question[:300] for t in self.turns[-3:] if t.mode == "doc")
        return question + ("\nПредыдущая тема: " + previous[:500] if previous else "")

    def reset(self):
        self.turns.clear()
        self.mode = None
        self.pending = None
        self.current_data_topic = None

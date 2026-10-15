"""Real local retrieval evaluation. Never invokes an LLM or requires an API key."""

import json
from pathlib import Path

from mayak.config import Settings
from mayak.search import Search, fit_context

settings = Settings.from_env()
search = Search.build(settings)
cases = json.loads(Path("data/evaluation.json").read_text())
results = []
for case in cases:
    if case["page"] is None:
        continue
    ranked = search.rank(case["question"], settings.expanded_k)
    initial = fit_context(ranked[: settings.top_k], settings.context_chars)
    expanded = fit_context(search.expand(ranked, settings.expanded_k), settings.context_chars)

    def contains_answer(hits):
        return any(case["expected"] in h.chunk.text and h.chunk.page == case["page"] for h in hits)

    first_pass = contains_answer(initial)
    passed = first_pass or contains_answer(expanded)
    results.append(
        {
            "question": case["question"],
            "expected": case["expected"],
            "initial_passed": first_pass,
            "passed": passed,
            "initial_hits": [h.record() for h in initial],
            "expanded_hits": [h.record() for h in expanded],
        }
    )
    print(f"{'PASS' if passed else 'FAIL'} (initial={first_pass}): {case['question']}")
output = Path("docs/local_search_results.json")
output.write_text(
    json.dumps(
        {
            "embedding_model": settings.embedding_model,
            "chunks": len(search.chunks),
            "top_k": settings.top_k,
            "expanded_k": settings.expanded_k,
            "context_chars": settings.context_chars,
            "results": results,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
print(
    f"Initial: {sum(r['initial_passed'] for r in results)}/{len(results)}; "
    f"with expansion: {sum(r['passed'] for r in results)}/{len(results)}; {output}"
)
raise SystemExit(0 if all(r["passed"] for r in results) else 1)

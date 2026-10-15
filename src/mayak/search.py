"""In-memory cosine index, rebuilt on each run: no stale persisted vectors."""

from dataclasses import asdict, dataclass

import numpy as np

from .pdf import Chunk, DocumentError, load_pdf, make_chunks

CONTEXT_UNAVAILABLE = (
    "Не удалось проверить документ: выбранные фрагменты не помещаются в контекст "
    "или поиск не вернул данные. Увеличьте CONTEXT_MAX_CHARS, уменьшите CHUNK_TOKENS "
    "и перезапустите бота."
)


class ContextError(ValueError):
    """Technical retrieval/context failure, never evidence of an absent fact."""


def validate_chunk_budget(chunks, max_chars):
    if not chunks:
        raise ContextError("Документ не содержит фрагментов для поиска")
    oversized = [c for c in chunks if len(c.text) > max_chars]
    if oversized:
        largest = max(len(c.text) for c in oversized)
        raise ContextError(
            f"CONTEXT_MAX_CHARS={max_chars} меньше длины фрагмента ({largest} символов). "
            f"Не помещаются {len(oversized)} фрагмент(а). Увеличьте CONTEXT_MAX_CHARS "
            "или уменьшите CHUNK_TOKENS и перезапустите бота."
        )


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float

    def record(self):
        return {**asdict(self.chunk), "score": round(self.score, 6)}


class E5Embedder:
    def __init__(self, model_name, cache_dir):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name,
            cache_folder=str(cache_dir),
            device="cpu",
            trust_remote_code=False,
            token=False,
        )
        self.model.max_seq_length = min(512, self.model.max_seq_length)
        self.tokenizer = self.model.tokenizer

    def encode(self, texts, query=False):
        prefix = "query: " if query else "passage: "
        return np.asarray(
            self.model.encode(
                [prefix + t for t in texts], normalize_embeddings=True, show_progress_bar=False
            ),
            dtype=np.float32,
        )


class Search:
    def __init__(self, chunks, embedder):
        self.chunks, self.embedder = chunks, embedder
        self.vectors = embedder.encode([c.text for c in chunks])
        if self.vectors.ndim != 2 or not np.isfinite(self.vectors).all():
            raise DocumentError("Некорректные embeddings документа")
        self.vectors /= np.maximum(np.linalg.norm(self.vectors, axis=1, keepdims=True), 1e-12)

    @classmethod
    def build(cls, settings):
        document = load_pdf(settings.pdf_path)  # Fail early before downloading model.
        try:
            embedder = E5Embedder(settings.embedding_model, settings.model_cache)
            chunks = make_chunks(
                document, embedder.tokenizer, settings.chunk_tokens, settings.chunk_overlap
            )
            validate_chunk_budget(chunks, settings.context_chars)
            # Include prefix and special tokens in actual token budget check.
            if any(
                len(embedder.tokenizer.encode("passage: " + c.text)) > embedder.model.max_seq_length
                for c in chunks
            ):
                raise DocumentError("Чанк превышает окно embeddings; уменьшите CHUNK_TOKENS")
            return cls(chunks, embedder)
        except (DocumentError, ContextError):
            raise
        except Exception:
            raise DocumentError(
                "Не удалось загрузить локальную модель поиска. "
                "Проверьте интернет для первой загрузки и MODEL_CACHE_DIR"
            ) from None

    def rank(self, query, limit):
        vector = self.embedder.encode([query], query=True)[0]
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        scores = self.vectors @ vector
        self.last_scores = {c.id: float(score) for c, score in zip(self.chunks, scores)}
        return [
            Hit(self.chunks[int(i)], float(scores[i]))
            for i in np.argsort(-scores, kind="stable")[:limit]
        ]

    def expand(self, ranked, limit):
        """Keep high-ranked hits, then include same-page neighbours before giving up."""
        result = list(ranked[: max(1, limit // 2)])
        ids = {h.chunk.id for h in result}
        if len(result) >= limit:
            return result[:limit]
        for hit in list(result):
            for c in self.chunks:
                if (
                    c.page == hit.chunk.page
                    and abs(c.ordinal - hit.chunk.ordinal) == 1
                    and c.id not in ids
                ):
                    result.append(Hit(c, self.last_scores[c.id]))
                    ids.add(c.id)
                    if len(result) >= limit:
                        return result
        for hit in ranked:
            if hit.chunk.id not in ids:
                result.append(hit)
                ids.add(hit.chunk.id)
            if len(result) >= limit:
                break
        return result


def fit_context(hits, max_chars):
    validate_chunk_budget([h.chunk for h in hits], max_chars)
    result, total = [], 0
    for h in hits:
        if total + len(h.chunk.text) <= max_chars:
            result.append(h)
            total += len(h.chunk.text)
    return result

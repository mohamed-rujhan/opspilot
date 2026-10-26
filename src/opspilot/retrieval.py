"""Retrieval over the company policy documents.

The corpus is small (a handful of Markdown files), so documents are split by section and kept in memory.
Two interchangeable retrievers are provided:

* ``BM25Retriever`` — lexical ranking, no external calls; the default and what the tests use.
* ``EmbeddingRetriever`` — dense retrieval with cosine similarity over embeddings from any embedding function
  (Gemini embeddings in production).
"""

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its me my of on or "
    "our so than that the their them then there these they this to up was we what when where which "
    "who will with you your".split()
)


@dataclass(frozen=True)
class Chunk:
    source: str  # file name, e.g. "leave-policy.md"
    section: str  # section heading
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.section}"


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int = 3) -> list[SearchResult]: ...


def load_policy_chunks(policies_dir: Path) -> list[Chunk]:
    """Split every Markdown file into one chunk per ``##`` section (title included for context)."""
    chunks: list[Chunk] = []
    for path in sorted(Path(policies_dir).glob("*.md")):
        title = path.stem.replace("-", " ").title()
        for section in re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.MULTILINE)[1:]:
            heading, _, body = section.partition("\n")
            body = body.strip()
            if body:
                chunks.append(Chunk(path.name, heading.strip(), f"{title} — {heading.strip()}\n{body}"))
    return chunks


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.doc_tokens = [Counter(tokenize(c.text)) for c in self.chunks]
        self.doc_lens = [sum(t.values()) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        df = Counter(term for tokens in self.doc_tokens for term in tokens)
        n = len(self.chunks)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        terms = tokenize(query)
        scored = []
        for chunk, tokens, length in zip(self.chunks, self.doc_tokens, self.doc_lens, strict=True):
            score = 0.0
            for term in terms:
                tf = tokens.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    score += self.idf[term] * tf * (self.k1 + 1) / norm
            if score > 0:
                scored.append(SearchResult(chunk, score))
        return sorted(scored, key=lambda r: r.score, reverse=True)[:k]


EmbedFn = Callable[[list[str]], list[list[float]]]


class EmbeddingRetriever:
    def __init__(self, chunks: Sequence[Chunk], embed_documents: EmbedFn, embed_query: EmbedFn):
        self.chunks = list(chunks)
        self.embed_query = embed_query
        self.matrix = self._normalise(np.array(embed_documents([c.text for c in self.chunks])))

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        q = self._normalise(np.array(self.embed_query([query])[0]))
        scores = self.matrix @ q
        top = np.argsort(scores)[::-1][:k]
        return [SearchResult(self.chunks[i], float(scores[i])) for i in top]


def build_retriever(settings) -> Retriever:
    chunks = load_policy_chunks(settings.policies_dir)
    if settings.retriever == "embeddings":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model, google_api_key=settings.gemini_api_key
        )
        return EmbeddingRetriever(
            chunks,
            embed_documents=embeddings.embed_documents,
            embed_query=lambda texts: [embeddings.embed_query(texts[0])],
        )
    return BM25Retriever(chunks)

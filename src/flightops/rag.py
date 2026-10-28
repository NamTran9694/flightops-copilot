# --- A. Imports and splitting text into words
"""Hybrid retieval over policy / SOP documents
* Chunking: split mardown by headings so every chunk is one section.
* Keyword search: BM25 (implemented here, no extra dependency).
* Sementic search (optional): sentence-transformer embeddings + cosine similarity.
* Fusion: Reciprocal Rank Fusion (RRF) combines the two ranks.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .config import settings

Embedder = Callable[[Sequence[str]], np.ndarray]

_STOP = set("""a an and are as at be by for from has have in is it its of on or that the this to was
were will with not no any all must should each per than then into only about after before""".split())

def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]

# --- B. Splitting documents into sections("chuncking")
@dataclass
class Chunk:
    doc: str
    section: str
    text: str

    @property
    def source(self) -> str:
        return f"{self.doc}#{self.section}"

def chunk_markdown(doc_name: str, text: str, max_chars: int = 1200) -> list[Chunk]:
    """One chunk per heading section; long sections are split by paragraphs."""
    chunks: list[Chunk] = []
    title, section, buf = doc_name, "Overview", []

    def flush():
        body = "\n".join(buf).strip()
        if not body:
            return
        paras, cur = body.split("\n\n"), ""
        for p in paras:
            if cur and len(cur) + len(p) > max_chars:
                chunks.append(Chunk(doc_name, section, f"{title} - {section}\n{cur.strip()}"))
                cur = ""
            cur += p + "\n\n"
        if cur.strip():
            chunks.append(Chunk(doc_name, section, f"{title} - {section}\n{cur.strip()}"))

    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            section, buf = line[3:].strip(), []
        else:
            buf.append(line)
    flush()
    return chunks

# --- C. Keyword search (BM25)
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(map(len, docs)) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n -f +0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            denom_base = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            for t in query:
                if t in tf:
                    out[i] += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + denom_base)
        return out


# --- D. Combining Rankings (RRF) and optional meaning search
def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank))."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=score.get, reverse=True)

def default_embedder() -> Embedder | None:
    """Return a sentence-transformers embedder if enabled and installed."""
    if not settings.use_embeddings:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("USE_EMBEDDINGS=1 but sentence-transformer is not installed, using BM25 only.")
        return None
    model = SentenceTransformer(settings.embedding_model)
    return lambda texts: np.asarray(model.encode(list(texts), normalize_embeddings=True))

# --- E. The Index, putting it all together
class DocIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        self.chunks = chunks
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        self.embedder = embedder
        self.vectors = None
        if embedder is not None and chunks:
            v = embedder([c.text for c in chunks])
            self.vectors = v / np.linalg.norm(v, axis=1, keepdims=True)

    @classmethod
    def from_dir(cls, docs_dir: Path | None = None, embedder: Embedder | None = None) -> "DocIndex":
        d = Path(docs_dir or settings.docs_dir)
        chunks: list[Chunk] = []
        for f in sorted(d.glob("*.md")):
            chunks.extend(chunk_markdown(f.stem, f.read_text(encoding="utf-8")))
        return cls(chunks, embedder)

    def search(self, query: str, k: int = 4) -> list[dict]:
        if not self.chunks:
            return[]
        kw = self.bm25.scores(tokenize(query))
        rankings = [[int(i) for i in np.argsort(-kw) if kw[i] > 0]]
        if self.vectors is not None:
            q = self.embedder([query])[0]
            sim = self.vectors @ (q / np.linalg.norm(q))
            rankings.append([int(i) for i in np.argsort(-sim)])
        order = rrf(rankings)[:k]
        return [{"source": self.chunks[i].source, "text": self.chunks[i].text} for i in order]
        
        
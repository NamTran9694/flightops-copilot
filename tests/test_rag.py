"""Tests for policy search (RAG). Run with: pytest -v"""
import numpy as np

from flightops.rag import BM25, DocIndex, chunk_markdown, rrf, tokenize


def test_tokenize_removes_stop_words():
    assert tokenize("The truck is on the ramp") == ["truck", "ramp"]


def test_chunking_by_heading():
    chunks = chunk_markdown("doc", "# Title\n\n## A\ntext a\n\n## B\ntext b")
    assert [c.section for c in chunks] == ["A", "B"]
    assert chunks[0].source == "doc#A"


def test_bm25_ranks_relevant_doc_first():
    docs = [tokenize("truck speed limit ramp"), tokenize("meal cart lifting injury")]
    scores = BM25(docs).scores(tokenize("speed limit"))
    assert scores[0] > scores[1]


def test_rrf_combines_rankings():
    # doc 0 and doc 1 are both near the top in the two rankings; doc 2 is last in both
    assert rrf([[0, 1, 2], [1, 0, 2]])[2] == 2


def test_search_finds_the_right_policy():
    idx = DocIndex.from_dir()
    top = idx.search("maximum speed near the aircraft", k=1)[0]
    assert top["source"] == "driver_safety_policy#Speed and Movement on the Ramp"


def test_hybrid_search_with_fake_embedder():
    def fake_embed(texts):
        return np.array([[t.lower().count("speed") + 0.1, t.lower().count("bonus") + 0.1] for t in texts])
    idx = DocIndex.from_dir(embedder=fake_embed)
    top = idx.search("maximum speed near aircraft", k=1)[0]
    assert top["source"].startswith("driver_safety_policy")
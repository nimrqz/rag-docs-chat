import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_eval import evaluate  # noqa: E402
from rag_chat.chat import NOT_FOUND, answer, cited_hits  # noqa: E402
from rag_chat.ingest import ingest_paths, split_text  # noqa: E402
from rag_chat.retriever import TfidfRetriever  # noqa: E402

class FakeClient:
    """Imita client.messages.create devolvendo um texto fixo."""

    def __init__(self, text):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self._text = text

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._text)])

@pytest.fixture(scope="module")
def retriever():
    return TfidfRetriever(ingest_paths(sorted((ROOT / "data").glob("*.md"))))

def test_split_text_respects_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = split_text(text, max_words=100, overlap=20)
    assert len(chunks) >= 3
    assert all(len(c.split()) <= 100 for c in chunks)
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]

def test_split_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        split_text("a b c", max_words=5, overlap=5)

def test_retriever_finds_relevant_chunk(retriever):
    top = retriever.search("valor do auxílio mensal de internet", k=1)[0]
    assert "R$ 150,00" in top.chunk.text

def test_answer_builds_context_and_extracts_citations(retriever):
    client = FakeClient("O auxílio é de R$ 150,00 [1].")
    result = answer("Qual o auxílio mensal?", retriever, client=client)
    assert "R$ 150,00" in client.calls[0]["messages"][0]["content"]
    assert len(result.cited) == 1 and result.cited[0] is result.hits[0]

def test_answer_without_context_skips_llm(retriever):
    client = FakeClient("não deveria ser chamado")
    result = answer("zzzz qqqq xxxx", retriever, client=client)
    assert result.text == NOT_FOUND and client.calls == []

def test_cited_hits_ignores_out_of_range():
    assert cited_hits("veja [9]", []) == []

def test_retrieval_quality_threshold(retriever):
    cases = json.loads((ROOT / "eval" / "eval_set.json").read_text(encoding="utf-8"))
    report = evaluate(retriever, cases, k=3, with_answers=False)
    assert report["hit@3"] >= 0.8, report

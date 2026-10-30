import os
import re
from dataclasses import dataclass, field

import anthropic

from .retriever import Hit, TfidfRetriever

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")

SYSTEM = """Você responde perguntas usando SOMENTE os trechos numerados fornecidos.
- Cite a fonte de cada afirmação com o número do trecho entre colchetes, ex.: [2].
- Se os trechos não contêm a resposta, diga exatamente: "Não encontrei isso nos documentos."
- Responda em português, de forma direta e curta."""

NOT_FOUND = "Não encontrei isso nos documentos."

@dataclass
class Answer:
    text: str
    hits: list[Hit]
    cited: list[Hit] = field(default_factory=list)

def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.chunk.label})\n{h.chunk.text}" for i, h in enumerate(hits, 1))

def cited_hits(text: str, hits: list[Hit]) -> list[Hit]:
    numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    return [hits[n - 1] for n in numbers if 1 <= n <= len(hits)]

def answer(
    question: str,
    retriever: TfidfRetriever,
    client: anthropic.Anthropic | None = None,
    k: int = 4,
) -> Answer:
    hits = retriever.search(question, k=k)
    if not hits:  # evita gastar tokens (e alucinar) sem nenhum contexto
        return Answer(NOT_FOUND, [])

    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Trechos:\n{build_context(hits)}\n\nPergunta: {question}",
            }
        ],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return Answer(text, hits, cited_hits(text, hits))

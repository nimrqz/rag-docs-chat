from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .ingest import Chunk

@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float

class TfidfRetriever:
    """Interface mínima (`search`) para poder trocar por embeddings/pgvector depois."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Nenhum chunk indexado")
        self.chunks = chunks
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode", lowercase=True
        )
        self._matrix = self._vectorizer.fit_transform(c.text for c in chunks)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        scores = linear_kernel(self._vectorizer.transform([query]), self._matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]

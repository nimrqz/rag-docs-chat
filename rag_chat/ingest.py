import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

@dataclass(frozen=True)
class Chunk:
    id: int
    text: str
    source: str
    page: int | None = None

    @property
    def label(self) -> str:
        return f"{self.source}" + (f", p.{self.page}" if self.page else "")

def load_pages(path: Path) -> list[tuple[int | None, str]]:
    """Retorna [(pagina, texto)]. Páginas só existem para PDF."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if suffix in {".md", ".txt"}:
        return [(None, path.read_text(encoding="utf-8"))]
    raise ValueError(f"Formato não suportado: {path.suffix}")

def split_text(text: str, max_words: int = 120, overlap: int = 25) -> list[str]:
    """Divide por parágrafo e agrupa até `max_words`, com sobreposição entre chunks."""
    if overlap >= max_words:
        raise ValueError("overlap deve ser menor que max_words")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        # parágrafo gigante: quebra por janela deslizante
        while len(words) > max_words:
            window, words = words[:max_words], words[max_words - overlap :]
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.append(" ".join(window))
        if len(current) + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.extend(words)
    if current:
        chunks.append(" ".join(current))
    return chunks

def ingest_paths(paths: list[Path], max_words: int = 120, overlap: int = 25) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in paths:
        for page, text in load_pages(path):
            for piece in split_text(text, max_words, overlap):
                chunks.append(Chunk(id=len(chunks), text=piece, source=path.name, page=page))
    return chunks

import tempfile
from pathlib import Path

import streamlit as st

from rag_chat.chat import answer
from rag_chat.ingest import ingest_paths
from rag_chat.retriever import TfidfRetriever

st.set_page_config(page_title="Chat com seus documentos", page_icon="📄")
st.title("📄 Chat com seus documentos (RAG)")

DATA = Path(__file__).parent / "data"
uploads = st.sidebar.file_uploader(
    "PDF, MD ou TXT", type=["pdf", "md", "txt"], accept_multiple_files=True
)
k = st.sidebar.slider("Trechos recuperados (k)", 1, 8, 4)

if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_resource(show_spinner="Indexando documentos...")
def build_retriever(names: tuple[str, ...], blobs: tuple[bytes, ...]):
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name, blob in zip(names, blobs):
            path = Path(tmp) / name
            path.write_bytes(blob)
            paths.append(path)
        if not paths:  # sem upload: usa o documento de exemplo
            paths = sorted(DATA.glob("*.md"))
        return TfidfRetriever(ingest_paths(paths))

retriever = build_retriever(
    tuple(u.name for u in uploads), tuple(u.getvalue() for u in uploads)
)
if not uploads:
    st.info("Usando o documento de exemplo (política de home office). Suba o seu na barra lateral.")

for role, text, sources in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)
        for label, snippet in sources:
            with st.expander(f"Fonte: {label}"):
                st.write(snippet)

if question := st.chat_input("Pergunte algo sobre os documentos"):
    st.session_state.history.append(("user", question, []))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            result = answer(question, retriever, k=k)
        st.markdown(result.text)
        sources = [(h.chunk.label, h.chunk.text) for h in (result.cited or result.hits)]
        for label, snippet in sources:
            with st.expander(f"Fonte: {label}"):
                st.write(snippet)
    st.session_state.history.append(("assistant", result.text, sources))

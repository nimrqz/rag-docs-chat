import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_chat.chat import NOT_FOUND, answer  # noqa: E402
from rag_chat.ingest import ingest_paths  # noqa: E402
from rag_chat.retriever import TfidfRetriever  # noqa: E402

def norm(s: str) -> str:
    return " ".join(s.lower().split())

def evaluate(retriever: TfidfRetriever, cases: list[dict], k: int, with_answers: bool) -> dict:
    hit_rank_sum, hits_at_k, answer_ok = 0.0, 0, 0
    for case in cases:
        results = retriever.search(case["question"], k=k)
        needle = norm(case["must_contain"])
        rank = next((i for i, h in enumerate(results, 1) if needle in norm(h.chunk.text)), None)
        if rank:
            hits_at_k += 1
            hit_rank_sum += 1 / rank
        if with_answers:
            result = answer(case["question"], retriever, k=k)
            if result.text != NOT_FOUND and needle in norm(result.text):
                answer_ok += 1
    n = len(cases)
    report = {"n": n, f"hit@{k}": hits_at_k / n, "mrr": hit_rank_sum / n}
    if with_answers:
        report["answer_contains_fact"] = answer_ok / n
    return report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, default=ROOT / "data")
    parser.add_argument("--cases", type=Path, default=ROOT / "eval" / "eval_set.json")
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--answers", action="store_true", help="também gera respostas (custa tokens)")
    args = parser.parse_args()

    files = sorted(p for p in args.docs.iterdir() if p.suffix in {".md", ".txt", ".pdf"})
    retriever = TfidfRetriever(ingest_paths(files))
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(retriever, cases, args.k, args.answers), indent=2))

if __name__ == "__main__":
    main()

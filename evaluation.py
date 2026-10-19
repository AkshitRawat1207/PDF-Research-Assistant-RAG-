"""
evaluation.py — a retrieval-quality harness for the PDF RAG pipeline.

Answer quality in a RAG system is capped by *retrieval* quality: if the right
page never makes it into the context, the LLM cannot cite it. This module
measures retrieval directly, at the same (source, page) citation granularity the
app shows to users, against a small labelled gold set.

Metrics (all standard IR measures, computed per question then averaged):
  * recall@k     — fraction of the relevant pages that appear in the top-k.
  * precision@k  — fraction of the returned top-k that are relevant
                   (this is exactly the app's *citation accuracy*).
  * hit@k        — did at least one relevant page make the top-k?
  * MRR          — mean reciprocal rank of the first relevant page.
  * MAP          — mean average precision over the ranked results.

The retriever is injected as a callable ``question -> [doc_id, ...]`` (ranked),
so the metrics are fully testable with a stub. ``faiss_retriever`` adapts the
real FAISS pipeline; it is imported lazily so this module (and its tests) run
with no API key and no built index.

CLI:
    python evaluation.py --gold gold.json --k 5           # against the real index
    python evaluation.py --gold gold.json --k 5 --json
"""

from __future__ import annotations

import argparse
import json
from typing import Callable, Dict, Iterable, List, Sequence

Retriever = Callable[[str], List[str]]


# --------------------------------------------------------------------------- #
#  Document identity — one id per (source, page), matching the app's citations
# --------------------------------------------------------------------------- #

def doc_id(source: str, page) -> str:
    return f"{source}::p{page}"


def ids_from_docs(docs: Iterable) -> List[str]:
    """Map retrieved LangChain documents to ranked, de-duplicated page ids.

    De-duplication preserves rank order and mirrors the app's ``_dedupe_sources``
    so a page that contributes several chunks counts once — the unit a user
    actually sees cited.
    """

    seen = set()
    ids: List[str] = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        ident = doc_id(meta.get("source", "document"), meta.get("page", "?"))
        if ident not in seen:
            seen.add(ident)
            ids.append(ident)
    return ids


# --------------------------------------------------------------------------- #
#  Pure IR metrics — operate on ranked ids + a relevant set
# --------------------------------------------------------------------------- #

def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    topk = retrieved[:k]
    if not topk:
        return 0.0
    hits = sum(1 for r in topk if r in relevant)
    return hits / len(topk)


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    relevant = set(relevant)
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(relevant)


# --------------------------------------------------------------------------- #
#  Harness
# --------------------------------------------------------------------------- #

def evaluate(retriever: Retriever, gold: Sequence[Dict], k: int = 5) -> Dict:
    """Run ``retriever`` over each gold item and aggregate retrieval metrics.

    ``gold`` items are ``{"question": str, "relevant": [doc_id, ...]}``.
    Returns aggregate means plus a per-question breakdown.
    """

    per_question = []
    for item in gold:
        question = item["question"]
        relevant = item["relevant"]
        retrieved = retriever(question)
        per_question.append({
            "question": question,
            "recall@k": recall_at_k(retrieved, relevant, k),
            "precision@k": precision_at_k(retrieved, relevant, k),
            "hit@k": hit_at_k(retrieved, relevant, k),
            "rr": reciprocal_rank(retrieved, relevant),
            "ap": average_precision(retrieved, relevant),
        })

    n = len(per_question)

    def mean(field: str) -> float:
        return round(sum(q[field] for q in per_question) / n, 4) if n else 0.0

    return {
        "k": k,
        "questions": n,
        "recall@k": mean("recall@k"),
        "precision@k": mean("precision@k"),
        "hit_rate@k": mean("hit@k"),
        "mrr": mean("rr"),
        "map": mean("ap"),
        "per_question": per_question,
    }


def format_report(result: Dict) -> str:
    k = result["k"]
    lines = [
        f"Retrieval Evaluation  (k={k}, {result['questions']} questions)",
        "=" * 48,
        f"  Recall@{k}      {result['recall@k']:.3f}",
        f"  Precision@{k}   {result['precision@k']:.3f}   (citation accuracy)",
        f"  Hit-rate@{k}    {result['hit_rate@k']:.3f}",
        f"  MRR            {result['mrr']:.3f}",
        f"  MAP            {result['map']:.3f}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Real-pipeline adapter (lazy import; needs a built FAISS index)
# --------------------------------------------------------------------------- #

def faiss_retriever(vector_store, k: int) -> Retriever:
    """Adapt the real FAISS pipeline to the ``question -> [doc_id]`` interface."""

    import rag_pipeline  # lazy: keeps this module importable with no deps/index

    def _retrieve(question: str) -> List[str]:
        docs = rag_pipeline.retrieve(vector_store, question, k=k)
        return ids_from_docs(docs)

    return _retrieve


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate PDF RAG retrieval quality.")
    parser.add_argument("--gold", required=True, help="Path to a gold JSON file.")
    parser.add_argument("--k", type=int, default=5, help="Cutoff for @k metrics.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    with open(args.gold, encoding="utf-8") as fh:
        gold = json.load(fh)

    import rag_pipeline  # lazy
    vector_store = rag_pipeline.load_vector_store()
    if vector_store is None:
        raise SystemExit("No FAISS index found — process PDFs in the app first.")

    result = evaluate(faiss_retriever(vector_store, args.k), gold, k=args.k)
    print(json.dumps(result, indent=2) if args.json else format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

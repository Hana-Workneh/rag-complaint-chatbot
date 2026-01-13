import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from src.llm_providers import LocalHFGenerator

PROMPT_TEMPLATE = """You are a financial complaint analyst.

Write 3–5 SHORT themes as bullet points. Each theme must be 6–12 words.
Do NOT quote the signals. Do NOT write paragraphs.

You MUST start with 'Themes:' and then bullet points.

Themes:
- ...
- ...
- ...

Signals:
{signals}

Question: {question}

Answer:
"""


def make_signals(docs: List[str], metas: List[Dict[str, Any]], max_items: int = 3) -> str:
    lines = []
    for i, (d, m) in enumerate(zip(docs, metas), start=1):
        text = d.strip().replace("\n", " ")
        text = text[:160]
        lines.append(f"- ({m.get('product_category','?')}) {text}")
        if i >= max_items:
            break
    return "\n".join(lines)


def pick_quotes(docs: List[str], max_quotes: int = 3) -> List[str]:
    quotes = []
    for d in docs:
        t = " ".join(d.strip().split())
        q = t[:180]
        if q:
            quotes.append(q)
        if len(quotes) >= max_quotes:
            break
    return quotes


def fallback_themes(docs: List[str]) -> List[str]:
    text = " ".join(" ".join(d.split()) for d in docs).lower()

    candidates = [
        ("poor customer service and misleading information", ["customer service", "misinformation", "representatives"]),
        ("credit limit reductions increasing utilization ratios", ["reduce my credit limit", "credit limit", "usage shoot up", "utilization"]),
        ("unexpected interest rate increases and higher payments", ["interest rate", "apr", "increase in my interest rate"]),
        ("fees and billing disputes causing frustration", ["fee", "charged", "billing", "late fee"]),
        ("poor communication during hardship or crisis periods", ["stress", "pandemic", "recession", "depression"]),
    ]

    themes = []
    for label, keys in candidates:
        if any(k in text for k in keys):
            themes.append(label)

    if len(themes) < 3:
        themes = (themes + ["account management frustrations", "slow or unhelpful dispute resolution", "unexpected account changes"])[:3]

    return themes[:5]


@dataclass
class RetrievalResult:
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    ids: List[str]


class ComplaintRAG:
    def __init__(
        self,
        persist_dir: str = "vector_store/full_chroma_task3",
        collection: str = "complaints_full",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.col = self.client.get_collection(name=collection)
        self.embedder = SentenceTransformer(embedding_model)
        self.llm = LocalHFGenerator()

    def retrieve(self, question: str, top_k: int = 5, product_filters: Optional[List[str]] = None) -> RetrievalResult:
        q_emb = self.embedder.encode([question], normalize_embeddings=True).tolist()[0]
        where = {"product_category": {"$in": product_filters}} if product_filters else None

        res = self.col.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas"],
        )

        return RetrievalResult(
            documents=res["documents"][0],
            metadatas=res["metadatas"][0],
            ids=res.get("ids", [[]])[0],
        )

    def answer(self, question: str, top_k: int = 5, product_filters: Optional[List[str]] = None):
        rr = self.retrieve(question, top_k=top_k, product_filters=product_filters)

        signals = make_signals(rr.documents, rr.metadatas, max_items=min(3, len(rr.documents)))
        prompt = PROMPT_TEMPLATE.format(signals=signals, question=question)
        theme_text = self.llm.generate(prompt).text.strip()

        if ("Themes:" not in theme_text) or ("-" not in theme_text):
            themes = fallback_themes(rr.documents)
            theme_text = "Themes:\n" + "\n".join([f"- {t}" for t in themes])

        quotes = pick_quotes(rr.documents, max_quotes=3)

        final = theme_text.strip()
        final = final + "\n\nEvidence:\n" + "\n".join([f'- "{q}"' for q in quotes])

        return final, rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--product", action="append", default=[], help='Repeatable filter. e.g. --product "Credit Card"')
    args = ap.parse_args()

    rag = ComplaintRAG()
    answer, rr = rag.answer(args.question, top_k=args.top_k, product_filters=(args.product or None))

    print("\n=== ANSWER ===\n")
    print(answer)

    print("\n=== SOURCES (top retrieved chunks) ===\n")
    for i, (doc, md) in enumerate(zip(rr.documents, rr.metadatas), start=1):
        print(
            f"[{i}] product_category={md.get('product_category')} "
            f"complaint_id={md.get('complaint_id')} chunk_index={md.get('chunk_index')}"
        )
        print(doc[:600].replace("\n", " ") + ("..." if len(doc) > 600 else ""))
        print()


if __name__ == "__main__":
    main()

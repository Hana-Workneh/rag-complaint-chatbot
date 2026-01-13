from datetime import datetime
from pathlib import Path

import pandas as pd

from src.task3_rag_pipeline import ComplaintRAG


QUESTIONS = [
    ("Why are customers unhappy with Credit Cards?", ["Credit Card"]),
    ("What are the most common issues reported for Money Transfers?", ["Money Transfer"]),
    ("What recurring problems do users report about Savings Accounts?", ["Savings Account"]),
    ("What are the top complaint themes for Personal Loans?", ["Personal Loan"]),
    ("What complaints indicate possible fraud or identity theft?", None),
    ("What kinds of billing disputes do customers mention?", None),
    ("Why do customers complain about fees and interest?", ["Credit Card", "Personal Loan"]),
    ("Compare issues in Credit Cards vs Money Transfers.", ["Credit Card", "Money Transfer"]),
]


def main():
    rag = ComplaintRAG()
    rows = []

    for q, filt in QUESTIONS:
        ans, rr = rag.answer(q, top_k=5, product_filters=filt)

        src_preview = ""
        if rr.documents:
            src_preview = rr.documents[0][:240].replace("\n", " ") + ("..." if len(rr.documents[0]) > 240 else "")

        rows.append({
            "Question": q,
            "Generated Answer": ans,
            "Retrieved Source (sample)": src_preview,
            "Quality Score (1-5)": "",
            "Comments/Analysis": "",
        })

    df = pd.DataFrame(rows)
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / f"task3_eval_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

    print(f"✅ Wrote evaluation table to: {out_path}")


if __name__ == "__main__":
    main()

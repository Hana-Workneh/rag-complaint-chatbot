import argparse
import math
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm


def stratified_sample(df: pd.DataFrame, strata_col: str, n: int, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    df[strata_col] = df[strata_col].fillna("UNKNOWN")
    counts = df[strata_col].value_counts()
    total = len(df)

    alloc = {k: max(1, int(math.floor((v / total) * n))) for k, v in counts.items()}

    # adjust to exactly n
    current = sum(alloc.values())
    while current < n:
        for k in counts.index:
            alloc[k] += 1
            current += 1
            if current >= n:
                break
    while current > n:
        for k in reversed(counts.index.tolist()):
            if alloc[k] > 1:
                alloc[k] -= 1
                current -= 1
                if current <= n:
                    break

    parts = []
    for k, k_n in alloc.items():
        g = df[df[strata_col] == k]
        parts.append(g.sample(n=min(k_n, len(g)), random_state=seed))

    return pd.concat(parts, ignore_index=True)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/filtered_complaints.csv")
    ap.add_argument("--out-sample", default="data/processed/task2_sample.csv")
    ap.add_argument("--out-chunks", default="data/processed/task2_chunks.csv")
    ap.add_argument("--product-col", default="Product")
    ap.add_argument("--text-col", default="narrative_clean")
    ap.add_argument("--id-col", default="Complaint ID")
    ap.add_argument("--n-sample", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=50)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # sample
    sample_df = stratified_sample(df, args.product_col, args.n_sample, args.seed)
    sample_df.to_csv(args.out_sample, index=False)
    print("✅ Saved sample:", args.out_sample)
    print(sample_df[args.product_col].value_counts())

    # chunk into rows
    rows = []
    for _, r in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Chunking sample"):
        cid = str(r.get(args.id_col, ""))
        product = str(r.get(args.product_col, "UNKNOWN"))
        text = r.get(args.text_col, "")
        chunks = chunk_text(text, args.chunk_size, args.overlap)
        for i, ch in enumerate(chunks):
            rows.append({
                "complaint_id": cid,
                "product_category": product,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_text": ch,
            })

    chunks_df = pd.DataFrame(rows)
    chunks_df.to_csv(args.out_chunks, index=False)
    print("✅ Saved chunks:", args.out_chunks)
    print("✅ Total chunks:", len(chunks_df))


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import chromadb
from sentence_transformers import SentenceTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/task2_chunks.csv")
    ap.add_argument("--persist-dir", default="vector_store/sample_chroma_task2")
    ap.add_argument("--collection", default="complaints_task2_sample")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Missing chunks file: {chunks_path}\n"
            "Run: python -m src.task2_prepare_sample_and_chunks --n-sample 12000"
        )

    out_dir = Path(args.persist_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(chunks_path)
    required = {"complaint_id", "product_category", "chunk_index", "total_chunks", "chunk_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Chunks file missing columns: {missing}")

    ids = (df["complaint_id"].astype(str) + "__" + df["chunk_index"].astype(int).astype(str)).tolist()
    docs = df["chunk_text"].astype(str).tolist()
    metas = df[["complaint_id", "product_category", "chunk_index", "total_chunks"]].to_dict(orient="records")

    print(f"Loaded {len(docs)} chunks from {chunks_path}")

    client = chromadb.PersistentClient(path=str(out_dir))

    # Clean reruns
    try:
        client.delete_collection(args.collection)
    except Exception:
        pass

    col = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    model = SentenceTransformer(args.model)

    bs = args.batch_size
    for start in tqdm(range(0, len(docs), bs), desc="Embedding+Indexing"):
        end = min(len(docs), start + bs)
        batch_docs = docs[start:end]
        batch_ids = ids[start:end]
        batch_metas = metas[start:end]

        embs = model.encode(batch_docs, normalize_embeddings=True).tolist()
        col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=embs)

    # sanity query
    q = "charged twice late fee"
    res = col.query(query_texts=[q], n_results=5)
    print("\nSanity query:", q)
    print("Top IDs:", res["ids"][0])
    print("Top products:", [m.get("product_category") for m in res["metadatas"][0]])

    print(f"\n✅ Persisted ChromaDB at: {out_dir}")
    print(f"✅ Collection: {args.collection}")


if __name__ == "__main__":
    main()

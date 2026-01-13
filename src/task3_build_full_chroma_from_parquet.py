import argparse
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import pyarrow.parquet as pq
from tqdm import tqdm


def _to_py_embedding(x: Any) -> List[float]:
    # embedding may be numpy array, list, or pyarrow scalar
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, list):
        return x
    return list(x)


def _normalize_meta(m: Any) -> Dict[str, Any]:
    # metadata may be dict-like; ensure JSON-serializable
    if m is None:
        return {}
    if not isinstance(m, dict):
        # fallback: store as string
        return {"metadata": str(m)}

    out = {}
    for k, v in m.items():
        if v is None:
            continue
        if isinstance(v, (int, float, bool, str)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/raw/complaint_embeddings.parquet")
    ap.add_argument("--persist-dir", default="vector_store/full_chroma_task3")
    ap.add_argument("--collection", default="complaints_full")
    ap.add_argument("--batch-rows", type=int, default=5000)
    ap.add_argument("--max-rows", type=int, default=0, help="0=all rows; use for testing")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet: {parquet_path}")

    out_dir = Path(args.persist_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(parquet_path)

    # Inspect first batch to detect real columns
    first_batch = next(pf.iter_batches(batch_size=5))
    df0 = first_batch.to_pandas()
    cols0 = set(df0.columns)

    if "id" not in cols0 or "document" not in cols0:
        raise ValueError(f"Expected columns 'id' and 'document'. Found: {list(df0.columns)}")

    has_embedding = "embedding" in cols0
    has_metadata_dict = "metadata" in cols0

    embedder = None
    if not has_embedding:
        # only needed if embeddings are not present in parquet
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(args.model)

    client = chromadb.PersistentClient(path=str(out_dir))
    try:
        client.delete_collection(args.collection)
    except Exception:
        pass

    col = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    total_rows = pf.metadata.num_rows
    if args.max_rows and args.max_rows > 0:
        total_rows = min(total_rows, args.max_rows)

    written = 0
    pbar = tqdm(total=total_rows, desc="Indexing parquet -> Chroma")

    # IMPORTANT: do NOT pass columns=... (schema.names is misleading for this file)
    for batch in pf.iter_batches(batch_size=args.batch_rows):
        if args.max_rows and args.max_rows > 0 and written >= args.max_rows:
            break

        df = batch.to_pandas()

        # Trim if max_rows hit mid-batch
        if args.max_rows and args.max_rows > 0 and (written + len(df)) > args.max_rows:
            df = df.iloc[: (args.max_rows - written)]

        ids = df["id"].astype(str).tolist()
        docs = df["document"].astype(str).tolist()

        # Embeddings: use prebuilt if present, else compute
        if "embedding" in df.columns:
            embs = [_to_py_embedding(e) for e in df["embedding"].tolist()]
        else:
            embs = embedder.encode(docs, normalize_embeddings=True).tolist()

        # Metadata: prefer dict column if present; else build from remaining columns
        if "metadata" in df.columns:
            metadatas = [_normalize_meta(m) for m in df["metadata"].tolist()]
        else:
            meta_cols = [c for c in df.columns if c not in ("id", "document", "embedding")]
            metadatas = []
            for _, row in df[meta_cols].iterrows():
                m = {}
                for k, v in row.to_dict().items():
                    if v is None:
                        continue
                    if isinstance(v, (int, float, bool, str)):
                        m[k] = v
                    else:
                        m[k] = str(v)
                metadatas.append(m)

        col.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embs)

        written += len(ids)
        pbar.update(len(ids))

    pbar.close()
    print(f"\n✅ Persisted Chroma index at: {out_dir}")
    print(f"✅ Collection: {args.collection}")
    print(f"✅ Rows indexed: {written}")
    print(f"✅ Detected parquet layout: embedding={'yes' if has_embedding else 'no'} metadata_dict={'yes' if has_metadata_dict else 'no'}")


if __name__ == "__main__":
    main()

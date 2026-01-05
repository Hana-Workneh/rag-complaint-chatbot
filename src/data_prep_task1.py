import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PRODUCTS = [
    "Credit card",
    "Credit card or prepaid card",
    "Checking or savings account",
    "Money transfer, virtual currency, or money service",
    "Money transfers",
    "Payday loan, title loan, or personal loan",
    "Payday loan, title loan, personal loan, or advance loan",
    "Consumer Loan",
]

BOILERPLATE_PATTERNS = [
    r"\bi am writing to file a complaint\b",
    r"\bi am filing a complaint\b",
    r"\bthis is a complaint\b",
    r"\bdear (sir|madam)\b",
]


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip().lower()

    for pat in BOILERPLATE_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)

    # remove odd symbols, keep basic punctuation
    t = re.sub(r"[^a-z0-9\s\.\,\;\:\?\!\-\'\/]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def word_count(s: str) -> int:
    if not isinstance(s, str) or not s.strip():
        return 0
    return len(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/complaints.csv")
    ap.add_argument("--output", default="data/filtered_complaints.csv")
    ap.add_argument("--product-col", default="Product")
    ap.add_argument("--narrative-col", default="Consumer complaint narrative")
    ap.add_argument("--id-col", default="Complaint ID")
    ap.add_argument("--products", nargs="*", default=DEFAULT_PRODUCTS)
    ap.add_argument("--eda-out", default="notebooks/outputs_task1")
    ap.add_argument("--chunksize", type=int, default=200000, help="Chunk size for reading large CSV")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    eda_out = Path(args.eda_out)
    eda_out.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # We’ll compute EDA stats in streaming mode (since file is ~6GB)
    product_counts = {}
    missing_narr = 0
    empty_narr = 0
    wc_list = []

    total_rows = 0

    # Read in chunks
    for chunk in pd.read_csv(in_path, chunksize=args.chunksize, low_memory=False):
        total_rows += len(chunk)

        # product distribution
        vc = chunk[args.product_col].value_counts(dropna=False)
        for k, v in vc.items():
            product_counts[k] = product_counts.get(k, 0) + int(v)

        # narrative stats
        narr = chunk[args.narrative_col]
        missing_narr += narr.isna().sum()
        narr_filled = narr.fillna("").astype(str)
        empty_narr += (narr_filled.str.strip() == "").sum()

        # wordcount (sample to avoid huge memory)
        # take up to 20k rows per chunk for wc
        sample_n = min(20000, len(narr_filled))
        wc = narr_filled.head(sample_n).apply(word_count).tolist()
        wc_list.extend(wc)

    # Save EDA summary
    product_series = pd.Series(product_counts).sort_values(ascending=False)
    summary_path = eda_out / "eda_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Total rows (raw): {total_rows}\n")
        f.write(f"Missing narratives (NaN): {missing_narr}\n")
        f.write(f"Empty narratives (''): {empty_narr}\n\n")
        f.write("Top products:\n")
        f.write(product_series.head(25).to_string())
        f.write("\n\nNarrative wordcount (sampled):\n")
        f.write(pd.Series(wc_list).describe().to_string())
        f.write("\n")

    # Plot product distribution (top 15)
    plt.figure()
    product_series.head(15).plot(kind="bar")
    plt.title("Top 15 Products by Complaint Count (raw)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(eda_out / "product_distribution.png", dpi=160)
    plt.close()

    # Plot wordcount distribution (sampled)
    plt.figure()
    pd.Series(wc_list).clip(upper=2000).plot(kind="hist", bins=50)
    plt.title("Narrative Word Count Distribution (sampled, clipped at 2000)")
    plt.xlabel("Word count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(eda_out / "narrative_length_hist.png", dpi=160)
    plt.close()

    # Now create filtered + cleaned CSV in streaming mode
    print("Creating filtered_complaints.csv (streaming)...")
    first_write = True

    for chunk in pd.read_csv(in_path, chunksize=args.chunksize, low_memory=False):
        # filter product
        chunk = chunk[chunk[args.product_col].isin(args.products)].copy()

        # remove empty narratives
        chunk[args.narrative_col] = chunk[args.narrative_col].fillna("").astype(str)
        chunk = chunk[chunk[args.narrative_col].str.strip() != ""].copy()

        # clean
        chunk["narrative_clean"] = chunk[args.narrative_col].apply(normalize_text)
        chunk = chunk[chunk["narrative_clean"].str.strip() != ""].copy()

        # keep useful columns
        keep_cols = [
            args.id_col,
            args.product_col,
            "Issue",
            "Sub-issue",
            "Company",
            "State",
            "Date received",
            "Submitted via",
            "narrative_clean",
        ]
        keep_cols = [c for c in keep_cols if c in chunk.columns]
        chunk_out = chunk[keep_cols]

        chunk_out.to_csv(out_path, index=False, mode="w" if first_write else "a", header=first_write)
        first_write = False

    print(f"✅ EDA outputs saved to: {eda_out}")
    print(f"✅ Filtered + cleaned dataset saved to: {out_path}")


if __name__ == "__main__":
    main()

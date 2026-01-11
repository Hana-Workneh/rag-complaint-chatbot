# RAG Complaint Chatbot — Intelligent Complaint Analysis for Financial Services

This repository contains a Retrieval-Augmented Generation (RAG) pipeline to transform unstructured customer complaint narratives into actionable insights for internal stakeholders (Product, Support, Compliance).

The project is built around the CFPB complaint dataset and includes:

- **Task 1:** Exploratory Data Analysis (EDA) + preprocessing and filtering
- **Task 2:** Stratified sampling + chunking + embeddings + vector store indexing (ChromaDB)
- **Task 3 (next):** RAG core logic using the provided pre-built embeddings/vector store
- **Task 4 (next):** Interactive UI (Gradio/Streamlit) with source display

---

## Repository Structure

```text
rag-complaint-chatbot/
├── .github/workflows/unittests.yml
├── data/
│   ├── raw/                         # raw datasets (not tracked by git)
│   └── processed/                   # intermediate artifacts (not tracked by git)
├── notebooks/
│   └── outputs_task1/               # EDA plots + summary
├── src/
│   ├── data_prep_task1.py           # Task 1: EDA + preprocessing
│   ├── task2_prepare_sample_and_chunks.py
│   └── task2_embed_and_index_chroma.py
├── tests/
│   └── test_sanity.py
├── vector_store/                    # persisted ChromaDB index (not tracked by git)
├── REPORT_INTERIM.md
├── app.py                           # (Task 4) UI entrypoint
├── requirements.txt
└── README.md
Data Requirements

Place the following files under data/raw/:

Raw CFPB complaints dataset

data/raw/complaints.csv (unzipped from complaints.csv.zip)

Pre-built embeddings file (for Tasks 3–4)

data/raw/complaint_embeddings.parquet

Expected raw CSV columns include:

Product

Consumer complaint narrative

Complaint ID

plus other metadata fields (Issue, Sub-issue, Company, State, Date received, etc.)

Environment Setup (Windows)
Important: Python versions

Task 1 + sampling/chunking (Task 2 prep): works with Python 3.13

Task 2 embeddings/indexing: requires Python 3.11 on Windows (recommended for PyTorch / sentence-transformers compatibility)

Create a Python 3.11 virtual environment (recommended)
py -3.11 -m venv .venv311
.\.venv311\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt


Verify:

python -c "import pandas, chromadb; import sentence_transformers; print('ok')"

Task 1 — EDA + Preprocessing
Goal

Understand complaint distribution and narrative availability

Filter to project-relevant product categories

Remove empty narratives

Clean narrative text for embedding quality

Save a cleaned dataset: data/filtered_complaints.csv

Run Task 1
python -m src.data_prep_task1

Outputs

data/filtered_complaints.csv (generated locally)

notebooks/outputs_task1/eda_summary.txt

notebooks/outputs_task1/product_distribution.png

notebooks/outputs_task1/narrative_length_hist.png

data/filtered_complaints.csv is not committed due to size.

Task 2 — Sampling + Chunking + Embeddings + Vector Store Indexing

Task 2 is implemented in two steps:

Prepare a stratified sample + chunk narratives

Embed chunks and persist in ChromaDB

2.1 Stratified sample + chunk generation

This creates a proportional stratified sample (default: 12,000 complaints) by Product and splits each narrative into overlapping character chunks.

python -m src.task2_prepare_sample_and_chunks --n-sample 12000


Generated files (local only):

data/processed/task2_sample.csv

data/processed/task2_chunks.csv

Chunking parameters:

chunk_size = 500

chunk_overlap = 50

2.2 Embed + index into ChromaDB (persisted vector store)

This step embeds data/processed/task2_chunks.csv using:

sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)

and writes a persisted ChromaDB index to:

vector_store/sample_chroma_task2/

collection: complaints_task2_sample

Run:

python -m src.task2_embed_and_index_chroma


Sanity check is included in the script (example query like "charged twice late fee").

vector_store/ is not committed to git due to size.

Reproducibility Notes

If you are re-running indexing and want a clean index, the indexing script recreates the Chroma collection automatically.

If you want to change sample size:

python -m src.task2_prepare_sample_and_chunks --n-sample 15000
python -m src.task2_embed_and_index_chroma

Running Tests (CI)
pytest -q


GitHub Actions runs tests on push/PR via .github/workflows/unittests.yml.

Next Steps (Tasks 3–4)
Task 3: RAG Core Logic

Load the pre-built embeddings/vector store (complaint_embeddings.parquet)

Implement retriever (top-k similarity search)

Prompt template + LLM generation

Evaluation table with 5–10 representative questions

Task 4: UI

Build a Streamlit or Gradio app (app.py)

Must show:

question input

generated answer

retrieved sources under the answer (for trust)

clear/reset button

Troubleshooting
Hugging Face symlink warning on Windows

You may see a warning about symlinks when downloading models. This does not affect correctness; it only impacts disk usage. To remove the warning, enable Windows Developer Mode or run the terminal as Administrator.

Python 3.13 issues with embeddings

If PyTorch/sentence-transformers fails on Python 3.13, use Python 3.11:

py -0
py -3.11 -m venv .venv311

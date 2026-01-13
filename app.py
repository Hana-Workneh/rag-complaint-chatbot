import gradio as gr

from src.task3_rag_pipeline import ComplaintRAG


# Load once (so the vector store + embedding model aren't reloaded every query)
rag = ComplaintRAG(
    persist_dir="vector_store/full_chroma_task3",
    collection="complaints_full",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
)


PRODUCT_CHOICES = ["All", "Credit Card", "Personal Loan", "Savings Account", "Money Transfer"]


def format_sources(rr, max_chars: int = 650) -> str:
    if rr is None or not rr.documents:
        return "No sources retrieved."

    blocks = []
    for i, (doc, md) in enumerate(zip(rr.documents, rr.metadatas), start=1):
        prod = md.get("product_category", "?")
        cid = md.get("complaint_id", "?")
        chunk_idx = md.get("chunk_index", "?")
        snippet = " ".join(str(doc).split())
        snippet = snippet[:max_chars] + ("..." if len(snippet) > max_chars else "")
        blocks.append(
            f"**[{i}] product_category={prod} | complaint_id={cid} | chunk_index={chunk_idx}**\n\n"
            f"> {snippet}\n"
        )
    return "\n\n".join(blocks)


def answer_question(question: str, product: str, top_k: int):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""

    filters = None if product == "All" else [product]

    answer, rr = rag.answer(question, top_k=top_k, product_filters=filters)
    sources_md = format_sources(rr)
    return answer, sources_md


with gr.Blocks(title="CrediTrust Complaint Intelligence (RAG)") as demo:
    gr.Markdown(
        "# Intelligent Complaint Analysis (RAG)\n"
        "Ask a question about customer complaints. The system retrieves relevant complaint excerpts and generates a grounded answer."
    )

    with gr.Row():
        product = gr.Dropdown(choices=PRODUCT_CHOICES, value="All", label="Product filter")
        top_k = gr.Slider(minimum=3, maximum=10, value=5, step=1, label="Top-K sources")

    question = gr.Textbox(
        label="Your question",
        placeholder='e.g., "Why are customers unhappy with Credit Cards?"',
        lines=2
    )

    with gr.Row():
        ask_btn = gr.Button("Ask", variant="primary")
        clear_btn = gr.Button("Clear")

    answer_out = gr.Markdown(label="Answer")
    sources_out = gr.Markdown(label="Sources")

    ask_btn.click(
        fn=answer_question,
        inputs=[question, product, top_k],
        outputs=[answer_out, sources_out],
    )

    clear_btn.click(
        fn=lambda: ("", "", "All", 5),
        inputs=[],
        outputs=[question, answer_out, product, top_k],
    )

demo.launch()

import argparse
from typing import List, Dict

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from index_qasper import build_and_index, DEFAULT_MODEL


def build_retrieval_pipeline(document_store: InMemoryDocumentStore, model_name: str) -> Pipeline:
    """
    Build a simple question-aware retrieval pipeline:
    - Embed the question using the same model as the documents
    - Retrieve top_k paragraphs from the in-memory store
    Returns the pipeline.
    """
    text_embedder = SentenceTransformersTextEmbedder(model=model_name)
    retriever = InMemoryEmbeddingRetriever(document_store)

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    return pipeline


essential_meta = ["id", "para_idx"]


def _neighbor_indices(idx: int) -> List[int]:
    # Return neighbor relative offsets [-1, 1]
    return [idx - 1, idx + 1]


def retrieve(model_name: str, split: str, question: str, top_k: int) -> List[Dict]:
    """
    Build the index in-memory (fresh for this run), then retrieve paragraphs relevant to the question.
    Returns a list of retrieved document dicts with content, meta, score, and neighbors when available.
    """
    store = build_and_index(model_name=model_name, split=split)
    pipeline = build_retrieval_pipeline(store, model_name)

    # Warm-up the text embedder
    text_component = pipeline.get_component("text_embedder")
    if hasattr(text_component, "warm_up"):
        text_component.warm_up()

    result = pipeline.run({"text_embedder": {"text": question}, "retriever": {"top_k": top_k}})
    documents = result["retriever"]["documents"]

    outputs: List[Dict] = []
    for d in documents:
        meta = d.meta or {}
        para_idx = meta.get("para_idx")
        paper_id = meta.get("paper_id")

        neighbors: List[Dict] = []
        if isinstance(para_idx, int) and paper_id is not None:
            # Fallback neighbor lookup: scan documents and match meta in Python for compatibility across Haystack versions
            try:
                all_docs = store.get_all_documents()  # type: ignore[attr-defined]
            except Exception:
                all_docs = []
            index_to_doc = {}
            for nd in all_docs:
                m = nd.meta or {}
                if m.get("paper_id") == paper_id:
                    idx_val = m.get("para_idx")
                    if isinstance(idx_val, int):
                        index_to_doc[idx_val] = nd
            for n_idx in _neighbor_indices(para_idx):
                n_doc = index_to_doc.get(n_idx)
                if n_doc is not None:
                    neighbors.append({"content": n_doc.content, "meta": n_doc.meta})

        outputs.append({
            "content": d.content,
            "meta": meta,
            "score": getattr(d, "score", None),
            "neighbors": neighbors,
        })
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Question-aware paragraph retrieval on QASPER")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="SentenceTransformer model for embeddings")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to index")
    parser.add_argument("--question", type=str, required=True, help="User question")
    parser.add_argument("--top_k", type=int, default=5, help="Number of paragraphs to retrieve")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hits = retrieve(model_name=args.model, split=args.split, question=args.question, top_k=args.top_k)
    for i, h in enumerate(hits, 1):
        para_idx = h["meta"].get("para_idx")
        paper_id = h["meta"].get("paper_id")
        title = h["meta"].get("title")
        display = paper_id if paper_id else (title if title else "<no-id>")
        score = h.get("score")
        print(f"[{i}] paper={display} para={para_idx} score={score}\n{h['content']}\n")
        # Print neighbors if any
        for j, nb in enumerate(h.get("neighbors", []), 1):
            n_idx = nb["meta"].get("para_idx")
            print(f"  [neighbor {j}] para={n_idx}\n  {nb['content']}\n")

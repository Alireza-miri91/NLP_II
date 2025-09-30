import argparse
from typing import List

from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter

from qasper_loader import load_qasper_haystack_documents


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_and_index(model_name: str = DEFAULT_MODEL, split: str = "train", 
                   batch_size: int = 32, granularity: str = "paragraph") -> InMemoryDocumentStore:
    """
    Build an in-memory document store, embed QASPER paragraph documents, and write them.
    Optimized for RTX A6000 with batch processing for better GPU utilization.

    Args:
        model_name: Sentence transformer model name
        split: Dataset split to index
        batch_size: Batch size for embedding (optimized for 49GB VRAM)

    Returns:
        The initialized InMemoryDocumentStore so it can be reused immediately by a retriever.
    """
    import torch
    
    document_store = InMemoryDocumentStore()

    # Collect some (or all) documents. Keeping it as a list so the embedder can process them.
    documents: List = list(load_qasper_haystack_documents(split=split, granularity=granularity))
    print(f"📚 Loaded {len(documents)} documents for indexing")

    indexing = Pipeline()
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model=model_name,
        batch_size=batch_size  # Use larger batch size for RTX A6000
    )
    writer = DocumentWriter(document_store=document_store)

    indexing.add_component(instance=doc_embedder, name="embedder")
    indexing.add_component(instance=writer, name="writer")
    indexing.connect("embedder.documents", "writer.documents")

    # Warm-up and run embedding + write
    print(f"🔥 Warming up embedder on {doc_embedder.device}...")
    doc_embedder.warm_up()
    
    print(f"⚡ Starting batch embedding with batch_size={batch_size}...")
    indexing.run({"documents": documents})

    print(f"✅ Indexed {document_store.count_documents()} documents")
    return document_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index QASPER documents with sentence embeddings")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--split", type=str, default="train", help="HF dataset split (train/validation/test)")
    parser.add_argument("--granularity", type=str, default="paragraph", choices=["paragraph", "sentence"],
                        help="Index paragraphs or sentences")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    store = build_and_index(model_name=args.model, split=args.split, granularity=args.granularity)
    print(f"Indexed {store.count_documents()} documents into InMemoryDocumentStore.")

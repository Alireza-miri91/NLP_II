from datasets import load_dataset
from typing import Iterable, List, Dict
import hashlib

try:
    # Import Document from Haystack if available in the environment
    from haystack import Document
except Exception:
    Document = None  # type: ignore


def _flatten_full_text(doc: Dict) -> List[str]:
    """
    Extract paragraphs from QASPER sample.

    The HF parquet schema has full_text as a dict with keys 'section_name' and 'paragraphs'.
    - 'section_name' is a list of strings (section titles)
    - 'paragraphs' is a list of lists, where each inner list contains strings (paragraphs) for that section

    We produce a linear list of paragraph strings. We also prepend title and abstract if present.
    """
    paragraphs: List[str] = []
    title = doc.get("title") or ""
    abstract = doc.get("abstract") or ""
    full_text = doc.get("full_text") or {}

    # Title and abstract as separate paragraphs if present
    if title:
        paragraphs.append(title)
    if abstract:
        paragraphs.append(abstract)

    # Full text paragraphs per observed schema
    if isinstance(full_text, dict):
        para_groups = full_text.get("paragraphs")
        if isinstance(para_groups, list):
            for group in para_groups:
                if isinstance(group, list):
                    for p in group:
                        if isinstance(p, str) and p.strip():
                            paragraphs.append(p.strip())
    else:
        # Fallback: older/simple schemas
        if isinstance(full_text, list):
            for p in full_text:
                if isinstance(p, str) and p.strip():
                    paragraphs.append(p.strip())

    return paragraphs


def _split_into_sentences(text: str) -> List[str]:
    """
    Very simple sentence splitter to avoid adding heavy dependencies.
    Splits on '.', '!' and '?' while keeping it robust for our use-case.
    """
    import re
    # Replace newlines with spaces and collapse multiple spaces
    normalized = re.sub(r"\s+", " ", text.strip())
    # Split on sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    # Filter empty fragments
    return [p.strip() for p in parts if p.strip()]


def load_qasper_documents(split: str = "train", revision: str = "refs/convert/parquet", 
                          granularity: str = "paragraph") -> Iterable[Dict]:
    """
    Load QASPER dataset and yield paragraph-level items as simple dicts compatible with Haystack `Document`.

    - Uses the same HF dataset call pattern as seen in ds.ipynb
    - Produces one item per paragraph with meta including paper id and paragraph index

    Returned dict format: {"content": str, "meta": {"id": str, "para_idx": int}}
    If Haystack is installed, you can convert to Document objects in your caller.
    """
    ds = load_dataset("allenai/qasper", revision=revision, split=split)
    for sample in ds:
        title = sample.get("title") or ""
        abstract = sample.get("abstract") or ""

        # Prefer provided ids; otherwise, create a stable hash from title+abstract
        paper_id = sample.get("id") or sample.get("paper_id") or ""
        if not paper_id:
            raw = f"{title}|{abstract}"
            digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
            paper_id = f"hash_{digest}"

        paragraphs = _flatten_full_text(sample)
        if granularity == "sentence":
            sent_idx = 0
            for p_idx, content in enumerate(paragraphs):
                sentences = _split_into_sentences(content)
                for s in sentences:
                    yield {
                        "content": s,
                        "meta": {
                            "paper_id": str(paper_id),
                            "para_idx": p_idx,
                            "sent_idx": sent_idx,
                            "title": title,
                        },
                    }
                    sent_idx += 1
        else:
            for idx, content in enumerate(paragraphs):
                yield {
                    "content": content,
                    "meta": {"paper_id": str(paper_id), "para_idx": idx, "title": title},
                }


def load_qasper_haystack_documents(split: str = "train", revision: str = "refs/convert/parquet", 
                                   granularity: str = "paragraph") -> Iterable["Document"]:
    """
    Convenience wrapper that returns Haystack Document objects if Haystack is available.
    Falls back to raising if Haystack is not installed.
    """
    if Document is None:
        raise RuntimeError("Haystack is not available. Please install haystack-ai to use this helper.")

    for item in load_qasper_documents(split=split, revision=revision, granularity=granularity):
        yield Document(content=item["content"], meta=item["meta"])  # type: ignore


if __name__ == "__main__":
    # Minimal smoke test: load a few documents and print counts
    count = 0
    for _ in load_qasper_documents(split="train", granularity="paragraph"):
        count += 1
        if count >= 5:
            break
    print(f"Loaded example paragraphs: {count}")
    count_s = 0
    for _ in load_qasper_documents(split="train", granularity="sentence"):
        count_s += 1
        if count_s >= 5:
            break
    print(f"Loaded example sentences: {count_s}")

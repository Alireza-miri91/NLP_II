import argparse
from typing import List, Dict, Tuple, Set, Iterable

from datasets import load_dataset

from index_qasper import build_and_index, DEFAULT_MODEL
from retrieve_qasper import build_retrieval_pipeline
from baselines import TfidfRetriever, LsiRetriever, build_dpr_retriever, SpladeRetriever
from answerability_eval import AnswerabilityEvaluator
from qasper_loader import load_qasper_documents

from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


# Notebook-style evaluation.
#
# Paragraph-level metrics (Precision@k / Recall@k / F1@k) are computed in evaluate().
# Paper-level Hit@k logic is preserved for comparison utilities via a separate helper.
#
# Supported modes:
# - embeddings: SentenceTransformersTextEmbedder + InMemoryEmbeddingRetriever
# - bm25: InMemoryBM25Retriever
# - tfidf: classical term-based cosine similarity
# - lsi: truncated SVD over TF-IDF (LSA)
# - dpr: dual-encoder retriever using transformer encoders
# - splade: learned sparse representations with neural networks
# - answerability: Answerability evaluation with precision/recall/F1 metrics


def _get_qasper_questions(split: str) -> List[Dict]:
    ds = load_dataset("allenai/qasper", revision="refs/convert/parquet", split=split)
    items: List[Dict] = []
    for row in ds:
        paper_id = row.get("id") or row.get("paper_id") or None
        title = row.get("title") or ""
        qas = row.get("qas") or {}
        questions = qas.get("question") or []
        for q_text in questions:
            if isinstance(q_text, str) and q_text.strip():
                items.append({
                    "paper_id": paper_id,
                    "title": title,
                    "question": q_text.strip(),
                })
    return items


# -------------------- Paragraph-level gold mapping helpers --------------------

def _flatten_text_evidence(evidence: Iterable) -> List[str]:
    """
    Best-effort flattening of the QASPER evidence field into a list of strings.
    The dataset's evidence objects can vary; we avoid assuming a strict schema.
    We extract any string entries or dict entries with a 'text' field.
    """
    texts: List[str] = []
    if evidence is None:
        return texts
    stack = list(evidence) if isinstance(evidence, (list, tuple)) else [evidence]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            s = item.strip()
            if s:
                texts.append(s)
        elif isinstance(item, dict):
            # Common patterns: {'text': '...'} or nested under keys
            for key in ["text", "evidence", "span", "content"]:
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    texts.append(val.strip())
                elif isinstance(val, (list, tuple)):
                    stack.extend(val)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return texts


def _build_paragraph_cache(split: str, granularity: str = "paragraph") -> Dict[str, List[Tuple[int, str]]]:
    """
    Build a cache mapping paper_id -> list of (para_idx, paragraph_text).
    Kept lightweight by storing only paragraph text and index.
    """
    cache: Dict[str, List[Tuple[int, str]]] = {}
    for item in load_qasper_documents(split=split, granularity=granularity):
        meta = item.get("meta", {})
        pid = meta.get("paper_id")
        pidx = meta.get("para_idx")
        if pid is None or not isinstance(pidx, int):
            continue
        cache.setdefault(str(pid), []).append((pidx, item.get("content", "")))
    return cache


def _derive_gold_para_indices(question_item: Dict, para_cache: Dict[str, List[Tuple[int, str]]]) -> Set[Tuple[str, int]]:
    """
    Produce the gold paragraph indices for a question by matching evidence text to
    paragraph contents. We consider any paragraph that contains an evidence span as gold.

    Returns a set of (paper_id, para_idx).
    """
    gold: Set[Tuple[str, int]] = set()
    paper_id = question_item.get("paper_id")
    if not paper_id:
        return gold

    paragraphs = para_cache.get(str(paper_id), [])

    # Gather candidate evidence strings from 'evidence' and fall back to 'extractive_spans'
    evidence_texts = _flatten_text_evidence(question_item.get("evidence"))
    if not evidence_texts:
        evidence_texts = _flatten_text_evidence(question_item.get("extractive_spans"))

    if not evidence_texts:
        return gold

    # For each evidence string, find paragraphs containing it
    for ev in evidence_texts:
        ev_lower = ev.lower()
        for para_idx, text in paragraphs:
            try:
                if ev_lower and isinstance(text, str) and ev_lower in text.lower():
                    gold.add((str(paper_id), para_idx))
            except Exception:
                # Be defensive against unexpected encodings
                continue
    return gold


def _build_bm25_pipeline(store: InMemoryDocumentStore) -> Pipeline:
    pipe = Pipeline()
    retriever = InMemoryBM25Retriever(document_store=store)
    pipe.add_component("retriever", retriever)
    return pipe


def evaluate(mode: str, model: str, split: str, k: int, limit: int, granularity: str = "paragraph") -> Tuple[Dict, int]:
    """
    Paragraph-level evaluation using QASPER evidence to derive gold paragraph sets.

    Returns (metrics_dict, num_questions):
    metrics_dict = {"precision@k": float, "recall@k": float, "f1@k": float}
    """
    # Load questions with evidence
    evaluator = AnswerabilityEvaluator()
    qa_items = evaluator.load_qasper_questions_with_answers(split, limit)
    if limit > 0:
        qa_items = qa_items[:limit]

    if not qa_items:
        return {"precision@k": 0.0, "recall@k": 0.0, "f1@k": 0.0}, 0

    # Build paragraph cache for gold mapping
    para_cache = _build_paragraph_cache(split=split, granularity="paragraph")

    # Build index once with optimized batch size
    batch_size = 64 if mode in ["embeddings", "dpr"] else 32
    store = build_and_index(model_name=model, split=split, batch_size=batch_size, granularity=granularity)

    # Configure retriever
    if mode == "embeddings":
        pipe = build_retrieval_pipeline(store, model)
        text_component = pipe.get_component("text_embedder")
        if hasattr(text_component, "warm_up"):
            text_component.warm_up()
    elif mode == "bm25":
        pipe = _build_bm25_pipeline(store)
    elif mode == "tfidf":
        pipe = None  # type: ignore[assignment]
        tfidf = TfidfRetriever(); tfidf.fit(split=split, granularity=granularity)
    elif mode == "lsi":
        pipe = None  # type: ignore[assignment]
        lsi = LsiRetriever(); lsi.fit(split=split, granularity=granularity)
    elif mode == "dpr":
        _, pipe = build_dpr_retriever(model_name=model)
    elif mode == "splade":
        pipe = None  # type: ignore[assignment]
        splade = SpladeRetriever(); splade.fit(split=split, granularity=granularity)
    else:
        raise ValueError("mode must be one of: 'embeddings', 'bm25', 'tfidf', 'lsi', 'dpr', 'splade'")

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for ex in qa_items:
        question = ex["question"]
        gold_para_set = _derive_gold_para_indices(ex, para_cache)

        # If no gold evidence found, skip this question (can't compute recall)
        if not gold_para_set:
            continue

        # Retrieve
        if mode == "embeddings":
            result = pipe.run({"text_embedder": {"text": question}, "retriever": {"top_k": k}})
            docs = result["retriever"]["documents"]
            retrieved = [(d.meta.get("paper_id"), d.meta.get("para_idx")) for d in docs]
        elif mode == "bm25":
            result = pipe.run({"retriever": {"query": question, "top_k": k}})
            docs = result["retriever"]["documents"]
            retrieved = [(d.meta.get("paper_id"), d.meta.get("para_idx")) for d in docs]
        elif mode == "tfidf":
            hits = tfidf.retrieve(question, top_k=k)
            retrieved = [(h["meta"].get("paper_id"), h["meta"].get("para_idx")) for h in hits]
        elif mode == "lsi":
            hits = lsi.retrieve(question, top_k=k)
            retrieved = [(h["meta"].get("paper_id"), h["meta"].get("para_idx")) for h in hits]
        elif mode == "dpr":
            result = pipe.run({"text_embedder": {"text": question}, "retriever": {"top_k": k}})
            docs = result["retriever"]["documents"]
            retrieved = [(d.meta.get("paper_id"), d.meta.get("para_idx")) for d in docs]
        elif mode == "splade":
            hits = splade.retrieve(question, top_k=k)
            retrieved = [(h["meta"].get("paper_id"), h["meta"].get("para_idx")) for h in hits]
        else:
            retrieved = []

        # Normalize and take top-k
        retrieved_pairs = []
        for pid, pidx in retrieved[:k]:
            if pid is None or not isinstance(pidx, int):
                continue
            retrieved_pairs.append((str(pid), int(pidx)))

        # Compute metrics
        retrieved_set = set(retrieved_pairs)
        tp = len(retrieved_set.intersection(gold_para_set))
        precision = tp / k if k > 0 else 0.0
        recall = tp / len(gold_para_set) if gold_para_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    n_eval = len(f1s)
    metrics = {
        "precision@k": float(sum(precisions) / n_eval) if n_eval else 0.0,
        "recall@k": float(sum(recalls) / n_eval) if n_eval else 0.0,
        "f1@k": float(sum(f1s) / n_eval) if n_eval else 0.0,
        "evaluated_questions": n_eval,
    }
    return metrics, len(qa_items)


# -------------------- Paper-level Hit@k (for comparisons) --------------------

def _evaluate_paper_hit_at_k(mode: str, model: str, split: str, k: int, limit: int, granularity: str = "paragraph") -> Tuple[float, int]:
    batch_size = 64 if mode in ["embeddings", "dpr"] else 32
    store = build_and_index(model_name=model, split=split, batch_size=batch_size, granularity=granularity)

    if mode == "embeddings":
        pipe = build_retrieval_pipeline(store, model)
        text_component = pipe.get_component("text_embedder")
        if hasattr(text_component, "warm_up"):
            text_component.warm_up()
    elif mode == "bm25":
        pipe = _build_bm25_pipeline(store)
    elif mode == "tfidf":
        pipe = None  # type: ignore[assignment]
        tfidf = TfidfRetriever(); tfidf.fit(split=split, granularity=granularity)
    elif mode == "lsi":
        pipe = None  # type: ignore[assignment]
        lsi = LsiRetriever(); lsi.fit(split=split, granularity=granularity)
    elif mode == "dpr":
        _, pipe = build_dpr_retriever(model_name=model)
    elif mode == "splade":
        pipe = None  # type: ignore[assignment]
        splade = SpladeRetriever(); splade.fit(split=split, granularity=granularity)
    else:
        raise ValueError("mode must be one of: 'embeddings', 'bm25', 'tfidf', 'lsi', 'dpr', 'splade'")

    qa_items = _get_qasper_questions(split)
    if limit > 0:
        qa_items = qa_items[:limit]

    paper_hits = 0
    for ex in qa_items:
        question = ex["question"]
        gold_paper = ex["paper_id"]
        if mode == "embeddings":
            result = pipe.run({"text_embedder": {"text": question}, "retriever": {"top_k": k}})
            docs = result["retriever"]["documents"]
            retrieved_papers = [d.meta.get("paper_id") for d in docs]
        elif mode == "bm25":
            result = pipe.run({"retriever": {"query": question, "top_k": k}})
            docs = result["retriever"]["documents"]
            retrieved_papers = [d.meta.get("paper_id") for d in docs]
        elif mode == "tfidf":
            hits = tfidf.retrieve(question, top_k=k)
            retrieved_papers = [h["meta"].get("paper_id") for h in hits]
        elif mode == "lsi":
            hits = lsi.retrieve(question, top_k=k)
            retrieved_papers = [h["meta"].get("paper_id") for h in hits]
        elif mode == "dpr":
            result = pipe.run({"text_embedder": {"text": question}, "retriever": {"top_k": k}})
            docs = result["retriever"]["documents"]
            retrieved_papers = [d.meta.get("paper_id") for d in docs]
        elif mode == "splade":
            hits = splade.retrieve(question, top_k=k)
            retrieved_papers = [h["meta"].get("paper_id") for h in hits]
        else:
            retrieved_papers = []

        if gold_paper and gold_paper in retrieved_papers:
            paper_hits += 1

    n = len(qa_items)
    paper_hit_at_k = paper_hits / n if n else 0.0
    return paper_hit_at_k, n


def evaluate_answerability(mode: str, model: str, split: str, k: int, limit: int, granularity: str = "paragraph",
                           min_confidence: float = 0.3, min_answer_words: int = 2,
                           overlap_max_ratio: float = 0.7, short_context_tokens: int = 50,
                           short_context_min_conf: float = 0.7) -> Tuple[Dict, int]:
    """
    Evaluate answerability prediction performance using retrieved context.
    
    Args:
        mode: Retrieval mode ('embeddings', 'bm25', 'tfidf', 'lsi', 'dpr', 'splade')
        model: Model name for embeddings/DPR
        split: Dataset split ('train', 'validation', 'test')
        k: Number of top documents to retrieve
        limit: Maximum number of questions to evaluate (0 = all)
        
    Returns:
        Tuple of (metrics_dict, num_questions_evaluated)
    """
    # Initialize answerability evaluator
    evaluator = AnswerabilityEvaluator(
        min_confidence=min_confidence,
        min_answer_words=min_answer_words,
        overlap_max_ratio=overlap_max_ratio,
        short_context_tokens=short_context_tokens,
        short_context_min_conf=short_context_min_conf,
    )
    
    # Load questions with answerability labels
    questions_data = evaluator.load_qasper_questions_with_answers(split, limit)
    if not questions_data:
        return {"error": "No questions loaded"}, 0
    
    # Build index and retrieval pipeline (reuse existing code)
    batch_size = 64 if mode in ["embeddings", "dpr"] else 32
    store = build_and_index(model_name=model, split=split, batch_size=batch_size, granularity=granularity)
    
    # Set up retrieval pipeline based on mode
    if mode == "embeddings":
        pipe = build_retrieval_pipeline(store, model)
        text_component = pipe.get_component("text_embedder")
        if hasattr(text_component, "warm_up"):
            text_component.warm_up()
    elif mode == "bm25":
        pipe = _build_bm25_pipeline(store)
    elif mode == "tfidf":
        pipe = None
        tfidf = TfidfRetriever()
        tfidf.fit(split=split, granularity=granularity)
    elif mode == "lsi":
        pipe = None
        lsi = LsiRetriever()
        lsi.fit(split=split, granularity=granularity)
    elif mode == "dpr":
        _, pipe = build_dpr_retriever(model_name=model)
    elif mode == "splade":
        pipe = None
        splade = SpladeRetriever()
        splade.fit(split=split, granularity=granularity)
    else:
        raise ValueError(f"Unsupported mode for answerability evaluation: {mode}")
    
    # Retrieve context for each question
    contexts = []
    for question_data in questions_data:
        question = question_data["question"]
        
        try:
            if mode == "embeddings":
                result = pipe.run({"text_embedder": {"text": question}, "retriever": {"top_k": k}})
                context = " ".join([doc.content for doc in result["retriever"]["documents"]])
            elif mode == "bm25":
                result = pipe.run({"retriever": {"query": question, "top_k": k}})
                context = " ".join([doc.content for doc in result["retriever"]["documents"]])
            elif mode == "tfidf":
                hits = tfidf.retrieve(question, top_k=k)
                context = " ".join([hit["content"] for hit in hits])
            elif mode == "lsi":
                hits = lsi.retrieve(question, top_k=k)
                context = " ".join([hit["content"] for hit in hits])
            elif mode == "dpr":
                result = pipe.run({"text_embedder": {"text": question}, "retriever": {"top_k": k}})
                context = " ".join([doc.content for doc in result["retriever"]["documents"]])
            elif mode == "splade":
                hits = splade.retrieve(question, top_k=k)
                context = " ".join([hit["content"] for hit in hits])
            else:
                context = ""
        except Exception as e:
            print(f"Warning: Error retrieving context for question: {e}")
            context = ""
        
        contexts.append(context)
    
    # Evaluate answerability
    results = evaluator.evaluate_answerability_batch(questions_data, contexts)
    metrics = evaluator.calculate_answerability_metrics(results)
    
    return metrics, len(questions_data)


def compare_lexical_vs_semantic(model: str, split: str, k: int, limit: int, granularity: str = "paragraph") -> Dict:
    """
    Run retrieval with lexical methods (bm25, tfidf, splade) vs semantic methods
    (embeddings, lsi, dpr) and report per-question win counts.
    A method "wins" if it retrieves the gold paper within top-k while the other side does not.
    Ties and double-wins are tracked.
    """
    # Build index for methods that use Haystack store
    batch_size = 64
    store = build_and_index(model_name=model, split=split, batch_size=batch_size, granularity=granularity)

    # Initialize retrievers
    # Lexical
    bm25_pipe = _build_bm25_pipeline(store)
    tfidf = TfidfRetriever(); tfidf.fit(split=split, granularity=granularity)
    splade = SpladeRetriever(); splade.fit(split=split, granularity=granularity)
    # Semantic
    emb_pipe = build_retrieval_pipeline(store, model)
    text_component = emb_pipe.get_component("text_embedder")
    if hasattr(text_component, "warm_up"):
        text_component.warm_up()
    lsi = LsiRetriever(); lsi.fit(split=split, granularity=granularity)
    _, dpr_pipe = build_dpr_retriever(model_name=model)

    qa_items = _get_qasper_questions(split)
    if limit > 0:
        qa_items = qa_items[:limit]

    stats = {
        "lexical_win": 0,
        "semantic_win": 0,
        "both_win": 0,
        "both_lose": 0,
        "total": len(qa_items),
    }

    for ex in qa_items:
        q = ex["question"]
        gold = ex["paper_id"]

        # Lexical retrieved gold?
        bm25_docs = bm25_pipe.run({"retriever": {"query": q, "top_k": k}})["retriever"]["documents"]
        bm25_hit = gold in [d.meta.get("paper_id") for d in bm25_docs]
        tfidf_docs = tfidf.retrieve(q, top_k=k)
        tfidf_hit = gold in [d["meta"].get("paper_id") for d in tfidf_docs]
        splade_docs = splade.retrieve(q, top_k=k)
        splade_hit = gold in [d["meta"].get("paper_id") for d in splade_docs]
        lexical_hit = bm25_hit or tfidf_hit or splade_hit

        # Semantic retrieved gold?
        emb_docs = emb_pipe.run({"text_embedder": {"text": q}, "retriever": {"top_k": k}})["retriever"]["documents"]
        emb_hit = gold in [d.meta.get("paper_id") for d in emb_docs]
        lsi_docs = lsi.retrieve(q, top_k=k)
        lsi_hit = gold in [d["meta"].get("paper_id") for d in lsi_docs]
        dpr_docs = dpr_pipe.run({"text_embedder": {"text": q}, "retriever": {"top_k": k}})["retriever"]["documents"]
        dpr_hit = gold in [d.meta.get("paper_id") for d in dpr_docs]
        semantic_hit = emb_hit or lsi_hit or dpr_hit

        if lexical_hit and not semantic_hit:
            stats["lexical_win"] += 1
        elif semantic_hit and not lexical_hit:
            stats["semantic_win"] += 1
        elif lexical_hit and semantic_hit:
            stats["both_win"] += 1
        else:
            stats["both_lose"] += 1

    return stats


def compare_granularity(mode: str, model: str, split: str, k: int, limit: int) -> Dict:
    """
    Compare Hit@k at paper-level between paragraph and sentence granularity for a given mode.
    """
    hit_para, n = _evaluate_paper_hit_at_k(mode, model, split, k, limit, granularity="paragraph")
    hit_sent, _ = _evaluate_paper_hit_at_k(mode, model, split, k, limit, granularity="sentence")
    return {"mode": mode, "n": n, "paragraph_hit": hit_para, "sentence_hit": hit_sent}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate retrieval on QASPER (Hit@k or Answerability)")
    p.add_argument("--mode", type=str, default="embeddings", 
                   choices=["embeddings", "bm25", "tfidf", "lsi", "dpr", "splade", "answerability"]) 
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--limit", type=int, default=200, help="evaluate on first N questions (0 = all)")
    p.add_argument("--granularity", type=str, default="paragraph", choices=["paragraph", "sentence"],
                   help="Index and retrieve at paragraph or sentence level")
    # Answerability config
    p.add_argument("--ans-min-confidence", type=float, default=0.3)
    p.add_argument("--ans-min-words", type=int, default=2)
    p.add_argument("--ans-overlap-max", type=float, default=0.7)
    p.add_argument("--ans-short-context", type=int, default=50)
    p.add_argument("--ans-short-min-conf", type=float, default=0.7)
    p.add_argument("--compare", type=str, default="none", choices=["none", "granularity", "lex_sem", "both"],
                   help="Run comparison analyses instead of a single evaluation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.compare != "none":
        # Collect results to print a consolidated summary at the end
        granularity_results = []
        lex_sem_stats = None

        if args.compare in ("granularity", "both"):
            for m in ["embeddings", "bm25", "tfidf", "lsi", "dpr", "splade"]:
                try:
                    res = compare_granularity(m, args.model, args.split, args.k, args.limit)
                    granularity_results.append(res)
                    print(f"[granularity] mode={res['mode']} n={res['n']} paragraph={res['paragraph_hit']:.3f} sentence={res['sentence_hit']:.3f}")
                except Exception as e:
                    print(f"[granularity] Skipped mode={m} due to error: {e}")

        if args.compare in ("lex_sem", "both"):
            try:
                lex_sem_stats = compare_lexical_vs_semantic(args.model, args.split, args.k, args.limit, args.granularity)
                print(f"[lex_vs_sem] total={lex_sem_stats['total']} lexical_win={lex_sem_stats['lexical_win']} semantic_win={lex_sem_stats['semantic_win']} both_win={lex_sem_stats['both_win']} both_lose={lex_sem_stats['both_lose']}")
            except Exception as e:
                print(f"[lex_vs_sem] Error: {e}")

        # Final consolidated summary footer
        print("\n" + "=" * 80)
        print("EVAL SUMMARY (footer)")
        print("=" * 80)
        print(f"model={args.model} | split={args.split} | k={args.k} | limit={args.limit} | granularity={args.granularity} | compare={args.compare}")

        if granularity_results:
            print("\n[granularity results]")
            for res in granularity_results:
                print(f"  mode={res['mode']:<12} n={res['n']:<6} paragraph_hit={res['paragraph_hit']:.3f} sentence_hit={res['sentence_hit']:.3f}")

        if lex_sem_stats is not None:
            print("\n[lexical vs semantic]")
            print(f"  total={lex_sem_stats['total']} lexical_win={lex_sem_stats['lexical_win']} semantic_win={lex_sem_stats['semantic_win']} both_win={lex_sem_stats['both_win']} both_lose={lex_sem_stats['both_lose']}")
        print("=" * 80)
    elif args.mode == "answerability":
        # Run answerability evaluation
        metrics, n = evaluate_answerability(
            args.mode, args.model, args.split, args.k, args.limit, args.granularity,
            args.ans_min_confidence, args.ans_min_words, args.ans_overlap_max,
            args.ans_short_context, args.ans_short_min_conf,
        )
        
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
        else:
            print(f"\nMode={args.mode} | Evaluated {n} questions | Answerability Evaluation Results:")
            print(f"Accuracy: {metrics['overall']['accuracy']:.4f}")
            print(f"Macro F1: {metrics['overall']['macro_f1']:.4f}")
            print(f"Weighted F1: {metrics['overall']['weighted_f1']:.4f}")
            print(f"Answerable F1: {metrics['answerable']['f1']:.4f}")
            print(f"Unanswerable F1: {metrics['unanswerable']['f1']:.4f}")
            
            # Print detailed report
            evaluator = AnswerabilityEvaluator()
            evaluator.print_detailed_report(metrics)
    else:
        # Run paragraph-level Precision/Recall/F1@k evaluation
        metrics, n = evaluate(args.mode, args.model, args.split, args.k, args.limit, args.granularity)
        print(f"Mode={args.mode} | Evaluated {metrics['evaluated_questions']}/{n} questions | "+
              f"Precision@{args.k}: {metrics['precision@k']:.3f} | "+
              f"Recall@{args.k}: {metrics['recall@k']:.3f} | "+
              f"F1@{args.k}: {metrics['f1@k']:.3f}")

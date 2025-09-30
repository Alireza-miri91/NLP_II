import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any

import torch

from eval_qasper import (
    evaluate as evaluate_paragraph_metrics,
    compare_granularity,
    compare_lexical_vs_semantic,
    evaluate_answerability,
    DEFAULT_MODEL,
)


def _detect_reasonable_limit(default: int = 500) -> int:
    """
    Pick a question limit based on available GPU/CPU resources.
    Conservative to avoid OOM while still being large.
    """
    try:
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            gb = total / (1024 ** 3)
            # Heuristic tiers for question count
            if gb >= 40:
                return 1500
            if gb >= 24:
                return 1000
            if gb >= 16:
                return 700
            return 400
        else:
            # CPU-only: keep moderate
            return 300
    except Exception:
        return default


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_comparison(
    split: str,
    k: int,
    limit: int,
    model: str,
    modes: List[str],
    granularities: List[str],
    include_answerability: bool,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "config": {
            "split": split,
            "k": k,
            "limit": limit,
            "model": model,
            "modes": modes,
            "granularities": granularities,
            "include_answerability": include_answerability,
        },
        "paragraph_metrics": {},  # metrics per mode x granularity
        "granularity_compare": {},  # paper hit compare per mode
        "lex_vs_sem": None,
        "answerability": {},  # answerability metrics per mode x granularity
        "started_at": time.time(),
    }

    # Paragraph-level Precision/Recall/F1@k per mode and granularity
    for mode in modes:
        summary["paragraph_metrics"].setdefault(mode, {})
        for gran in granularities:
            try:
                metrics, n = evaluate_paragraph_metrics(
                    mode=mode, model=model, split=split, k=k, limit=limit, granularity=gran
                )
                summary["paragraph_metrics"][mode][gran] = {
                    "metrics": metrics,
                    "n_questions": n,
                }
                print(f"[prf@k] mode={mode} gran={gran} eval={metrics['evaluated_questions']}/{n} "
                      f"P@{k}={metrics['precision@k']:.3f} R@{k}={metrics['recall@k']:.3f} F1@{k}={metrics['f1@k']:.3f}")
            except Exception as e:
                summary["paragraph_metrics"][mode][gran] = {"error": str(e)}
                print(f"[prf@k] Skipped mode={mode} gran={gran} due to error: {e}")

    # Paper-level granularity comparison per mode
    for mode in modes:
        try:
            res = compare_granularity(mode, model, split, k, limit)
            summary["granularity_compare"][mode] = res
            print(f"[granularity] mode={mode} n={res['n']} paragraph={res['paragraph_hit']:.3f} sentence={res['sentence_hit']:.3f}")
        except Exception as e:
            summary["granularity_compare"][mode] = {"error": str(e)}
            print(f"[granularity] Skipped mode={mode} due to error: {e}")

    # Lexical vs semantic (paper-level)
    try:
        summary["lex_vs_sem"] = compare_lexical_vs_semantic(model, split, k, limit, granularities[0])
        print(f"[lex_vs_sem] total={summary['lex_vs_sem']['total']} "
              f"lexical_win={summary['lex_vs_sem']['lexical_win']} "
              f"semantic_win={summary['lex_vs_sem']['semantic_win']} "
              f"both_win={summary['lex_vs_sem']['both_win']} "
              f"both_lose={summary['lex_vs_sem']['both_lose']}")
    except Exception as e:
        summary["lex_vs_sem"] = {"error": str(e)}
        print(f"[lex_vs_sem] Error: {e}")

    # Answerability evaluation per mode and granularity
    if include_answerability:
        for mode in modes:
            summary["answerability"].setdefault(mode, {})
            for gran in granularities:
                try:
                    metrics, n = evaluate_answerability(
                        mode=mode, model=model, split=split, k=k, limit=limit, granularity=gran
                    )
                    summary["answerability"][mode][gran] = {
                        "metrics": metrics,
                        "n_questions": n,
                    }
                    print(f"[answerability] mode={mode} gran={gran} n={n} accuracy={metrics['overall']['accuracy']:.3f} macro_f1={metrics['overall']['macro_f1']:.3f}")
                except Exception as e:
                    summary["answerability"][mode][gran] = {"error": str(e)}
                    print(f"[answerability] Skipped mode={mode} gran={gran} due to error: {e}")

    summary["finished_at"] = time.time()
    summary["duration_sec"] = summary["finished_at"] - summary["started_at"]
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comprehensive QASPER retrieval comparison runner")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--limit", type=int, default=-1, help="Questions to evaluate (-1 to auto, 0 for all)")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument(
        "--modes", type=str,
        default="embeddings,bm25,tfidf,lsi,dpr,splade",
        help="Comma-separated retrieval modes"
    )
    p.add_argument(
        "--granularities", type=str, default="paragraph,sentence",
        help="Comma-separated granularities to test"
    )
    p.add_argument("--include-answerability", action="store_true")
    p.add_argument("--out", type=str, default=f"compare_results_{_timestamp()}.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    granularities = [g.strip() for g in args.granularities.split(",") if g.strip()]

    # Decide limit
    if args.limit == -1:
        limit = _detect_reasonable_limit()
    else:
        limit = args.limit

    print(f"Running comparison: split={args.split} k={args.k} limit={limit} model={args.model}")
    print(f"Modes: {modes}")
    print(f"Granularities: {granularities}")

    results = run_comparison(
        split=args.split,
        k=args.k,
        limit=limit,
        model=args.model,
        modes=modes,
        granularities=granularities,
        include_answerability=args.include_answerability,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {os.path.abspath(args.out)}")



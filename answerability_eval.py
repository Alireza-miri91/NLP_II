#!/usr/bin/env python3
"""
Answerability evaluation module for QASPER dataset.

This module provides functionality to:
1. Generate answers for questions given retrieved context
2. Determine if a question is answerable or unanswerable
3. Evaluate answerability prediction performance with precision/recall/F1 metrics

Following the project's existing patterns and optimization strategies.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerabilityEvaluator:
    """
    Main class for answerability evaluation following the project's patterns.
    Optimized for RTX A6000 with batch processing and GPU utilization.
    """
    
    def __init__(self, qa_model_name: str = "distilbert-base-cased-distilled-squad", 
                 device: str = "auto",
                 min_confidence: float = 0.3,
                 min_answer_words: int = 2,
                 overlap_max_ratio: float = 0.7,
                 short_context_tokens: int = 50,
                 short_context_min_conf: float = 0.7,
                 no_answer_patterns: Optional[List[str]] = None):
        """
        Initialize the answerability evaluator.
        
        Args:
            qa_model_name: Name of the QA model to use for answer generation
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.qa_model_name = qa_model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        # Configurable thresholds
        self.min_confidence = min_confidence
        self.min_answer_words = min_answer_words
        self.overlap_max_ratio = overlap_max_ratio
        self.short_context_tokens = short_context_tokens
        self.short_context_min_conf = short_context_min_conf
        
        # Initialize QA pipeline
        logger.info(f"Loading QA model: {qa_model_name} on {self.device}")
        self.qa_pipeline = pipeline(
            "question-answering",
            model=qa_model_name,
            device=0 if self.device == "cuda" else -1
        )
        
        # Answerability patterns for rule-based classification
        self.no_answer_patterns = no_answer_patterns or [
            "cannot be determined", "not mentioned", "not specified",
            "not provided", "unclear", "unknown", "not found",
            "not available", "not stated", "not given"
        ]
        
        logger.info("AnswerabilityEvaluator initialized successfully")
    
    def load_qasper_questions_with_answers(self, split: str, limit: int = 0) -> List[Dict]:
        """
        Load QASPER questions with their answerability labels and answers.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            limit: Maximum number of questions to load (0 = all)
            
        Returns:
            List of question dictionaries with answerability information
        """
        logger.info(f"Loading QASPER questions from {split} split...")
        ds = load_dataset("allenai/qasper", revision="refs/convert/parquet", split=split)
        items = []
        
        for row in ds:
            paper_id = row.get("id") or row.get("paper_id")
            title = row.get("title", "")
            qas = row.get("qas", {})
            questions = qas.get("question", [])
            answers = qas.get("answers", [])

            for i, question in enumerate(questions):
                if i >= len(answers) or not isinstance(question, str) or not question.strip():
                    continue

                # Each entry in answers[i] is a dict with key 'answer' (list of annotations)
                entry = answers[i]
                annotations = entry.get("answer", []) if isinstance(entry, dict) else []

                # Aggregate fields across annotations
                evidence_texts = []
                extractive_spans = []
                free_form_candidates = []
                yes_no_candidates = []
                unanswerable_flags = []

                if isinstance(annotations, list):
                    for ann in annotations:
                        if not isinstance(ann, dict):
                            continue
                        # Evidence: list of strings
                        ev = ann.get("evidence")
                        if isinstance(ev, list):
                            for e in ev:
                                if isinstance(e, str) and e.strip():
                                    evidence_texts.append(e.strip())
                        # Extractive spans: list of strings
                        spans = ann.get("extractive_spans")
                        if isinstance(spans, list):
                            for s in spans:
                                if isinstance(s, str) and s.strip():
                                    extractive_spans.append(s.strip())
                        # Free-form answer
                        ffa = ann.get("free_form_answer")
                        if isinstance(ffa, str) and ffa.strip():
                            free_form_candidates.append(ffa.strip())
                        # Yes/No
                        yn = ann.get("yes_no")
                        if yn is not None:
                            yes_no_candidates.append(yn)
                        # Unanswerable flag
                        ua = ann.get("unanswerable")
                        if isinstance(ua, bool):
                            unanswerable_flags.append(ua)

                # Decide final fields
                # Unanswerable: majority vote if available, else False
                if unanswerable_flags:
                    is_unanswerable = sum(1 for x in unanswerable_flags if x) > (len(unanswerable_flags) / 2)
                else:
                    is_unanswerable = False

                gold_answer = free_form_candidates[0] if free_form_candidates else ""
                yes_no_answer = yes_no_candidates[0] if yes_no_candidates else None

                items.append({
                    "paper_id": paper_id,
                    "title": title,
                    "question": question.strip(),
                    "is_unanswerable": is_unanswerable,
                    "is_answerable": not is_unanswerable,
                    "gold_answer": gold_answer,
                    "extractive_spans": extractive_spans,
                    "yes_no_answer": yes_no_answer,
                    "evidence": evidence_texts,
                })
        
        if limit > 0:
            items = items[:limit]
        
        logger.info(f"Loaded {len(items)} questions with answerability labels")
        return items
    
    def generate_answer_with_context(self, question: str, context: str) -> Dict:
        """
        Generate answer using retrieved context and determine answerability.
        
        Args:
            question: The question to answer
            context: Retrieved paragraphs from retrieval system
            
        Returns:
            Dictionary with answer, answerability prediction, and confidence
        """
        try:
            # Generate answer using QA pipeline
            result = self.qa_pipeline(question=question, context=context)
            
            answer_text = result['answer']
            confidence = result['score']
            
            # Determine answerability using multiple heuristics
            is_answerable = self._determine_answerability(
                question, answer_text, confidence, context
            )
            
            return {
                "answer": answer_text,
                "is_answerable": is_answerable,
                "is_unanswerable": not is_answerable,
                "confidence": confidence,
                "context_length": len(context.split())
            }
            
        except Exception as e:
            logger.warning(f"Error generating answer for question: {e}")
            return {
                "answer": "",
                "is_answerable": False,
                "is_unanswerable": True,
                "confidence": 0.0,
                "context_length": len(context.split())
            }
    
    def _determine_answerability(self, question: str, answer: str, confidence: float, 
                                context: str) -> bool:
        """
        Determine answerability using multiple heuristics.
        
        Args:
            question: Original question
            answer: Generated answer
            confidence: Model confidence score
            context: Retrieved context
            
        Returns:
            True if question is answerable, False otherwise
        """
        # Heuristic 1: Low confidence threshold
        if confidence < self.min_confidence:
            return False
        
        # Heuristic 2: Check for "no answer" patterns
        answer_lower = answer.lower()
        if any(pattern in answer_lower for pattern in self.no_answer_patterns):
            return False
        
        # Heuristic 3: Very short answers might be unanswerable
        if len(answer.split()) < self.min_answer_words:
            return False
        
        # Heuristic 4: Check if answer repeats the question (common failure mode)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        if len(question_words) > 3:  # Only check for longer questions
            overlap_ratio = len(question_words.intersection(answer_words)) / len(question_words)
            if overlap_ratio > self.overlap_max_ratio:
                return False
        
        # Heuristic 5: Check if answer is too generic
        generic_answers = ["yes", "no", "maybe", "possibly", "perhaps", "it depends"]
        if answer_lower.strip() in generic_answers and len(answer.split()) == 1:
            return False
        
        # Heuristic 6: Check if context is too short (might indicate poor retrieval)
        if len(context.split()) < self.short_context_tokens:
            return confidence > self.short_context_min_conf
        
        return True
    
    def evaluate_answerability_batch(self, questions_data: List[Dict], 
                                   contexts: List[str]) -> List[Dict]:
        """
        Evaluate answerability for a batch of questions.
        Optimized for RTX A6000 with batch processing.
        
        Args:
            questions_data: List of question dictionaries
            contexts: List of retrieved contexts for each question
            
        Returns:
            List of evaluation results
        """
        logger.info(f"Evaluating answerability for {len(questions_data)} questions...")
        results = []
        
        # Process in batches for better GPU utilization
        batch_size = 32 if self.device == "cuda" else 16
        
        for i in range(0, len(questions_data), batch_size):
            batch_questions = questions_data[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]
            
            batch_results = []
            for j, (question_data, context) in enumerate(zip(batch_questions, batch_contexts)):
                question = question_data["question"]
                gold_unanswerable = question_data["is_unanswerable"]
                
                # Generate answer and predict answerability
                answer_result = self.generate_answer_with_context(question, context)
                predicted_unanswerable = answer_result["is_unanswerable"]
                
                result = {
                    "question": question,
                    "paper_id": question_data["paper_id"],
                    "gold_unanswerable": gold_unanswerable,
                    "predicted_unanswerable": predicted_unanswerable,
                    "gold_answerable": not gold_unanswerable,
                    "predicted_answerable": not predicted_unanswerable,
                    "answer": answer_result["answer"],
                    "confidence": answer_result["confidence"],
                    "context_length": answer_result["context_length"],
                    "gold_answer": question_data.get("gold_answer", ""),
                    "extractive_spans": question_data.get("extractive_spans", [])
                }
                
                batch_results.append(result)
            
            results.extend(batch_results)
            
            if (i + batch_size) % 100 == 0:
                logger.info(f"Processed {min(i + batch_size, len(questions_data))} questions...")
        
        logger.info(f"Completed answerability evaluation for {len(results)} questions")
        return results
    
    def calculate_answerability_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate comprehensive answerability evaluation metrics.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with detailed metrics
        """
        if not results:
            return {"error": "No results to evaluate"}
        
        # Extract labels
        gold_unanswerable = [r["gold_unanswerable"] for r in results]
        pred_unanswerable = [r["predicted_unanswerable"] for r in results]
        
        # Calculate metrics for both classes
        precision, recall, f1, support = precision_recall_fscore_support(
            gold_unanswerable, pred_unanswerable, average=None, labels=[False, True]
        )
        
        # Macro F1 (average of both classes)
        macro_f1 = (f1[0] + f1[1]) / 2
        
        # Weighted F1 (weighted by support)
        weighted_f1 = np.average(f1, weights=support)
        
        # Confusion matrix
        cm = confusion_matrix(gold_unanswerable, pred_unanswerable, labels=[False, True])
        
        # Additional statistics
        total_questions = len(results)
        answerable_questions = sum(1 for r in results if not r["gold_unanswerable"])
        unanswerable_questions = sum(1 for r in results if r["gold_unanswerable"])
        
        # Accuracy
        accuracy = sum(1 for r in results if r["gold_unanswerable"] == r["predicted_unanswerable"]) / total_questions
        
        # Confidence statistics
        confidences = [r["confidence"] for r in results]
        avg_confidence = np.mean(confidences)
        
        metrics = {
            "overall": {
                "total_questions": total_questions,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "average_confidence": avg_confidence
            },
            "answerable": {
                "count": answerable_questions,
                "precision": precision[0],
                "recall": recall[0],
                "f1": f1[0],
                "support": int(support[0])
            },
            "unanswerable": {
                "count": unanswerable_questions,
                "precision": precision[1],
                "recall": recall[1],
                "f1": f1[1],
                "support": int(support[1])
            },
            "confusion_matrix": {
                "true_negative": int(cm[0, 0]),  # Predicted answerable, actually answerable
                "false_positive": int(cm[0, 1]),  # Predicted unanswerable, actually answerable
                "false_negative": int(cm[1, 0]),  # Predicted answerable, actually unanswerable
                "true_positive": int(cm[1, 1])   # Predicted unanswerable, actually unanswerable
            }
        }
        
        return metrics
    
    def print_detailed_report(self, metrics: Dict):
        """
        Print a detailed evaluation report.
        
        Args:
            metrics: Metrics dictionary from calculate_answerability_metrics
        """
        print("\n" + "="*80)
        print("ANSWERABILITY EVALUATION REPORT")
        print("="*80)
        
        # Overall metrics
        overall = metrics["overall"]
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Questions: {overall['total_questions']}")
        print(f"  Accuracy: {overall['accuracy']:.4f}")
        print(f"  Macro F1: {overall['macro_f1']:.4f}")
        print(f"  Weighted F1: {overall['weighted_f1']:.4f}")
        print(f"  Average Confidence: {overall['average_confidence']:.4f}")
        
        # Answerable questions
        answerable = metrics["answerable"]
        print(f"\nANSWERABLE QUESTIONS ({answerable['count']} total):")
        print(f"  Precision: {answerable['precision']:.4f}")
        print(f"  Recall: {answerable['recall']:.4f}")
        print(f"  F1-Score: {answerable['f1']:.4f}")
        
        # Unanswerable questions
        unanswerable = metrics["unanswerable"]
        print(f"\nUNANSWERABLE QUESTIONS ({unanswerable['count']} total):")
        print(f"  Precision: {unanswerable['precision']:.4f}")
        print(f"  Recall: {unanswerable['recall']:.4f}")
        print(f"  F1-Score: {unanswerable['f1']:.4f}")
        
        # Confusion matrix
        cm = metrics["confusion_matrix"]
        print(f"\nCONFUSION MATRIX:")
        print(f"  True Negatives (Answerable → Answerable): {cm['true_negative']}")
        print(f"  False Positives (Answerable → Unanswerable): {cm['false_positive']}")
        print(f"  False Negatives (Unanswerable → Answerable): {cm['false_negative']}")
        print(f"  True Positives (Unanswerable → Unanswerable): {cm['true_positive']}")
        
        print("="*80)


def main():
    """Test the answerability evaluator with a small sample."""
    evaluator = AnswerabilityEvaluator()
    
    # Load a small sample
    questions = evaluator.load_qasper_questions_with_answers("train", limit=10)
    
    # Create dummy contexts (in real usage, these would come from retrieval)
    contexts = ["This is a sample context for testing answerability evaluation."] * len(questions)
    
    # Evaluate
    results = evaluator.evaluate_answerability_batch(questions, contexts)
    metrics = evaluator.calculate_answerability_metrics(results)
    
    # Print report
    evaluator.print_detailed_report(metrics)


if __name__ == "__main__":
    main()

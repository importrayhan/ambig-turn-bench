"""Evaluation metrics for conversational Q&A."""
import re
import string
from collections import Counter
from typing import List, Dict, Any


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score.
    
    Args:
        prediction: Predicted answer
        ground_truth: Gold answer
        
    Returns:
        F1 score (0-100)
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return 100.0 * f1


def compute_em(prediction: str, ground_truth: str) -> float:
    """Compute exact match score.
    
    Args:
        prediction: Predicted answer
        ground_truth: Gold answer
        
    Returns:
        EM score (0 or 100)
    """
    return 100.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_intent_metrics(predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute intent detection metrics.
    
    Args:
        predictions: List of prediction dicts with 'ambiguous_utterance', 'total_candidates'
        references: List of reference dicts
        
    Returns:
        Dict with metrics:
            - ambiguity_precision
            - ambiguity_recall
            - ambiguity_f1
            - avg_candidates
    """
    true_pos = 0
    false_pos = 0
    false_neg = 0
    total_candidates = []
    
    for pred, ref in zip(predictions, references):
        pred_ambig = pred.get("ambiguous_utterance", False)
        ref_ambig = ref.get("ambiguous_utterance", False)
        
        if pred_ambig and ref_ambig:
            true_pos += 1
        elif pred_ambig and not ref_ambig:
            false_pos += 1
        elif not pred_ambig and ref_ambig:
            false_neg += 1
        
        total_candidates.append(pred.get("total_candidates", 1))
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "ambiguity_precision": 100.0 * precision,
        "ambiguity_recall": 100.0 * recall,
        "ambiguity_f1": 100.0 * f1,
        "avg_candidates": sum(total_candidates) / len(total_candidates) if total_candidates else 0.0,
    }


def compute_heq1(predictions: List[str], ground_truths: List[List[str]]) -> float:
    """Compute HEQ-Q (Human Equivalence Question) metric.
    
    Measures whether prediction matches ANY of the reference answers.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of lists of acceptable answers
        
    Returns:
        HEQ-Q score (0-100)
    """
    correct = 0
    for pred, gts in zip(predictions, ground_truths):
        if any(compute_em(pred, gt) == 100.0 for gt in gts):
            correct += 1
    
    return 100.0 * correct / len(predictions) if predictions else 0.0

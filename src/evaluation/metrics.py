"""
Evaluation metrics module for Cricket Analytics RAG system
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def calculate_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """
    Calculate accuracy for categorical predictions
    
    Args:
        predictions: List of predicted values
        ground_truth: List of ground truth values
    
    Returns:
        Accuracy score between 0 and 1
    """
    if len(predictions) != len(ground_truth):
        logger.warning(f"Predictions and ground truth have different lengths: {len(predictions)} vs {len(ground_truth)}")
        return 0.0
    
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(predictions) if len(predictions) > 0 else 0.0

def calculate_f1_score(predictions: List[str], ground_truth: List[str], positive_class: str) -> float:
    """
    Calculate F1 score for binary classification
    
    Args:
        predictions: List of predicted values
        ground_truth: List of ground truth values
        positive_class: The class to consider as positive
        
    Returns:
        F1 score between 0 and 1
    """
    if len(predictions) != len(ground_truth):
        logger.warning(f"Predictions and ground truth have different lengths: {len(predictions)} vs {len(ground_truth)}")
        return 0.0
    
    true_positives = sum(1 for p, g in zip(predictions, ground_truth) if p == positive_class and g == positive_class)
    false_positives = sum(1 for p, g in zip(predictions, ground_truth) if p == positive_class and g != positive_class)
    false_negatives = sum(1 for p, g in zip(predictions, ground_truth) if p != positive_class and g == positive_class)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

def calculate_mrr(relevance_rankings: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR)
    
    Args:
        relevance_rankings: List where each element is the rank position of the first relevant item
            (1-indexed, 0 if no relevant items)
    
    Returns:
        MRR score between 0 and 1
    """
    reciprocal_ranks = [1/rank if rank > 0 else 0 for rank in relevance_rankings]
    return sum(reciprocal_ranks) / len(relevance_rankings) if len(relevance_rankings) > 0 else 0

def calculate_ndcg(rankings: List[List[float]], k: Optional[int] = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG)
    
    Args:
        rankings: List of lists where each inner list contains relevance scores
        k: Optional cutoff for calculation
    
    Returns:
        NDCG score between 0 and 1
    """
    def dcg(ranking: List[float], k_val: Optional[int] = None) -> float:
        """Calculate DCG for a single ranking"""
        k_val = k_val or len(ranking)
        ranking = ranking[:k_val]
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(ranking))
    
    ndcg_scores = []
    for ranking in rankings:
        k_val = k or len(ranking)
        ideal_ranking = sorted(ranking, reverse=True)
        
        dcg_val = dcg(ranking, k_val)
        idcg_val = dcg(ideal_ranking, k_val)
        
        ndcg_scores.append(dcg_val / idcg_val if idcg_val > 0 else 0)
    
    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0

def calculate_retrieval_precision(retrieved_docs: List[str], 
                                relevant_docs: List[str]) -> float:
    """
    Calculate precision for retrieval tasks
    
    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs
    
    Returns:
        Precision score between 0 and 1
    """
    if not retrieved_docs:
        return 0.0
    
    relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]
    return len(relevant_retrieved) / len(retrieved_docs)

def calculate_retrieval_recall(retrieved_docs: List[str], 
                             relevant_docs: List[str]) -> float:
    """
    Calculate recall for retrieval tasks
    
    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs
    
    Returns:
        Recall score between 0 and 1
    """
    if not relevant_docs:
        return 0.0
    
    relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]
    return len(relevant_retrieved) / len(relevant_docs)

def calculate_cricket_insight_accuracy(predictions: List[Dict[str, Any]], 
                                     ground_truth: List[Dict[str, Any]],
                                     metrics: List[str]) -> Dict[str, float]:
    """
    Calculate cricket-specific insight accuracy
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        metrics: List of metrics to calculate
        
    Returns:
        Dictionary with accuracy scores for each metric
    """
    results = {}
    
    for metric in metrics:
        pred_values_raw = [p.get(metric) for p in predictions]
        gt_values_raw = [g.get(metric) for g in ground_truth]

        if not pred_values_raw or not gt_values_raw:
            logger.warning(f"Metric {metric} has no values to compare")
            continue

        # Skip metrics that are not present
        if any(value is None for value in pred_values_raw) or any(value is None for value in gt_values_raw):
            logger.warning(f"Metric {metric} not present in all predictions/ground truth")
            continue

        # Handle different types of metrics
        if all(isinstance(value, (int, float)) for value in pred_values_raw + gt_values_raw):
            pred_values = [float(value) for value in pred_values_raw]
            gt_values = [float(value) for value in gt_values_raw]

            # For numerical metrics, calculate mean absolute error
            mae = sum(abs(p - g) for p, g in zip(pred_values, gt_values)) / len(pred_values)
            results[f"{metric}_mae"] = mae

            # Calculate normalized error
            combined_values = pred_values + gt_values
            max_val = max(combined_values)
            min_val = min(combined_values)
            range_val = max_val - min_val

            if range_val > 0:
                norm_mae = mae / range_val
                results[f"{metric}_norm_mae"] = norm_mae
                results[f"{metric}_accuracy"] = 1 - norm_mae
        elif all(isinstance(value, str) for value in pred_values_raw + gt_values_raw):
            pred_values = [str(value) for value in pred_values_raw]
            gt_values = [str(value) for value in gt_values_raw]

            # For categorical metrics, calculate accuracy
            accuracy = calculate_accuracy(pred_values, gt_values)
            results[f"{metric}_accuracy"] = accuracy
        else:
            logger.warning(f"Unsupported metric type for {metric}; skipping")
            
    return results
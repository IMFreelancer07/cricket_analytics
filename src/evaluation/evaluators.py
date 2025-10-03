"""
Evaluators module for Cricket Analytics RAG system
"""

from typing import List, Dict, Any, Optional
import json
import pandas as pd
import logging
from pathlib import Path

# Import metrics functions
from .metrics import (
    calculate_accuracy, 
    calculate_f1_score,
    calculate_mrr,
    calculate_ndcg,
    calculate_retrieval_precision,
    calculate_retrieval_recall,
    calculate_cricket_insight_accuracy
)

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluator for RAG system performance"""
    
    def __init__(self, ground_truth_path: Optional[str] = None):
        self.ground_truth: List[Dict[str, Any]] = []
        if ground_truth_path:
            self.load_ground_truth(ground_truth_path)
    
    def load_ground_truth(self, ground_truth_path: str):
        """Load ground truth data from file"""
        try:
            gt_path = Path(ground_truth_path)
            if gt_path.suffix == '.json':
                with open(gt_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                if not isinstance(data, list):
                    raise ValueError("Ground truth JSON must be a list or object")
                parsed_items: List[Dict[str, Any]] = []
                for item in data:
                    if isinstance(item, dict):
                        parsed_items.append({str(key): value for key, value in item.items()})
                    else:
                        logger.warning("Skipping non-dict item in ground truth JSON: %s", item)
                self.ground_truth = parsed_items
                logger.info(f"Loaded ground truth data from {ground_truth_path}")
            elif gt_path.suffix == '.csv':
                self.ground_truth = pd.read_csv(gt_path).to_dict(orient='records')
                logger.info(f"Loaded ground truth data from {ground_truth_path}")
            else:
                logger.error(f"Unsupported file format: {gt_path.suffix}")
        except Exception as e:
            logger.error(f"Failed to load ground truth data: {e}")
    
    def evaluate_retrieval(self, 
                          queries: List[str],
                          retrieved_docs: List[List[str]],
                          relevant_docs: List[List[str]]) -> Dict[str, float]:
        """
        Evaluate retrieval performance
        
        Args:
            queries: List of query strings
            retrieved_docs: List of lists of retrieved document IDs for each query
            relevant_docs: List of lists of relevant document IDs for each query
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {}
        
        # Calculate precision and recall for each query
        precisions = []
        recalls = []
        
        for q_retrieved, q_relevant in zip(retrieved_docs, relevant_docs):
            precision = calculate_retrieval_precision(q_retrieved, q_relevant)
            recall = calculate_retrieval_recall(q_retrieved, q_relevant)
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate mean metrics
        results['mean_precision'] = sum(precisions) / len(precisions) if precisions else 0
        results['mean_recall'] = sum(recalls) / len(recalls) if recalls else 0
        
        # Calculate F1 from mean precision and recall
        if results['mean_precision'] + results['mean_recall'] > 0:
            results['f1_score'] = 2 * results['mean_precision'] * results['mean_recall'] / (
                results['mean_precision'] + results['mean_recall'])
        else:
            results['f1_score'] = 0
        
        # Calculate MRR
        mrr_ranks = []
        for q_retrieved, q_relevant in zip(retrieved_docs, relevant_docs):
            # Find position of first relevant document (1-indexed)
            rank = 0
            for i, doc_id in enumerate(q_retrieved, 1):
                if doc_id in q_relevant:
                    rank = i
                    break
            mrr_ranks.append(rank)
        
        results['mrr'] = calculate_mrr(mrr_ranks)
        
        return results
    
    def evaluate_cricket_insights(self, 
                                predictions: List[Dict[str, Any]], 
                                ground_truth: Optional[List[Dict[str, Any]]] = None,
                                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate cricket-specific insights
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: Optional list of ground truth dictionaries (uses loaded ground truth if None)
            metrics: Optional list of metrics to calculate (uses all common keys if None)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if ground_truth is None:
            if not self.ground_truth:
                raise ValueError("No ground truth data available")
            ground_truth = self.ground_truth
        
        metrics_to_use = metrics
        if metrics_to_use is None:
            metrics_to_use = []
            if predictions and ground_truth:
                pred_keys = set(predictions[0].keys())
                gt_keys = set(ground_truth[0].keys())
                metrics_to_use = list(pred_keys.intersection(gt_keys))
        
        results = calculate_cricket_insight_accuracy(predictions, ground_truth, metrics_to_use)
        
        # Calculate overall accuracy as mean of individual accuracies
        accuracy_keys = [k for k in results.keys() if k.endswith('_accuracy')]
        if accuracy_keys:
            results['overall_accuracy'] = sum(results[k] for k in accuracy_keys) / len(accuracy_keys)
        
        return results
    
    def evaluate_rag_responses(self, 
                             queries: List[str],
                             responses: List[str],
                             ground_truth_responses: List[str],
                             retrieved_docs: Optional[List[List[str]]] = None,
                             relevant_docs: Optional[List[List[str]]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of RAG system responses
        
        Args:
            queries: List of query strings
            responses: List of RAG system responses
            ground_truth_responses: List of ground truth responses
            retrieved_docs: Optional list of lists of retrieved document IDs
            relevant_docs: Optional list of lists of relevant document IDs
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        results = {}
        
        # Evaluate response quality (placeholder for more sophisticated metrics)
        # In a real implementation, this could use semantic similarity or LLM-based evaluation
        response_match = [1 if r == gt else 0 for r, gt in zip(responses, ground_truth_responses)]
        results['response_accuracy'] = sum(response_match) / len(response_match) if response_match else 0
        
        # Add retrieval evaluation if documents are provided
        if retrieved_docs and relevant_docs:
            retrieval_results = self.evaluate_retrieval(queries, retrieved_docs, relevant_docs)
            for k, v in retrieval_results.items():
                results[f"retrieval_{k}"] = v
        
        return results
    
    def evaluate_langgraph_reasoning(self,
                                    queries: List[str],
                                    reasoning_steps: List[List[Dict[str, Any]]],
                                    expected_reasoning: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        Evaluate multi-hop reasoning quality
        
        Args:
            queries: List of query strings
            reasoning_steps: List of lists of reasoning step dictionaries
            expected_reasoning: List of lists of expected reasoning step dictionaries
            
        Returns:
            Dictionary with reasoning evaluation metrics
        """
        results = {}
        
        # Compare reasoning path lengths
        path_lengths = [len(steps) for steps in reasoning_steps]
        expected_lengths = [len(steps) for steps in expected_reasoning]
        
        length_diff = [abs(p - e) for p, e in zip(path_lengths, expected_lengths)]
        results['mean_path_length_diff'] = sum(length_diff) / len(length_diff) if length_diff else 0
        
        # Calculate path similarity (simple implementation)
        path_similarities = []
        for actual_steps, expected_steps in zip(reasoning_steps, expected_reasoning):
            # Calculate Jaccard similarity between steps
            actual_types = [step.get('node_type', '') for step in actual_steps]
            expected_types = [step.get('node_type', '') for step in expected_steps]
            
            # Get unique steps
            unique_actual = set(actual_types)
            unique_expected = set(expected_types)
            
            # Calculate Jaccard similarity
            if unique_actual or unique_expected:
                similarity = len(unique_actual.intersection(unique_expected)) / len(unique_actual.union(unique_expected))
                path_similarities.append(similarity)
        
        results['mean_path_similarity'] = sum(path_similarities) / len(path_similarities) if path_similarities else 0
        
        return results
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file"""
        try:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            if out_path.suffix == '.json':
                with open(out_path, 'w') as f:
                    json.dump(results, f, indent=2)
            elif out_path.suffix == '.csv':
                pd.DataFrame([results]).to_csv(out_path, index=False)
            else:
                logger.error(f"Unsupported output format: {out_path.suffix}")
                return
            
            logger.info(f"Saved evaluation results to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
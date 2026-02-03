"""Evaluation engine for conversational Q&A."""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .data_loader import get_tasks
from .models import ConversationalQAModel
from .metrics import compute_f1, compute_em, compute_intent_metrics, compute_heq1


class ConQAEval:
    """Evaluation suite for conversational Q&A models."""
    
    def __init__(
        self,
        tasks: Optional[List[str]] = None,
        batch_size: int = 1,
        split: str = "validation",
    ):
        """Initialize evaluator.
        
        Args:
            tasks: List of task names (default: all)
            batch_size: Batch size for prediction
            split: Dataset split
        """
        self.tasks = get_tasks(tasks=tasks, split=split)
        self.batch_size = batch_size
        self.split = split
    
    def run(
        self,
        model: ConversationalQAModel,
        output_folder: str = "results",
        save_predictions: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """Run evaluation on all tasks.
        
        Args:
            model: Model to evaluate
            output_folder: Where to save results
            save_predictions: Whether to save predictions
            
        Returns:
            Dict of {task_name: metrics}
        """
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        for task in self.tasks:
            print(f"\n{'='*60}")
            print(f"Evaluating on {task.name}")
            print(f"{'='*60}")
            
            results = self.evaluate_task(task, model)
            all_results[task.name] = results
            
            # Save results
            result_file = output_path / f"{task.name}_{self.split}.json"
            with open(result_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults for {task.name}:")
            for metric, value in results["metrics"].items():
                print(f"  {metric}: {value:.2f}")
        
        # Save summary
        summary_file = output_path / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        return all_results
    
    def evaluate_task(
        self,
        task,
        model: ConversationalQAModel,
    ) -> Dict[str, Any]:
        """Evaluate model on a single task.
        
        Args:
            task: ConversationalTask object
            model: Model to evaluate
            
        Returns:
            Dict with 'metrics' and 'predictions'
        """
        conversations = task.get_conversations()
        
        predictions = []
        references = []
        
        for conv in tqdm(conversations, desc=f"Processing {task.name}"):
            history = []
            
            for turn in conv.turns:
                # Format input
                input_data = task.format_input(turn, history)
                
                # Predict
                pred = model.predict(input_data)
                
                # Store predictions
                predictions.append({
                    "question_id": turn.question_id,
                    "conversation_id": turn.conversation_id,
                    "turn_id": turn.turn_id,
                    "question": turn.question,
                    "predicted_answer": pred["answer"],
                    "gold_answer": turn.answer,
                    "ambiguous_utterance": pred.get("ambiguous_utterance", False),
                    "total_candidates": pred.get("total_candidates", 1),
                    "explanation": pred.get("explanation", ""),
                })
                
                references.append({
                    "answer": turn.answer,
                    "ambiguous_utterance": task._is_ambiguous(turn),
                })
                
                # Update history with PREDICTED answer (realistic setting)
                history.append({
                    "question": turn.question,
                    "answer": pred["answer"],
                })
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, references)
        
        return {
            "metrics": metrics,
            "predictions": predictions,
        }
    
    @staticmethod
    def _compute_metrics(predictions: List[Dict], references: List[Dict]) -> Dict[str, float]:
        """Compute all metrics."""
        f1_scores = []
        em_scores = []
        
        for pred, ref in zip(predictions, references):
            f1 = compute_f1(pred["predicted_answer"], ref["answer"])
            em = compute_em(pred["predicted_answer"], ref["answer"])
            f1_scores.append(f1)
            em_scores.append(em)
        
        # Intent metrics
        intent_metrics = compute_intent_metrics(predictions, references)
        
        return {
            "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "em": sum(em_scores) / len(em_scores) if em_scores else 0.0,
            **intent_metrics,
        }

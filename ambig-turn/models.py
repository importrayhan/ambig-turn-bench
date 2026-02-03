"""Model wrappers for conversational Q&A."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class ConversationalQAModel(ABC):
    """Abstract base class for conversational Q&A models."""
    
    @abstractmethod
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer for a conversation turn.
        
        Args:
            input_data: Dict with fields:
                - prompt: Current question
                - context: Background + history
                - can_retrieve: Whether retrieval allowed
                - tools: Available tools
                - conversation: History
                
        Returns:
            Dict with fields:
                - answer: Generated answer
                - ambiguous_utterance: bool
                - total_candidates: int
                - explanation: str
        """
        pass
    
    @abstractmethod
    def batch_predict(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch prediction."""
        pass


class DummyModel(ConversationalQAModel):
    """Dummy model for testing."""
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "answer": "I don't know.",
            "ambiguous_utterance": False,
            "total_candidates": 1,
            "explanation": "Dummy model always returns 'I don't know.'",
        }
    
    def batch_predict(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict(inp) for inp in inputs]

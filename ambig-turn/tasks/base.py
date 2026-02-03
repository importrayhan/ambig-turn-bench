"""Base class for conversational ambig-turn tasks."""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    question: str
    answer: str
    question_id: str
    conversation_id: str
    turn_id: int
    context: Optional[str] = None
    yesno: Optional[str] = None  # 'y', 'n', 'x' (not applicable)
    followup: Optional[str] = None  # 'y', 'n', 'm' (maybe)
    can_retrieve: bool = True
    tools: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "question_id": self.question_id,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "context": self.context,
            "yesno": self.yesno,
            "followup": self.followup,
            "can_retrieve": self.can_retrieve,
            "tools": self.tools or [],
        }


@dataclass
class Conversation:
    """Full conversation with multiple turns."""
    conversation_id: str
    turns: List[ConversationTurn]
    background: Optional[str] = None
    title: Optional[str] = None
    
    def get_history(self, up_to_turn: int) -> List[Dict[str, str]]:
        """Get conversation history up to a specific turn.
        
        Args:
            up_to_turn: Turn index
            
        Returns:
            List of {"question": ..., "answer": ...} dicts
        """
        return [
            {"question": turn.question, "answer": turn.answer}
            for turn in self.turns[:up_to_turn]
        ]


class ConversationalTask:
    """Base class for conversational Q&A tasks."""
    
    def __init__(
        self,
        name: str,
        split: str = "validation",
        cache_dir: Optional[str] = None
    ):
        self.name = name
        self.split = split
        self.cache_dir = cache_dir
        self.conversations: List[Conversation] = []
        self.corpus: Dict[str, str] = {}  # doc_id -> text
        
    def load_data(self):
        """Load dataset. Override in subclasses."""
        raise NotImplementedError
        
    def get_conversations(self) -> List[Conversation]:
        """Get all conversations."""
        if not self.conversations:
            self.load_data()
        return self.conversations
    
    def get_corpus(self) -> Dict[str, str]:
        """Get document corpus for retrieval."""
        if not self.corpus:
            self.load_data()
        return self.corpus
    
    def format_input(self, turn: ConversationTurn, history: List[Dict]) -> Dict[str, Any]:
        """Format input for model.
        
        Args:
            turn: Current conversation turn
            history: Previous turns
            
        Returns:
            Input dict with fields:
                - prompt: Current question
                - context: Background + conversation history
                - can_retrieve: Whether retrieval is allowed
                - tools: Available tools
                - conversation: Full conversation history
        """
        # Build context string
        context_parts = []
        if turn.context:
            context_parts.append(f"Background: {turn.context}")
        
        if history:
            context_parts.append("Conversation history:")
            for i, h in enumerate(history, 1):
                context_parts.append(f"Q{i}: {h['question']}")
                context_parts.append(f"A{i}: {h['answer']}")
        
        context = "\n".join(context_parts)
        
        return {
            "prompt": turn.question,
            "context": context,
            "can_retrieve": turn.can_retrieve,
            "tools": turn.tools or [],
            "conversation": history,
        }
    
    def format_output(self, prediction: str, turn: ConversationTurn) -> Dict[str, Any]:
        """Format model output for evaluation.
        
        Args:
            prediction: Model's answer
            turn: Ground truth turn
            
        Returns:
            Output dict with fields:
                - answer: Predicted answer
                - ambiguous_utterance: Whether question is ambiguous
                - total_candidates: Number of candidate answers
                - explanation: Reasoning for the answer
        """
        return {
            "answer": prediction,
            "ambiguous_utterance": self._is_ambiguous(turn),
            "total_candidates": 1,  # Override in subclass if applicable
            "explanation": "",
        }
    
    @staticmethod
    def _is_ambiguous(turn: ConversationTurn) -> bool:
        """Check if utterance is ambiguous."""
        # Simple heuristic: questions with pronouns without context
        pronouns = ["it", "that", "this", "they", "them", "he", "she"]
        question_lower = turn.question.lower()
        return any(p in question_lower for p in pronouns) and turn.turn_id == 0

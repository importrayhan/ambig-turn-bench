"""QuAC (Question Answering in Context) task."""
from typing import List, Dict
from datasets import load_dataset

from .base import ConversationalTask, Conversation, ConversationTurn


class QuACTask(ConversationalTask):
    """QuAC benchmark task."""
    
    def __init__(self, split: str = "validation", cache_dir: str = None):
        super().__init__(name="quac", split=split, cache_dir=cache_dir)
        
    def load_data(self):
        """Load QuAC dataset from HuggingFace."""
        dataset = load_dataset("quac", split=self.split, cache_dir=self.cache_dir)
        
        for example in dataset:
            conv_id = example["id"]
            background = example["background"]
            title = example["title"]
            
            turns = []
            for i in range(len(example["questions"])):
                turn = ConversationTurn(
                    question=example["questions"][i],
                    answer=example["answers"]["texts"][i][0],  # First answer span
                    question_id=f"{conv_id}_q{i}",
                    conversation_id=conv_id,
                    turn_id=i,
                    context=background,
                    yesno=example["answers"]["answer_starts"][i],  # Simplified
                    followup=None,  # Not in HF dataset
                    can_retrieve=True,
                )
                turns.append(turn)
            
            conversation = Conversation(
                conversation_id=conv_id,
                turns=turns,
                background=background,
                title=title,
            )
            self.conversations.append(conversation)
            
            # Add background to corpus
            self.corpus[conv_id] = background

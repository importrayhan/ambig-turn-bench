"""CoQA (Conversational Question Answering) task."""
from datasets import load_dataset

from .base import ConversationalTask, Conversation, ConversationTurn


class CoQATask(ConversationalTask):
    """CoQA benchmark task."""
    
    def __init__(self, split: str = "validation", cache_dir: str = None):
        super().__init__(name="coqa", split=split, cache_dir=cache_dir)
        
    def load_data(self):
        """Load CoQA dataset."""
        dataset = load_dataset("coqa", split=self.split, cache_dir=self.cache_dir)
        
        for example in dataset:
            conv_id = example["id"]
            story = example["story"]
            
            turns = []
            for i in range(len(example["questions"])):
                turn = ConversationTurn(
                    question=example["questions"][i],
                    answer=example["answers"]["input_text"][i],
                    question_id=f"{conv_id}_q{i}",
                    conversation_id=conv_id,
                    turn_id=i,
                    context=story,
                    can_retrieve=False,  # CoQA is extractive
                )
                turns.append(turn)
            
            conversation = Conversation(
                conversation_id=conv_id,
                turns=turns,
                background=story,
                title=example.get("source", ""),
            )
            self.conversations.append(conversation)
            
            self.corpus[conv_id] = story

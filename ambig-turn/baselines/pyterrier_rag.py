"""PyTerrier + RAG baseline for conversational Q&A."""
import pyterrier as pt
from typing import Dict, Any, List, Optional

from ..models import ConversationalQAModel

# Initialize PyTerrier
if not pt.started():
    pt.init()


class PyTerrierRAGModel(ConversationalQAModel):
    """PyTerrier retrieval + RAG generation baseline."""
    
    def __init__(
        self,
        index_path: str,
        rag_backend: str = "openai",
        model_name: str = "gpt-4o-mini",
        top_k: int = 5,
    ):
        """Initialize PyTerrier RAG model.
        
        Args:
            index_path: Path to PyTerrier index
            rag_backend: RAG backend ('openai', 'hf', 'llama')
            model_name: LLM model name
            top_k: Number of documents to retrieve
        """
        self.index = pt.IndexFactory.of(index_path)
        self.retriever = pt.BatchRetrieve(self.index, wmodel="BM25", num_results=top_k)
        self.top_k = top_k
        
        # Initialize RAG backend
        try:
            import pyterrier_rag
            if rag_backend == "openai":
                self.rag = pyterrier_rag.OpenAIBackend(model=model_name)
            elif rag_backend == "hf":
                self.rag = pyterrier_rag.HuggingFaceBackend(model=model_name)
            else:
                raise ValueError(f"Unknown backend: {rag_backend}")
        except ImportError:
            print("Warning: pyterrier_rag not installed. Using dummy backend.")
            self.rag = None
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer using retrieval + RAG.
        
        Args:
            input_data: Input with 'prompt', 'context', etc.
            
        Returns:
            Prediction dict
        """
        query = input_data["prompt"]
        context = input_data.get("context", "")
        conversation = input_data.get("conversation", [])
        
        # Build conversational query
        if conversation:
            query_with_history = self._rewrite_query(query, conversation)
        else:
            query_with_history = query
        
        # Retrieve documents
        if input_data.get("can_retrieve", True):
            results = self.retriever.search(query_with_history)
            retrieved_docs = "\n\n".join([
                f"[Doc {i+1}] {row['text']}"
                for i, (_, row) in enumerate(results.head(self.top_k).iterrows())
            ])
        else:
            retrieved_docs = context
        
        # Generate answer with RAG
        if self.rag:
            prompt = f"""Context: {retrieved_docs}

Conversation history:
{self._format_history(conversation)}

Question: {query}

Answer the question based on the context. If you cannot answer, say "CANNOTANSWER"."""
            
            answer = self.rag.generate(prompt)
        else:
            # Fallback: use first retrieved doc
            answer = retrieved_docs.split("\n")[0] if retrieved_docs else "CANNOTANSWER"
        
        # Detect ambiguity
        is_ambiguous = self._detect_ambiguity(query, conversation)
        
        return {
            "answer": answer,
            "ambiguous_utterance": is_ambiguous,
            "total_candidates": len(results) if input_data.get("can_retrieve") else 1,
            "explanation": f"Retrieved {self.top_k} documents, generated answer with {self.rag.__class__.__name__ if self.rag else 'fallback'}.",
        }
    
    def batch_predict(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch prediction (currently sequential)."""
        return [self.predict(inp) for inp in inputs]
    
    @staticmethod
    def _rewrite_query(query: str, history: List[Dict[str, str]]) -> str:
        """Rewrite query with conversation history."""
        if not history:
            return query
        
        # Simple heuristic: prepend last question
        last_q = history[-1]["question"]
        return f"{last_q} {query}"
    
    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        """Format conversation history."""
        lines = []
        for i, turn in enumerate(history, 1):
            lines.append(f"Q{i}: {turn['question']}")
            lines.append(f"A{i}: {turn['answer']}")
        return "\n".join(lines)
    
    @staticmethod
    def _detect_ambiguity(query: str, history: List[Dict[str, str]]) -> bool:
        """Simple ambiguity detector."""
        pronouns = ["it", "that", "this", "they", "them", "he", "she", "him", "her"]
        query_lower = query.lower()
        has_pronoun = any(f" {p} " in f" {query_lower} " for p in pronouns)
        return has_pronoun and len(history) == 0


def build_index(corpus: Dict[str, str], index_path: str):
    """Build PyTerrier index from corpus.
    
    Args:
        corpus: Dict of {doc_id: text}
        index_path: Output index path
    """
    import pandas as pd
    
    # Convert corpus to DataFrame
    docs = [
        {"docno": doc_id, "text": text}
        for doc_id, text in corpus.items()
    ]
    df = pd.DataFrame(docs)
    
    # Create index
    indexer = pt.IterDictIndexer(index_path, overwrite=True)
    indexer.index(df.to_dict("records"))
    
    print(f"Index created at {index_path}")

"""Data loading utilities for ambig-turn-benchmarks."""

import json
from typing import List, Optional, Dict
from pathlib import Path
from datasets import load_dataset

from .tasks.base import ConversationalTask
from .tasks.quac import QuACTask
from .tasks.coqa import CoQATask


AVAILABLE_TASKS = {
    "quac": QuACTask,
    "coqa": CoQATask,
}


def list_tasks() -> List[str]:
    """List all available tasks.
    
    Returns:
        List of task names
    """
    return list(AVAILABLE_TASKS.keys())


def get_tasks(
    tasks: Optional[List[str]] = None,
    split: str = "validation",
    cache_dir: Optional[str] = None
) -> List[ConversationalTask]:
    """Load conversational Q&A tasks.
    
    Args:
        tasks: List of task names. If None, loads all tasks.
        split: Dataset split ('train', 'validation', 'test')
        cache_dir: Directory to cache downloaded datasets
        
    Returns:
        List of ConversationalTask objects
        
    Example:
        >>> from conqa_bench import get_tasks
        >>> tasks = get_tasks(tasks=["quac", "coqa"])
        >>> print(tasks[0].name)
        'quac'
    """
    if tasks is None:
        tasks = list_tasks()
    
    loaded_tasks = []
    for task_name in tasks:
        if task_name not in AVAILABLE_TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available: {list_tasks()}"
            )
        
        task_class = AVAILABLE_TASKS[task_name]
        task = task_class(split=split, cache_dir=cache_dir)
        loaded_tasks.append(task)
    
    return loaded_tasks


def load_custom_task(
    name: str,
    data_path: str,
    format: str = "json"
) -> ConversationalTask:
    """Load a custom conversational Q&A dataset.
    
    Args:
        name: Task name
        data_path: Path to data file
        format: Data format ('json', 'jsonl')
        
    Returns:
        ConversationalTask object
    """
    from .tasks.base import ConversationalTask
    
    with open(data_path) as f:
        if format == "json":
            data = json.load(f)
        elif format == "jsonl":
            data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    return ConversationalTask(name=name, data=data)

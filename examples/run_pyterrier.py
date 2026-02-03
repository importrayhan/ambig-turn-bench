"""Example: Run PyTerrier RAG baseline."""
import argparse
from conqa_bench import get_tasks, ConQAEval
from conqa_bench.baselines.pyterrier_rag import PyTerrierRAGModel, build_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["quac"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--index-path", default="./indexes/quac_index")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--build-index", action="store_true")
    args = parser.parse_args()
    
    # Load tasks
    tasks = get_tasks(tasks=args.tasks, split=args.split)
    
    # Build index if needed
    if args.build_index:
        print("Building index...")
        for task in tasks:
            corpus = task.get_corpus()
            index_path = f"{args.index_path}_{task.name}"
            build_index(corpus, index_path)
    
    # Initialize model
    model = PyTerrierRAGModel(
        index_path=f"{args.index_path}_{args.tasks[0]}",
        rag_backend="openai",
        model_name="gpt-4o-mini",
        top_k=5,
    )
    
    # Run evaluation
    evaluator = ConQAEval(tasks=args.tasks, split=args.split)
    results = evaluator.run(model, output_folder=args.output_dir)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for task_name, metrics in results.items():
        print(f"\n{task_name}:")
        for metric, value in metrics["metrics"].items():
            print(f"  {metric}: {value:.2f}")


if __name__ == "__main__":
    main()

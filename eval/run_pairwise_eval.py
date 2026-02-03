import argparse
from eval_utils import get_device, load_model, evaluate_pairwise, print_results, print_task_summary
from dataset_loaders import load_dalaj_minpairs, load_scala_swedish, load_swedish_idioms, load_bananer_translationese


DATASET_LOADERS = {
    'dalaj': load_dalaj_minpairs,
    'scala-sv': load_scala_swedish,
    'idioms': load_swedish_idioms,
    'bananer': load_bananer_translationese,
}


def run_evaluation(dataset_name, model_name='HuggingFaceTB/SmolLM-135M', batched=False,
                   save_results=False, output_dir="results"):
    print(f"\n{'='*60}")
    print(f"Running evaluation: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    if dataset_name not in DATASET_LOADERS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available datasets: {list(DATASET_LOADERS.keys())}")
        return

    loader = DATASET_LOADERS[dataset_name]
    data = loader()

    device = get_device()
    tokenizer, model = load_model(model_name, device)

    all_results = {}

    if 'correct' in data and 'wrong' in data:
        results = evaluate_pairwise(
            data['correct'], data['wrong'],
            tokenizer, model, batched=batched
        )
        print_results(results, f"{data['name']}")
        all_results['overall'] = results

    elif 'setups' in data:
        for setup_name, setup_data in data['setups'].items():
            print(f"\nEvaluating setup: {setup_name}")
            results = evaluate_pairwise(
                setup_data['correct'], setup_data['wrong'],
                tokenizer, model, batched=batched
            )
            print_results(results, f"{data['name']} - {setup_name}")
            all_results[setup_name] = results

    elif 'error_types' in data:
        for error_type, error_data in data['error_types'].items():
            print(f"\nEvaluating error type: {error_type}")
            results = evaluate_pairwise(
                error_data['correct'], error_data['wrong'],
                tokenizer, model, batched=batched
            )
            print_results(results, f"{data['name']} - {error_type}")
            all_results[error_type] = results

    elif dataset_name == 'scala-sv':
        for error_type, error_data in data.items():
            if error_type in ['name', 'language']:
                continue
            print(f"\nEvaluating: {error_type}")
            results = evaluate_pairwise(
                error_data['correct'], error_data['wrong'],
                tokenizer, model, batched=batched
            )
            print_results(results, f"{data['name']} - {error_type}")
            all_results[error_type] = results

    print_task_summary(all_results, summary_title=f"Summary for {dataset_name}")

    if save_results:
        from eval_utils import save_multi_task_results

        language = data.get('language', None)

        metadata = {
            'batched': batched,
            'num_tasks': len(all_results)
        }

        filepath = save_multi_task_results(
            all_results, model_name, dataset_name,
            output_dir=output_dir, language=language, metadata=metadata
        )
        print(f"Results saved to: {filepath}\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run pairwise perplexity evaluations')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(DATASET_LOADERS.keys()),
        help='Dataset to evaluate on'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='HuggingFaceTB/SmolLM-135M',
        help='Model to evaluate'
    )
    parser.add_argument(
        '--batched',
        action='store_true',
        help='Use batched perplexity computation'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results (default: results)'
    )

    args = parser.parse_args()

    run_evaluation(
        args.dataset,
        model_name=args.model,
        batched=args.batched,
        save_results=args.save_results,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

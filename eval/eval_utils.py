import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_model(model_name, device, for_generation=False):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if for_generation:
        tokenizer.padding_side = "left"

    if device.type == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map=device.type if device.type != "mps" else None
        )
        if device.type != "cuda":
            model = model.to(device)

    model.eval()

    if for_generation:
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


def compute_perplexity_batched(texts_a, texts_b, tokenizer, model, batch_size=16, show_progress=True):
    assert len(texts_a) == len(texts_b), "Text lists must have same length"

    all_texts = []
    for i in range(len(texts_a)):
        all_texts.extend([texts_a[i], texts_b[i]])

    results_flat = []
    loss_fct = CrossEntropyLoss(reduction="none")

    num_batches = (len(all_texts) + batch_size - 1) // batch_size
    iterator = range(0, len(all_texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=num_batches, desc="Computing perplexity")

    for i in iterator:
        batch = all_texts[i:i+batch_size]

        tok = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            logits = model(**tok).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tok.input_ids[..., 1:].contiguous()

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size())

            mask = shift_labels != tokenizer.pad_token_id
            ppl = torch.exp((loss * mask).sum(dim=1) / mask.sum(dim=1))

        results_flat.extend(ppl.tolist())

    results = [(results_flat[i*2], results_flat[i*2+1]) for i in range(len(texts_a))]

    return results


def evaluate_pairwise(correct_texts, wrong_texts, tokenizer, model, batched=False, batch_size=16):
    if batched:
        results = compute_perplexity_batched(correct_texts, wrong_texts, tokenizer, model, batch_size)
    else:
        results = compute_perplexity_unbatched(correct_texts, wrong_texts, tokenizer, model)

    correct_guesses = sum(1 for ppl_a, ppl_b in results if ppl_a < ppl_b)
    accuracy = correct_guesses / len(results)

    return {
        'accuracy': accuracy,
        'correct_guesses': correct_guesses,
        'total': len(results),
        'results': results
    }


def compute_perplexity_unbatched(texts_a, texts_b, tokenizer, model, show_progress=True):
    results = []
    iterator = zip(texts_a, texts_b)
    if show_progress:
        iterator = tqdm(iterator, total=len(texts_a), desc="Computing perplexity")

    for a, b in iterator:
        pair = []
        for text in (a, b):
            text = str(text).strip()
            encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = encoding["input_ids"].to(model.device)
            attention_mask = encoding["attention_mask"].to(model.device)

            labels = input_ids.clone()
            labels[input_ids == tokenizer.pad_token_id] = -100

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            num_tokens = (labels != -100).sum().item()
            total_loss = loss.item() * num_tokens
            average_token_loss = total_loss / num_tokens
            ppl = torch.exp(torch.tensor(average_token_loss)).item()
            pair.append(ppl)

        results.append(tuple(pair))

    return results


def compute_confidence_interval(accuracy, n_samples, confidence=0.95):
    try:
        from scipy import stats
    except ImportError:
        import math
        z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
        se = math.sqrt(accuracy * (1 - accuracy) / n_samples)
        margin = z * se
        return (max(0, accuracy - margin), min(1, accuracy + margin))

    z = stats.norm.ppf((1 + confidence) / 2)

    if n_samples == 0:
        return (0.0, 1.0)
    if accuracy == 0:
        return (0.0, stats.binom.ppf(1 - (1 - confidence) / 2, n_samples, 1/n_samples) / n_samples)
    if accuracy == 1:
        return (stats.binom.ppf((1 - confidence) / 2, n_samples, 1 - 1/n_samples) / n_samples, 1.0)

    p = accuracy
    n = n_samples
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denominator

    return (max(0, centre - margin), min(1, centre + margin))


def print_results(results_dict, task_name="Evaluation", show_ci=True, confidence=0.95):
    print(f"\n{'='*60}")
    print(f"{task_name}")
    print(f"{'='*60}")

    if 'accuracy' in results_dict:
        acc = results_dict['accuracy']
        correct = results_dict.get('correct_guesses', 0)
        total = results_dict.get('total', 0)

        if show_ci and total > 0:
            ci_low, ci_high = compute_confidence_interval(acc, total, confidence)
            ci_percent = int(confidence * 100)
            print(f"Accuracy: {acc:.4f} ({correct}/{total})")
            print(f"{ci_percent}% CI:  [{ci_low:.4f}, {ci_high:.4f}]")
        else:
            print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    else:
        for key, value in results_dict.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    print(f"{'='*60}\n")


def print_task_summary(all_results, summary_title="Summary", confidence=0.95):
    print(f"\n{'='*60}")
    print(f"{summary_title}")
    print(f"{'='*60}")

    for task_name, results in all_results.items():
        if isinstance(results, dict) and 'accuracy' in results:
            acc = results['accuracy']
            total = results.get('total', 0)
            if total > 0:
                ci_low, ci_high = compute_confidence_interval(acc, total, confidence)
                ci_percent = int(confidence * 100)
                print(f"{task_name}: {acc:.4f} ({ci_percent}% CI: [{ci_low:.4f}, {ci_high:.4f}])")
            else:
                print(f"{task_name}: {acc:.4f}")
        elif isinstance(results, (int, float)):
            print(f"{task_name}: {results:.4f}")

    print(f"{'='*60}\n")


def save_multi_task_results(all_results, model_name, dataset_name, output_dir="results",
                             language=None, metadata=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    clean_model_name = model_name.replace("/", "_")
    clean_dataset_name = dataset_name.replace("/", "_")

    filename = f"{clean_model_name}_{clean_dataset_name}_all_{timestamp}.json"
    filepath = output_path / filename

    output_data = {
        "metadata": {
            "model": model_name,
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "num_tasks": len(all_results)
        },
        "tasks": {}
    }

    if language:
        output_data["metadata"]["language"] = language
    if metadata:
        output_data["metadata"].update(metadata)

    for task_name, results in all_results.items():
        task_data = {}

        if isinstance(results, dict) and 'accuracy' in results:
            acc = results['accuracy']
            total = results.get('total', 0)
            correct = results.get('correct_guesses', 0)

            task_data["accuracy"] = acc
            task_data["correct"] = correct
            task_data["total"] = total

            if total > 0:
                ci_low, ci_high = compute_confidence_interval(acc, total)
                task_data["confidence_interval"] = {
                    "lower": ci_low,
                    "upper": ci_high,
                    "confidence_level": 0.95
                }
        else:
            task_data = results

        output_data["tasks"][task_name] = task_data

    accuracies = []
    for task_name, task_data in output_data["tasks"].items():
        if isinstance(task_data, dict) and 'accuracy' in task_data:
            accuracies.append(task_data['accuracy'])

    if accuracies:
        output_data["summary"] = {
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "num_tasks_evaluated": len(accuracies)
        }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return str(filepath)

# Idiomatic Language Acquisition
Code and data for the paper "Preferences for Idiomatic Language are Acquired Slowly --- and Forgotten Quickly: A Case Study on Swedish", TACL 2026.

### Models
Models and checkpoints are available on HuggingFace in the collection [Idiomatic Language Acquisition](https://huggingface.co/collections/jekunz/idiomatic-language-acquisition). 

## Run evals 

```bash
python run_pairwise_eval.py --dataset dalaj --model HuggingFaceTB/SmolLM-135M
```
Arguments: 
- `--dataset`: Dataset to evaluate (dalaj, scala-sv, bananer, idioms)
- `--model`: Model to evaluate (hf identifier)
- `--save-results`: Save results to JSON
- `--output-dir`: Output directory for results

### Citation
Coming soon.

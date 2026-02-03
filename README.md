# Idiomatic Language Acquisition
Code and data for the paper "Preferences for Idiomatic Language are Acquired Slowly --- and Forgotten Quickly: A Case Study on Swedish", TACL 2026.

### Models
Models and checkpoints are available on HuggingFace in the collection [Idiomatic Language Acquisition](https://huggingface.co/collections/jekunz/idiomatic-language-acquisition). 

## Data

All publicly releasible eval data is available as csv files in the data folder. 

## Run evals 

```bash
python run_pairwise_eval.py --dataset dalaj --model HuggingFaceTB/SmolLM-135M
```
Arguments: 
- `--dataset`: Dataset to evaluate (dalaj, scala-sv, bananer, idioms)
- `--model`: Model to evaluate (hf identifier)
- `--save-results`: Save results to JSON
- `--output-dir`: Output directory for results

## Citations

### Paper
Coming soon.

```bibtex
```

### DaLAJ Dataset 

```bibtex
  @inproceedings{volodina-etal-2021-dalaj,
      title = "{D}a{LAJ} {--} a dataset for linguistic acceptability judgments for {S}wedish",
      author = "Volodina, Elena  and
        Mohammed, Yousuf Ali  and
        Klezl, Julia",
      editor = {Alfter, David  and
        Volodina, Elena  and
        Pilan, Ildik{\'o}  and
        Gra{\"e}n, Johannes  and
        Borin, Lars},
      booktitle = "Proceedings of the 10th Workshop on NLP for Computer Assisted Language Learning",
      month = may,
      year = "2021",
      address = "Online",
      publisher = "LiU Electronic Press",
      url = "https://aclanthology.org/2021.nlp4call-1.3/",
      pages = "28--37"
  }
```

### ScaLA Dataset 
```bibtex
  @inproceedings{nielsen-2023-scandeval,
      title = "{S}cand{E}val: A Benchmark for {S}candinavian Natural Language Processing",
      author = "Nielsen, Dan",
      editor = {Alum{\"a}e, Tanel  and
        Fishel, Mark},
      booktitle = "Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)",
      month = may,
      year = "2023",
      address = "T{\'o}rshavn, Faroe Islands",
      publisher = "University of Tartu Library",
      url = "https://aclanthology.org/2023.nodalida-1.20/",
      pages = "185--201",
      abstract = "This paper introduces a Scandinavian benchmarking platform, ScandEval, which can benchmark any pretrained model on four different tasks in the Scandinavian languages. The datasets used in two of the tasks, linguistic acceptability and question answering, are new. We develop and release a Python package and command-line interface, scandeval, which can benchmark any model that has been uploaded to the Hugging Face Hub, with reproducible results. Using this package, we benchmark more than 80 Scandinavian or multilingual models and present the results of these in an interactive online leaderboard, as well as provide an analysis of the results. The analysis shows that there is substantial cross-lingual transfer among the the Mainland Scandinavian languages (Danish, Swedish and Norwegian), with limited cross-lingual transfer between the group of Mainland Scandinavian languages and the group of Insular Scandinavian languages (Icelandic and Faroese). The benchmarking results also show that the investment in language technology in Norway and Sweden has led to language models that outperform massively multilingual models such as XLM-RoBERTa and mDeBERTaV3. We release the source code for both the package and leaderboard."
  }
```

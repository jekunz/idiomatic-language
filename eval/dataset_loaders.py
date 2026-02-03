import pandas as pd
from pathlib import Path

CSV_DIR = Path(__file__).parent / "datasets_csv"


def load_scala_swedish():
    dfs = []
    for split in ['train', 'validation', 'test']:
        df = pd.read_csv(CSV_DIR / f"scala_swedish_{split}.csv")
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)

    filtered_flip = [
        (row['correct'], row['incorrect'])
        for _, row in all_data.iterrows()
        if row['type'] == 'flip_neighbours'
    ]

    filtered_delete = [
        (row['correct'], row['incorrect'])
        for _, row in all_data.iterrows()
        if row['type'] == 'delete'
    ]

    return {
        'flip_neighbours': {
            'correct': [x[0] for x in filtered_flip],
            'wrong': [x[1] for x in filtered_flip],
        },
        'delete': {
            'correct': [x[0] for x in filtered_delete],
            'wrong': [x[1] for x in filtered_delete],
        },
        'name': 'scala-swedish',
        'language': 'swedish'
    }


def load_dalaj_minpairs():
    import ast

    dfs = []
    for split in ['train', 'validation', 'test']:
        df = pd.read_csv(CSV_DIR / f"dalaj_{split}.csv")
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)

    error_types = {
        'Morphology': 'M',
        'Syntax': 'S',
        'Orthography': 'O',
        'Lexical': 'L',
        'Punctuation': 'P'
    }

    result = {}
    for error_name, error_label in error_types.items():
        filtered = [
            (row['corrected_sentence'], row['sentence'])
            for _, row in all_data.iterrows()
            if ast.literal_eval(row['meta'])['error_label'] == error_label
        ]
        result[error_name] = {
            'correct': [x[0] for x in filtered],
            'wrong': [x[1] for x in filtered],
        }

    return {
        'error_types': result,
        'name': 'dalaj-minpairs',
        'language': 'swedish'
    }


def load_bananer_translationese(tokenizer=None):
    from transformers import AutoTokenizer

    df = pd.read_csv(CSV_DIR / "bananer_translationese.csv")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    setups = {}

    filtered = [(row['Alternativ'], row['Översättningssvenska']) for _, row in df.iterrows()]
    setups['all'] = {
        'correct': [x[0] for x in filtered],
        'wrong': [x[1] for x in filtered]
    }

    filtered = [
        (row['Alternativ'], row['Översättningssvenska'])
        for _, row in df.iterrows()
        if len(tokenizer(text=str(row["Alternativ"]))["input_ids"]) ==
           len(tokenizer(text=str(row["Översättningssvenska"]))["input_ids"])
        and row['Val'] == 'Alternativ'
    ]
    setups['token_length_and_manual'] = {
        'correct': [x[0] for x in filtered],
        'wrong': [x[1] for x in filtered]
    }

    filtered = [
        (row['Alternativ'], row['Översättningssvenska'])
        for _, row in df.iterrows()
        if row['Val'] == 'Alternativ'
    ]
    setups['manual_only'] = {
        'correct': [x[0] for x in filtered],
        'wrong': [x[1] for x in filtered]
    }

    filtered = [
        (row['Alternativ'], row['Översättningssvenska'])
        for _, row in df.iterrows()
        if len(tokenizer(text=str(row["Alternativ"]))["input_ids"]) ==
           len(tokenizer(text=str(row["Översättningssvenska"]))["input_ids"])
    ]
    setups['token_length_only'] = {
        'correct': [x[0] for x in filtered],
        'wrong': [x[1] for x in filtered]
    }

    filtered = [
        (row['Alternativ'], row['Översättningssvenska'])
        for _, row in df.iterrows()
        if len(row["Alternativ"].split()) == len(row["Översättningssvenska"].split())
        and row['Val'] == 'Alternativ'
    ]
    setups['whitespace_and_manual'] = {
        'correct': [x[0] for x in filtered],
        'wrong': [x[1] for x in filtered]
    }

    filtered = [
        (row['Alternativ'], row['Översättningssvenska'])
        for _, row in df.iterrows()
        if len(row["Alternativ"].split()) == len(row["Översättningssvenska"].split())
    ]
    setups['whitespace_only'] = {
        'correct': [x[0] for x in filtered],
        'wrong': [x[1] for x in filtered]
    }

    return {
        'setups': setups,
        'name': 'bananer-translationese',
        'language': 'swedish'
    }


def load_swedish_idioms():
    df = pd.read_csv(CSV_DIR / "swedish_idioms.csv")

    filtered = [
        (row['Positive (Minimal Pairs Setup)'], row['Negative (Minimal Pairs Setup)'])
        for _, row in df.iterrows()
    ]

    correct, wrong = zip(*filtered) if filtered else ([], [])

    return {
        'correct': list(correct),
        'wrong': list(wrong),
        'name': 'swedish-idioms',
        'language': 'swedish'
    }

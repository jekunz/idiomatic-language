from datasets import load_dataset, concatenate_datasets


def load_scala_swedish():
    ds = load_dataset("jekunz/scala_sv_minpairs")
    all_data = concatenate_datasets([ds['train'], ds['validation'], ds['test']])

    filtered_flip = [
        (row['correct'], row['incorrect'])
        for row in all_data
        if row['type'] == 'flip_neighbours'
    ]

    filtered_delete = [
        (row['correct'], row['incorrect'])
        for row in all_data
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
    ds = load_dataset("jekunz/dalaj_minpairs_sim")
    all_data = concatenate_datasets([ds['train'], ds['validation'], ds['test']])

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
            for row in all_data
            if row['meta']['error_label'] == error_label
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

    ds = load_dataset("jekunz/bananer")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    setups = {}

    filtered = [(row['Alternativ'], row['Översättningssvenska']) for row in ds['train']]
    setups['all'] = {
        'correct': [x[0] for x in filtered],
        'wrong': [x[1] for x in filtered]
    }

    filtered = [
        (row['Alternativ'], row['Översättningssvenska'])
        for row in ds['train']
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
        for row in ds['train']
        if row['Val'] == 'Alternativ'
    ]
    setups['manual_only'] = {
        'correct': [x[0] for x in filtered],
        'wrong': [x[1] for x in filtered]
    }

    filtered = [
        (row['Alternativ'], row['Översättningssvenska'])
        for row in ds['train']
        if len(tokenizer(text=str(row["Alternativ"]))["input_ids"]) ==
           len(tokenizer(text=str(row["Översättningssvenska"]))["input_ids"])
    ]
    setups['token_length_only'] = {
        'correct': [x[0] for x in filtered],
        'wrong': [x[1] for x in filtered]
    }

    filtered = [
        (row['Alternativ'], row['Översättningssvenska'])
        for row in ds['train']
        if len(row["Alternativ"].split()) == len(row["Översättningssvenska"].split())
        and row['Val'] == 'Alternativ'
    ]
    setups['whitespace_and_manual'] = {
        'correct': [x[0] for x in filtered],
        'wrong': [x[1] for x in filtered]
    }

    filtered = [
        (row['Alternativ'], row['Översättningssvenska'])
        for row in ds['train']
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
    ds = load_dataset("liu-nlp/swedish-idioms")

    filtered = [
        (row['Positive (Minimal Pairs Setup)'], row['Negative (Minimal Pairs Setup)'])
        for row in ds['train']
    ]

    correct, wrong = zip(*filtered) if filtered else ([], [])

    return {
        'correct': list(correct),
        'wrong': list(wrong),
        'name': 'swedish-idioms',
        'language': 'swedish'
    }

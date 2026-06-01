import pandas as pd
from datasets import load_dataset

from train import get_or_build_tokenizer

config = {
    "datasource": 'opus_books',
    "lang_src": "en",
    "lang_tgt": "it",
    "tokenizer_file": "tokenizer_{}.json"
}

dataset = load_dataset("opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train")

tokenizer_src = get_or_build_tokenizer(config, dataset, config['lang_src'])
tokenizer_tgt = get_or_build_tokenizer(config, dataset, config['lang_tgt'])

# Analyze the length of the sentences in the dataset
lengths_src = []
lengths_tgt = []

for item in dataset:
    src = item['translation'][config['lang_src']]
    tgt = item['translation'][config['lang_tgt']]

    lengths_src.append(len(tokenizer_src.encode(src).ids))
    lengths_tgt.append(len(tokenizer_tgt.encode(tgt).ids))

lengths_src.sort()
lengths_tgt.sort()

print("SRC")
print("max:", max(lengths_src))
print("mean:", sum(lengths_src)/len(lengths_src))
print("p95:", lengths_src[int(0.95 * len(lengths_src))])
print("p99:", lengths_src[int(0.99 * len(lengths_src))])

print("TGT")
print("max:", max(lengths_tgt))
print("mean:", sum(lengths_tgt)/len(lengths_tgt))
print("p95:", lengths_tgt[int(0.95 * len(lengths_tgt))])
print("p99:", lengths_tgt[int(0.99 * len(lengths_tgt))])

# Flatten the 'translation' column into a DataFrame
df = pd.DataFrame(dataset['translation'])

# Calculate max character lengths
max_char_en = df[config['lang_src']].str.len().max()
max_char_it = df[config['lang_tgt']].str.len().max()

# Calculate max word counts
max_word_en = df[config['lang_src']].str.split().str.len().max()
max_word_it = df[config['lang_tgt']].str.split().str.len().max()

print(f"Max English words: {max_word_en}, Max Italian words: {max_word_it}")
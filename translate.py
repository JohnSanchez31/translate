from pathlib import Path
from config import get_config, latest_weights_file_path 
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset, causal_mask
import torch
import sys

from train import greedy_decode

def translate(sentence: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config = get_config()

    tokenizer_src = Tokenizer.from_file(
        str(Path(config['tokenizer_file'].format(config['lang_src'])))
    )
    tokenizer_tgt = Tokenizer.from_file(
        str(Path(config['tokenizer_file'].format(config['lang_tgt'])))
    )

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"]
    ).to(device)

    # Load checkpoint
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    label = ""

    # Optional: use dataset example by index
    if isinstance(sentence, int) or (isinstance(sentence, str) and sentence.isdigit()):
        idx = int(sentence)

        ds = load_dataset(
            config['datasource'],
            f"{config['lang_src']}-{config['lang_tgt']}",
            split='all'
        )

        ds = BilingualDataset(
            ds,
            tokenizer_src,
            tokenizer_tgt,
            config['lang_src'],
            config['lang_tgt'],
            config['seq_len']
        )

        sentence = ds[idx]['src_text']
        label = ds[idx]['tgt_text']

    model.eval()

    with torch.no_grad():

        # Tokenize source sentence
        src_tokens = tokenizer_src.encode(sentence).ids

        source = torch.cat([
            torch.tensor(
                [tokenizer_src.token_to_id('[SOS]')],
                dtype=torch.int64
            ),
            torch.tensor(src_tokens, dtype=torch.int64),
            torch.tensor(
                [tokenizer_src.token_to_id('[EOS]')],
                dtype=torch.int64
            ),
            torch.tensor(
                [tokenizer_src.token_to_id('[PAD]')] *
                (config['seq_len'] - len(src_tokens) - 2),
                dtype=torch.int64
            )
        ])

        # Match validation shapes
        source = source.unsqueeze(0).to(device)  # (1, seq_len)

        source_mask = (
            (source != tokenizer_src.token_to_id('[PAD]'))
            .unsqueeze(1)
            .unsqueeze(2)
            .int()
            .to(device)
        )  # (1,1,1,seq_len)

        output_tokens = greedy_decode(
            model=model,
            source=source,
            source_mask=source_mask,
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
            max_len=config['seq_len'],
            device=device
        )

        output_text = tokenizer_tgt.decode(
            output_tokens.detach().cpu().numpy()
        )

        print(f"{'SOURCE: ':>12}{sentence}")

        if label:
            print(f"{'TARGET: ':>12}{label}")

        print(f"{'PREDICTED: ':>12}{output_text}")

        return output_text
    
#read sentence from argument
translate(sys.argv[1] if len(sys.argv) > 1 else "I am not a very good a student.")
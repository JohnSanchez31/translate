import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_config, get_weights_file_path, latest_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

def greedy_decode(
    model,
    source,
    source_mask,
    tokenizer_src, 
    tokenizer_tgt,
    max_len,
    device
):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every token we get from the
    # decoder
    encoder_output = model.encode(source, source_mask)

    # Initilize the decoder input with the sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) >= max_len:
            print("Reached max length, stopping decoding")
            break

        # Build mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:, -1])
        # Select the token with the max probability (because it's a greedy search)
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model, 
    validation_ds, 
    tokenizer_src, 
    tokenizer_tgt,
    max_len,
    device, 
    print_msg, 
    global_step, 
    writer, 
    loss_fn,
    num_examples=2
):
    model.eval()
    count = 0

    # Size of the control window (just use a default value)
    console_width = 80

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in validation_ds:

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            # Forward pass
            encoder_output = model.encode(
                encoder_input,
                encoder_mask
            )

            decoder_output = model.decode(
                encoder_output,
                encoder_mask,
                decoder_input,
                decoder_mask
            )

            proj_output = model.project(decoder_output)

            loss = loss_fn(
                proj_output.view(
                    -1,
                    tokenizer_tgt.get_vocab_size()
                ),
                label.view(-1)
            )

            total_loss += loss.item()
            total_batches += 1

            count += 1

            if count <= num_examples:
                assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

                model_out = greedy_decode(
                    model, 
                    encoder_input, 
                    encoder_mask, 
                    tokenizer_src, 
                    tokenizer_tgt, 
                    max_len, 
                    device
                )

                source_text = batch['src_text'][0]
                target_text = batch['tgt_text'][0]
                model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

                # Print on the console
                print_msg('-'*console_width)
                print_msg(f"SOURCE: {source_text}")
                print_msg(f"TARGET: {target_text}")
                print_msg(f"PREDICTED: {model_out_text}")


        
        val_loss = val_loss = total_loss / max(total_batches, 1)

        writer.add_scalar(
            'validation_loss',
            val_loss,
            global_step
        )

        print_msg(f"Validation Loss: {val_loss:.4f}")

        return val_loss


def run_examples(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        # min_frecuency: For a word to appear in our vocabulary is the min frecuency
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)

        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer


def filter_dataset(ds, tokenizer_src, tokenizer_tgt, config):
    filtered_examples = []
    for item in ds:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids

        if len(src_ids) <= config['seq_len'] - 2 and len(tgt_ids) <= config['seq_len'] - 1:
            filtered_examples.append(item)

    removed = len(ds) - len(filtered_examples)
    print(f"Removed {removed} examples ({100*removed/len(ds):.2f}%)")
    
    return filtered_examples

def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split="train")

    # Build the tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    filtered_ds = filter_dataset(ds_raw, tokenizer_src, tokenizer_tgt, config)

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(filtered_ds))
    valid_ds_size = len(filtered_ds) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(filtered_ds, [train_ds_size, valid_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src, 
        tokenizer_tgt, 
        config['lang_src'], 
        config["lang_tgt"], 
        config['seq_len']
    )

    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src, 
        tokenizer_tgt, 
        config['lang_src'], 
        config["lang_tgt"], 
        config['seq_len']
    )

    max_len_src = 0
    max_len_tgt = 0

    for item in filtered_ds:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    
    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len, 
        vocab_tgt_len, 
        config['seq_len'], 
        config['seq_len'], 
        config['d_model']
    )
    return model


def train_model(config):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(
        config, 
        tokenizer_src.get_vocab_size(), 
        tokenizer_tgt.get_vocab_size()
    ).to(device)

    # Tensorloss (allows you to watch the loss the graph)
    writer = SummaryWriter(config["experiment_name"])

    # Change to 1.0 because the scheduler will adjust it, and we don't want to start with a very low learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9) 
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, min_lr=1e-6)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = (
        latest_weights_file_path(config) 
        if preload == 'latest' 
        else get_weights_file_path(config, preload) if preload else None
    )
    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        # scheduler.load_state_dict(state['scheduler_state_dict']) # Loading the scheduler
        global_step = state['global_step']
    else:
        print("No model to preload, training from scratch")

    
    # For loss function we are using the CrossEntropyLoss
    # Label_smoothing
    loss_fn = nn.CrossEntropyLoss(
        # ignore_index=tokenizer_src.token_to_id('[PAD]'), 
        ignore_index=tokenizer_tgt.token_to_id('[PAD]'), # Ignore the padding token in the target language
        label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache() # Clear the cache to avoid OOM errors
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1 Seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)    
            proj_output = model.project(decoder_output) # (B, seq_len, tgt_vocab_size)

            # Compare the output with the label and calculate the loss
            label = batch['label'].to(device) # (B, seq_len)

            # (B, seq_len, tgt_vocab_size) --> (B * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"}) # Show loss on progress bar

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Gradient clipping (to avoid exploding gradients)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            # Update the weights 
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1


        if epoch % 2 == 0:
            # Run validation every 2 epochs (you can change this value if you want to run it more or less often)
            val_loss = run_validation(
                model, 
                val_dataloader, 
                tokenizer_src, 
                tokenizer_tgt,
                config['seq_len'], 
                device, 
                lambda msg: batch_iterator.write(msg), 
                global_step, 
                writer,
                loss_fn,
                num_examples=3
            )

            # scheduler.step(val_loss) # Step the scheduler with the validation loss


        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    # warnings.filterwarnings('ignore') # see this at least once

    config = get_config()
    train_model(config)



from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 10,
        "lr": 10**-4,
        "seq_len": 128,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "es",
        "model_folder": "models/en_es_v8",
        "model_basename": "en_es_v8_",
        "preload": "05",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/en_es_v8_"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename) # How this works???

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split

import dataset
from model import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]

def get_or_create_tokenizer(config, dataset, lang):
    # Construct the path for the tokenizer file based on the provided language
    # Example: config['tokenizer_file'].format(lang) = 'tokenizer_{lang}.json'
    path = Path(config['tokenizer_file'].format(lang))
    
    # Check if the tokenizer file already exists
    if not Path.exists(path):
        # If the tokenizer file does not exist, create a new tokenizer
        tokenizer = Tokenizer(WordLevel(unkown_token="[UNK]"))  # Initialize tokenizer with unknown token
        tokenizer.pre_tokenizer = Whitespace()  # Set pre-tokenization method to whitespace

        # Initialize the trainer with special tokens and minimum frequency for training
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        
        # Train the tokenizer on the provided dataset files
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        
        # Save the trained tokenizer model to a file
        tokenizer.save(str(path))
    else:
        # If the tokenizer file already exists, load it
        tokenizer = Tokenizer.from_file(str(path))
        
    return tokenizer  # Return the created tokenizer

def get_ds(config, tokenizer: Tokenizer, src_lang: str, target_lang: str):
    # Load the dataset from Hugging Face using the specified source and target language
    dataset = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # Create tokenizers for both source and target languages using the provided configuration
    tokenizer_src = get_or_create_tokenizer(config, dataset, src_lang)
    tokenizer_tgt = get_or_create_tokenizer(config, dataset, target_lang)

    # Calculate the sizes for training and validation datasets (90% for training, 10% for validation)
    train_ds_size = int(len(dataset) * 0.9)  # Size of the training dataset
    val_ds_size = len(dataset) - train_ds_size  # Size of the validation dataset
    train_ds, val_ds = random_split(dataset, [train_ds_size, val_ds_size])  # Split the dataset

    # Create a bilingual dataset for training and validation
    train_ds = BillingualDataset(train_ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BillingualDataset(val_ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Initialize variables to track the maximum lengths of source and target texts
    max_len_src = 0
    max_len_tgt = 0

    # Iterate through the dataset to find the maximum lengths of source and target texts
    for item in dataset:
        src_text = item['translation'][config['lang_src']]  # Get source text
        tgt_text = item['translation'][config['lang_tgt']]  # Get target text
        src_ids = tokenizer_src.encode(src_text).ids  # Encode source text to IDs
        tgt_ids = tokenizer_tgt.encode(tgt_text).ids  # Encode target text to IDs

        # Update maximum lengths if current lengths are greater
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    # Print the maximum lengths of source and target texts
    print(f"Maximum length of source text: {max_len_src}")
    print(f"Maximum length of target text: {max_len_tgt}")

    # Create data loaders for training and validation datasets
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)  # Training data loader
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)  # Validation data loader

    # Return the data loaders and tokenizers for source and target languages
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(src_vocab_size, tgt_vocab_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config, train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt):
    pass
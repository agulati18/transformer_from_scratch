import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split
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

    # Map the training dataset to include tokenized input and label for both source and target languages
    train_ds = dataset['train'].map(
        lambda x: {
            'input': tokenizer.encode(x[src_lang]).ids,  # Tokenized input for source language
            'label': tokenizer.encode(x[target_lang]).ids  # Tokenized label for target language
        },
        remove_columns=[src_lang, target_lang]  # Remove original columns from the dataset
    )
    
    return dataset, train_ds  # Return the original dataset and the processed training dataset
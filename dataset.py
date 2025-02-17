import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
       super().__init__()

       # Initialize dataset and language parameters
       self.dataset = dataset
       self.src_lang = src_lang
       self.tgt_lang = tgt_lang
       self.seq_len = seq_len
       
       # Initialize tokenizers for source and target languages
       self.tokenizer_src = tokenizer_src
       self.tokenizer_tgt = tokenizer_tgt
       
       # Define special tokens for start of sequence, end of sequence, and padding
       self.sos_token = torch.tensor([tokenizer_src.token_to_id('<sos>')], dtype=torch.int64)
       self.eos_token = torch.tensor([tokenizer_src.token_to_id('<eos>')], dtype=torch.int64)
       self.pad_token = torch.tensor([tokenizer_src.token_to_id('<pad>')], dtype=torch.int64)

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.dataset)

    def __getitem__(self, idx: Any) -> Any:
        # Retrieve the source-target pair from the dataset using the provided index
        src_target_pair = self.dataset[idx] 

        # Extract source and target texts based on the specified source and target languages
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Encode the source and target texts, adding start and end tokens
        enc_input = [self.sos_token] + self.tokenizer_src.encode(src_text).ids + [self.eos_token]
        dec_input = [self.sos_token] + self.tokenizer_tgt.encode(tgt_text).ids 

        # Calculate the number of padding tokens needed for encoder and decoder inputs
        enc_num_padding_tokens = self.seq_len - len(enc_input) - 2  # -2 for the sos and eos tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input) - 1  # -1 for the sos token

        # Raise an error if the input sentences exceed the maximum length
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Pad the encoder and decoder inputs with the pad token
        enc_input = enc_input + [self.pad_token] * enc_num_padding_tokens
        dec_input = dec_input + [self.pad_token] * dec_num_padding_tokens
        
        # Prepare the label for the target text, including the end token and padding
        label = self.tokenizer_tgt.encode(tgt_text).ids + [self.eos_token] + [self.pad_token] * dec_num_padding_tokens

        # Ensure the final tensor sizes match the expected sequence length
        assert len(enc_input) == self.seq_len, "Encoder input size mismatch"
        assert len(dec_input) == self.seq_len, "Decoder input size mismatch"
        assert len(label) == self.seq_len, "Label size mismatch"

        return {
            "enc_input": torch.tensor(enc_input, dtype=torch.int64),
            "dec_input": torch.tensor(dec_input, dtype=torch.int64),
            "label": torch.tensor(label, dtype=torch.int64),
        }

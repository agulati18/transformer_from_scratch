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
            # Convert the encoded source input to a tensor
            "enc_input": torch.tensor(enc_input, dtype=torch.int64),  # (Seq_Len number of tokens)
            
            # Convert the encoded target input to a tensor
            "dec_input": torch.tensor(dec_input, dtype=torch.int64),  # (Seq_Len number of tokens)
            
            # Create a mask for the encoder input to identify padding tokens
            "encoder_mask": (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # Mask where pad tokens are 0
            
            # Create a mask for the decoder input to identify padding tokens and apply causal masking
            "decoder_mask": (dec_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(dec_input.size(0)),  # Mask where pad tokens are 0, with causal masking
            
            # Convert the target labels to a tensor, including the end token and padding
            "label": torch.tensor(label, dtype=torch.int64)  # (Seq_Len number of tokens)

            # Store the original source and target texts for debugging
            "src_text": src_text,
            "tgt_text": tgt_text
        }

    def causal_mask(size):
        mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
        # Create a 2D tensor of shape (1, size, size) filled with ones.
        # Use torch.triu to get the upper triangular part of the tensor,
        # setting all elements below the diagonal to zero. The diagonal itself is excluded
        # from the upper triangular part due to diagonal=1.

        return mask == 0
        # Return a boolean mask where elements that are zero in the 'mask' tensor are marked as True,
        # and elements that are one are marked as False. This indicates which positions in the sequence
        # can be attended to during decoding, preventing the model from attending to future tokens.
    


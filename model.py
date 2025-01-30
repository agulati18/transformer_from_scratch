import torch
import torch.nn 
import math

class InputEmbedding(nn.Module):
    """
    NN module that converts token indices into continuous vector representations (embeddings).
    Commonly used as the first layer in transformer-based models.
    """
    def __init__(self, d_model, vocab_size):
        """
        Args:
            d_model: The dimension of the embedding vectors (how many features each token will be represented by)
            vocab_size: The size of the vocabulary (how many different tokens can be embedded)
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # PyTorch's embedding layer that converts indices into dense vectors. Mapping each token to a dense vector of size d_model.
        self.embedding = nn.Embedding(vocab_size: int, d_model: int)

    def forward(self, x):
        """
        Convert input tokens to embeddings and apply scaling factor.
        
        Args:
            x: Tensor of token indices
            
        Returns:
            Scaled embedding vectors. Shape: (batch_size, sequence_length, d_model)
        """
        # Convert tokens to embeddings and multiply by sqrt(d_model); from Section 3.4 of the Transformer paper
        # This scaling prevents dot products in attention from growing too large
        return self.embedding(x) * math.sqrt(self.d_model) 

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings using sinusoidal (sin/cos) functions.
    This allows the model to understand the order/position of tokens in the sequence.
    The encoding uses alternating sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None: 
        super().__init__()
        self.d_model = d_model  # Dimension of the model/embeddings
        self.seq_len = seq_len  # Maximum sequence length
        self.dropout = nn.Dropout(dropout)
        
        # This parameter isn't actually used - can be removed
        self.encoding = nn.Parameter(torch.zeros(seq_len, d_model))
        self.encoding.requires_grad = False

        # Initialize positional encoding matrix
        pe = torch.zeros(seq_len, d_model)
        
        # Create position vector: [[0], [1], [2], ..., [seq_len-1]]
        # Shape: (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 
        
        # Calculate division terms for the positional encoding formula
        # For each dimension i: 1/(10000^(2i/d_model))
        # This creates wavelengths forming a geometric progression from 2π to 10000·2π
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # Fill the positional encoding matrix:
        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices get sin
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices get cos
        
        # Add batch dimension for broadcasting
        # Shape changes from (seq_len, d_model) to (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (won't be updated during training but saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input embeddings.
        
        Args:
            x: Input embeddings. Shape: (batch_size, seq_len, d_model)
            
        Returns:
            Embeddings with positional information. Shape: (batch_size, seq_len, d_model)
        """
        # Add positional encoding to embeddings
        # :x.shape[1] handles variable sequence lengths up to seq_len
        x = x + self.pe[:, :x.shape[1], :]
        
        # Apply dropout and return
        return self.dropout(x)


import torch
import torch.nn 
import math

class InputEmbedding(nn.Module):
    """
    A neural network module that converts token indices into continuous vector representations (embeddings).
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
        # PyTorch's embedding layer that converts indices into dense vectors
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Convert input tokens to embeddings and apply scaling factor.
        
        Args:
            x: Tensor of token indices
            
        Returns:
            Scaled embedding vectors. Shape: (batch_size, sequence_length, d_model)
        """
        # Convert tokens to embeddings and multiply by sqrt(d_model)
        # This scaling prevents dot products in attention from growing too large
        return self.embedding(x) * math.sqrt(self.d_model)
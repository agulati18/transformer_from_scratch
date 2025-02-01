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
        # Visualization of the process:
        # Input tokens (indices):  [5, 2, 941, 12, ...]
        #                           ↓   ↓    ↓    ↓
        # Embedding lookup:       [v₅, v₂, v₉₄₁, v₁₂, ...] where each v is a d_model dimensional vector
        #                           ↓   ↓    ↓    ↓
        # Scaling:               [v₅, v₂, v₉₄₁, v₁₂, ...] * √(d_model)
        
        embeddings = self.embedding(x) * math.sqrt(self.d_model)

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
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices get sin, starting at 0 to the end of the sequence, incrementing by 2 [0,2,4,6,8]
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices get cos, starting at 1 to the end of the sequence, incrementing by 2 [1,3,5,7,9]
        
        # We will have a batch of sequences, so add batch dimension 
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
        # Visualization of the process:
        # Input embeddings:     [e₁,    e₂,    e₃,    e₄,    ...]  # each e is a d_model vector
        # Positional encoding: +[pos₁,  pos₂,  pos₃,  pos₄,  ...]  # each pos is a d_model vector
        #                       ↓      ↓      ↓      ↓
        # Result:              [e₁+p₁, e₂+p₂, e₃+p₃, e₄+p₄, ...]
        # 
        # Where each posᵢ is composed of sine and cosine waves:
        # posᵢ = [sin(i/10000⁰), cos(i/10000⁰), sin(i/10000²), cos(i/10000²), ...]
        
        # Add positional encodings to the input embeddings using broadcasting:
        # - self.pe has shape (1, max_seq_len, d_model)
        # - We slice it to match x's sequence length: self.pe[:, :x.shape[1], :]
        # - The slice operation keeps all batches (:), matches sequence length (:x.shape[1]), keeps all dimensions (:)
        # - Broadcasting: the batch dim (1) of positional encodings expands to match x's batch size
        # - Example shapes for batch_size=32, seq_len=50, d_model=512:
        #   * x shape:        (32, 50, 512)
        #   * self.pe slice:  (1,  50, 512) -> broadcasts to (32, 50, 512)
        #   * result shape:   (32, 50, 512)
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

class LayerNormalization(nn.Module):
    """
    Applies layer normalization to the input tensor.
    """
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Scale factor (multiplicative); using Parameter to make it trainable/learnable (nn.Parameter)
        self.bias = nn.Parameter(torch.zeros(1)) # Bias term (additive)

    def forward(self, x):
        """
        Apply layer normalisation to the input tensor.
        
        Args:
            x: Input tensor. Shape: (batch_size, seq_len, d_model)
            
        Returns:
            Normalized tensor. Shape: (batch_size, seq_len, d_model)
        """
        # Visualization of the process:
        # Input:         [2.0,  -1.0,  5.0,   0.0]
        # Mean:          1.5 (mean of all values)
        # Std:          2.5 (standard deviation)
        # Normalize:    [(2.0-1.5)/2.5, (-1.0-1.5)/2.5, (5.0-1.5)/2.5, (0.0-1.5)/2.5]
        #              = [0.2,  -1.0,    1.4,   -0.6]
        # Scale+Shift:  [0.2α + β, -1.0α + β, 1.4α + β, -0.6α + β]  where α=self.alpha, β=self.bias
        
        mean = x.mean(dim = -1, keepdim=True) # Mean of the last dimension (dim = -1)
        std = x.std(dim = -1, keepdim=True) # Standard deviation of the last dimension (dim = -1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module):
    """
    Implements the feed-forward part of the transformer layer. Fully connected layers. Model uses this both in encoder and decoder.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # First linear layer, W1 & b1. Bias argument is set to True by default for nn.Linear; no need to specify
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # Second linear layer, W2 & b2. Bias argument is set to True by default for nn.Linear; no need to specify
    
    def forward(self, x):
        """
        Forward pass through the feed-forward layer.
        
        Args:
            x: Input tensor. Shape: (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor. Shape: (batch_size, seq_len, d_model)
        """
        
        # Visualization of the process:
        # Input:         [x₁, x₂, x₃, x₄, ...] # shape of (batch_size, seq_len, d_model)
        # Linear 1:     [W₁x₁ + b₁, W₁x₂ + b₁, W₁x₃ + b₁, W₁x₄ + b₁, ...] # shape of (batch_size, seq_len, d_ff)
        # ReLU:         [ReLU(W₁x₁ + b₁), ReLU(W₁x₂ + b₁), ReLU(W₁x₃ + b₁), ReLU(W₁x₄ + b₁), ...] # shape of (batch_size, seq_len, d_ff)
        # Dropout:      [dropout(ReLU(W₁x₁ + b₁)), dropout(ReLU(W₁x₂ + b₁)), ...] # shape of (batch_size, seq_len, d_ff)
        # Linear 2:     [W₂(dropout(ReLU(W₁x₁ + b₁))) + b₂, W₂(dropout(ReLU(W₁x₂ + b₁))) + b₂, ...] # shape of (batch_size, seq_len, d_model)
        
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) # Function Composition of Layers through the feed-forward pass

class MultiHeadAttention(nn.Module):
    """
    Implements the multi-head attention mechanism.
    """
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h" # Ensures that the number of dimensions in the input can be evenly divided into heads (no remainder)

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
        
        self.w_o = nn.Linear(d_model, d_model) # Wo, h * d_v = d_model - d_v is the same as d_k (d_model/h). Given this, d_v = d_k = d_model/h, so h * d_v = d_model

        self.dropout = nn.Dropout(dropout)
    
    @staticmethod # Static method is a method that belongs to the class itself, not an instance of the class. So it can be called without creating an instance of the class.
    def attention(query, key, value, mask=None, dropout: nn.Dropout):
        """
        Compute scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
        
        Args:
            query: Query matrix (Q). Shape: (..., seq_len_q, d_k)
            key: Key matrix (K). Shape: (..., seq_len_k, d_k)
            value: Value matrix (V). Shape: (..., seq_len_k, d_v)
            mask: Optional mask to prevent attention to certain positions. Shape: (..., seq_len_q, seq_len_k)
            dropout: Optional dropout layer
            
        Returns:
            tuple: (weighted sum of values, attention weights)
        """
        # Get the dimension of the key vectors (used for scaling)
        d_k = query.size(-1)
        
        # Step 1: Calculate attention scores
        # - Multiply Q with K^T (transpose) to get raw attention scores
        # - Scale by 1/√d_k to prevent softmax from having extremely small gradients
        # Shape: (..., seq_len_q, seq_len_k)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # Transposes the last two dimensions of the key matrix, meaning that it swaps the last two axes. This is done to align the dimensions for the matrix multiplication
        
        # Step 2: Apply mask (if provided)
        # - Replace scores with -inf where mask is 0
        # - This ensures these positions will have ~0 probability after softmax
        # - Purpose: Ensures that the model does not attend to padding tokens (which are added to make sequences the same length) and prevents a token from attending to future tokens in decoder self-attention
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 3: Apply softmax to get attention weights
        # - Convert scores to probabilities (0 to 1)
        # - Each query will have a probability distribution over all keys
        attention_scores = attention_scores.softmax(dim=-1) # Applies the softmax function to the last dimension of the attention scores tensor
        
        # Step 4: Apply dropout (if provided)
        # - Randomly zero out some attention weights during training
        # - Helps prevent overfitting
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # Step 5: Multiply attention weights with values
        # - Weighted sum of values based on attention weights
        # - First output: weighted values. Shape: (..., seq_len_q, d_v)
        # - Second output: attention weights (useful for visualization)
        return torch.matmul(attention_scores, value), attention_scores
    
    def forward(self, q, k, v, mask=None):
        """
        Multi-head attention forward pass.
        
        Args:
            q, k, v: Query, Key, and Value tensors (batch_size, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # STEP 1: Linear Projections
        # Transform input tensors into query, key, and value representations
        # Creating 3 different "views" of the same input:
        # - Query (q): "What am I looking for?"
        # - Key (k): "What do I contain?"
        # - Value (v): "What information do I give if matched?"
        # Just like in a database query, we match queries with keys to determine which values are important
        query = self.w_q(q)  # Project query
        key = self.w_k(k)    # Project key
        value = self.w_v(v)  # Project value

        # STEP 2: Split heads
        # Reshape tensors to separate the heads dimension
        # Example for batch_size=32, seq_len=50, h=8, d_model=512:
        # Input shape:  (32, 50, 512)
        # Split shape:  (32, 50, 8, 64)  # 512 split into 8 heads of 64 dimensions
        # Final shape:  (32, 8, 50, 64)  # Transpose to put heads dimension second
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # STEP 3: Apply Attention
        # Calculate attention scores and get weighted values
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # STEP 4: Merge heads
        # Reshape from separate heads back to single d_model dimension
        # (batch_size, heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2)                         # Move seq_len back to position 2
        x = x.contiguous()                            # Ensure tensor is contiguous in memory
        x = x.view(x.shape[0], -1, self.h * self.d_k) # Combine heads

        # STEP 5: Final Linear Projection
        # Transform the merged heads back to d_model dimensions
        return self.w_o(x)

class ResidualConnection(nn.Module):
    """
    Implements the residual connection mechanism (skip connection). 
    Allows information to directly flow from earlier layers to later layers, bypassing some layers in between, 
    which helps to alleviate the vanishing gradient problem and enables training much deeper networks by facilitating 
    better gradient flow through the network

    Without the residual connections, a large part of the training signal would get lost during back-propagation. 
    Residual connections reduce effect because summation is linear with respect to derivative, so each residual block also gets a signal 
    that is not affected by the vanishing gradient

    Another effect of residual connections is that the information stays local in the Transformer layer stack. 
    The residual connections, however, always "remind" the representation of what the original state was.
    """
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Regularization to prevent overfitting. Randomly zeros out some values during training to prevent overfitting
        self.norm = LayerNormalization()    # Normalizes inputs to have zero mean and unit variance

    def forward(self, x, sublayer):
        """
        Applies sublayer to input, then adds the result to the original input.
        
        Args:
            x: Input tensor
            sublayer: Neural network layer/function to be applied (e.g., self-attention or feed-forward)
            
        Returns:
            Tensor after applying sublayer and residual connection: x + dropout(sublayer(x))
        """
        return x + self.dropout(sublayer(x))  # Skip connection: Original input + Transformed input
    
class EncoderBlock(nn.Module):
    """
    A single encoder block in the transformer architecture.
    Each encoder block performs these operations in sequence:
    1. Multi-head self-attention
    2. Add & Normalize (residual connection)
    3. Feed-forward neural network
    4. Add & Normalize (residual connection)
    """
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        # First residual connection: after self-attention
        self.residual_connection_1 = ResidualConnection(dropout)
        # Multi-head attention layer
        self.attention = self_attention_block
        # Second residual connection: after feed-forward
        self.residual_connection_2 = ResidualConnection(dropout)
        # Feed-forward neural network
        self.feed_forward = feed_forward_block

    def forward(self, x, src_mask):
        """
        Process input through one encoder block.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            src_mask: Mask for padding tokens (1 for tokens, 0 for padding)
            
        Flow:
            1. Self-attention: Each token attends to all other tokens
            2. Residual connection + normalisation
            3. Feed-forward network
            4. Residual connection + normalisation
        """
        # Step 1 & 2: Self-attention with residual connection
        # - In self-attention, query, key, and value all come from the same source (x)
        # - That's why we pass x three times: (x, x, x)
        x = self.residual_connection_1(x, lambda x: self.attention(x, x, x, src_mask))
        
        # Step 3 & 4: Feed-forward with residual connection
        # - Transform each token's representation independently
        x = self.residual_connection_2(x, self.feed_forward)
        return x

class Encoder(nn.Module):
    """
    Complete encoder consisting of multiple encoder blocks stacked on top of each other.
    The encoder processes the entire input sequence and produces contextual representations
    for each token that incorporate information from the entire sequence.
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        # Stack of encoder blocks
        self.layers = layers
        # Final layer normalization
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        """
        Process input through all encoder blocks in sequence.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Padding mask
            
        Returns:
            Contextual representations for each token
        """
        # Pass input through each encoder block sequentially
        for layer in self.layers:
            x = layer(x, mask)
        # Final normalization before output
        return self.norm(x)

class DecoderBlock(nn.Module):
    """
    A single decoder block in the transformer architecture.
    Each decoder block contains:
    1. Masked self-attention (prevents attending to future tokens)
    2. Cross-attention (attends to encoder output)
    3. Feed-forward neural network
    Each of these has its own residual connection
    """
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        # Self-attention layer: allows decoder to attend to previous positions in decoder
        self.self_attention_block = self_attention_block
        # Cross-attention layer: allows decoder to attend to encoder's output
        self.cross_attention_block = cross_attention_block
        # Feed-forward network for additional processing
        self.feed_forward_block = feed_forward_block
        # Three residual connections, one for each sub-layer
        self.residual_connection_1 = ResidualConnection(dropout)  # For self-attention
        self.residual_connection_2 = ResidualConnection(dropout)  # For cross-attention
        self.residual_connection_3 = ResidualConnection(dropout)  # For feed-forward
    
    def forward(self, x, enc_output, src_mask, target_mask):
        """
        Process input through one decoder block.
        
        Args:
            x: Input tensor from previous decoder layer or embedding
            enc_output: Output from the encoder
            src_mask: Mask for encoder output (padding mask)
            target_mask: Mask for decoder self-attention (combines padding and future mask)
        """
        # Step 1: Self-attention with masking
        # Uses target_mask to prevent attending to future tokens and padding
        x = self.residual_connection_1(x, lambda x: self.self_attention_block(x, x, x, target_mask))
        
        # Step 2: Cross-attention with encoder output
        # Query comes from decoder (x), Key and Value come from encoder (enc_output)
        # This is how the decoder "looks at" the encoded source sequence
        x = self.residual_connection_2(x, lambda x: self.cross_attention_block(x, enc_output, enc_output, src_mask))
        
        # Step 3: Feed-forward network
        # Final processing of the attention outputs
        x = self.residual_connection_3(x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    """
    Complete decoder consisting of multiple decoder blocks stacked on top of each other.
    The decoder generates the output sequence one token at a time, using:
    - Previously generated tokens
    - Encoder's output
    - Masking to prevent attending to future tokens
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        # Stack of decoder blocks
        self.layers = layers
        # Final layer normalization
        self.norm = LayerNormalization()

    def forward(self, x, enc_output, src_mask, target_mask):
        """
        Process input through all decoder blocks in sequence.
        
        Args:
            x: Input tensor (typically embedded target tokens)
            enc_output: Output from the encoder
            src_mask: Mask for encoder output (padding mask)
            target_mask: Mask for decoder self-attention
        """
        # Pass through each decoder block sequentially
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, target_mask)
        # Final normalization before output
        return self.norm(x)
import torch
from torch import nn

# Custom child class of nn.Module, where we set the MultiheadAttention and forward pass methods but inherit everything else
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        # Sets the internal Multihead attention method
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
    # Setting the attention behaviour through the forward pass method
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Identify padding tokens
        key_padding_mask = attention_mask == 0 
        seq_len = x.size(1)
        # A lower triangular matrix sets causal attention
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        # Run a forward pass of the MultiheadAttention method over the batched token representation tensor x, produced in the previous layer
        out, _ = self.attn(
            x, # query projection produced here
            x, # key
            x, # value 
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, d_hidden: int) -> None:
        super().__init__() # Initialise from parent class, then we will replace a few methods
        # Set layer normalisation, attention and MLP methods
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )
    # In this transformer block, attention and MLP layers will update the current sequence representations
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class TinyCausalTransformer(nn.Module):
    # Set defining variables of the exact transformer layout to use
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        d_hidden: int = 512, # Width of intermediary MLP layers between input and output of each transformer block
        dropout: float = 0.1,
    ) -> None:
        # Specify exact methods in parent class for this use case
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout, d_hidden) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        # Produces a positions tensor matching the dimensions of the current batch
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        # This positional embedding is then added to the initial token embedding
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        # Update the initial embedding sequentially through all the transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        # normalise
        x = self.ln_final(x)
        # Unembed to move from hidden state representation to weights over possible output tokens
        return self.unembed(x)

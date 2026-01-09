import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

class LinearAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Project queries, keys and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Linear attention
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            k = k * mask.unsqueeze(-1)
            v = v * mask.unsqueeze(-1)
        
        # Compute attention scores
        kv = torch.einsum('nhsd,nhsv->nhdv', k, v)
        z = 1.0 / (torch.einsum('nhld,nhd->nhl', q, k.sum(dim=2)) + 1e-6)
        out = torch.einsum('nhld,nhdv,nhd->nhlv', q, kv, z)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Project back and apply dropout
        out = self.out_proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = LinearAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, hidden_dim, dropout)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.feed_forward(self.norm2(x))
        return x

class LinearTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 500
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(embed_dim)
        self.bp_head = nn.Linear(embed_dim, 1)  # BP estimation head
        self.pulse_head = nn.Linear(embed_dim, 2)  # Pulse waveform prediction
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)
        
    def forward(
        self, 
        x: Tensor, 
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.size()
        
        # Project input
        x = self.input_proj(x)  # [batch_size, seq_len, embed_dim]
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Apply layer norm
        x = self.norm(x)
        
        # Get outputs
        bp = self.bp_head(x).squeeze(-1)  # [batch_size, seq_len]
        pulse = self.pulse_head(x)  # [batch_size, seq_len, 2]
        
        return bp, pulse

    def load_pretrained_attention_masks(self, lodestar_weights_path: str):
        """Load attention masks from a pretrained LodeSTAR model"""
        # This is a placeholder. You'll need to implement the actual loading logic
        # based on how the LodeSTAR model stores its attention masks
        print(f"Loading attention masks from {lodestar_weights_path}")
        # Example: Load the state dict and extract attention masks
        # state_dict = torch.load(lodestar_weights_path)
        # self.attention_mask = state_dict['attention_mask']
        pass
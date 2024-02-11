import torch
import torch.nn as nn
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pickle
i = 0
class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed_dim)  # Layer Normalization 추가

    def forward(self, x):
        return self.norm(self.transformer(x))  # Layer Normalization 적용

class PredictionLayer(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):
        return self.linear(self.dropout(x))

class AutoInt(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.transformer_block = TransformerBlock(args.embed_dim, args.num_heads, args.num_layers)
        self.prediction_layer = PredictionLayer(args.embed_dim, args.dropout)

    def forward(self, x):
        embed = self.embedding(x)
        out = self.transformer_block(embed)
        out = torch.sum(out, dim=1)
        _out = self.prediction_layer(out)
        
        return _out.squeeze(1), out
    

import torch
from torch import nn
from torch.nn import functional as F

class Transformer_Attention(nn.Module):
    """One head of self/cross-attention"""
    def __init__(self, embed_dim:int, masked:bool, dropout:float, kdim:int, vdim:int):
        super().__init__()

        self.WQ = nn.Linear(embed_dim, kdim, bias=False)
        self.WK = nn.Linear(embed_dim, kdim, bias=False)
        self.WV = nn.Linear(embed_dim, vdim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.masked = masked

    def forward(self, *x):
        # x[0] for WQ
        # x[1] or x[-1] for WK and WV
        assert x[0].size(-1) == x[-1].size(-1), "mismatched shapes"
        q = self.WQ(x[0]) # (B,T,kdim)
        k = self.WK(x[-1]) # (B,T,kdim)

        attention_act = q @ k.transpose(1, 2) * k.size(-1) ** -0.5
        attention_weights = self.dropout(F.softmax(attention_act, dim=-1)) # (B,T,T)

        if self.masked:
            tril = torch.tril(torch.ones_like(attention_weights, requires_grad=False)).to(x[-1].device)
            attention_weights.masked_fill(tril==0, float("-inf"))

        v = self.WV(x[-1]) # (B,T,vdim)
        ## Weighted sum
        out = attention_weights @ v
        return out
    

class MultiHead_TransformerAttention(nn.Module):
    """Multiple heads of self/cross-attention in parallel"""
    def __init__(self, embed_dim, num_heads, masked:bool, dropout=0.0, kdim=None, vdim=None):
        super().__init__()
        
        if not kdim:
            kdim = embed_dim

        if not vdim:
            vdim = embed_dim
        self.head_list = nn.ModuleList([Transformer_Attention(embed_dim, masked, dropout, kdim, vdim) for _ in range(num_heads)])
        self.WO = nn.Linear(num_heads*vdim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, *x):
        y = torch.cat([head(*x) for head in self.head_list], dim=-1)
        
        y = self.dropout(self.WO(y))
        return y
    

class FeedForward_Network(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                nn.ReLU(),
                                nn.Linear(dim_feedforward, d_model),
                                nn.Dropout(dropout))
    def forward(self, x):
        return self.ff(x)
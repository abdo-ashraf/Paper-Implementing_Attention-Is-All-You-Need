from torch import nn
from AttentionAndFeedForward import MultiHead_TransformerAttention, FeedForward_Network


class EncoderLayer(nn.Module):
    def __init__(self, d_model:int, nhead:int, dim_feedforward:int, dropout=0.0):
        super().__init__()
        head_features = d_model // nhead
        self.mha = MultiHead_TransformerAttention(d_model, nhead, False, dropout, head_features, head_features)
        self.ln_mha = nn.LayerNorm(d_model)

        self.ffn = FeedForward_Network(d_model, dim_feedforward, dropout)
        self.ln_ffn = nn.LayerNorm(d_model)
        
    def forward(self, x):
        ## (x +) is for residual connections
        ## It observed that doing layernorm before computation layers is better, but for now we only do re-implementing for the paper which do layernorm after computation layer.
        x = self.ln_mha(x + self.mha(x))
        x = self.ln_ffn(x + self.ffn(x))
        return x
    

class Encoder(nn.Module):
    def __init__(self, num_layers: int, d_model:int, nhead:int, dim_feedforward:int, dropout=0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        
    def forward(self, x):
        return self.blocks(x)
from torch import nn
from AttentionAndFeedForward import MultiHead_TransformerAttention, FeedForward_Network


class DecoderLayer(nn.Module):
    def __init__(self, d_model:int, nhead:int, dim_feedforward:int, dropout=0.0):
        super().__init__()
        head_features = d_model // nhead

        self.self_mha = MultiHead_TransformerAttention(d_model, nhead, True, dropout, kdim=head_features, vdim=head_features)
        self.ln_self_mha = nn.LayerNorm(d_model)

        ## Cross attention
        self.cross_mha = MultiHead_TransformerAttention(d_model, nhead, False, dropout, kdim=head_features, vdim=head_features)
        self.ln_cross_mha = nn.LayerNorm(d_model)

        self.ffn = FeedForward_Network(d_model, dim_feedforward, dropout)
        self.ln_ffn = nn.LayerNorm(d_model)

    def forward(self, *x):
        # x[0] should be decoder input embedings
        # x[1] should be encoder output embedings
        x_dec = x[0]
        x_enc = x[1] ## x[1] instead of x[-1] to raise an error incase user forgot to pass encoder rich-tokens

        x_dec = self.ln_self_mha(x_dec + self.self_mha(x_dec))
        oo = self.ln_cross_mha(x_dec + self.cross_mha(x_dec, x_enc))
        oo = self.ln_ffn(oo + self.ffn(oo))
        return oo
    

class Decoder(nn.Module):
    def __init__(self, num_layers: int, d_model:int, nhead:int, dim_feedforward:int, dropout=0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        
    def forward(self, *x):
        x_dec = x[0]
        x_enc = x[1]
        for block in self.blocks:
            x_dec = block(x_dec, x_enc)
        return x_dec
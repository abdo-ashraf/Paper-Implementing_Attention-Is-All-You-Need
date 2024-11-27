import torch
from torch import nn
from torch.nn import functional as F
from Decoder import Decoder
from Encoder import Encoder

## Hyperparameters (for test)
layers_repeat = 6
heads = 8 ## number of heads for each multi-head Attention
d_model = 512
dropout = 0.1
dim_feedforward = d_model*4
enc_vocab_size = 30000
dec_vocab_size = 40000


class Identity(nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, x):
        return torch.zeros_like(x)

class Transformer(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        self.encEmbed = nn.Embedding(enc_vocab_size, d_model)
        self.decEmbed = nn.Embedding(dec_vocab_size, d_model)
        self.pos = Identity() # Pass for now

        self.encoder = Encoder(num_layers=layers_repeat, d_model=d_model,
                                nhead=heads, dim_feedforward=dim_feedforward, dropout=dropout)
        
        self.decoder = Decoder(num_layers=layers_repeat, d_model=d_model,
                                nhead=heads, dim_feedforward=dim_feedforward, dropout=dropout)

        self.classifier = nn.Linear(d_model, dec_vocab_size)
    
    def forward(self, src_tokens, trgt_tokens):

        src_embed = self.encEmbed(src_tokens)
        src_pos = self.pos(src_embed)
        encoder_input = src_embed + src_pos
        encoder_context = self.encoder(encoder_input)
        
        trgt_embed = self.decEmbed(trgt_tokens)
        trgt_pos = self.pos(trgt_embed)
        decoder_input = trgt_embed + trgt_pos
        decoder_context = self.decoder(decoder_input, encoder_context)
        
        logits = self.classifier(decoder_context)
        return logits
    
    @torch.no_grad()
    def inference(self, src_tokens):
        src_embed = self.encEmbed(src_tokens)
        src_pos = self.pos(src_embed)
        encoder_input = src_embed + src_pos
        encoder_context = self.encoder(encoder_input)

        ## Assume <pad>:0, <unk>:1 <s>:2, </s>:3
        token_list = [2]
        confidences = []
        token = token_list[0]
        maxtries = 0

        while token != 3 and maxtries <= src_tokens.size(-1) + 5:
            trgt_embed = self.decEmbed(torch.tensor([token_list]).to(src_tokens.device))
            trgt_pos = self.pos(trgt_embed)
            decoder_input = trgt_embed + trgt_pos
            decoder_context = self.decoder(decoder_input, encoder_context)
            logits = self.classifier(decoder_context) # (B,T,vocab_size) often B=1
            
            softmax = F.softmax(logits[:,-1,:], dim=-1) # (B,vocab_size)
            confidence, token = torch.max(softmax, dim=-1)
            
            token = token.item()
            confidence = confidence.item()
            token_list.append(token)
            confidences.append(confidence)
            maxtries += 1
        
        return token_list, confidences
    

transformer = Transformer()
src_tokens = torch.randint(low=0, high=enc_vocab_size, size=(32, 10))
trgt_tokens = torch.randint(low=0, high=enc_vocab_size, size=(32, 15))

logits = transformer(src_tokens, trgt_tokens)
print(logits.shape)


x_text = torch.tensor([[2,500,100,8564, 21, 1, 754, 3]]) ## dumb
transformer.eval()
tokens, confidences = transformer.inference(x_text)
transformer.train()
print(tokens, confidences, sep='\n')
print("Not Trained yet.")


def get_parameters_info(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad==True)
    nontrainable = sum(p.numel() for p in model.parameters() if p.requires_grad==False)

    return trainable, nontrainable

tr, nontr = get_parameters_info(transformer)
print(f"Total trainable parameters= {tr:,}\nTotal non-trainable parameters= {nontr:,}") 
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

'''
CLIP Text Encoder is similar to the Encoder Layer of the Vanilla Transformer 
'''

class CLIPEmbedding(nn.Module):
    
    def __init__(self, embed_size, vocab_size, ntokens):
        super().__init__()
        
        self.embed_size = torch.tensor(embed_size)
        self.vocab_size = vocab_size
        self.input_embedding = nn.Embedding(vocab_size, embed_size)
        
        #Learnable Position Encoding
        self.position_embedding = nn.Parameter(torch.zeros(ntokens, embed_size))
        
    def forward(self, tokens):
        
        #(Batch_size, seq_len) --> (Batch_size, seq_len, embed_size)
        x = self.embedding(tokens) #* torch.sqrt(self.embed_size)
        return x + self.position_embedding(x)


class quick_GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x) 

class CLIPLayer(nn.Module):
    def __init__(self, embed_size, nheads):
        super().__init__()
        
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, nheads)
        self.fc1 = nn.Linear(embed_size, 4* embed_size)
        self.fc2 = nn.Linear(4* embed_size, embed_size)
        
        # self.gelu = nn.GELU(approximate='none')
        self.gelu = quick_GELU()
        
    def forward(self, x):
        #(Batch_size, seq_len, ndim)
        residue = x
        x = self.layernorm1(x)
        x = self.attention(x, causal_mask = True)
        x += residue
        
        residue = x
        x = self.layernorm2(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x += residue
        
        return x

class CLIP(nn.Module):
    
    def __init__(self):
        super().__init__(self)
        
        self.embedding = CLIPEmbedding(embed_size= 768, vocab_size= 49408, ntoken = 77)
        
        self.layers = nn.Module([
            CLIPLayer(12, 768) for _ in range(12)
        ])
        
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens):
        tokens = tokens.type(torch.long)
        
        #(Batch_size, seq_len) --> (Batch_size, seq_len, embed_size)
        state = self.embedding(tokens)
        
        for layer in self.layers:
            state = layer(state)
        
        
        return self.layernorm(state)
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention
from unet import UNET, UNET_Output


class TimeEmbedding(nn.Module):
    
    def __init__(self, embed_size):
        super().__init__()
        
        self.linear1 = nn.Linear(embed_size, 4 * embed_size)
        self.linear2 = nn.Linear(4 * embed_size, embed_size)
    
    def forward(self, x):
        
        x = F.silu(self.linear1(x))
        
        return self.linear2(x)


class Diffusion(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_Output(320, 4)
    
    def forward(self, latent, context, time):
        #latent: (Batch_size, 4, H/8, W/8) 
        #context: (Batch_size, seq_len, embed_size) #Prompt Embeddings
        #time: (1, 320)
        
        #(1, 320) --> (1, 1280)
        time = self.time_embedding(time)
        
        #(Batch_size, 4, H/8, W/8) --> (Batch_size, 320, H/8, W/8)
        output = self.unet(latent, context, time)
        
        #(Batch_size, 320, H/8, W/8) --> (Batch_size, 4, H/8, W/8)
        output = self.final(output)
        
        return output
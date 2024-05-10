import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, nheads, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        
        assert embed_size % nheads == 0, f"embed_size: {embed_size} (d_model) is not divisible by nheads: {nheads}"
        
        self.embed_size = embed_size
        self.nheads = nheads
        self.dk = embed_size // nheads
        
        '''
        Wq, Wk and Wv are of the shape (embed_size, embed_size)
        Taking a single weight matrix for it of the shape (embed_size, Wq.shape[1] * Wk.shape[1] * Wv.shape[1])
        '''
        self.in_proj = nn.Linear(embed_size, 3* embed_size, bias = in_proj_bias)
        
        #Wo weight matrix
        self.out_proj = nn.Linear(embed_size, embed_size, bias = out_proj_bias)
        
    
    def forward(self, x, causal_mask = False):
        #x: (Batch_size, seq_len, dim)
        input_shape = x.shape
        batch_size, seq_len, embed_size = input_shape
        
        intermim_shape = (batch_size, seq_len, self.nheads, self.dk)
        
        #(Batch_size, seq_len, dim) --> 3 tensors: [(Batch_size, seq_len, dim), (Batch_size, seq_len, dim), (Batch_size, seq_len, dim)]
        q, k, v = self.in_proj(x).chunk(3, dim = -1)
        
        #(Batch_size, seq_len, dim) --> (Batch_size, seq_len, nheads, dim/nheads) --> (Batch_size, nheads, seq_len, dim/nheads)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)
        
        #(Batch_size, nheads, seq_len, seq_len)
        w = torch.matmul(q, k.transpose(-1, -2))
        
        if causal_mask:
            mask = torch.ones_like(w, dtype = torch.bool).triu(1)
            w.masked_fill_(mask, -torch.inf)
            
        w /= torch.sqrt(torch.tensor(self.dk))
        
        w = F.softmax(w, dim = -1)
        
        #(Batch_size, nheads, seq_len, seq_len) @ (Batch_size, nheads, seq_len, dim/nheads) --> (Batch_size, nheads, seq_len, dim/nheads)
        output = torch.matmul(w, v)
        output
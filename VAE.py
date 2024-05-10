import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size= 3, padding= 1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size= 3, padding = 1)
        )
        # self.group_norm_in = nn.GroupNorm(32, in_channels)
        # self.group_norm_out = nn.GroupNorm(32, out_channels)
        
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size= 3, padding= 1)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size= 3, padding= 1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size= 1, padding= 0)

        self.silu = nn.SiLU()
        
    def forward(self, x):
        #x: (Batch_size, in_channels, H, W)
        residue = x
        
        # x = self.silu(self.group_norm_in(x))
        # x = self.conv1(x)
        # x = self.silu(self.group_norm_out(x))
        # x = self.conv2(x)
        x = self.block(x)
        
        return x + self.residual_layer(residue)

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
        
    def forward(self, x):
        #x : (Batch_size, channels, H, W)
        residue = x
        
        batch_size, c, H, W = x.shape
        
        #(Batch_size, channels, H, W) --> (Batch_size, channels, H * W)
        x = x.view(batch_size, c, H * W)
        
        #(Batch_size, channels, H * W) --> (Batch_size, H * W, channels)
        x = x.transpose(-1, -2)
        
        x = self.attention(x)
        
        #(Batch_size, H * W, channels) --> (Batch_size, channels, H * W)
        x = x.transpose(-1, -2)
        
        #(Batch_size, channels, H * W) --> (Batch_size, channels, H, W)
        x = x.view((batch_size, c, H, W))
        
        x += residue
        
        return


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            
            # (Batch_size, 3, H, W)  -> (Batch_size, 128, H, W)
            nn.Conv2d(in_channels= 3, out_channels= 128, kernel_size= 3, padding= 1),
            
            VAE_ResidualBlock(128, 128),
            
            VAE_ResidualBlock(128, 128),
            
            # (Batch_size, 128, H, W)  -> (Batch_size, 128, H/2, W/2)
            nn.Conv2d(in_channels= 128, out_channels= 128, kernel_size= 3, stride= 2, padding= 0),
            
            # (Batch_size, 128, H/2, W/2)  -> (Batch_size, 256, H/2, W/2)
            VAE_ResidualBlock(128, 256),
            
            VAE_ResidualBlock(256, 256),
            
            # (Batch_size, 256, H/2, W/2)  -> (Batch_size, 256, H/4, W/4)
            nn.Conv2d(in_channels= 256, out_channels= 256, kernel_size= 3, stride= 2, padding= 0),
            
            # (Batch_size, 256, H/4, W/4)  -> (Batch_size, 512, H/4, W/4)
            VAE_ResidualBlock(256, 512),
            
            VAE_ResidualBlock(512, 512),
            
            # (Batch_size, 512, H/4, W/4)  -> (Batch_size, 512, H/8, W/8)
            nn.Conv2d(in_channels= 512, out_channels= 512, kernel_size= 3, stride= 2, padding= 0),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_ResidualBlock(512, 512),
            
            # (Batch_size, 512, H/8, W/8)  -> (Batch_size, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),
            
            # (Batch_size, 512, H/8, W/8)  -> (Batch_size, 512, H/8, W/8)
            VAE_AttentionBlock(512),
            
            nn.GroupNorm(num_groups= 32, num_channels= 512),
            
            ##Sigmoid Linear Unit
            nn.SiLU(), 
            
            ##BottleNeck of the Encoder 
            # (Batch_size, 512, H/8, W/8)  -> (Batch_size, 8, H/8, W/8)
            nn.Conv2d(in_channels= 512, out_channels= 8, kernel_size= 3, padding= 1),
            
            # (Batch_size, 8, H/8, W/8)  -> (Batch_size, 512, H/8, W/8)
            nn.Conv2d(in_channels= 8, out_channels= 8, kernel_size= 3, padding= 1),
            
        )
    
    def forward(self, x, noise):
        
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1)) #(Pad Left, Pad Right, Pad Top, Pad Bottom)
            x = module(x)
        
        # (Batch_size, 8, H/8, W/8)  -> [(Batch_size, 4, H/8, W/8), (Batch_size, 4, H/8, W/8)]
        mean, log_var = torch.chunk(x, 2, dim = 1)
        
        log_var = torch.clamp(log_var, -30, 20)
        
        var = torch.exp(log_var)
        
        std_dev = torch.sqrt(var)
        
        x = mean + std_dev * noise
        
        x *= 0.18125
        
        return x
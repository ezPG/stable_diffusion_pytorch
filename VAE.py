import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self):
        return

class VAE_AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self):
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
        return super().forward(input)
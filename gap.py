import torch 
import torch.nn as nn 

from conv4d import Conv4d 

from typing import List 


def build_gap(args): 
    """ Build spatio-temporal global average pooling (GAP) using args. """
    if args.gap == 'Simple':  
        channels = [int(nc) for nc in args.channels.split(',')]
        _gap = SimpleSpatioTemporalGAP(channels, args.out_channels) 
    else: 
        raise NotImplementedError("gap {} has not been implemented".format(args.gap)) 
    
    return _gap 


class SimpleSpatioTemporalGAP(nn.Module): 
    """ Define a simple spatio-temporal global average pooling (GAP) module. """
    def __init__(self, channels: List, out_channels: int): 
        super().__init__() 
        
        self.dense = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2), 
            nn.Linear(channels[-1] // 2, out_channels)
        )

    def forward(self, x: torch.Tensor): 
        """ Forward function. 
        
        Parameters: 
            x (Tensor): input 6d feature tensor [B, L, C, D, H, W] 
        
        Returns: 
            x (Tensor): output 2d feature vector [B, C] """
        x = x.transpose(1, 2) 
        x = x.mean(dim=(-1, -2, -3, -4)) 
        x = self.dense(x) 
        return x  

    def __str__(self) -> str:
        return "Simple" 

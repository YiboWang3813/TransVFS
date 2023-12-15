import torch 
import torch.nn as nn 

from stfet import build_stfet 
from gap import build_gap 
from criterion import SetCriterion, init_weights 


def build_transvfs(args): 
    # build criterion 
    weight_dict = {'loss_mse_f': 1, 'loss_mse_t': args.lambda_}  
    losses = ['mse_f', 'mse_t', 'mae', 'pcc'] # specify loss or accuracy needed to be computed 
    criterion = SetCriterion(weight_dict, losses)  

    # build TransVFS 
    stfet = build_stfet(args) 
    gap = build_gap(args) 
    transvfs = TransVFS(stfet, gap) 
    init_weights(transvfs) 

    return transvfs, criterion 


class TransVFS(nn.Module): 
    def __init__(self, stfet: nn.Module, gap: nn.Module):
        super().__init__() 

        self.stfet = stfet 
        self.gap = gap 

    def forward(self, x: torch.Tensor): 
        """ Forward function. 
        
        Inputs: 
            x (Tensor): input image tensor [B, L, Ci, D, H, W] 
            
        Returns: 
            x (Tensor): output force tensor [B, Co]"""
        return self.gap(self.stfet(x)) 

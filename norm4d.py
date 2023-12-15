import torch 
import torch.nn as nn 


class BatchNorm4d(nn.Module): 
    def __init__(self, num_channels: int):
        super().__init__() 

        self.bn = nn.BatchNorm3d(num_channels) 
        
    def forward(self, x: torch.Tensor): 
        (b, l_i, c_i, d_i, h_i, w_i) = x.shape 
        results = [] 

        for i in range(l_i): 
            _x_normed = self.bn(x[:, i, ...].view(b, c_i, d_i, h_i, w_i)) 
            results.append(_x_normed.view(b, 1, c_i, d_i, h_i, w_i)) 

        y = torch.cat(results, dim=1) 
        return y 
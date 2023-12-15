import torch 
import torch.nn as nn 

from typing import Tuple 


class ConvCell3d(nn.Module):
    """ Define a 3d spatial convolutional cell called in Conv4d. """ 
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 use_relu_act=True, use_batch_norm=False):
        # type: (int, int, Tuple, Tuple, bool, bool) -> None 
        super().__init__() 

        blks = [] 
        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2) 
        blks.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)) 
        if use_relu_act: 
            blks.append(nn.ReLU()) 
        if use_batch_norm: 
            blks.append(nn.BatchNorm3d(out_channels)) 
        self.net = nn.Sequential(*blks) 

    def forward(self, x): 
        return self.net(x) 


class Conv4d(nn.Module):
    """ Define a 4d spatial-temporal convolutional module. 
    
    Parameters: 
        in_channels (int): channels of input tensor 
        out_channels (int): channels of output tensor 
        kernel_size (tuple): spatio-temporal kernel size e.g. (3, 3, 3, 3) 
        stride (tuple): spatio-temporal stride e.g. (1, 1, 1, 1) 
        use_relu_act (bool): is to use nn.ReLU as activation function 
        use_batch_norm (bool): is to use nn.BatchNorm as normalization function """ 
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 use_relu_act=True, use_batch_norm=False):
        # type: (int, int, Tuple, Tuple, bool, bool) -> None 
        super().__init__() 

        assert len(kernel_size) == len(stride) == 4, f"The length of kernel size should be same to the length of stride and equal to 4, but got {len(kernel_size)}"

        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.kernel_size = kernel_size 
        self.stride = stride 
        self.use_relu_act = use_relu_act 
        self.use_batch_norm = use_batch_norm 
        
        self.conv_layers = nn.ModuleList() 
        (l_k, d_k, h_k, w_k) = kernel_size 
        (l_s, d_s, h_s, w_s) = stride 
        for _ in range(l_k):
            self.conv_layers.append(ConvCell3d(in_channels, out_channels,
                                    (d_k, h_k, w_k), (d_s, h_s, w_s), use_relu_act, use_batch_norm))

    def forward(self, input): 
        """ Forward function. 
        
        Parameters: 
            input (Tensor): 6d spatio-temporal Tensor [B, Li, Ci, Di, Hi, Wi] 
        
        Returns: 
            output (Tensor): 6d spatio-temporal Tensor [B, Lo, Co, Do, Ho, Wo] 
        where B denotes batch size, Li and Lo denote the length of time of input and output tensor, repectively;
        Ci, Co denote the channels of input and output tensor, and Di, Hi, Wi, Do, Ho, Wo denote 
        the depth, height, width of input and output tensor, repectively """
        (b, l_i, c_i, d_i, h_i, w_i) = tuple(input.shape) 
        (l_k, d_k, h_k, w_k) = self.kernel_size 
        c_o = self.out_channels 
        (l_s, d_s, h_s, w_s) = self.stride
        (l_o, d_o, h_o, w_o) = (l_i // l_s, d_i // d_s, h_i // h_s, w_i // w_s) 
        frame_results = [torch.zeros((b, 1, c_o, d_o, h_o, w_o), dtype=torch.float32,
                                     device=input.device, requires_grad=False)] * l_o 

        for i in range(l_k): 
            for j in range(l_i): 
                # calculate output frame's position and avoid overflow 
                out_frame = j - (i - l_k // 2) + (l_i - l_o) // 2 
                if out_frame < 0 or out_frame >= l_o: 
                    continue 

                # convolve input frame j with kernel frame i 
                frame_conv3d = self.conv_layers[i](input[:, j, ...].view(b, c_i, d_i, h_i, w_i))
                frame_conv3d = frame_conv3d.view(b, 1, c_o, d_o, h_o, w_o)

                # add these overlapped frames together 
                if frame_results[out_frame] is None: 
                    frame_results[out_frame] = frame_conv3d
                else: 
                    frame_results[out_frame] += frame_conv3d
        
        output = torch.cat(frame_results[0::], dim=1) 
        return output 

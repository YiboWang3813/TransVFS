import torch 
import torch.nn as nn 
import numpy as np 
import copy

from conv4d import Conv4d 
from norm4d import BatchNorm4d 

from typing import List, Tuple, Dict 


class TLGAttention(nn.Module): 
    """ Temporal Local Global Attention """
    def __init__(self, dim: int, num_heads: int = 8, qk_scale: float = None, 
                 attn_drop: float = 0., proj_drop: float = 0., 
                 local_size: Tuple = (1, 2, 2, 2), stage: int = 0):
        super().__init__() 

        assert dim % num_heads == 0, "dim {} should can be divided by num_heads {}".format(dim, num_heads) 
        self.dim = dim 
        self.num_heads = num_heads 
        head_dim = dim // num_heads 
        self.scale = qk_scale or head_dim ** -0.5 

        self.q = Conv4d(dim, dim, (3, 1, 1, 1), (1, 1, 1, 1)) 
        self.k = Conv4d(dim, dim, (3, 1, 1, 1), (1, 1, 1, 1)) 
        self.v = Conv4d(dim, dim, (3, 1, 1, 1), (1, 1, 1, 1)) 
        self.attn_drop = nn.Dropout(attn_drop) 
        self.proj = Conv4d(dim, dim, (3, 1, 1, 1), (1, 1, 1, 1)) 
        self.proj_drop = nn.Dropout(proj_drop) 
        self.local_size = local_size 
        self.stage = stage 

    def forward(self, x: torch.Tensor): 
        B, L, C, D, H, W = x.shape 

        if self.stage < 2: 

            xs, xe = x[:, :L // 2, ...], x[:, L // 2:, ...]  # start, end  

            nl = L // 2 // self.local_size[0] 
            nd, nh, nw = D // self.local_size[1], H // self.local_size[2], W // self.local_size[3] 
            nlw = nl * nd * nh * nw # number of local windows 

            # q: start; k, v: end 
            qs = self.q(xs) 
            ke, ve = self.k(xe), self.v(xe) 

            # qs, ke, ve: [B, nlw, num_heads, prod(local_size), head_dim] 
            qs = qs.transpose(1, 2).reshape(B, C, nl, self.local_size[0], nd, self.local_size[1], 
                                            nh, self.local_size[2], nw, self.local_size[3]).permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1) 
            qs = qs.reshape(B, nlw, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4) # -1 = prod(local_size) 
            ke = ke.transpose(1, 2).reshape(B, C, nl, self.local_size[0], nd, self.local_size[1], 
                                            nh, self.local_size[2], nw, self.local_size[3]).permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1) 
            ke = ke.reshape(B, nlw, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4) 
            ve = ve.transpose(1, 2).reshape(B, C, nl, self.local_size[0], nd, self.local_size[1], 
                                            nh, self.local_size[2], nw, self.local_size[3]).permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1) 
            ve = ve.reshape(B, nlw, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4) 

            attnse = (qs @ ke.transpose(-1, -2)) * self.scale 
            attnse = attnse.softmax(dim=-1) 
            attnse = self.attn_drop(attnse) 
            yse = (attnse @ ve).transpose(2, 3).reshape(B, nl, nd, nh, nw, self.local_size[0], self.local_size[1],
                                                        self.local_size[2], self.local_size[3], C) 
            yse = yse.permute(0, 9, 1, 5, 2, 6, 3, 7, 4, 8).reshape(B, C, L // 2, D, H, W).transpose(1, 2) 
            yse = self.proj(yse) 
            yse = self.proj_drop(yse) 

            # q: end; k, v: start 
            qe = self.q(xe) 
            ks, vs = self.k(xs), self.v(xs) 

            # qe, ks, vs: [B, nlw, num_heads, prod(local_size), head_dim] 
            qe = qe.transpose(1, 2).reshape(B, C, nl, self.local_size[0], nd, self.local_size[1], 
                                            nh, self.local_size[2], nw, self.local_size[3]).permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1) 
            qe = qe.reshape(B, nlw, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4) # -1 = prod(local_size) 
            ks = ks.transpose(1, 2).reshape(B, C, nl, self.local_size[0], nd, self.local_size[1], 
                                            nh, self.local_size[2], nw, self.local_size[3]).permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1) 
            ks = ks.reshape(B, nlw, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4) 
            vs = vs.transpose(1, 2).reshape(B, C, nl, self.local_size[0], nd, self.local_size[1], 
                                            nh, self.local_size[2], nw, self.local_size[3]).permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1) 
            vs = vs.reshape(B, nlw, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4) 

            attnes = (qe @ ks.transpose(-1, -2)) * self.scale 
            attnes = attnes.softmax(dim=-1) 
            attnes = self.attn_drop(attnes) 
            yes = (attnes @ vs).transpose(2, 3).reshape(B, nl, nd, nh, nw, self.local_size[0], self.local_size[1],
                                                        self.local_size[2], self.local_size[3], C) 
            yes = yes.permute(0, 9, 1, 5, 2, 6, 3, 7, 4, 8).reshape(B, C, L // 2, D, H, W).transpose(1, 2) 
            yes = self.proj(yes) 
            yes = self.proj_drop(yes) 

            y = torch.cat([yse, yes], dim=1)  

        else: 

            nl = L // self.local_size[0] 
            nd, nh, nw = D // self.local_size[1], H // self.local_size[2], W // self.local_size[3] 
            nlw = nl * nd * nh * nw 

            q, k, v = self.q(x), self.k(x), self.v(x) 

            q = q.transpose(1, 2).reshape(B, C, nl, self.local_size[0], nd, self.local_size[1], 
                                          nh, self.local_size[2], nw, self.local_size[3]).permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1)  
            k = k.transpose(1, 2).reshape(B, C, nl, self.local_size[0], nd, self.local_size[1], 
                                          nh, self.local_size[2], nw, self.local_size[3]).permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1)  
            v = v.transpose(1, 2).reshape(B, C, nl, self.local_size[0], nd, self.local_size[1], 
                                          nh, self.local_size[2], nw, self.local_size[3]).permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1)  

            attn = (q @ k.transpose(-1, -2)) * self.scale 
            attn = attn.softmax(dim=-1) 
            attn = self.attn_drop(attn) 
            y = (attn @ v).transpose(2, 3).reshape(B, nl, nd, nh, nw, self.local_size[0], self.local_size[1],
                                                   self.local_size[2], self.local_size[3], C) 
            y = y.permute(0, 9, 1, 5, 2, 6, 3, 7, 4, 8).reshape(B, C, L, D, H, W).transpose(1, 2) 
            y = self.proj(y) 
            y = self.proj_drop(y) 

        return y 


class SLGAttention(nn.Module): 
    """ Spatial Local Global Attention """
    def __init__(self, dim: int, num_heads: int = 8, qk_scale: float = None, 
                 attn_drop: float = 0., proj_drop: float = 0., 
                 fine_pysize: Tuple = (2, 8, 8, 8), coarse_pysize: Tuple = (1, 2, 2, 2), 
                 resolution: Tuple = (4, 16, 16, 16), stage: int = 0): 
        super().__init__() 

        assert dim % num_heads == 0, "dim {} should can be divided by num_heads {}".format(dim, num_heads) 
        self.dim = dim 
        self.num_heads = num_heads 
        head_dim = dim // num_heads 
        self.scale = qk_scale or head_dim ** -0.5 

        self.q = Conv4d(dim, dim, (1, 3, 3, 3), (1, 1, 1, 1)) 
        self.k = Conv4d(dim, dim, (1, 3, 3, 3), (1, 1, 1, 1)) 
        self.v = Conv4d(dim, dim, (1, 3, 3, 3), (1, 1, 1, 1)) 
        self.attn_drop = nn.Dropout(attn_drop) 
        self.proj = Conv4d(dim, dim, (1, 3, 3, 3), (1, 1, 1, 1)) 
        self.proj_drop = nn.Dropout(proj_drop) 
        
        self.resolution = resolution 
        self.stage = stage 

        self.fine_pysize = fine_pysize 
        self.fine_kernel_size = [rs // fs for rs, fs in zip(resolution, fine_pysize)] 

        if np.prod(self.fine_kernel_size) > 1: # fine_pysize != resolution 
            
            self.fine_conv = Conv4d(dim, dim, kernel_size=(1, 3, 3, 3), stride=self.fine_kernel_size) 
            self.norm = nn.LayerNorm(dim) 

            self.coarse_pysize = coarse_pysize 
            self.coarse_kernel_size = [rs // cs for rs, cs in zip(resolution, coarse_pysize)] 

            if stage < 2:  
                self.coarse_conv = Conv4d(dim, dim, kernel_size=(1, 3, 3, 3), stride=self.coarse_kernel_size) 

    def forward(self, x: torch.Tensor): 
        B, L, C, D, H, W = x.shape 
        N = L * D * H * W 

        q = self.q(x).transpose(1, 2).reshape(B, C, -1).transpose(1, 2) # [B, N, C] 
        q = q.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # [B, num_heads, N, head_dim]

        if np.prod(self.fine_kernel_size) > 1: 
            if self.stage < 2: 
                xc = self.coarse_conv(x) 
                xf = self.fine_conv(x) 
                
                kc, vc = self.k(xc), self.v(xc) # [B, Lc, C, Dc, Hc, Wc] 
                kf, vf = self.k(xf), self.v(xf) # [B, Lf, C, Df, Hf, Wf] 

                kc = kc.transpose(1, 2).reshape(B, C, -1).transpose(1, 2) 
                kf = kf.transpose(1, 2).reshape(B, C, -1).transpose(1, 2) 
                k = torch.cat([kc, kf], dim=1) # [B, Nc + Nf, C] 
                k = self.norm(k)

                vc = vc.transpose(1, 2).reshape(B, C, -1).transpose(1, 2) 
                vf = vf.transpose(1, 2).reshape(B, C, -1).transpose(1, 2) 
                v = torch.cat([vc, vf], dim=1) # [B, Nc + Nf, C] 
                v = self.norm(v) 
            else: 
                xf = self.fine_conv(x) 

                kf, vf = self.k(xf), self.v(xf) # [B, Lf, C, Df, Hf, Wf] 

                kf = kf.transpose(1, 2).reshape(B, C, -1).transpose(1, 2) 
                k = self.norm(kf) # [B, Nf, C]

                vf = vf.transpose(1, 2).reshape(B, C, -1).transpose(1, 2) 
                v = self.norm(vf) # [B, Nf, C] 

            k = k.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
            v = v.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # -1 = Nf or Nf + Nc 
        
        else: 
            k, v = self.k(x), self.v(x) # [B, L, C, D, H, W] 

            k = k.transpose(1, 2).reshape(B, C, -1).transpose(1, 2) 
            k = k.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 

            v = v.transpose(1, 2).reshape(B, C, -1).transpose(1, 2) 
            v = v.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # -1 = N 

        attn = (q @ k.transpose(-1, -2)) * self.scale # [B, num_heads, N, Nk] 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  
        x = x.transpose(1, 2).reshape(B, C, L, D, H, W).transpose(1, 2) 
        x = self.proj(x) 
        y = self.proj_drop(x) 

        return y 


class ConvMlp4d(nn.Module): 
    def __init__(self, in_channels: int, hidden_channels: int = None, out_channels: int = None, 
                 act_layer: nn.Module = nn.ReLU, drop: float = 0., stmode: str = 'C'): 
        super().__init__() 

        kernel_size = None 
        if stmode in ('A', 'B', 'C', 'D'): 
            kernel_size = (3, 3, 3, 3) 
        elif stmode == 'E': 
            kernel_size = (1, 3, 3, 3) 
        elif stmode == 'F': 
            kernel_size = (3, 1, 1, 1) 
        else: 
            raise NotImplementedError("stmode {} has not been implemented".format(stmode)) 

        hidden_channels = hidden_channels or in_channels 
        out_channels = out_channels or in_channels 
        self.conv1 = Conv4d(in_channels, hidden_channels, kernel_size, (1, 1, 1, 1)) 
        self.act = act_layer() 
        self.conv2 = Conv4d(hidden_channels, out_channels, kernel_size, (1, 1, 1, 1)) 
        self.drop = nn.Dropout(drop) 

    def forward(self, x: torch.Tensor): 
        return self.drop(self.conv2(self.drop(self.act(self.conv1(x))))) 


class STLGTransformerBlock(nn.Module): 
    def __init__(self, dim: int, num_heads: int = 8, 
                 resolution: Tuple = (4, 16, 16, 16), mlp_rate: int = 2,  
                 qk_scale: float = None, attn_drop: float = 0., 
                 proj_drop: float = 0., drop: float = 0., 
                 act_layer: nn.Module = nn.ReLU, norm_layer: nn.Module = BatchNorm4d, 
                 local_size: Tuple = (1, 2, 2, 2), 
                 fine_pysize: Tuple = (2, 8, 8, 8), coarse_pysize: Tuple = (1, 2, 2, 2), 
                 stage: int = 0, stmode: str = 'C'):
        super().__init__() 

        self.slga, self.tlga = None, None 
        if stmode in ('A', 'B', 'C', 'D'): 
            self.slga = SLGAttention(dim, num_heads, qk_scale, attn_drop, proj_drop, 
                                    fine_pysize, coarse_pysize, resolution, stage) 
            self.tlga = TLGAttention(dim, num_heads, qk_scale, attn_drop, proj_drop, 
                                    local_size, stage) 
            self.norm1s = norm_layer(dim) 
            self.norm1t = norm_layer(dim) 
        elif stmode == 'E': 
            self.slga = SLGAttention(dim, num_heads, qk_scale, attn_drop, proj_drop, 
                                    fine_pysize, coarse_pysize, resolution, stage) 
            self.norm1s = norm_layer(dim) 
        elif stmode == 'F': 
            self.tlga = TLGAttention(dim, num_heads, qk_scale, attn_drop, proj_drop, 
                                    local_size, stage) 
            self.norm1t = norm_layer(dim) 
        else: 
            raise NotImplementedError("stmode {} has not been implemented".format(stmode))

        # self.norm1 = norm_layer(dim) 
        self.drop = nn.Dropout(drop) 
        self.norm2 = norm_layer(dim) 
        mlp_hidden_dim = int(dim * mlp_rate) 
        self.mlp = ConvMlp4d(dim, hidden_channels=mlp_hidden_dim, act_layer=act_layer, drop=drop, stmode=stmode)  

        self.stmode = stmode 

    def forward(self, x: torch.Tensor): 

        x_shortcut = x 
        
        if self.stmode == 'A': 
            x = self.slga(self.norm1s(x)) 
            x = self.tlga(self.norm1t(x)) 
        elif self.stmode == 'B': 
            x1 = self.tlga(self.norm1t(x)) 
            x2 = self.slga(self.norm1s(x)) 
            x = x1 + x2 
        elif self.stmode == 'C': 
            x1 = self.slga(self.norm1s(x)) 
            x2 = self.tlga(self.norm1t(x1))  
            x = x1 + x2 
        elif self.stmode == 'D': 
            x1 = self.tlga(self.norm1t(x)) 
            x2 = self.slga(self.norm1s(x1))  
            x = x1 + x2 
        elif self.stmode == 'E': 
            x = self.slga(self.norm1s(x))  
        elif self.stmode == 'F': 
            x = self.tlga(self.norm1t(x))  
        else: 
            raise NotImplementedError("st mode {} has not been implemented".format(self.stmode)) 
        
        x = x_shortcut + self.drop(x) 
        y = x + self.drop(self.mlp(self.norm2(x))) 

        return y  


class SpatioTemporalFeatureExtractionTransformer(nn.Module): 
    def __init__(self, 
                 in_channels: int = 1, 
                 video_size: Tuple = (4, 128, 128, 128), 
                 embed_dims: List[int] = [8, 16, 32, 64], 
                 num_heads: List[int] = [1, 2, 4, 8], 
                 mlp_rates: List[int] = [2, 2, 2, 2], 
                 qk_scale: float = None, 
                 drop_rate: float = 0., 
                 attn_drop_rate: float = 0., 
                 proj_drop_rate: float = 0., 
                 norm_layer: nn.Module = BatchNorm4d, 
                 depths: List[int] = [1, 1, 1, 1], 
                 local_sizes: List[Tuple] = [(1, 2, 2, 2), (1, 2, 2, 2), (1, 2, 2, 2), (1, 2, 2, 2)], 
                 fine_pysizes: List[Tuple] = [(2, 8, 8, 8), (2, 8, 8, 8), (2, 4, 4, 4), (1, 2, 2, 2)], 
                 coarse_pysizes: List[Tuple] = [(1, 2, 2, 2), (1, 2, 2, 2), (1, 2, 2, 2), (1, 2, 2, 2)], 
                 spatial_pooling: List[int] = [4, 2, 2, 2], 
                 temporal_pooling: List[int] = [1, 1, 2, 1],  
                 stmode: str = 'C', 
                 return_intermediate_features: bool = True):
        super().__init__() 

        self.feature_embeds = nn.ModuleList() 
        self.pos_embeds = nn.ModuleList() 
        self.pos_drops = nn.ModuleList() 
        self.trans_blocks = nn.ModuleList() 

        # generate a pre feature embedding to match different input image shape 
        pre_feature_stride = (1, 2, 2, 2) if video_size[1] == 128 else (1, 1, 1, 1) 
        self.pre_feature_embedding = Conv4d(in_channels, embed_dims[0], kernel_size=(3, 3, 3, 3), stride=pre_feature_stride) 
        adjusted_video_size = (video_size[0] // pre_feature_stride[0], 
                               video_size[1] // pre_feature_stride[1], 
                               video_size[2] // pre_feature_stride[2], 
                               video_size[3] // pre_feature_stride[3]) 

        # compute feature map's resolution of each stage 
        self.video_sizes = [] 
        vs = adjusted_video_size 
        for i in range(len(depths)): 
            vs = (vs[0] // temporal_pooling[i], 
                  vs[1] // spatial_pooling[i], 
                  vs[2] // spatial_pooling[i], 
                  vs[3] // spatial_pooling[i]) 
            self.video_sizes.append(vs) 

        # generate feature embeds and position embeds 
        for i in range(len(depths)): 
            if i == 0: 
                old_video_size = adjusted_video_size 
                new_video_size = self.video_sizes[0] 
                ks = tuple([ovs // nvs for ovs, nvs in zip(old_video_size, new_video_size)]) 
                ks1 = (1, 2, 2, 2) 
                ks2 = tuple([_ks // _ks1 for _ks, _ks1 in zip(ks, ks1)]) 
                self.feature_embeds.append(nn.Sequential(
                    Conv4d(embed_dims[0], embed_dims[0], kernel_size=(3, 3, 3, 3), stride=ks1), 
                    Conv4d(embed_dims[0], embed_dims[0], kernel_size=(3, 3, 3, 3), stride=ks2) 
                ))
            else: 
                old_video_size = self.video_sizes[i - 1] 
                new_video_size = self.video_sizes[i] 
                ks = tuple([ovs // nvs for ovs, nvs in zip(old_video_size, new_video_size)])
                self.feature_embeds.append(Conv4d(embed_dims[i - 1], embed_dims[i], kernel_size=(3, 3, 3, 3), stride=ks)) 

            self.pos_embeds.append(Conv4d(embed_dims[i], embed_dims[i], kernel_size=(3, 3, 3, 3), stride=(1, 1, 1, 1))) 
            self.pos_drops.append(nn.Dropout(proj_drop_rate)) 

        # generates transformer blocks 
        depths_copy = copy.deepcopy(depths) 
        for idx, num in enumerate(depths_copy): 
            if num is None: 
                depths_copy[idx] = 1 
        dpr = [x.item() for x in torch.linspace(0, drop_rate, sum(depths_copy))]
        cur = 0 # current number of blocks 

        for k in range(len(depths)): # stage 
            if depths[k] is not None: 
                _block = nn.ModuleList([STLGTransformerBlock(dim=embed_dims[k], 
                                                            num_heads=num_heads[k], 
                                                            resolution=self.video_sizes[k],
                                                            mlp_rate=mlp_rates[k], 
                                                            qk_scale=qk_scale, 
                                                            attn_drop=attn_drop_rate, 
                                                            proj_drop=proj_drop_rate, 
                                                            drop=dpr[cur + i], 
                                                            norm_layer=norm_layer, 
                                                            local_size=local_sizes[k], 
                                                            fine_pysize=fine_pysizes[k], 
                                                            coarse_pysize=coarse_pysizes[k], 
                                                            stmode=stmode, 
                                                            stage=k) for i in range(depths[k])]) 
                cur += depths[k] 
            else: 
                _block = nn.ModuleList([
                    Conv4d(embed_dims[k], embed_dims[k], (3, 3, 3, 3), (1, 1, 1, 1), False, True) 
                ])
                cur += 1 
            self.trans_blocks.append(_block) 
        
        self.norms = nn.ModuleList([norm_layer(dim) for dim in embed_dims]) 
        self.depths = depths 
        self.return_intermediate_features = return_intermediate_features 

    def forward(self, x: torch.Tensor): 
        """ Forward function. 
        
        Inputs: 
            x (Tensor): input image tensor [B, L, Ci, D, H, W] 
        
        Returns: 
            x (List[Tensor] or Tensor): output feature map or list [B, L, Co, Do, Ho, Wo] """
        x = self.pre_feature_embedding(x) 
        results = [] 
        for i in range(len(self.depths)): 
            x = self.feature_embeds[i](x) 
            x = self.pos_embeds[i](x) + x 
            x = self.pos_drops[i](x) 
            for j, blk in enumerate(self.trans_blocks[i]): 
                x = blk(x)  
            x = self.norms[i](x) 
            results.append(x) 
        
        if self.return_intermediate_features: 
            return results 
        else: 
            return results[-1] 


def build_stfet(args): 
    in_channels = args.in_channels 

    num_steps = args.num_steps 
    if args.is_box: 
        vol_shape = tuple([int(vs) for vs in args.cp_shape.split(',')])
    else: 
        vol_shape = tuple([int(vs) for vs in args.vol_shape.split(',')])
    video_size = (num_steps, *vol_shape) 

    embed_dims = [int(nc) for nc in args.channels.split(',')] 
    num_heads = [int(nd) for nd in args.num_heads.split(',')] 
    str_to_depths_map_dict = {
        'ATiny1': [None, 1, 1, 1], 
        'ATiny2': [1, None, 1, 1],
        'ATiny3': [1, 1, None, 1],
        'ATiny4': [1, 1, 1, None],
        'Tiny': [1, 1, 1, 1], 
        'ASmall': [2, 2, 1, 1], 
        'Small': [1, 1, 2, 2], 
        'ABase1': [1, 4, 2, 2], 
        'ABase2': [1, 2, 4, 2], 
        'Base': [1, 2, 2, 4], 
        'Large': [1, 2, 4, 8],
        'Huge': [4, 4, 8, 8]} 
    depths = str_to_depths_map_dict[args.depths] 

    local_sizes = [(1, 2, 2, 2), (1, 2, 2, 2), (1, 2, 2, 2), (1, 2, 2, 2)] 
    fine_pysizes = [(2, 8, 8, 8), (2, 8, 8, 8), (2, 4, 4, 4), (1, 2, 2, 2)]
    coarse_pysizes = [(1, 2, 2, 2), (1, 2, 2, 2), (1, 2, 2, 2), (1, 2, 2, 2)] 

    spatial_pooling = [int(sd) for sd in args.ssd.split(',')] 
    temporal_pooling = [int(sd) for sd in args.tsd.split(',')] 

    stmode = args.mode 
    mlp_rates = [2, 2, 2, 2]
    return_mid_features = True if args.gap == "Pyramid" else False 

    stfet = SpatioTemporalFeatureExtractionTransformer(
        in_channels, video_size, embed_dims, num_heads, mlp_rates, 
        None, 0.2, 0.1, 0.1, BatchNorm4d, depths, local_sizes, fine_pysizes, 
        coarse_pysizes, spatial_pooling, temporal_pooling, stmode, return_mid_features) 

    return stfet 

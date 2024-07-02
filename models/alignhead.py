import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Pyramid(nn.Module):
    # input: [B,C,H,W]
    # output: list of features from multi downsample layers
    def __init__(self, in_channels, n_feats, kernel_sizes=[3,3,3], strides=[1,2,2], paddings=[1,1,1]):
        super(Pyramid, self).__init__()
        self.in_channels = in_channels
        self.n_feats = n_feats

        layers = []
        in_channel = self.in_channels       
        for i in range(len(kernel_sizes)):
            cur_layer = nn.Sequential(
            nn.Conv2d(in_channel, self.n_feats, kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i]),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )
            layers.append(cur_layer)
            in_channel = self.n_feats
            
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        feature_list = []
        for layer in self.layers:
            x = layer(x)
            feature_list.append(x)
        return feature_list


class Cross_WindowAttention(nn.Module):
    # q and k are different
    r""" Window based multi-head cross attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super(Cross_WindowAttention,self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim , bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C), which maps query
            y: input features with shape of (num_windows*B, N, C), which maps key and value
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = self.k(y).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.v(y).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], k[0], v[0]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class Cross_WindowAttention_ReAtt(Cross_WindowAttention):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Cross_WindowAttention_ReAtt, self).__init__(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
    
    def forward(self, x, y, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C), which maps query
            y: input features with shape of (num_windows*B, N, C), which maps key and value
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = self.k(y).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.v(y).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], k[0], v[0]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn1 = self.attn_drop(attn)

        x = (attn1 @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Cross_WindowAttention2(Cross_WindowAttention):
    # the q and k are the same
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Cross_WindowAttention2, self).__init__(dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.)
        self.k = self.q

class Crossatten_align(nn.Module):
    r""" .
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn=Cross_WindowAttention):
        super(Crossatten_align, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.attn = attn(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        #self.register_buffer("attn_mask", attn_mask) 

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, y):
        # the shape of input and output :B, H, W, C
        #x: query (ref image)
        #y: kv    (to be aligned img)
        
        B, H, W, C = x.shape
        #assert L == H * W, "input feature has wrong size"

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            #attn_mask = self.calculate_mask((H,W))
        else:
            shifted_x = x
            shifted_y = y
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        y_windows = window_partition(shifted_y, self.window_size)  # nW*B, window_size, window_size, C
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size

        attn_windows_A = self.attn(x_windows, y_windows, mask=attn_mask)  # nW*B, window_size*window_size, C


        # merge windows
        attn_windows_A = attn_windows_A.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows_A, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class Crossatten_align_atttrans(Crossatten_align):
    # just return one more para: att
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn=Cross_WindowAttention_ReAtt):
        super(Crossatten_align_atttrans, self).__init__(dim, num_heads, window_size, shift_size,
                 qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer, attn)
    
    def forward(self, x, y):
        # the shape of input and output :B, H, W, C
        #x: query (ref image)
        #y: kv    (to be aligned img)
        
        B, H, W, C = x.shape
        #assert L == H * W, "input feature has wrong size"

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            #attn_mask = self.calculate_mask((H,W))
        else:
            shifted_x = x
            shifted_y = y
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        y_windows = window_partition(shifted_y, self.window_size)  # nW*B, window_size, window_size, C
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size

        attn_windows_A, att = self.attn(x_windows, y_windows, mask=attn_mask)  # nW*B, window_size*window_size, C


        # merge windows
        attn_windows_A = attn_windows_A.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows_A, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        return x, att


### Here begin the attention weight transfer version 
class Pyramid_CrossattAlign_Atttrans(nn.Module):
    # the original version num_heads=1, if the foldername of experiment has not indicate, the num_heads is 1
    def __init__(self, scales, num_feats, window_size, num_heads = 1, attn = Cross_WindowAttention_ReAtt):
        super(Pyramid_CrossattAlign_Atttrans, self).__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.scales = scales
        self.feat_conv = nn.ModuleDict()
        self.align = nn.ModuleDict()
        self.attn = attn
        self.window_size = window_size
        self.num_heads = num_heads
        for i in range(self.scales, 0, -1):
            level = f'l{i}'
            self.align[level] = Crossatten_align_atttrans(num_feats, num_heads, window_size, attn=self.attn)
            
            if i < self.scales:
                self.feat_conv[level] = nn.Conv2d(num_feats*3, num_feats, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, ref_feats_list, toalign_feats_list, patch_ratio_list):
        # input: B,C,H,W        
        upsample_feat = None
        last_att = None
        for i in range(self.scales, 0, -1):
            level = f'l{i}'
            ref_feat = ref_feats_list[i-1].permute(0,2,3,1)
            toalign_feat = toalign_feats_list[i-1].permute(0,2,3,1)
            
            aligned_feat, att = self.align[level](ref_feat, toalign_feat) 
            feat = aligned_feat.permute(0,3,1,2)
            
            if i < self.scales:
                patch_ratio = patch_ratio_list[i-1]
                atttransfer_feat = self.atttransfer_multiheads(toalign_feat, last_att, patch_ratio)
                atttransfer_feat = atttransfer_feat.permute(0,3,1,2)
                #print('atttransfer_feat shape', atttransfer_feat.shape)
                feat = self.feat_conv[level](
                    torch.cat([feat, upsample_feat, atttransfer_feat], dim=1)
                )
            if i > 1:
                feat = self.lrelu(feat)
                upsample_feat = self.upsample(feat)
            last_att = att
        feat = self.lrelu(feat)
        return feat

    def atttransfer(self, toalign_feat, att, patch_ratio):

        # att: [b*n_windows, num_heads, N, N],  N = window_size * window_size 
        # just work under num_heads=1
        # toalign_feat: [B, H, W, C]
        
        B, H, W, C = toalign_feat.shape
        window_size = self.window_size*patch_ratio
        x = toalign_feat.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C) # [b*n_windows, window_size, window_size,C]
        bn_windows = windows.shape[0]
        windows = windows.view(bn_windows, self.window_size, patch_ratio, self.window_size, patch_ratio, C)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().flatten(3) #[bn_windows, self.window_size, self.window_size, patch_ratio*patch_ratio*C]
        windows = windows.view(bn_windows, self.window_size*self.window_size, -1)

        att = att.squeeze(dim=1) # [bn_windows, N, N]

        atttransfer_feat = att@windows  # [bn_windows, N, patch_ratio*patch_ratio*C]

        # reverse
        atttransfer_feat = atttransfer_feat.view(bn_windows, self.window_size, self.window_size, patch_ratio, patch_ratio, C)
        atttransfer_feat = atttransfer_feat.permute(0,1,3,2,4,5).contiguous()
        atttransfer_feat = atttransfer_feat.view(bn_windows, window_size, window_size, C)
        atttransfer_feat = atttransfer_feat.view(B, H//window_size, W//window_size, window_size, window_size, C)
        atttransfer_feat = atttransfer_feat.permute(0,1,3,2,4,5).contiguous()
        atttransfer_feat = atttransfer_feat.view(B, H, W, C)
        return atttransfer_feat


    def atttransfer_multiheads(self, toalign_feat, att, patch_ratio):
        # att: [b*n_windows, num_heads, N, N],  N = window_size * window_size 
        # can work under 1<= num_heads <= C 
        # toalign_feat: [B, H, W, C]
        B, H, W, C = toalign_feat.shape
        window_size = self.window_size*patch_ratio
        x = toalign_feat.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C) # [b*n_windows, window_size, window_size,C]
        bn_windows = windows.shape[0]
        windows = windows.view(bn_windows, self.window_size, patch_ratio, self.window_size, patch_ratio, self.num_heads, C//self.num_heads)
        windows = windows.permute(0, 5, 1, 3, 2, 4, 6).contiguous().flatten(3) #[bn_windows, num_heads, self.window_size, self.window_size, patch_ratio*patch_ratio*C//num_heads]
        windows = windows.view(bn_windows, self.num_heads, self.window_size*self.window_size, -1)

        #att = att.squeeze(dim=1) # [bn_windows, N, N]

        atttransfer_feat = att@windows  # [bn_windows, self.num_heads, N, patch_ratio*patch_ratio*C//num_heads]

        # reverse
        atttransfer_feat = atttransfer_feat.view(bn_windows, self.num_heads, self.window_size, self.window_size, patch_ratio, patch_ratio, C//self.num_heads)
        atttransfer_feat = atttransfer_feat.permute(0,2,4,3,5,1,6).contiguous()
        atttransfer_feat = atttransfer_feat.view(bn_windows, window_size, window_size, C)
        atttransfer_feat = atttransfer_feat.view(B, H//window_size, W//window_size, window_size, window_size, C)
        atttransfer_feat = atttransfer_feat.permute(0,1,3,2,4,5).contiguous()
        atttransfer_feat = atttransfer_feat.view(B, H, W, C)
        return atttransfer_feat

 
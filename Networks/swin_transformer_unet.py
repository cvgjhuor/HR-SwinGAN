import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


import torch.nn.functional as F

class Mlp(nn.Module):
    """Multi-Layer Perceptron block for feature transformation in attention blocks."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """Initialize two-layer linear transformation with activation and dropout."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Apply MLP transformation."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network (GDFN) from Restormer"""
    def __init__(self, dim, expansion_factor=2.66, bias=False, drop=0.):
        super(GDFN, self).__init__()

        hidden_features = int(dim * expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, L, C)
        Returns:
            Output tensor of shape (B, L, C)
        """
        B, L, C = x.shape
        H = int(L ** 0.5)
        W = H
        assert L == H * W, "Input feature has wrong size"

        # (B, L, C) -> (B, C, H, W)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        # (B, C, H, W) -> (B, L, C)
        x = x.permute(0, 2, 3, 1).view(B, L, C)
        
        x = self.drop(x)
        return x


class ShiftConv2d(nn.Module):
    """
    Shift-Convolution module from ESTN.
    Uses pixel shifting + 1x1 Conv to simulate 3x3 Conv with low cost.
    """
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d, self).__init__()    
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        # Initialize weight for shifting (fixed, no gradient)
        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div
        
        # Define shift patterns
        self.weight[0*g:1*g, 0, 1, 2] = 1.0 ## left
        self.weight[1*g:2*g, 0, 1, 0] = 1.0 ## right
        self.weight[2*g:3*g, 0, 2, 1] = 1.0 ## up
        self.weight[3*g:4*g, 0, 0, 1] = 1.0 ## down
        self.weight[4*g:, 0, 1, 1] = 1.0 ## identity     

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, L, C)
        Returns:
            Output tensor of shape (B, L, C)
        """
        B, L, C = x.shape
        H = int(L ** 0.5)
        W = H
        
        # (B, L, C) -> (B, C, H, W)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        
        # Pixel Shift
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        
        # 1x1 Conv fusion
        y = self.conv1x1(y) 
        
        # (B, C, H, W) -> (B, L, C)
        y = y.permute(0, 2, 3, 1).view(B, L, C)
        
        return y


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape  # x.shape:(1,64,64,1)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)  # (1,8,8,8,8,1)
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


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
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

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 or qk_scale

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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    ########################################################################
        from .irpe import get_rpe_config, build_rpe
        rpe_config = get_rpe_config(
            ratio=1.9,
            method="product",
            mode='ctx',
            shared_head=True,  # irpe
            skip=1,
            rpe_on='k',
        )

        self.rpe_q, self.rpe_k, self.rpe_v = \
            build_rpe(rpe_config,
                      head_dim=self.head_dim,
                      num_heads=num_heads)

    #########################################################################
    # def forward(self, x, mask=None):
    #     """
    #     Args:
    #         x: input features with shape of (num_windows*B, N, C)
    #         mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
    #     """
    #     B_, N, C = x.shape
    #     qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    #     q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

    #     q = q * self.scale
    #     attn = (q @ k.transpose(-2, -1))

    #     relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
    #         self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    #     relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    #     attn = attn + relative_position_bias.unsqueeze(0)

    #     if mask is not None:
    #         nW = mask.shape[0]
    #         attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    #         attn = attn.view(-1, self.num_heads, N, N)
    #         attn = self.softmax(attn)
    #     else:
    #         attn = self.softmax(attn)

    #     attn = self.attn_drop(attn)

    #     x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x

    # # irpe_forward
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
    
        q = q * self.scale
    
        attn = (q @ k.transpose(-2, -1))
        if self.rpe_k is not None:
            attn += self.rpe_k(q)
    
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)
    
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)
    
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
    
        attn = self.attn_drop(attn)
        out = attn @ v
        if self.rpe_v is not None:
            out += self.rpe_v(attn)
    
        x = out.transpose(1, 2).reshape(B_, N, C)
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


class WindowMultiScaleAttention(nn.Module):
    """
    Window Multi-scale Self-Attention (W-MSSA/SW-MSSA) from ESTN.
    Splits channels into 3 scales with different window sizes: 4, 8, 16.
    """
    def __init__(self, dim, num_heads, input_resolution, window_sizes=[4, 8, 16], qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0., shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.input_resolution = input_resolution
        self.window_sizes = list(window_sizes) # Make a copy
        self.shift_size = shift_size
        
        # Check divisibility
        if num_heads % 3 != 0:
            print(f"Warning: num_heads ({num_heads}) is not divisible by 3. This may cause uneven split in MSSA.")
        
        # Split dim and heads into 3 parts
        self.dims = [dim // 3] * 3
        # Adjust for divisibility
        self.dims[2] += dim - sum(self.dims)
        
        self.heads = [num_heads // 3] * 3
        self.heads[2] += num_heads - sum(self.heads)
        
        self.attns = nn.ModuleList()
        self.masks = nn.ParameterList() # Use ParameterList to hold buffers effectively or just a list of buffers
        
        H, W = self.input_resolution
        
        for i in range(3):
            # Adjust window size if larger than input resolution
            if min(H, W) <= self.window_sizes[i]:
                self.window_sizes[i] = min(H, W)
                # If window size equals input size, shift size must be 0
                current_shift = 0
            else:
                # Use provided shift_size (global) to determine local shift
                # In ESTN, shift is typically window_size // 2
                current_shift = self.window_sizes[i] // 2 if self.shift_size > 0 else 0
                
            self.attns.append(
                WindowAttention(
                    dim=self.dims[i], 
                    window_size=to_2tuple(self.window_sizes[i]), 
                    num_heads=self.heads[i],
                    qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    attn_drop=attn_drop, proj_drop=proj_drop
                )
            )
            
            # Pre-calculate Mask
            if current_shift > 0:
                img_mask = torch.zeros((1, H, W, 1))
                h_slices = (slice(0, -self.window_sizes[i]),
                            slice(-self.window_sizes[i], -current_shift),
                            slice(-current_shift, None))
                w_slices = (slice(0, -self.window_sizes[i]),
                            slice(-self.window_sizes[i], -current_shift),
                            slice(-current_shift, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
                
                mask_windows = window_partition(img_mask, self.window_sizes[i])
                mask_windows = mask_windows.view(-1, self.window_sizes[i] * self.window_sizes[i])
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
                
                # Register as buffer
                self.register_buffer(f"attn_mask_{i}", attn_mask)
            else:
                self.register_buffer(f"attn_mask_{i}", None)
            
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        """
        Args:
            x: Input tensor of shape (B, L, C)
            H, W: Spatial resolution
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H == self.input_resolution[0] and W == self.input_resolution[1], "input resolution mismatch"
        
        x = x.view(B, H, W, C)
        
        # Split into 3 scales
        x_scales = torch.split(x, self.dims, dim=-1)
        out_scales = []
        
        for i in range(3):
            xi = x_scales[i]
            win_size = self.window_sizes[i]
            # Determine shift size based on window size and global config
            # We need to replicate the logic from __init__ to match the mask
            if min(H, W) <= win_size:
                s_size = 0
            else:
                s_size = win_size // 2 if self.shift_size > 0 else 0
            
            # 1. Cyclic shift
            if s_size > 0:
                xi = torch.roll(xi, shifts=(-s_size, -s_size), dims=(1, 2))
            
            # 2. Partition windows
            xi_windows = window_partition(xi, win_size) # nW*B, win, win, C_i
            xi_windows = xi_windows.view(-1, win_size * win_size, self.dims[i])
            
            # 3. Attention
            mask = getattr(self, f"attn_mask_{i}")
            attn_out = self.attns[i](xi_windows, mask=mask)
            
            # 4. Reverse partition
            attn_out = attn_out.view(-1, win_size, win_size, self.dims[i])
            xi = window_reverse(attn_out, win_size, H, W)
            
            # 5. Reverse shift
            if s_size > 0:
                xi = torch.roll(xi, shifts=(s_size, s_size), dims=(1, 2))
            
            out_scales.append(xi)
            
        # Concatenate and project
        x = torch.cat(out_scales, dim=-1)
        x = x.view(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

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

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_gdfn=False, use_shift_conv=False,
                 block_idx=0, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_gdfn = use_gdfn
        
        # [ESTN] Hard Isolation Switches
        # Priority: kwargs > config object > legacy use_shift_conv
        self.use_shift_conv_internal = kwargs.get('use_shift_conv_internal', False)
        self.use_shift_conv_external = kwargs.get('use_shift_conv_external', False)
        
        # Fallback for config object
        if 'config' in kwargs:
             config = kwargs['config']
             if 'use_shift_conv_internal' not in kwargs:
                 self.use_shift_conv_internal = getattr(config, 'use_shift_conv_internal', False)
             if 'use_shift_conv_external' not in kwargs:
                 self.use_shift_conv_external = getattr(config, 'use_shift_conv_external', False)
        
        # Legacy support: if use_shift_conv is True but specific flags are not set, enable external (default behavior of previous code)
        if use_shift_conv and not self.use_shift_conv_internal and not self.use_shift_conv_external:
             self.use_shift_conv_external = True
             
        self.block_idx = block_idx

        # Hard Isolation: ESTN Optional Settings
        # Legacy Alternating strategy (buggy; keeps old compatibility path)
        if kwargs.get('use_estn_alternating', False):
            self.use_estn_alternating = True

        # Fixed ESTN alternating strategy switch.
        if kwargs.get('use_estn_alternating_v2', False):
            self.use_estn_alternating_v2 = True
            
        if kwargs.get('use_estn_bsgpm', False):
            self.use_estn_bsgpm = True
            self.bsgpm = ESTN_BSGPM(dim=dim)
        if kwargs.get('use_estn_lrcam', False):
            self.use_estn_lrcam = True
            self.lrcam = ESTN_LRCAM(dim=dim)
            
        # Hard Isolation: RCAB (Official ESTN Implementation)
        if kwargs.get('use_estn_rcab', False):
            self.use_estn_rcab = True
            reduction = kwargs.get('rcab_reduction', 4)
            lrelu_slope = kwargs.get('rcab_lrelu_slope', 0.2)
            self.rcab = ESTN_RCAB(
                dim=dim,
                reduction=reduction,
                lrelu_slope=lrelu_slope,
                use_bias=True
            )
        
        # Hard Isolation: MSSA (Multi-scale Self-Attention)
        if kwargs.get('use_estn_mssa', False):
            self.use_estn_mssa = True
            self.mssa = WindowMultiScaleAttention(
                dim, num_heads, input_resolution=input_resolution,
                window_sizes=[4, 8, 16], 
                qkv_bias=qkv_bias, qk_scale=qk_scale, 
                attn_drop=attn_drop, proj_drop=drop, 
                shift_size=shift_size
            )

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        # Hard Isolation: Shift Convolution
        # 1. Internal ShiftConv (Inside Attention Residual)
        if self.use_shift_conv_internal:
            self.shift_conv_internal = ShiftConv2d(inp_channels=dim, out_channels=dim)
            
        # 2. External ShiftConv (Independent Residual Block)
        if self.use_shift_conv_external:
            self.shift_conv_external = ShiftConv2d(inp_channels=dim, out_channels=dim)
        
        # Hard Isolation: Only instantiate the selected module
        if self.use_gdfn:
            self.mlp = GDFN(dim=dim, expansion_factor=2.66, drop=drop)
        else:
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
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
            mask_windows = mask_windows.view(-1, (self.window_size) * (self.window_size))
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # --- ESTN Alternating Strategy Logic (Hard Isolation) ---
        # Read alternating-strategy switches from the block state.
        is_alternating_old = getattr(self, 'use_estn_alternating', False)  # Legacy alternating path (known buggy behavior)
        is_alternating_v2 = getattr(self, 'use_estn_alternating_v2', False)  # Fixed alternating path
        block_idx = getattr(self, 'block_idx', 0)
        
        # In alternating_v2, every block still runs attention.
        if is_alternating_v2:
            # Alternation is handled by shift_size in BasicLayer:
            # shift_size = 0 -> W-MSA on even blocks
            # shift_size = window_size // 2 -> SW-MSA on odd blocks
            run_attn = True  # All blocks keep attention enabled
            run_external_sc = True  # All blocks keep external ShiftConv enabled
        
        elif is_alternating_old:
            # Legacy alternating path keeps its historical buggy behavior.
            run_attn = True
            if block_idx % 2 == 0:
                run_attn = False  # Known legacy bug: disables attention on alternating blocks
            
            run_external_sc = False
            if hasattr(self, 'shift_conv_external'):
                run_external_sc = True
                if block_idx % 2 != 0:
                    run_external_sc = False
        else:
            # Default path: standard attention plus optional external ShiftConv.
            run_attn = True
            run_external_sc = hasattr(self, 'shift_conv_external')

        # 1. Attention Stage
        # Attention stage: BasicLayer already controls shift_size per block.
        if run_attn:
            shortcut = x
            x = self.norm1(x)
            
            if getattr(self, 'use_estn_mssa', False):
                # W-MSSA/SW-MSSA handles partitioning and shifting internally
                x = self.mssa(x, H, W)
            else:
                x = x.view(B, H, W, C)
                # cyclic shift
                if self.shift_size > 0:
                    shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                else:
                    shifted_x = x

                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)
                x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

                # W-MSA/SW-MSA
                attn_windows = self.attn(x_windows, mask=self.attn_mask)

                # merge windows
                attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)

                # reverse cyclic shift
                if self.shift_size > 0:
                    x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                else:
                    x = shifted_x
                x = x.view(B, H * W, C)
            
            # [ESTN] Internal ShiftConv (Hard Isolation)
            # Executed on the attention output BEFORE residual connection
            if hasattr(self, 'shift_conv_internal'):
                 # Note: No residual here, it modifies the attention output directly (like in shift.txt)
                 x = self.shift_conv_internal(x)

            x = shortcut + self.drop_path(x)
        # --- End of Attention Stage ---

        # 2. Local Feature Enhancement Stage (External ShiftConv)
        # [ESTN] External ShiftConv (Hard Isolation)
        # Executed as an independent residual block
        if run_external_sc and hasattr(self, 'shift_conv_external'):
            x = x + self.shift_conv_external(x)
            
        # 3. Feed-Forward Stage (MLP / GDFN / BSGPM)
        shortcut = x
        
        # BSGPM: Only in Global blocks when alternating
        use_bsgpm = False
        # BSGPM is only enabled on alternating global blocks when configured.
        if (is_alternating_old or is_alternating_v2) and (block_idx % 2 != 0) and getattr(self, 'use_estn_bsgpm', False):
            use_bsgpm = True
            
        if use_bsgpm:
            x = shortcut + self.drop_path(self.bsgpm(self.norm2(x), H, W))
        else:
            x = shortcut + self.drop_path(self.mlp(self.norm2(x)))

        # 4. LRCAM (Optional ESTN Improvement)
        # Hard Isolation Execution Logic
        # Priority: RCAB (Official) > LRCAM (Legacy)
        
        if getattr(self, 'use_estn_rcab', False):
            # Official RCAB implementation (includes residual connection internally)
            x = self.rcab(x, H, W)
        elif getattr(self, 'use_estn_lrcam', False):
            # Legacy LRCAM implementation
            x = x + self.lrcam(x, H, W)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, 
                 use_gdfn=False, use_shift_conv=False, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 use_gdfn=use_gdfn,
                                 use_shift_conv=use_shift_conv,
                                 block_idx=i,
                                 **kwargs)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False, 
                 use_gdfn=False, use_shift_conv=False, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 use_gdfn=use_gdfn,
                                 use_shift_conv=use_shift_conv,
                                 block_idx=i,
                                 **kwargs)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding - RSTB Helper (Hard Isolation)
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x


class RSTB(nn.Module):
    """
    Residual Swin Transformer Block (RSTB) from SwinIR/SwinMR.
    Hard Isolation: Only used when config.use_rstb is True.
    Structure: Input -> BasicLayer -> Conv2d -> Residual Add -> Output -> (Optional Down/Up Sample)
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv', **kwargs):
        super(RSTB, self).__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        
        # Reuse existing BasicLayer (Swin Transformer Blocks)
        # Note: We pass downsample=None because RSTB handles resizing AFTER the residual connection
        self.residual_group = BasicLayer(
            dim=dim, input_resolution=input_resolution, depth=depth, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer,
            downsample=None, use_checkpoint=use_checkpoint, **kwargs
        )
        
        self.resi_connection = resi_connection
        
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1)
            )
            
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        # Handle Downsample (Encoder) or Upsample (Decoder)
        self.downsample = downsample
        if self.downsample is not None:
             self.downsample_layer = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
             self.downsample_layer = None
             
        self.upsample = upsample
        if self.upsample is not None:
             # BasicLayer_up logic for PatchExpand
             # Note: BasicLayer_up uses 'dim_scale=2' by default inside, we need to match that if passing class
             # But here we assume 'upsample' is the class (PatchExpand)
             self.upsample_layer = upsample(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
             self.upsample_layer = None

    def forward(self, x):
        return self.forward_rstb(x, self.input_resolution)

    def forward_rstb(self, x, input_resolution):
        H, W = input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        
        # 1. Swin Transformer Blocks (BasicLayer)
        res = self.residual_group(x) # (B, L, C)
        
        # 2. Reshape for Conv
        # (B, L, C) -> (B, C, H, W)
        res = self.patch_unembed(res, (H, W)) 
        
        # 3. Convolution
        res = self.conv(res)
        
        # 4. Reshape back (Flatten)
        # (B, C, H, W) -> (B, L, C)
        res = res.flatten(2).transpose(1, 2)
        
        # 5. Residual Connection
        x = res + shortcut
        
        # 6. Optional Downsample / Upsample
        if self.downsample_layer is not None:
            x = self.downsample_layer(x)
            
        if self.upsample_layer is not None:
            x = self.upsample_layer(x)
            
        return x


class SwinTransformerSys(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        # Optional SE settings for skip, bottleneck, and decoder features.
        # Prefer explicit kwargs and only fall back to config when needed.
        # This keeps hard-isolation behavior predictable across call sites.
        self.use_se = kwargs.get('use_se_block', False)
        self.se_reduction = kwargs.get('se_reduction', 4)
        self.se_residual = kwargs.get('se_residual', False)
        self.use_se_bottleneck = kwargs.get('use_se_bottleneck', False)
        self.use_se_decoder = kwargs.get('use_se_decoder', False)
        use_standard_se = kwargs.get('use_standard_se', False)
        
        # Optional GDFN feed-forward replacement.
        self.use_gdfn = kwargs.get('use_gdfn', False)
        
        # Optional ESTN ShiftConv switches.
        self.use_shift_conv = kwargs.get('use_shift_conv', False)
        self.use_shift_conv_internal = kwargs.get('use_shift_conv_internal', False)
        self.use_shift_conv_external = kwargs.get('use_shift_conv_external', False)

        # Optional ESTN feature modules.
        self.use_estn_bsgpm = kwargs.get('use_estn_bsgpm', False)
        self.use_estn_lrcam = kwargs.get('use_estn_lrcam', False)
        self.use_estn_alternating = kwargs.get('use_estn_alternating', False)
        self.use_estn_mssa = kwargs.get('use_estn_mssa', False)
        
        # Optional RSTB / SwinMR settings.
        self.use_rstb = kwargs.get('use_rstb', False)
        self.rstb_depth = kwargs.get('rstb_depth', 6)
        self.rstb_num_heads = kwargs.get('rstb_num_heads', 6)
        self.rstb_resi_connection = kwargs.get('rstb_resi_connection', '1conv')
        self.use_dynamic_heads = kwargs.get('use_dynamic_heads', False)
        self.use_symmetric_decoder = kwargs.get('use_symmetric_decoder', False)
        self.use_full_decoder = kwargs.get('use_full_decoder', False)
        
        # Hard Isolation: RCAB Config
        self.use_estn_rcab = kwargs.get('use_estn_rcab', False)
        self.rcab_reduction = kwargs.get('rcab_reduction', 4)
        self.rcab_lrelu_slope = kwargs.get('rcab_lrelu_slope', 0.2)

        # Fixed alternating_v2 switch.
        self.use_estn_alternating_v2 = kwargs.get('use_estn_alternating_v2', False)

        # Use the config object only as a fallback for missing kwargs.
        if 'config' in kwargs:
            config = kwargs['config']
            # Respect explicit kwargs first to preserve hard-isolation behavior.
            if 'use_se_block' not in kwargs:
                self.use_se = getattr(config, 'use_se_block', False)
            if 'se_reduction' not in kwargs:
                self.se_reduction = getattr(config, 'se_reduction', 4)
            if 'se_residual' not in kwargs:
                self.se_residual = getattr(config, 'se_residual', False)
            if 'use_se_bottleneck' not in kwargs:
                self.use_se_bottleneck = getattr(config, 'use_se_bottleneck', False)
            if 'use_se_decoder' not in kwargs:
                self.use_se_decoder = getattr(config, 'use_se_decoder', False)
            if 'use_standard_se' not in kwargs:
                use_standard_se = getattr(config, 'use_standard_se', False)
            if 'use_gdfn' not in kwargs:
                self.use_gdfn = getattr(config, 'use_gdfn', False)
            if 'use_shift_conv' not in kwargs:
                self.use_shift_conv = getattr(config, 'use_shift_conv', False)

            # ESTN ShiftConv Switches
            if 'use_shift_conv_internal' not in kwargs:
                self.use_shift_conv_internal = getattr(config, 'use_shift_conv_internal', False)
            if 'use_shift_conv_external' not in kwargs:
                self.use_shift_conv_external = getattr(config, 'use_shift_conv_external', False)

            # Fallback ESTN module settings from config.
            if 'use_estn_bsgpm' not in kwargs:
                self.use_estn_bsgpm = getattr(config, 'use_estn_bsgpm', False)
            if 'use_estn_lrcam' not in kwargs:
                self.use_estn_lrcam = getattr(config, 'use_estn_lrcam', False)
            if 'use_estn_alternating' not in kwargs:
                self.use_estn_alternating = getattr(config, 'use_estn_alternating', False)
            if 'use_estn_mssa' not in kwargs:
                self.use_estn_mssa = getattr(config, 'use_estn_mssa', False)

            # RSTB (SwinMR)
            if 'use_rstb' not in kwargs:
                self.use_rstb = getattr(config, 'use_rstb', False)
            if 'rstb_depth' not in kwargs:
                self.rstb_depth = getattr(config, 'rstb_depth', 6)
            if 'rstb_num_heads' not in kwargs:
                self.rstb_num_heads = getattr(config, 'rstb_num_heads', 6)
            if 'rstb_resi_connection' not in kwargs:
                self.rstb_resi_connection = getattr(config, 'rstb_resi_connection', '1conv')

            # ESTN V2 Config
            if 'use_estn_alternating_v2' not in kwargs:
                self.use_estn_alternating_v2 = getattr(config, 'use_estn_alternating_v2', False)

            # ESTN RCAB Config
            if 'use_estn_rcab' not in kwargs:
                self.use_estn_rcab = getattr(config, 'use_estn_rcab', False)
            if 'rcab_reduction' not in kwargs:
                self.rcab_reduction = getattr(config, 'rcab_reduction', 4)
            if 'rcab_lrelu_slope' not in kwargs:
                self.rcab_lrelu_slope = getattr(config, 'rcab_lrelu_slope', 0.2)

            # [Hard Isolation Switches]
            if 'use_dynamic_heads' not in kwargs:
                self.use_dynamic_heads = getattr(config, 'use_dynamic_heads', False)
            if 'use_symmetric_decoder' not in kwargs:
                self.use_symmetric_decoder = getattr(config, 'use_symmetric_decoder', False)
            if 'use_full_decoder' not in kwargs:
                self.use_full_decoder = getattr(config, 'use_full_decoder', False)

            # Fix: Ensure Decoder is symmetric to Encoder (unless explicitly configured)
            # This matches the behavior of the original BasicLayer path which uses depths[::-1]
            if hasattr(config, 'depths_decoder'):
                self.depths_decoder = config.depths_decoder
            elif self.use_symmetric_decoder:
                self.depths_decoder = list(depths)[::-1]
                # print(f"[Config Auto-Fix] Set symmetric decoder depths: {self.depths_decoder}")
            else:
                # If symmetric decoder is disabled, use default [1,2,2,2] passed to __init__
                self.depths_decoder = depths_decoder

            # Sync local variable for downstream usage (dpr calc, loop, etc.)
            depths_decoder = self.depths_decoder

        # Pass ESTN switches down to child blocks.
        kwargs['use_estn_bsgpm'] = self.use_estn_bsgpm
        kwargs['use_estn_lrcam'] = self.use_estn_lrcam
        kwargs['use_estn_alternating'] = self.use_estn_alternating
        kwargs['use_estn_mssa'] = self.use_estn_mssa
        
        # Pass alternating_v2 explicitly to child blocks.
        kwargs['use_estn_alternating_v2'] = getattr(self, 'use_estn_alternating_v2', False)
        
        # Inject RCAB params
        kwargs['use_estn_rcab'] = self.use_estn_rcab
        kwargs['rcab_reduction'] = self.rcab_reduction
        kwargs['rcab_lrelu_slope'] = self.rcab_lrelu_slope
        
        # Ensure GDFN and ShiftConv are also in kwargs (and consistent with self attributes)
        kwargs['use_gdfn'] = self.use_gdfn
        kwargs['use_shift_conv'] = self.use_shift_conv
        kwargs['use_shift_conv_internal'] = self.use_shift_conv_internal
        kwargs['use_shift_conv_external'] = self.use_shift_conv_external

        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes))
        
        if use_standard_se:
            se_mode = "standard"
        else:
            se_mode = "residual" if self.se_residual else "scaling"
        if self.use_se:
            print(f"SE Block Enabled in Skip Connections with reduction={self.se_reduction}, mode={se_mode}")
        if self.use_se_bottleneck:
            print(f"SE Block Enabled in Bottleneck with reduction={self.se_reduction}, mode={se_mode}")
        if self.use_se_decoder:
            print(f"SE Block Enabled in Decoder Stages with reduction={self.se_reduction}, mode={se_mode}")
        if self.use_gdfn:
            print(f"GDFN (Gated-Dconv Feed-Forward Network) Enabled in Swin Transformer Blocks")
        if self.use_shift_conv:
            print(f"Shift Convolution (ESTN) Enabled in Swin Transformer Blocks (Legacy Switch)")
            
        if self.use_shift_conv_internal:
            print(f"[ESTN] Internal ShiftConv ENABLED (Inside Attention)")
        if self.use_shift_conv_external:
            print(f"[ESTN] External ShiftConv ENABLED (Independent Residual Block)")
        
        if self.use_estn_bsgpm:
            print(f"ESTN Improvement: BSGPM (Block Sparse Global Perception Module) Enabled")
        if self.use_estn_lrcam:
            print(f"ESTN Improvement: LRCAM (Low-parameter Residual Channel Attention) Enabled")
        if self.use_estn_alternating:
            print(f"ESTN Strategy: Alternating Local-Global Feature Aggregation Enabled")
            
        if self.use_estn_alternating_v2:
            print("[ESTN] Alternating Strategy V2 (Fixed) Enabled: All blocks perform Attention with alternating shift_size")
        elif self.use_estn_alternating:
            print("[WARNING] Using old Alternating Strategy (Bug: disables 50% Attention)")
            
        if self.use_estn_mssa:
            print(f"ESTN Improvement: Multi-scale Self-Attention (W-MSSA/SW-MSSA) Enabled")
        
        # Hard Isolation: RCAB Logging
        if self.use_estn_rcab:
            print(f"ESTN Improvement: RCAB (Official Implementation) Enabled with reduction={self.rcab_reduction}, slope={self.rcab_lrelu_slope}")

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        if self.use_rstb:
             # RSTB mode: depth varies by layer (optimized)
             # Use the actual depths list to calculate total depth
             total_depth = sum(depths) + sum(depths_decoder) # Approximate total depth
             dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        else:
             # BasicLayer mode: depth varies by layer
             dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ============================================
        # Build the encoder and decoder stacks.
        # ============================================
        if self.use_rstb:
            # RSTB path: both encoder and decoder use RSTB blocks.
            print(f"[Hard Isolation] RSTB Enabled: Dynamic Depths from config={depths}, dynamic heads={self.use_dynamic_heads}, symmetric decoder={self.use_symmetric_decoder}, full decoder={self.use_full_decoder}")

            # ---- Encoder RSTB Layers ----
            self.layers = nn.ModuleList()
            dpr_offset = 0
            for i_layer in range(self.num_layers):
                current_depth = depths[i_layer] # Use config depth instead of fixed rstb_depth
                
                # Dynamic Heads Logic
                if self.use_dynamic_heads:
                    current_num_heads = num_heads[i_layer]
                else:
                    current_num_heads = self.rstb_num_heads
                
                layer = RSTB(
                    dim=int(embed_dim * 2 ** i_layer),
                    input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                      patches_resolution[1] // (2 ** i_layer)),
                    depth=current_depth, # Dynamic depth
                    num_heads=current_num_heads, # Dynamic num_heads
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[dpr_offset : dpr_offset + current_depth], # Correct dpr slice
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    img_size=img_size,
                    patch_size=patch_size,
                    resi_connection=self.rstb_resi_connection,
                    **kwargs
                )
                self.layers.append(layer)
                dpr_offset += current_depth

            # ---- Decoder RSTB Layers ----
            self.layers_up = nn.ModuleList()
            self.concat_back_dim = nn.ModuleList()
            # Decoder depths are usually symmetric or specified in depths_decoder
            # Assuming depths_decoder is [1, 2, 2, 2] corresponding to layers [3, 2, 1, 0] reversed?
            # Or usually decoder layers correspond to encoder layers reversed.
            # Let's use depths_decoder provided in __init__
            
            # Need to manage dpr for decoder separately or continue from encoder?
            # Original code only calculated dpr for encoder (sum(depths)). 
            # But RSTB adds blocks to decoder too. 
            # Let's assume dpr continues increasing for decoder layers.
            
            for i_layer in range(self.num_layers):
                concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                          int(embed_dim * 2 ** (
                                                      self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
                if i_layer == 0 and not self.use_full_decoder:
                    # Decoder stage 0 can use plain PatchExpand when the full decoder is disabled.
                    layer_up = PatchExpand(
                        input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                          patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                        dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
                else:
                    # Other decoder stages use RSTB blocks with decoder-specific depth.
                    # Use depths_decoder for depth control
                    # i_layer=1 corresponds to depths_decoder[1] etc.
                    current_depth = depths_decoder[i_layer]
                    
                    # Dynamic Heads Logic
                    if self.use_dynamic_heads:
                        current_num_heads = num_heads[self.num_layers - 1 - i_layer]
                    else:
                        current_num_heads = self.rstb_num_heads

                    layer_up = RSTB(
                        dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                        input_resolution=(
                            patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                            patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                        depth=current_depth, # Dynamic depth
                        num_heads=current_num_heads, # Dynamic num_heads
                        window_size=window_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[dpr_offset : dpr_offset + current_depth], # Continue dpr
                        norm_layer=norm_layer,
                        downsample=None,  # Decoder path does not downsample here
                        upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,  # Decoder upsamples except at the final stage
                        img_size=img_size,
                        patch_size=patch_size,
                        resi_connection=self.rstb_resi_connection,
                        **kwargs
                    )
                    dpr_offset += current_depth
                self.layers_up.append(layer_up)
                self.concat_back_dim.append(concat_linear)

        else:
            # ====== Standard Backbone: BasicLayer + BasicLayer_up ======

            # ---- Encoder BasicLayer ----
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                     patches_resolution[1] // (2 ** i_layer)),
                                   depth=depths[i_layer],
                                   num_heads=num_heads[i_layer],
                                   window_size=window_size,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                   use_checkpoint=use_checkpoint,
                                   **kwargs)
                self.layers.append(layer)

            # ---- Decoder BasicLayer_up ----
            self.layers_up = nn.ModuleList()
            self.concat_back_dim = nn.ModuleList()
            for i_layer in range(self.num_layers):
                concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                          int(embed_dim * 2 ** (
                                                      self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
                if i_layer == 0:
                    layer_up = PatchExpand(
                        input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                          patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                        dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
                else:
                    layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                             input_resolution=(
                                             patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                             patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                             depth=depths[(self.num_layers - 1 - i_layer)],
                                             num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                             window_size=window_size,
                                             mlp_ratio=self.mlp_ratio,
                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop_rate, attn_drop=attn_drop_rate,
                                             drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                                 depths[:(self.num_layers - 1 - i_layer) + 1])],
                                             norm_layer=norm_layer,
                                             upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                             use_checkpoint=use_checkpoint,
                                             **kwargs)
                self.layers_up.append(layer_up)
                self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        # Optional SE blocks for skip connections.
        if self.use_se:
            self.se_blocks = nn.ModuleList()
            from .lrcab import LRCAB as SEBlock
            # Skip features are consumed in reverse encoder order: 384 -> 192 -> 96.
            # layers_up[0] matches encoder stage 2, then stage 1, then stage 0.
            # This keeps skip-channel sizes aligned with the decoder stages.
            # forward_up_features reads x_downsample in that reverse order.
            
            # inx=1..3 maps to x_downsample[2], x_downsample[1], x_downsample[0].
            # inx=1 (i=0) -> skip: x_downsample[2] (384)
            # inx=2 (i=1) -> skip: x_downsample[1] (192)
            # inx=3 (i=2) -> skip: x_downsample[0] (96)
            
            # Apply SE to encoder skip features before concatenation.
            # Encoder features are stored in x_downsample list
            # x_downsample[0]: 96, [1]: 192, [2]: 384
            
            skip_channels = [
                int(embed_dim * 2 ** 2), # 384 (for inx=1)
                int(embed_dim * 2 ** 1), # 192 (for inx=2)
                int(embed_dim * 2 ** 0)  # 96  (for inx=3)
            ]
            
            if use_standard_se:
                from .lrcab import StandardSEBlock
                for ch in skip_channels:
                    self.se_blocks.append(StandardSEBlock(ch, reduction=self.se_reduction))
            else:
                from .lrcab import LRCAB as SEBlock
                for ch in skip_channels:
                    self.se_blocks.append(SEBlock(ch, reduction=self.se_reduction, use_residual=self.se_residual))

        # Optional SE block at the bottleneck.
        if self.use_se_bottleneck:
            # Bottleneck channels = embed_dim * 2**(num_layers-1) = 96 * 8 = 768
            bottleneck_dim = int(embed_dim * 2 ** (self.num_layers - 1))
            if use_standard_se:
                from .lrcab import StandardSEBlock
                self.se_bottleneck = StandardSEBlock(bottleneck_dim, reduction=self.se_reduction)
            else:
                from .lrcab import LRCAB as SEBlock
                self.se_bottleneck = SEBlock(bottleneck_dim, reduction=self.se_reduction, use_residual=self.se_residual)

        # Optional SE blocks after decoder stages.
        if self.use_se_decoder:
            self.se_decoder_layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                # Input dim to layer_up
                dim = int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
                
                if i_layer == 0:
                    # PatchExpand: reduces by 2
                    out_dim = dim // 2
                else:
                    # BasicLayer_up
                    # check upsample condition: i_layer < self.num_layers - 1
                    if i_layer < self.num_layers - 1:
                        out_dim = dim // 2
                    else:
                        out_dim = dim
                
                if use_standard_se:
                    from .lrcab import StandardSEBlock
                    self.se_decoder_layers.append(StandardSEBlock(out_dim, reduction=self.se_reduction))
                else:
                    from .lrcab import LRCAB as SEBlock
                    self.se_decoder_layers.append(SEBlock(out_dim, reduction=self.se_reduction, use_residual=self.se_residual))

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed  # absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        # Apply bottleneck SE after encoder normalization.
        if self.use_se_bottleneck:
            x = self.se_bottleneck(x)

        return x, x_downsample

    # Decoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                # Fetch the matching encoder skip feature.
                skip_feat = x_downsample[3 - inx]
                
                # Apply SE to skip features when enabled.
                if self.use_se:
                    # se_blocks[0] -> inx=1
                    # se_blocks[1] -> inx=2
                    # se_blocks[2] -> inx=3
                    skip_feat = self.se_blocks[inx-1](skip_feat)
                
                x = torch.cat([x, skip_feat], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

            # Apply decoder-side SE after each up block.
            if self.use_se_decoder:
                x = self.se_decoder_layers[inx](x)

        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)

        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class ESTN_BSGPM(nn.Module):
    """
    Block Sparse Global Perception Module (BSGPM) from ESTN.
    Reorganizes spatial info to capture global dependencies via a dense layer.
    """
    def __init__(self, dim, ratio=2):
        super(ESTN_BSGPM, self).__init__()
        self.dim = dim
        self.ratio = ratio
        # PixelUnshuffle: (B, C, H, W) -> (B, C*r^2, H/r, W/r)
        self.unshuffle = nn.PixelUnshuffle(ratio)
        mid_dim = dim * (ratio ** 2)
        self.dense = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, mid_dim)
        )
        self.shuffle = nn.PixelShuffle(ratio)

    def forward(self, x, H, W):
        # x: (B, L, C)
        B, L, C = x.shape
        # Ensure H, W are divisible by ratio for PixelUnshuffle
        # Fallback: If dimensions mismatch, return input directly to avoid crash
        if H % self.ratio != 0 or W % self.ratio != 0:
            return x
            
        feat = x.view(B, H, W, C).permute(0, 3, 1, 2) # B, C, H, W
        
        # Spatial Reorganization
        out = self.unshuffle(feat) # B, C*r^2, H/r, W/r
        
        # Dense Perception
        B2, C2, H2, W2 = out.shape
        out = out.permute(0, 2, 3, 1).reshape(-1, C2) # (B*H2*W2, C2)
        out = self.dense(out)
        out = out.view(B2, H2, W2, C2).permute(0, 3, 1, 2)
        
        # Restore
        out = self.shuffle(out) # B, C, H, W
        out = out.permute(0, 2, 3, 1).reshape(B, L, C)
        return out


# ESTN helper blocks: CALayer and RCAB.
# These modules are used when the official ESTN RCAB path is enabled.


class Conv_1X1(nn.Module):
    """1x1 convolution for channel projection."""
    def __init__(self, In_ch=1, Out_ch=1, use_bias=True):
        super(Conv_1X1, self).__init__()
        self.conv = nn.Conv2d(In_ch, Out_ch, kernel_size=1, bias=use_bias)

    def forward(self, x):
        return self.conv(x)


class Conv_3X3(nn.Module):
    """3x3 convolution with same padding."""
    def __init__(self, In_ch=1, Out_ch=1, use_bias=True):
        super(Conv_3X3, self).__init__()
        self.conv = nn.Conv2d(In_ch, Out_ch, kernel_size=3, padding='same', bias=use_bias)

    def forward(self, x):
        return self.conv(x)


class ESTN_CALayer(nn.Module):
    """
    Channel Attention Layer using 1x1 convolutions.
    Input: (B, C, H, W)
    Output: (B, C, H, W)
    """
    def __init__(self, features, reduction=4, use_bias=True):
        super(ESTN_CALayer, self).__init__()
        self.features = features
        self.conv_squeeze = Conv_1X1(features, features // reduction, use_bias)
        self.conv_excitation = Conv_1X1(features // reduction, features, use_bias)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, dim=[2, 3], keepdim=True)
        y = self.conv_squeeze(y)
        y = self.relu(y)
        y = self.conv_excitation(y)
        y = self.sigmoid(y)
        return x * y


class ESTN_RCAB(nn.Module):
    """
    Residual Channel Attention Block.
    Input: (B, L, C)
    Output: (B, L, C)
    """
    def __init__(self, dim, reduction=4, lrelu_slope=0.2, use_bias=True):
        super(ESTN_RCAB, self).__init__()
        self.dim = dim
        self.conv_1 = Conv_3X3(dim, dim * 2, use_bias)
        self.conv_2 = Conv_3X3(dim * 2, dim, use_bias)
        self.leakyrelu = nn.LeakyReLU(lrelu_slope, inplace=True)
        self.ca = ESTN_CALayer(dim, reduction=reduction, use_bias=use_bias)

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        feat = x.view(B, H, W, C).permute(0, 3, 1, 2)
        feat = self.conv_1(feat)
        feat = self.leakyrelu(feat)
        feat = self.conv_2(feat)
        feat = self.ca(feat)
        feat = feat.permute(0, 2, 3, 1).view(B, L, C)
        return shortcut + feat


class ESTN_LRCAM(nn.Module):
    """
    Low-parameter Residual Channel Attention Module.
    """
    def __init__(self, dim, reduction=4):
        super(ESTN_LRCAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )
        nn.init.zeros_(self.fc[2].weight)

    def forward(self, x, H, W):
        B, L, C = x.shape
        feat = x.view(B, H, W, C).permute(0, 3, 1, 2)
        y = self.avg_pool(feat).view(B, C)
        y = self.fc(y).view(B, 1, C)
        return x * y.expand_as(x)


class SwinUnet(nn.Module):
    """Swin Transformer UNet model wrapper."""
    def __init__(self, config, img_size=256, num_classes=1, in_chans=1, **kwargs):
        """Initialize Swin Transformer UNet architecture from configuration parameters."""
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.config = config

        self.swin_unet = SwinTransformerSys(
            img_size=getattr(config, 'img_size', img_size),
            patch_size=getattr(config, 'patch_size', 4),
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=getattr(config, 'embed_dim', 96),
            depths=getattr(config, 'depths', [2, 2, 6, 2]),
            num_heads=getattr(config, 'num_heads', [3, 6, 12, 24]),
            window_size=getattr(config, 'window_size', 8),
            mlp_ratio=getattr(config, 'mlp_ratio', 4.0),
            qkv_bias=getattr(config, 'qkv_bias', True),
            qk_scale=getattr(config, 'qk_scale', None),
            drop_rate=getattr(config, 'drop_rate', 0.0),
            drop_path_rate=getattr(config, 'drop_path_rate', 0.1),
            ape=getattr(config, 'ape', False),
            patch_norm=getattr(config, 'patch_norm', True),
            use_checkpoint=getattr(config, 'use_checkpoint', False),
            config=config,
            **kwargs
        )

    def forward(self, x):
        """Forward pass through Swin Transformer UNet."""
        return self.swin_unet(x)

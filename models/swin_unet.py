import torch
import torch.nn as nn
from typing import Tuple, List
from einops import rearrange

from utils.utils_models import trunc_normal_
from models.swin_transformer_3d import SwinTransformer3DLayer

class SwinUNet(nn.Module):
    """
    Args:
        in_chans (int): Number of input channels. Default: 3
        embed_dim (int): Dimension of the token embeddings. Default: 96
        num_neighbour_frames (int): Number of neigbour frames to each side
        depths (List[int]): Depths of the Swin Transformer layers. Default: [2, 2, 6, 2].
        num_heads (List[int]): Number of attention heads for each layer. Default: [8, 8, 8, 8].
        window_size (Tuple[int]): Window size for each layer. Default: (3, 6, 6).
        mlp_ratio (float): Ratio of the mlp hidden dimension to the embedding dimension. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        use_checkpoint (bool): If True, use gradient checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 in_chans: int = 3,
                 embed_dim: int = 96,
                 num_neighbour_frames: int = 3,
                 depths: List[int] = [2, 2, 2, 6, 2],
                 num_heads: List[int] = [8, 8, 8, 8, 8],
                 window_size: Tuple[int] = (3, 6, 6),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.2,
                 norm_layer: nn.Module = nn.LayerNorm,
                 use_checkpoint: bool = False):

        super(SwinUNet, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_neighbour_frames = num_neighbour_frames

        self.conv_input = nn.Conv3d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1)
        self.conv_output = nn.Conv3d(embed_dim // 4, in_chans, kernel_size=(num_neighbour_frames + 1, 3, 3), stride=1, padding=(0,1,1))

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.num_layers = len(depths)

        self.encoding_layers = nn.ModuleList()
        for i_layer in range(1, self.num_layers - 1):
            layer = SwinTransformer3DLayer(
                dim=int(embed_dim * 2 ** (i_layer - 1)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                sampling_operation="downsample",
                use_checkpoint=use_checkpoint)
            self.encoding_layers.append(layer)

        self.decoding_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SwinTransformer3DLayer(
                dim=int(embed_dim * 2 ** (self.num_layers - 2 - i_layer)),
                depth=depths[self.num_layers - 1 - i_layer],
                num_heads=num_heads[self.num_layers - 1 - i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                          sum(depths[:self.num_layers - 1 - i_layer]):sum(depths[:self.num_layers - 1 - i_layer + 1])],
                norm_layer=norm_layer,
                sampling_operation="upsample",
                use_checkpoint=use_checkpoint)
            self.decoding_layers.append(layer)

        self.apply(self._init_weights)

    def forward_encoding(self, imgs_lq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        restored = rearrange(imgs_lq, 'b t c h w -> b c t h w')
        restored = self.conv_input(restored)

        # UNet encoder
        residual = [restored]
        for layer in self.encoding_layers:
            restored = layer(restored.contiguous())
            residual.append(restored)

        return restored, residual

    def forward_decoding(self, restored: torch.Tensor, residual: List[torch.Tensor]) -> torch.Tensor:

        # UNet decoder
        B, _, T, _, _ = restored.shape
        for i, layer in enumerate(self.decoding_layers):
            if i == 0 or i == self.num_layers - 1:
                restored = layer(restored)  # Bottleneck layer
            else:
                restored += residual[-1 - i]  # Encoder-decoder skip connection
                restored = layer(restored)  # Decoder layer

        restored = self.conv_output(restored)
        restored = rearrange(restored, 'b c t h w -> b t c h w')
        restored = restored.squeeze(dim=1)
        return restored

    def forward(self, imgs_lq: torch.Tensor) -> torch.Tensor:
        """
        Forward function.
        Args:
            imgs_lq (Tensor): Input frames with shape (b, t, c, h, w).
        """
        restored, residual = self.forward_encoding(imgs_lq)
        restored = self.forward_decoding(restored, residual)
        return restored

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

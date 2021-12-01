'''
Author: jianzhnie
Date: 2021-12-01 14:28:09
LastEditTime: 2021-12-01 17:26:36
LastEditors: jianzhnie
Description:

'''

import torch
from timm.models.layers import to_2tuple
from torch import nn as nn


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=True):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])

        self.num_pathchs = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        assert H == self.img_size[
            0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."

        assert W == self.img_size[
            1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."

        x = self.proj(x)  # trandsform image to patchs
        # torch.Size([1, 3, 224, 224]) ---> torch.Size([1, 768, 14, 14])
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        #  torch.Size([1, 768, 14, 14]) ---> torch.Size([1, 196, 768])
        x = self.norm(x)

        return x


if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    print(x.shape)
    patch_embed = PatchEmbed()
    out = patch_embed(x)
    print(out.shape)

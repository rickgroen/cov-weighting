#
# Author : Alwyn Mathew
# Monodepth in pytorch(https://github.com/alwynmathew/monodepth-pytorch)
# Bilinear sampler in pytorch(https://github.com/alwynmathew/bilinear-sampler-pytorch)
# Corrected from issue: https://github.com/alwynmathew/bilinear-sampler-pytorch/issues/1
#

import torch


def apply_disparity(img, depth, padding_mode='zeros'):
    # img: the source image (where to sample pixels) -- [B, 3, H, W]
    # depth: depth map of the target image -- [B, 1, H, W]
    # Returns: Source image warped to the target image

    b, _, h, w = depth.size()
    i_range = torch.autograd.Variable(torch.linspace(-1.0, 1.0, h).view(1, h, 1).expand(1, h, w),
                                      requires_grad=False)  # [1, H, W]  copy 0-height for w times : y coord
    j_range = torch.autograd.Variable(torch.linspace(-1.0, 1.0, w).view(1, 1, w).expand(1, h, w),
                                      requires_grad=False)  # [1, H, W]  copy 0-width for h times  : x coord

    pixel_coords = torch.stack((j_range, i_range), dim=1).float().cuda()  # [1, 2, H, W]
    batch_pixel_coords = pixel_coords[:, :, :, :].expand(b, 2, h, w).contiguous().view(b, 2, -1)  # [B, 2, H*W]

    X = batch_pixel_coords[:, 0, :] + (depth.contiguous().view(b, -1) * 2)  # [B, H*W]
    Y = batch_pixel_coords[:, 1, :]

    pixel_coords = torch.stack([X, Y], dim=2)  # [B, H*W, 2]
    pixel_coords = pixel_coords.view(b, h, w, 2)  # [B, H, W, 2]

    projected_img = torch.nn.functional.grid_sample(img, pixel_coords, padding_mode=padding_mode, align_corners=False)
    return projected_img

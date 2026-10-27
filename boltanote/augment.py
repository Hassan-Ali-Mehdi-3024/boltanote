"""Data augmentation done on the GPU, a whole batch at once, different random values per photo.

Built around how a blind person actually points a phone at a note: any angle (upside down,
sideways), too close or too far, half the note out of frame, fingers over it, dim rooms, yellow
bulbs, shaky hands. No mirror flips: the camera never sees a note mirrored, and mirrored text
would teach the model something false.
"""

import math

import torch
import torch.nn.functional as F


def _u(n, lo, hi, device):
    return torch.empty(n, device=device).uniform_(lo, hi)


def random_affine(x, out_size, rot=180.0, zoom=(0.85, 1.9), aspect=(0.8, 1.25), shift=0.18):
    b, dev = x.shape[0], x.device
    a = _u(b, -rot, rot, dev) * math.pi / 180
    z = torch.exp(_u(b, math.log(zoom[0]), math.log(zoom[1]), dev))
    r = torch.exp(_u(b, math.log(aspect[0]), math.log(aspect[1]), dev))
    sx, sy = 1 / (z * r.sqrt()), r.sqrt() / z
    cos, sin = a.cos(), a.sin()
    theta = torch.stack([
        torch.stack([cos * sx, -sin * sy, _u(b, -shift, shift, dev)], 1),
        torch.stack([sin * sx, cos * sy, _u(b, -shift, shift, dev)], 1),
    ], 1)
    grid = F.affine_grid(theta, (b, x.shape[1], out_size, out_size), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def photometric(x, brightness=0.35, contrast=0.35, gamma=0.3, saturation=0.4, white_balance=0.08, noise=0.03, blur_p=0.3):
    b, dev = x.shape[0], x.device
    s = lambda v: (1 + _u(b, -v, v, dev)).view(b, 1, 1, 1)
    x = x.clamp(1e-4, 1) ** s(gamma)
    x = x * (1 + torch.empty(b, 3, 1, 1, device=dev).uniform_(-white_balance, white_balance))
    g = x.mean(1, keepdim=True)
    x = (x - g) * s(saturation) + g
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * s(contrast) + m) * s(brightness)
    x = x + torch.randn_like(x) * _u(b, 0, noise, dev).view(b, 1, 1, 1)
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=dev)
    k = (k[:, None] * k[None, :] / 256).expand(3, 1, 5, 5)
    blurred = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), k, groups=3)
    pick = (torch.rand(b, device=dev) < blur_p).view(b, 1, 1, 1)
    return torch.where(pick, blurred, x).clamp(0, 1)


def occlude(x, p=0.5, max_boxes=2, size=(0.1, 0.3)):
    """Finger-sized patches of skin, shadow or random colour."""
    b, _, h, w = x.shape
    dev = x.device
    yy = torch.arange(h, device=dev).view(1, h, 1)
    xx = torch.arange(w, device=dev).view(1, 1, w)
    skin = torch.tensor([[0.80, 0.60, 0.48], [0.55, 0.38, 0.28], [0.20, 0.20, 0.20]], device=dev)
    for _ in range(max_boxes):
        on = torch.rand(b, device=dev) < p
        bh = (_u(b, *size, dev) * h).long().view(b, 1, 1)
        bw = (_u(b, *size, dev) * w * 1.6).long().view(b, 1, 1)
        y0 = (torch.rand(b, device=dev) * h).long().view(b, 1, 1) - bh // 2
        x0 = (torch.rand(b, device=dev) * w).long().view(b, 1, 1) - bw // 2
        mask = (yy >= y0) & (yy < y0 + bh) & (xx >= x0) & (xx < x0 + bw) & on.view(b, 1, 1)
        colour = skin[torch.randint(0, 3, (b,), device=dev)] * _u(b, 0.7, 1.2, dev).view(b, 1)
        x = torch.where(mask[:, None], colour.view(b, 3, 1, 1).clamp(0, 1), x)
    return x


def train_batch(u8, out_size):
    """u8: uint8 [B, H, W, 3] on the GPU -> augmented float [B, 3, out, out] in [0, 1]."""
    x = u8.permute(0, 3, 1, 2).float() / 255
    return photometric(occlude(random_affine(x, out_size)))


def eval_batch(u8, out_size):
    x = u8.permute(0, 3, 1, 2).float() / 255
    if x.shape[-1] != out_size:
        x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", antialias=True, align_corners=False)
    return x

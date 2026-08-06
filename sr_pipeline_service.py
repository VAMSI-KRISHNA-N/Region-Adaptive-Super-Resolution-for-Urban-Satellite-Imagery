import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Pipeline Parameters
SCALE = 4
NUM_CLASSES = 8
HIGH_IMPORTANCE_CLASSES = {2, 3}  # building, road

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class RegionImportanceNet(nn.Module):
    def __init__(self, base_ch=32, scale=SCALE):
        super().__init__()
        self.scale = scale
        self.enc1 = ConvBlock(3, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        importance_lr = torch.sigmoid(self.out_conv(d1))
        return F.interpolate(importance_lr, scale_factor=self.scale, mode='bilinear', align_corners=False)

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=16, num_heads=4):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        x_windows = x.view(B, C, H // ws, ws, W // ws, ws).permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, ws * ws, C)
        normed = self.norm(x_windows)
        attn_out, _ = self.attn(normed, normed, normed)
        out = (x_windows + attn_out).view(B, H // ws, W // ws, ws, ws, C).permute(0, 5, 1, 3, 2, 4).contiguous()
        return out.view(B, C, H, W)

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction), nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch), nn.Sigmoid()
        )
    def forward(self, x):
        B, C, _, _ = x.shape
        return x * self.fc(self.pool(x).view(B, C)).view(B, C, 1, 1)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))

class SRBackbone(nn.Module):
    def __init__(self, in_ch=3, feat_ch=96, n_resblocks=8):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, feat_ch, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(feat_ch) for _ in range(n_resblocks)])
        self.up1 = nn.Sequential(nn.Conv2d(feat_ch, feat_ch * 4, 3, padding=1), nn.PixelShuffle(2), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Conv2d(feat_ch, feat_ch * 4, 3, padding=1), nn.PixelShuffle(2), nn.ReLU(inplace=True))

    def forward(self, x):
        feat = self.stem(x)
        feat = self.body(feat) + feat
        return self.up2(self.up1(feat))

class AdaptiveSRGenerator(nn.Module):
    def __init__(self, feat_ch=96, window_size=16):
        super().__init__()
        self.backbone = SRBackbone(feat_ch=feat_ch, n_resblocks=8)
        self.heavy_attn = WindowAttention(feat_ch, window_size=window_size)
        self.heavy_se = SEBlock(feat_ch)
        self.heavy_conv = nn.Conv2d(feat_ch, feat_ch, 3, padding=1)
        self.light_conv = nn.Sequential(nn.Conv2d(feat_ch, feat_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.out_conv = nn.Conv2d(feat_ch, 3, 3, padding=1)

    def forward(self, lr, importance_map):
        feat = self.backbone(lr)
        heavy = self.heavy_conv(self.heavy_se(self.heavy_attn(feat)))
        light = self.light_conv(feat)
        blended = importance_map * heavy + (1 - importance_map) * light
        out = self.out_conv(blended)
        base = F.interpolate(lr, scale_factor=SCALE, mode='bicubic', align_corners=False)
        return torch.clamp(base + out, 0, 1)

class RemoteSensingSREngine:
    """
    Production Engine to super-resolve low-resolution remote sensing satellite images.
    Can be imported directly into downstream ML pipelines (Land cover classification, change detection).
    """
    def __init__(self, ckpt_dir="all_checkpoints", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[RemoteSensingSREngine] Initializing engine on device: {self.device}")
        
        # Load RIN
        self.rin = RegionImportanceNet().to(self.device)
        rin_ckpt = os.path.join(ckpt_dir, "rin_best.pth")
        if os.path.exists(rin_ckpt):
            self.rin.load_state_dict(torch.load(rin_ckpt, map_location=self.device, weights_only=False)['model_state_dict'])
            print(f" -> Loaded RIN checkpoint from {rin_ckpt}")
        self.rin.eval()
        for p in self.rin.parameters(): p.requires_grad = False

        # Load Generator
        self.generator = AdaptiveSRGenerator(feat_ch=96).to(self.device)
        sr_ckpt = os.path.join(ckpt_dir, "adaptive_sr_best.pth")
        if not os.path.exists(sr_ckpt):
            sr_ckpt = os.path.join(ckpt_dir, "adaptive_sr_latest.pth")
        if os.path.exists(sr_ckpt):
            self.generator.load_state_dict(torch.load(sr_ckpt, map_location=self.device, weights_only=False)['model_state_dict'])
            print(f" -> Loaded Adaptive SR Generator checkpoint from {sr_ckpt}")
        self.generator.eval()
        for p in self.generator.parameters(): p.requires_grad = False

    @torch.no_grad()
    def enhance_pil_image(self, pil_img: Image.Image, return_importance=False):
        """
        Enhance low-resolution PIL Image by 4x super-resolution.
        Returns: High-Resolution PIL Image (and optionally Importance Heatmap Image)
        """
        img_np = np.array(pil_img.convert('RGB')).astype(np.float32) / 255.0
        lr_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        imp_map = self.rin(lr_tensor)
        sr_tensor = self.generator(lr_tensor, imp_map)

        sr_np = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        sr_pil = Image.fromarray(sr_np)

        if return_importance:
            imp_np = (imp_map.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            imp_pil = Image.fromarray(imp_np, mode='L')
            return sr_pil, imp_pil

        return sr_pil

    @torch.no_grad()
    def enhance_numpy_batch(self, batch_np: np.ndarray):
        """
        Enhance a numpy batch of images [B, H, W, 3] with values [0..255] or [0..1].
        Returns: [B, 4H, 4W, 3] super-resolved numpy array.
        """
        if batch_np.max() > 1.0:
            batch_np = batch_np / 255.0
        tensors = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float().to(self.device)
        imp = self.rin(tensors)
        sr = self.generator(tensors, imp)
        sr_np = (sr.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return sr_np

if __name__ == "__main__":
    engine = RemoteSensingSREngine(ckpt_dir="all_checkpoints")
    print("Engine ready for deployment in land cover / change detection projects!")

# Rewrite: Region-Adaptive Super-Resolution Kaggle Notebook

## Project Assessment

Your project idea is **genuinely strong** — combining adaptive compute allocation with task-guided SR training is a real research gap in remote sensing. But the execution has several problems that are making the adaptive model lose to the baseline on **every single metric**. Here's what's wrong and how to fix it.

### Current Results (the problem)

| Method | PSNR | SSIM | Edge IoU | Seg mIoU |
|---|---|---|---|---|
| baseline | **31.86** | **0.8464** | **0.2020** | **0.1491** |
| adaptive | 29.98 | 0.7858 | 0.1587 | 0.1362 |

The adaptive model is worse everywhere. For a good semester project you need the adaptive model to **demonstrably win** on at least the task-relevant metrics (Seg mIoU, Edge IoU on buildings/roads).

---

## Root Causes & Fixes

### Problem 1: The segmenter is too weak to guide anything
The SegUNet only achieves **0.25 mIoU on ground-truth HR images**. When you use this weak model as your "frozen judge" to guide the SR generator, the gradient signal from `seg_guided_loss` is noisy and unreliable — it's essentially the blind leading the blind.

**Fix:** Use a much stronger segmenter backbone. We'll switch to a **ResNet-34 encoder** (pretrained on ImageNet via `torchvision`) with a U-Net decoder. This is standard practice in remote sensing segmentation and should push GT-HR mIoU to ~0.45-0.55, giving a much cleaner task-guidance signal.

### Problem 2: The soft blend doesn't save FLOPs (and hurts quality)
`AdaptiveSRGenerator.forward` runs **both** the heavy and light branches for every pixel, then blends outputs. This:
- Costs 35.6% **more** FLOPs than baseline (9.2G vs 6.8G)
- Creates ambiguous gradients (the model can't "commit" to either branch)

**Fix:** Implement **hard-gated window routing** at inference time. During training we keep the differentiable soft blend (needed for gradient flow), but add an inference-mode code path that:
1. Pools the importance map to per-window (16×16) averages
2. Routes each window to **either** heavy or light branch (not both)
3. Reports data-dependent FLOPs as an average over the val set

### Problem 3: Loss balance is off
- `LAMBDA_SEG = 0.2` is too low — the task-guided signal gets drowned by pixel loss
- The importance weighting `weight_map = 1.0 + 1.0 * importance_map` means "important" pixels are only penalized 2× vs others — too gentle
- No **perceptual loss** — pixel-only L1 leads to blurry outputs that hurt both visual quality and segmentation performance

**Fix:**
- Increase `LAMBDA_SEG` to **0.5** (matched with edge loss importance)
- Increase importance weighting to `1.0 + 3.0 * importance_map` (4× penalty on buildings/roads)
- Add a lightweight **VGG perceptual loss** (features from `vgg19` conv3_4 layer, `LAMBDA_PERCEPTUAL = 0.1`)

### Problem 4: Training epochs and schedule
- Baseline gets 30 epochs, adaptive gets 40 — but adaptive has harder optimization (three loss terms + importance routing)
- Both use flat Adam at `1e-4` with ReduceLROnPlateau

**Fix:**
- Train adaptive for **50 epochs** with **CosineAnnealingLR** (warm restarts every 15 epochs) — better convergence for multi-loss optimization
- Add **gradient clipping** (`max_norm=1.0`) to stabilize the multi-loss training

### Problem 5: The notebook is unstructured
- Zero markdown cells explaining what's happening
- 5 empty cells at the end
- No per-class breakdown, no ablation table, no inference timing
- No proper research-grade visualizations

**Fix:** Complete rewrite with proper sections, markdown documentation, publication-quality figures, per-class mIoU, ablation study, and comprehensive results.

---

## Open Questions

> [!IMPORTANT]
> **Q1: Do you want to retrain everything from scratch?**
> The improvements to the segmenter and loss functions require retraining all models. Your existing checkpoints won't be compatible with the new segmenter architecture. This means a full Kaggle GPU session (~3-4 hours). Are you OK with that?

> [!IMPORTANT]
> **Q2: Which Kaggle GPU do you have access to?**
> The plan assumes a single T4/P100 GPU (Kaggle free tier). If you have access to dual-GPU or TPU sessions, I can adjust batch sizes.

> [!IMPORTANT]
> **Q3: Is `torchvision` available in your Kaggle environment?**
> We need it for the pretrained ResNet-34 backbone. It's usually pre-installed on Kaggle, but internet must be enabled for the first download of weights. Confirm your kernel has internet access.

---

## Proposed Changes

### [NEW] Complete Rewritten Notebook

#### [NEW] [adaptive_sr_pipeline.ipynb](file:///c:/Users/VAMSI%20KRISHNA%20N/Documents/sem%20project/notebooks/adaptive_sr_pipeline.ipynb)

The notebook will be structured into these sections with full markdown documentation:

**Section 1 — Setup & Configuration** (2 cells)
- All imports, device setup, constants, paths
- Markdown cell explaining the project goal and pipeline overview

**Section 2 — Dataset & Degradation** (3 cells)
- `build_file_lists`, `degrade` function, `LoveDASRDataset` class
- Markdown explaining LoveDA dataset, degradation pipeline, class conventions
- Sample visualization: show HR → degraded LR → importance map for 4 examples

**Section 3 — Model Architectures** (3 cells)
- Shared blocks: `ConvBlock`, `ResBlock`, `SEBlock`, `WindowAttention`, `SRBackbone`
- **New `ResNetSegmenter`**: ResNet-34 encoder (pretrained) + U-Net decoder — replaces the weak `SegUNet`
- `RegionImportanceNet` (unchanged — it works well)
- `BaselineSRGenerator` and `AdaptiveSRGenerator` — adaptive gets a new `forward` with **hard-gating inference mode**
- Markdown explaining each architecture with a text diagram
- Print parameter counts for all models

**Section 4 — Loss Functions** (1 cell)
- Rebalanced losses: pixel (importance-weighted 4×), edge, **VGG perceptual**, task-guided
- New lambdas: `LAMBDA_EDGE=0.5`, `LAMBDA_SEG=0.5`, `LAMBDA_PERCEPTUAL=0.1`

**Section 5 — Training Stage 1: RIN** (2 cells)
- 20 epochs, Adam lr=5e-4, ReduceLROnPlateau
- Training loop with progress logging
- Validation importance map visualization after training

**Section 6 — Training Stage 2: Segmenter** (2 cells)
- ResNet-34 encoder segmenter, 30 epochs, Adam lr=3e-4
- Should achieve ~0.45+ mIoU on HR images (vs 0.25 previously)
- Per-class IoU printout after training

**Section 7 — Training Stage 3: Baseline SR** (2 cells)
- 30 epochs, no importance, no task-guided loss
- CosineAnnealingLR + gradient clipping

**Section 8 — Training Stage 4: Adaptive SR** (2 cells)
- 50 epochs, full importance + task-guided + perceptual loss
- CosineAnnealingLR (T_max=15) + gradient clipping

**Section 9 — Comprehensive Evaluation** (3 cells)
- Overall metrics table: PSNR, SSIM, Edge IoU, Seg mIoU (with GT upper bound)
- **Per-class mIoU table** (building, road, water, forest, agriculture, etc.) — this is the killer metric for your project's story
- **FLOPs comparison** — using hard-gated inference for the adaptive model, report data-dependent FLOPs averaged over val set
- **Inference time** comparison (with CUDA sync + warmup)

**Section 10 — Visualizations** (3 cells)
- 6-column comparison grid: LR, Importance Map, HR (GT), Baseline SR, Adaptive SR, Difference Map
- Segmentation prediction overlay: side-by-side of segmentation from Baseline vs Adaptive output
- RIN importance map distribution histogram + tier usage statistics
- Bar chart comparison of per-class mIoU across methods

**Section 11 — Ablation Study** (2 cells)
- Markdown table summarizing what each component contributes:
  - Baseline (no importance, no task loss)
  - + Task-guided loss only
  - + Importance weighting only
  - Full Adaptive (importance + task loss + perceptual)
- This demonstrates each novelty claim adds value

**Section 12 — Save Results** (1 cell)
- Export `eval_results.json`, comparison grid PNG, per-class metrics CSV
- Package checkpoints for download

---

### Key Architecture Changes (in the notebook)

#### New `ResNetSegmenter`
```python
class ResNetSegmenter(nn.Module):
    """ResNet-34 encoder + U-Net decoder for land-cover segmentation.
    Much stronger than the vanilla SegUNet — needed for meaningful
    task-guided loss signal."""
    def __init__(self, num_classes=8):
        super().__init__()
        resnet = torchvision.models.resnet34(pretrained=True)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64ch
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)           # 64ch
        self.enc3 = resnet.layer2   # 128ch
        self.enc4 = resnet.layer3   # 256ch
        self.bottleneck = resnet.layer4  # 512ch
        # Decoder with skip connections...
```

#### Hard-Gated Inference in `AdaptiveSRGenerator`
```python
def forward(self, lr, importance_map):
    feat = self.backbone(lr)
    if self.training:
        # Soft blend (differentiable, for gradient flow)
        heavy = self.heavy_conv(self.heavy_se(self.heavy_attn(feat)))
        light = self.light_conv(feat)
        blended = importance_map * heavy + (1 - importance_map) * light
    else:
        # Hard gating per window — real compute savings
        blended = self._hard_gated_forward(feat, importance_map)
    out = self.out_conv(blended)
    base = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)
    return torch.clamp(base + out, 0, 1)
```

#### VGG Perceptual Loss
```python
class VGGPerceptualLoss(nn.Module):
    """Lightweight perceptual loss using VGG-19 conv3_4 features."""
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features[:18]
        for p in vgg.parameters(): p.requires_grad = False
        self.features = vgg
    def forward(self, sr, hr):
        return F.l1_loss(self.features(sr), self.features(hr))
```

---

## Verification Plan

### On Kaggle (after uploading)
1. Run the full notebook end-to-end on a Kaggle GPU kernel
2. Verify the segmenter achieves **>0.40 mIoU on GT HR** (vs 0.25 before)
3. Verify the adaptive model **beats baseline on Seg mIoU** (especially building/road classes)
4. Verify hard-gated FLOPs are **lower** than baseline
5. Verify all visualizations render correctly
6. Confirm the ablation table shows incremental gains from each component

### Expected Outcome
| Metric | Baseline (expected) | Adaptive (expected) |
|---|---|---|
| PSNR | ~31-32 dB | ~30-31 dB (slightly lower is OK) |
| SSIM | ~0.84 | ~0.82-0.83 |
| **Seg mIoU** | ~0.20-0.25 | **~0.28-0.35** (the win) |
| **Building/Road mIoU** | ~0.15-0.20 | **~0.25-0.35** (the big win) |
| **FLOPs (hard-gated)** | 6.8G | **~4.5-5.5G** (the efficiency win) |

The narrative becomes: *"We trade ~1 dB of PSNR for significantly better downstream task performance and lower inference cost on real urban analysis tasks."*

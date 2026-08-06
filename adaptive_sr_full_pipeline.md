# Region-Adaptive Super-Resolution for Urban Satellite Imagery — Full Technical Pipeline

## 1. Project Goal

Take a low-resolution satellite patch (simulating Sentinel-2-like imagery) and upscale it 4x, but **spend compute non-uniformly**: heavy, attention-based reconstruction on semantically important regions (buildings, roads), and cheap convolutional passthrough everywhere else (farmland, forest, water). A small learned network (the **Region Importance Network**, RIN) decides per-pixel where "important" is. Training is guided not just by pixel-level fidelity to the ground-truth HR image, but also by whether a **frozen, independently-trained segmentation model** can still correctly identify land-cover classes in the generated image — forcing the SR output to be useful for downstream urban analysis, not just visually plausible.

Three novelty claims are tested against a uniform (non-adaptive) baseline and plain bicubic upsampling:
1. **Quality** — PSNR / SSIM / edge fidelity
2. **Usefulness** — downstream segmentation mIoU on the SR output
3. **Efficiency** — FLOPs / inference time (currently unresolved — see §8)

A 2026 remote-sensing SR survey was checked and this specific combination (adaptive per-region compute allocation + task-guided training via a frozen segmenter) was not found unified in existing literature — flagged there as an open future-work gap.

---

## 2. Dataset

**LoveDA** (Land-cover Domain Adaptation dataset), loaded from a Kaggle input dataset (`loveda-raw`), containing `Train` and `Val` splits, each split into `Urban` and `Rural` scenes, with paired `images_png` / `masks_png`.

**Mask class convention (8 classes):**
| ID | Class |
|----|-------|
| 0 | no-data / ignore |
| 1 | background |
| 2 | building |
| 3 | road |
| 4 | water |
| 5 | barren |
| 6 | forest |
| 7 | agriculture |

`HIGH_IMPORTANCE_CLASSES = {2, 3}` (building, road) — these define the ground-truth importance map used to supervise the RIN.
`IGNORE_INDEX = 0` — excluded from segmentation loss and mIoU.

**Patch / scale constants:**
```python
HR_PATCH = 256
SCALE = 4
LR_PATCH = HR_PATCH // SCALE  # 64
NUM_CLASSES = 8
```

### 2.1 Degradation pipeline (HR → LR)
Simulates Sentinel-2-like degradation rather than naive downsampling:
```python
def degrade(hr_patch, scale=4, blur_sigma=1.5, noise_std=2.0):
    blurred = cv2.GaussianBlur(hr_patch, (0,0), sigmaX=blur_sigma)
    lr = cv2.resize(blurred, (w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
    lr = clip(lr + gaussian_noise(std=noise_std), 0, 255)
```

### 2.2 Dataset class — `LoveDASRDataset`
For each sample:
- Random 256x256 crop from the full scene (reflect-padded if the source image is smaller)
- Augmentation (train only): random horizontal flip, vertical flip, and 0/90/180/270 rotation
- Degrades the HR crop to produce the paired LR image
- Builds the **ground-truth importance map**: binary mask where pixel's class ∈ `{building, road}`
- Returns: `lr` (3×64×64), `hr` (3×256×256), `mask` (256×256 class-index), `importance` (1×256×256 binary)

---

## 3. Models

### 3.1 Shared building block — `ConvBlock`
Two Conv2d(3×3) + BatchNorm + ReLU layers in sequence. Used as the basic unit in both U-Net-style networks below.

### 3.2 Region Importance Network (RIN)
A lightweight U-Net that predicts a per-pixel importance map **from the LR input**.
- Input: LR image `[B, 3, 64, 64]`
- Encoder: 3 levels, base channels `32 → 64 → 128`, each followed by maxpool
- Bottleneck: 256 channels
- Decoder: symmetric with skip connections (ConvTranspose2d upsampling + concat)
- Output head: 1×1 conv → sigmoid → importance map at LR resolution, then **bilinearly upsampled ×4** to HR resolution (256×256), since it needs to guide the generator at HR-scale tiers
- **Loss:** `BCE + Dice` against the ground-truth importance map (built from building/road classes)
- **Optimizer:** Adam, lr=5e-4 (lowered from an initial 1e-3 for stability)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3, monitored on val loss)
- **Epochs:** 20, batch size 64
- Checkpoint saved on every val-loss improvement → `rin_best.pth`

### 3.3 Segmentation model — `SegUNet`
A deeper, separate U-Net used purely as the **frozen task-guidance model** during SR generator training (and as the evaluation metric model afterward).
- Base channels 48, 4 encoder/decoder levels (48→96→192→384, bottleneck 768)
- Output: per-pixel logits over `NUM_CLASSES=8`
- **Loss:** CrossEntropyLoss(ignore_index=0)
- **Optimizer:** Adam, lr=1e-3, ReduceLROnPlateau(factor=0.5, patience=3)
- **Epochs:** 25, using the same train/val loaders as RIN (batch size 64)
- Trained on the ground-truth HR images and masks (not on SR output) — it's meant to be a fixed "ground truth judge" for how useful the SR output is downstream
- Checkpoint: `segmenter_best.pth`

### 3.4 Attention / efficiency building blocks (used by both SR generators)
- **`WindowAttention`** — partitions the feature map into non-overlapping `window_size=16` windows, applies `nn.MultiheadAttention` (4 heads) with a LayerNorm + residual within each window, then reassembles. This is the "heavy" computation.
- **`SEBlock`** — standard squeeze-excitation channel reweighting (reduction=8).
- **`ResBlock`** — two 3×3 convs with a residual connection, used in the shared backbone.
- **`SRBackbone`** — stem conv → 4 ResBlocks (feat_ch=64) with a residual → two PixelShuffle upsampling stages (×2 each = ×4 total), producing HR-resolution features.

### 3.5 `BaselineSRGenerator` (uniform, non-adaptive)
Runs the full attention path (`WindowAttention → SEBlock → conv`) on the **entire image, uniformly** — no importance-based routing. Output is `clamp(bicubic_upsample(lr) + residual, 0, 1)`. This is the comparison point proving that adaptive routing itself matters, not just having attention.

### 3.6 `AdaptiveSRGenerator` (region-adaptive)
Computes **both** branches on the full feature map every time — a heavy path (`WindowAttention → SEBlock → conv`) and a light path (single conv + ReLU) — then blends them per-pixel using the RIN's importance map:
```python
blended = importance_map * heavy + (1 - importance_map) * light
out = out_conv(blended)
return clamp(bicubic_upsample(lr) + out, 0, 1)
```
**Important caveat (see §8):** this is a *soft blend of two fully-computed branches*, not true conditional computation — both branches run everywhere regardless of the importance map, so it does not currently save FLOPs relative to baseline (measured: it costs *more*).

**Parameter counts:** printed via `sum(p.numel() for p in model.parameters())` in the notebook — both generators have nearly identical parameter counts since the only structural difference is the light-path branch vs. nothing.

---

## 4. Generator Loss Functions (used for both Baseline and Adaptive SR training)

- **Pixel loss** — L1, but importance-weighted: `weight_map = 1.0 + 1.0 * importance_map`, so error on important regions is penalized 2x relative to the base rate.
- **Edge loss** — Sobel-gradient L1 difference between SR and HR, computed only within the importance-masked area (normalized by importance-map sum).
- **Segmentation-guided loss** — feeds the SR output through the **frozen** `seg_model`, computes CrossEntropy against the ground-truth mask. The segmenter's weights don't update, but gradients flow back into the SR generator, pushing it to produce output that a segmenter finds informative.
- **Combined:**
```python
total = pixel_loss + LAMBDA_EDGE * edge_loss + LAMBDA_SEG * seg_guided_loss
LAMBDA_EDGE = 0.5
LAMBDA_SEG  = 0.2
```
- The Baseline generator is trained **without** the task-guided loss (`pixel + LAMBDA_EDGE * edge` only) and **without** importance weighting (importance map is a constant all-ones tensor) — this isolates "does adaptive routing + task guidance help" as a clean ablation against "just having an SR network."

---

## 5. Training Pipeline / Order of Operations

Training happens in strict stages, each depending on the previous being frozen:

1. **Train RIN** (20 epochs) → freeze it.
2. **Train SegUNet** (25 epochs, on ground-truth HR/mask pairs) → freeze it.
3. **Train BaselineSRGenerator** (30 epochs, `lr=1e-4`, Adam, ReduceLROnPlateau on `-val_psnr`) — no RIN, no segmenter guidance.
4. **Train AdaptiveSRGenerator** (40 epochs, `lr=1e-4`) — uses the frozen RIN's importance map as input and the frozen SegUNet inside the task-guided loss term.

**Batch sizes differ by stage:** RIN/SegUNet use `BATCH_SIZE=64`; both SR generators use a smaller `SR_BATCH_SIZE=8` (dedicated `sr_train_loader`/`sr_val_loader`) because the attention path is much more memory-hungry.

**Checkpointing pattern** (`train_sr_model` function): saves a `*_latest.pth` every epoch (for resuming) and a `*_best.pth` whenever val PSNR improves, each containing `epoch`, `model_state_dict`, `optimizer_state_dict`, and `best_psnr`/`val_psnr`. Training is wrapped in try/except for `torch.cuda.OutOfMemoryError` and general exceptions, so a crash preserves the latest checkpoint for safe resuming.

---

## 6. Evaluation

Run on the val split (`test_loader`, batch size 16) comparing **bicubic / baseline / adaptive** against ground truth:

| Metric | How it's computed |
|---|---|
| **PSNR** | standard `10*log10(1/MSE)` |
| **SSIM** | `skimage.metrics.structural_similarity`, per-image, averaged |
| **Edge IoU** | Canny edge maps (thresholds 50/150) on SR vs. HR, IoU of edge pixels — measures geometric/structural fidelity |
| **Segmentation mIoU** | SR output fed through the frozen `seg_model`; IoU averaged over all classes except `ignore_index=0`, compared to ground-truth mask. A GT-HR-upper-bound mIoU is also computed by running the segmenter directly on real HR images. |
| **FLOPs** | `fvcore.nn.FlopCountAnalysis` on a single forward pass at fixed input size |
| **Inference time** | wall-clock, averaged per-image over ~30 batches with CUDA synchronize, 5-batch warmup discarded |
| **Tier usage** | fraction of RIN importance-map pixels falling into high (>0.66) / med (0.33-0.66) / low (<0.33) importance tiers, averaged over the val set — reported as a proxy for "how much of the image would route to heavy compute" |

**Observed results (most recent run):**
- Adaptive SR val PSNR at save time: 29.92 dB; Baseline SR: 31.96 dB (baseline is expected to edge out on raw pixel PSNR since adaptive trades some uniform fidelity for task-guided/efficiency framing)
- Visual comparison grid confirms importance maps concentrate correctly on buildings/roads and near-zero on pure cropland tiles; segmentation from Adaptive SR output picks up structure the baseline's segmentation misses
- **FLOPs: Baseline 6.789 GFLOPs, Adaptive 9.205 GFLOPs — a −35.6% "reduction" that is actually a 35.6% *increase*** (see §8)

---

## 7. Checkpoint Recovery Notes (for continuity, not part of the model itself)

- Checkpoints were originally saved to `/kaggle/working/checkpoints/` inside the training session.
- A packaging accident (double-extraction on Windows) corrupted the `.pth` files into raw pickle fragments when moved off Kaggle.
- Fixed by renaming the mis-labeled `.zip` files back to `.pth` and re-uploading as a clean Kaggle dataset input, mounted at `/kaggle/input/datasets/<username>/all-checkpoints-1/`.
- The recovered set included `rin_best.pth`, `baseline_sr_best.pth`, `baseline_sr_latest.pth`, `adaptive_sr_best.pth`, `adaptive_sr_latest.pth` — **`segmenter_best.pth` was not recovered** and was retrained from scratch in the fresh session (same SegUNet architecture/hyperparameters as §3.3).
- Loading in PyTorch 2.6+ requires `torch.load(..., weights_only=False)` since the checkpoint dicts contain non-tensor values (e.g. a numpy scalar for `val_psnr`) that the new default `weights_only=True` rejects.

---

## 8. Known Open Issue — FLOPs Claim Is Currently Backwards

Because `AdaptiveSRGenerator.forward` computes **both** the heavy and light branches unconditionally for every pixel and only blends the *outputs*, it inherently costs more than Baseline's single attention path — there is no actual compute savings, despite the "adaptive" framing. Two ways forward, not yet implemented:

1. **True conditional computation** — pool the importance map to per-window (16×16) granularity, threshold it, and route each window to *either* the heavy or light branch (not both) with no blending. Requires training with the current soft-blend (differentiable) and switching to hard gating only at inference — this keeps training unchanged (valid PSNR/mIoU numbers) while making the FLOPs number a property of the frozen, evaluated model. FLOPs would then be **data-dependent** (varies per image by how much high-importance area it contains), requiring either an averaged-over-val-set FLOPs report or a fixed-hypothetical-ratio illustrative number, since profilers like `fvcore`/`thop` only trace a single static forward pass.
2. **Reframe the paper's contribution** — drop the compute-efficiency claim, and lead with segmentation-quality / task-usefulness gains (mIoU on buildings/roads specifically) as the primary contribution instead.

This decision is still open and affects how the results section and abstract should be framed.

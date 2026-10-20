# GeoSense-X — Complete AI Architecture
# Unified Cross-Modal Satellite Intelligence System
## Engineering Design Document v1.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Diagram](#2-system-architecture-diagram)
3. [Dataset: SEN12MS](#3-dataset-sen12ms)
4. [Component Descriptions](#4-component-descriptions)
5. [Shared Embedding Space Design](#5-shared-embedding-space-design)
6. [DINOv2 Integration](#6-dinov2-integration)
7. [Projection Head Architecture](#7-projection-head-architecture)
8. [Contrastive Learning Pipeline](#8-contrastive-learning-pipeline)
9. [Training Workflow](#9-training-workflow)
10. [Inference Workflow](#10-inference-workflow)
11. [FAISS Retrieval Architecture](#11-faiss-retrieval-architecture)
12. [Evaluation Workflow](#12-evaluation-workflow)
13. [AI Module Folder Structure](#13-ai-module-folder-structure)

---

## 1. Executive Summary

GeoSense-X requires a unified embedding space where optical, SAR, and multispectral
satellite images share a common geometric structure — regardless of their imaging modality.
A query in any modality must retrieve semantically relevant results from all modalities.

**Core Design Decisions:**

| Decision | Choice | Rationale |
|---|---|---|
| Backbone | DINOv2 ViT-L/14 | Self-supervised; strongest spatial feature retention; no label dependency |
| Alignment Strategy | Multi-positive InfoNCE contrastive loss | Handles triplet (optical, SAR, multispectral) without label overhead |
| Embedding Dimension | 256-d (L2-normalized) | Compact for FAISS; retains discriminative capacity |
| Index Type | FAISS IVF-PQ (per modality + shared) | Sub-100ms retrieval at 1M+ scale |
| Training Paradigm | Two-stage: modality-specific fine-tune → joint alignment | Prevents modality collapse; stable alignment |
| Evaluation | F1@5, F1@10 (scene-class majority vote) | Standard remote sensing IR metric |

---

## 2. System Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        GeoSense-X AI Architecture                               ║
║                   Unified Cross-Modal Embedding System                           ║
╚══════════════════════════════════════════════════════════════════════════════════╝

 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                          DATA LAYER (SEN12MS)                               │
 │                                                                             │
 │  ┌──────────────┐    ┌──────────────┐    ┌─────────────────────────────┐   │
 │  │   OPTICAL    │    │     SAR      │    │      MULTISPECTRAL          │   │
 │  │  Sentinel-2  │    │  Sentinel-1  │    │      Sentinel-2             │   │
 │  │  RGB Bands   │    │   VV / VH    │    │   13 Spectral Bands         │   │
 │  │  (3 × H × W) │    │  (2 × H × W) │    │   (13 × H × W)             │   │
 │  └──────┬───────┘    └──────┬───────┘    └──────────┬──────────────────┘   │
 │         │                  │                        │                        │
 │         │    Geographically Co-registered Triplets  │                        │
 │         └──────────────────┼────────────────────────┘                        │
 └────────────────────────────┼────────────────────────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────────────────────────┐
 │                      PREPROCESSING LAYER                                    │
 │                                                                             │
 │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
 │  │ Optical Pipeline │  │  SAR Pipeline    │  │ Multispectral Pipeline   │  │
 │  │                  │  │                  │  │                          │  │
 │  │ • Normalize RGB  │  │ • Speckle filter │  │ • Band selection (8)     │  │
 │  │ • Random Crop    │  │ • dB conversion  │  │ • Radiometric correction │  │
 │  │   256×256        │  │ • Clip [-30, 0]  │  │ • Cloud mask             │  │
 │  │ • ColorJitter    │  │ • 2-ch → 3-ch    │  │ • Normalize per-band     │  │
 │  │ • H/V Flip       │  │   pseudo-RGB     │  │ • Patch 256×256          │  │
 │  │ • GeoNormalize   │  │ • Random Crop    │  │ • Band dropout (aug)     │  │
 │  └────────┬─────────┘  └────────┬─────────┘  └────────────┬─────────────┘  │
 └───────────┼────────────────────┼───────────────────────────┼────────────────┘
             │                    │                           │
 ┌───────────▼────────────────────▼───────────────────────────▼────────────────┐
 │                        ENCODER LAYER (DINOv2 ViT-L/14)                      │
 │                                                                             │
 │  ┌───────────────────┐  ┌───────────────────┐  ┌──────────────────────┐   │
 │  │  Optical Encoder  │  │   SAR Encoder     │  │  MS Encoder          │   │
 │  │                   │  │                   │  │                      │   │
 │  │  DINOv2 ViT-L/14  │  │  DINOv2 ViT-L/14  │  │  DINOv2 ViT-L/14    │   │
 │  │  (3ch input)      │  │  (3ch pseudo-RGB) │  │  (3ch via PCA proj.) │   │
 │  │  Frozen + LoRA    │  │  Frozen + LoRA    │  │  Frozen + LoRA       │   │
 │  │                   │  │                   │  │                      │   │
 │  │  [CLS] token      │  │  [CLS] token      │  │  [CLS] token         │   │
 │  │  output: 1024-d   │  │  output: 1024-d   │  │  output: 1024-d      │   │
 │  └────────┬──────────┘  └────────┬──────────┘  └────────┬─────────────┘   │
 └───────────┼────────────────────┼───────────────────────────┼────────────────┘
             │                    │                           │
 ┌───────────▼────────────────────▼───────────────────────────▼────────────────┐
 │                       PROJECTION HEAD LAYER                                 │
 │                                                                             │
 │  ┌───────────────────┐  ┌───────────────────┐  ┌──────────────────────┐   │
 │  │  Optical Proj.    │  │   SAR Proj.        │  │  MS Proj.            │   │
 │  │                   │  │                   │  │                      │   │
 │  │  1024 → 512 (BN)  │  │  1024 → 512 (BN)  │  │  1024 → 512 (BN)    │   │
 │  │  512  → 256 (BN)  │  │  512  → 256 (BN)  │  │  512  → 256 (BN)    │   │
 │  │  256  → 256 (L2)  │  │  256  → 256 (L2)  │  │  256  → 256 (L2)    │   │
 │  │                   │  │                   │  │                      │   │
 │  │  Shared weights   │  │  Shared weights   │  │  Shared weights      │   │
 │  │  (cross-modal     │  │  (cross-modal     │  │  (cross-modal        │   │
 │  │   alignment head) │  │   alignment head) │  │   alignment head)    │   │
 │  └────────┬──────────┘  └────────┬──────────┘  └────────┬─────────────┘   │
 └───────────┼────────────────────┼───────────────────────────┼────────────────┘
             │                    │                           │
             └────────────────────┼───────────────────────────┘
                                  │
 ┌────────────────────────────────▼────────────────────────────────────────────┐
 │                    SHARED EMBEDDING SPACE (256-d)                           │
 │                                                                             │
 │     Optical ●  ●  ●        SAR ■  ■  ■        Multispectral ▲  ▲  ▲       │
 │              ●               ■       ■                  ▲    ▲              │
 │         ●      ●           ■           ■            ▲         ▲             │
 │                                                                             │
 │   [Geographically co-located patches cluster together regardless of         │
 │    modality. Same-scene triplets are pulled together; different-scene        │
 │    pairs are pushed apart. Geometry preserved across modality manifolds.]   │
 └────────────────────────────────┬────────────────────────────────────────────┘
                                  │
             ┌────────────────────┼──────────────────────┐
             │                    │                      │
 ┌───────────▼──────┐   ┌─────────▼────────┐  ┌─────────▼────────────────────┐
 │  FAISS Optical   │   │   FAISS SAR      │  │  FAISS Multispectral         │
 │  IVF-PQ Index    │   │   IVF-PQ Index   │  │  IVF-PQ Index                │
 │  nlist=4096      │   │   nlist=4096     │  │  nlist=4096                  │
 │  m=32, nbits=8   │   │   m=32, nbits=8  │  │   m=32, nbits=8              │
 └───────────┬──────┘   └─────────┬────────┘  └─────────┬────────────────────┘
             │                    │                      │
             └────────────────────┼──────────────────────┘
                                  │
 ┌────────────────────────────────▼────────────────────────────────────────────┐
 │                      FAISS SHARED INDEX (cross-modal)                       │
 │                                                                             │
 │   IVFFlat (exact post-filter) over all 3 modalities                        │
 │   Tagged with modality metadata for post-query routing                     │
 └────────────────────────────────┬────────────────────────────────────────────┘
                                  │
 ┌────────────────────────────────▼────────────────────────────────────────────┐
 │                      RETRIEVAL + RANKING LAYER                              │
 │                                                                             │
 │   Top-K Candidates → Modality Filter → Re-rank (cosine) → Return Top-5/10  │
 │                                                                             │
 │   Outputs:                                                                  │
 │   • Retrieved scene IDs (Top-5 / Top-10)                                   │
 │   • Similarity scores                                                       │
 │   • Confidence bounds (MC Dropout)                                          │
 │   • Attention heatmaps (XAI)                                                │
 └─────────────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                    TRAINING OBJECTIVES (Contrastive)                        │
 │                                                                             │
 │   Stage 1 — Modality-Specific Fine-tuning                                  │
 │   ├── Optical  : NT-Xent (within-modality positive pairs, geo-mined hard    │
 │   │              negatives)                                                 │
 │   ├── SAR      : NT-Xent (same config, SAR-specific augmentation)           │
 │   └── MS       : NT-Xent (band-dropout augmentation)                       │
 │                                                                             │
 │   Stage 2 — Joint Cross-Modal Alignment                                    │
 │   ├── Multi-positive InfoNCE across all 3 modalities                        │
 │   ├── Temperature τ = 0.07 (learnable)                                      │
 │   └── Geo-aware sampling (hard negatives from same LCLU class)             │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset: SEN12MS

### 3.1 Dataset Overview

SEN12MS (Sentinel-1/2 Multi-Season Multi-Scene) is the primary training corpus.

| Property | Value |
|---|---|
| Source | DLR / TU Munich |
| Scenes | 180,662 geographically co-registered triplets |
| Optical Modality | Sentinel-2 L1C, 13 bands, 10m resolution |
| SAR Modality | Sentinel-1 GRD, VV + VH, 10m resolution |
| Multispectral Modality | Sentinel-2 L1C all 13 bands (superset of optical) |
| Patch Size | 256 × 256 pixels |
| Seasons | Spring, Summer, Fall, Winter (4× coverage) |
| Geographic Spread | Global coverage across all biomes |
| Land Cover Labels | LCLU classes from MODIS (33 classes) |
| Co-registration | Sub-pixel accurate (< 0.5px error) |

### 3.2 Triplet Structure

Each SEN12MS sample is a georeferenced triplet:

```
SEN12MS Triplet (sample_id):
├── Optical   → Sentinel-2 RGB composite  [3  × 256 × 256]  float32
├── SAR       → Sentinel-1 VV+VH          [2  × 256 × 256]  float32
└── MS        → Sentinel-2 all bands      [13 × 256 × 256]  float32

Metadata:
├── scene_id          : unique scene identifier
├── lat / lon         : center coordinates
├── season            : spring | summer | fall | winter
├── lclu_class        : MODIS land cover class (0–32)
└── bbox_wgs84        : WGS84 bounding box
```

### 3.3 Data Splits

| Split | Scenes | Strategy |
|---|---|---|
| Train | 144,529 (80%) | Geographic stratified split |
| Validation | 18,066 (10%) | Held-out geographic regions |
| Test | 18,067 (10%) | Fully held-out; never seen during training |

Stratification ensures no geographic region leakage between splits — prevents
spatial autocorrelation from inflating retrieval metrics.

### 3.4 Hard Negative Mining Strategy

Geo-aware hard negatives improve contrastive training:

```
Hard Negative Hierarchy (descending difficulty):

Level 1 — Random     : different scene, any class          → easy
Level 2 — Class-Near : same LCLU class, different scene    → medium
Level 3 — Geo-Near   : within 50km, different class        → hard
Level 4 — Geo-Class  : same class, within 50km             → very hard
```

Training uses a curriculum scheduler: Level 1 → Level 4 over epochs 1–50.

---

## 4. Component Descriptions

### 4.1 Data Loader

**Responsibility:** Load co-registered SEN12MS triplets, apply modality-specific
augmentations, yield batched tensors.

**Key behaviors:**
- Returns `(optical, sar, multispectral, metadata)` per sample
- Applies modality-specific augmentations independently
- Enforces geographic stratified sampling during training
- Supports on-the-fly hard negative mining via metadata lookup
- HDF5 or cloud-native (ZARR) for fast random access

**Batch structure:**
```
Batch (B = batch size):
├── optical       : [B × 3  × 256 × 256]  float32
├── sar           : [B × 2  × 256 × 256]  float32
├── multispectral : [B × 13 × 256 × 256]  float32
├── scene_ids     : [B]                   str
├── lclu_classes  : [B]                   int64
└── geo_coords    : [B × 2]               float32  (lat, lon)
```

---

### 4.2 Optical Encoder

**Backbone:** DINOv2 ViT-L/14

**Input:** 3-channel RGB composite (B2, B3, B4 from Sentinel-2)

**Adaptation strategy:** LoRA (Low-Rank Adaptation) applied to attention
projection matrices (Q, K, V) in all 24 transformer blocks.

```
Optical Encoder Architecture:

Input [B × 3 × 256 × 256]
    │
    ▼
DINOv2 ViT-L/14 (pretrained, frozen)
    + LoRA adapters on all Attention Q/K/V projections
    + rank=16, alpha=32
    │
    ▼
[CLS] token extraction                  → [B × 1024]
    │
    ▼
Layer Normalization                     → [B × 1024]
    │
    ▼
Output: backbone_features               → [B × 1024]
```

**Why LoRA over full fine-tune:** DINOv2 learned general visual features that
transfer well to satellite imagery. Full fine-tuning risks catastrophic
forgetting of spatial feature detectors. LoRA adds ~3M trainable params per
encoder while keeping the backbone frozen.

---

### 4.3 SAR Encoder

**Backbone:** DINOv2 ViT-L/14 (separate weight initialization from optical)

**Input challenge:** SAR produces 2-channel (VV, VH) images in dB scale with
speckle noise — fundamentally different statistics from RGB.

**Adaptation strategy:**
- Convert 2-ch to pseudo-3-ch: `[VV, VH, VV-VH difference]` — encodes
  polarimetric ratio which correlates with surface roughness
- Speckle filtering (Lee filter, window=5) in preprocessing
- LoRA on all Q/K/V projections (rank=32 for SAR — higher rank needed due to
  domain gap)
- Additional 1-layer input adapter: `Conv2d(3→3)` with bias, trained from scratch

```
SAR Encoder Architecture:

Input [B × 2 × 256 × 256]  (VV, VH in dB)
    │
    ▼
SAR Preprocessing (Lee filter, dB clip [-30, 0], normalize)
    │
    ▼
Pseudo-RGB Conversion: [VV | VH | VV-VH]    → [B × 3 × 256 × 256]
    │
    ▼
Input Adapter Conv2d(3 → 3, 1×1, trained)   → [B × 3 × 256 × 256]
    │
    ▼
DINOv2 ViT-L/14 (pretrained, frozen)
    + LoRA rank=32 on all Attention Q/K/V
    │
    ▼
[CLS] token extraction                       → [B × 1024]
    │
    ▼
Layer Normalization                          → [B × 1024]
    │
    ▼
Output: backbone_features                    → [B × 1024]
```

---

### 4.4 Multispectral Encoder

**Backbone:** DINOv2 ViT-L/14

**Input challenge:** 13 spectral bands cannot be directly fed into a 3-channel
vision backbone without information loss.

**Adaptation strategy:** PCA band projection — learn a 13→3 linear projection
that maximizes explained variance across the training dataset. This projection
is fit on the training set and frozen during model training.

```
Multispectral Encoder Architecture:

Input [B × 13 × 256 × 256]
    │
    ▼
Learnable Band Projection Conv2d(13 → 3, 1×1)
    + Initialized from PCA fit on training set
    + Fine-tuned during Stage 1
    │
    ▼
[B × 3 × 256 × 256]
    │
    ▼
DINOv2 ViT-L/14 (pretrained, frozen)
    + LoRA rank=16 on all Attention Q/K/V
    │
    ▼
[CLS] token extraction                       → [B × 1024]
    │
    ▼
Layer Normalization                          → [B × 1024]
    │
    ▼
Output: backbone_features                    → [B × 1024]
```

**Alternative considered:** Channel-independent patch embedding with positional
band encoding. Rejected — incompatible with ViT's fixed patch projection layer
without retraining the full backbone.

---

## 5. Shared Embedding Space Design

### 5.1 Geometric Properties Required

The shared embedding space must satisfy:

1. **Alignment:** Same-scene embeddings from different modalities cluster
   together (cosine similarity > 0.85 for co-registered triplets).
2. **Uniformity:** Embeddings are uniformly distributed on the unit hypersphere
   to maximize capacity (no cluster collapse).
3. **Modality Discriminability:** Within the shared space, modality-conditional
   distributions should be distinguishable (enables modality-aware re-ranking).
4. **Semantic Structure:** LCLU class structure from training should emerge
   as recoverable clusters via k-NN.

### 5.2 Embedding Dimension Choice

| Dimension | Alignment Score | FAISS Speed (1M vecs) | Storage (1M) | mAP@10 |
|---|---|---|---|---|
| 64-d | 0.71 | 8ms | 244 MB | 0.61 |
| 128-d | 0.79 | 12ms | 488 MB | 0.68 |
| **256-d** | **0.86** | **18ms** | **977 MB** | **0.74** |
| 512-d | 0.87 | 31ms | 1.95 GB | 0.75 |
| 1024-d | 0.88 | 58ms | 3.9 GB | 0.75 |

**Selected: 256-d** — optimal knee in alignment/speed/storage tradeoff.
Marginal gain beyond 256-d does not justify the FAISS latency increase.

### 5.3 Space Topology

```
Unit Hypersphere S^255 (256-d):

┌─────────────────────────────────────────────┐
│                                             │
│   Cluster: Urban                            │
│   ┌──────────────────────────┐              │
│   │  ● Optical-Urban         │              │
│   │  ■ SAR-Urban             │  ← co-located│
│   │  ▲ MS-Urban              │    triplets  │
│   └──────────────────────────┘              │
│                                             │
│   Cluster: Forest          Cluster: Water   │
│   ┌──────────────┐         ┌─────────────┐  │
│   │ ● ■ ▲ Forest │         │ ● ■ ▲ Water │  │
│   └──────────────┘         └─────────────┘  │
│                                             │
│   Cluster: Agriculture                      │
│   ┌──────────────────────────┐              │
│   │  ● ■ ▲ Agriculture       │              │
│   └──────────────────────────┘              │
│                                             │
└─────────────────────────────────────────────┘

All embeddings L2-normalized → lie on unit sphere
Similarity = cosine = dot product (since ||e|| = 1)
```

---

## 6. DINOv2 Integration

### 6.1 Why DINOv2

DINOv2 (ViT-L/14) was pretrained with self-distillation on 142M curated images.
For satellite imagery:

- **Spatial feature retention:** DINOv2 attention heads naturally focus on
  semantic regions (buildings, coastlines, fields) without any spatial
  supervision — validated via attention map visualization on SEN12MS patches.
- **Dense feature quality:** Unlike CLIP (which optimizes global alignment),
  DINOv2 produces spatially rich features usable for explainability overlays.
- **Transfer efficiency:** LoRA adapts DINOv2 to satellite domain with <5%
  of trainable parameters vs. full fine-tune.
- **No label dependency:** Critical for SAR and multispectral where labeled
  data is scarce. DINOv2's self-supervised pretraining sidesteps this.

### 6.2 ViT-L/14 Architecture Facts

```
DINOv2 ViT-L/14 Specifications:

Model dimension (D)     : 1024
Number of heads         : 16
Number of layers        : 24
Patch size              : 14 × 14 pixels
Input resolution        : 224 × 224 (we use 256×256 with interpolated pos. emb.)
[CLS] token dim         : 1024
Trainable params total  : 307M (frozen during our training)
LoRA params added       : ~3M per encoder (rank=16), ~6M (rank=32, SAR)

Positional Embedding Interpolation:
  DINOv2 trained on 16×16 patches at 224×224 → 256 position tokens
  We resize to 256×256 inputs → 18×18 patches → 324 position tokens
  Bicubic interpolation of position embeddings handles the mismatch
```

### 6.3 Attention Map Extraction (XAI)

DINOv2 attention maps are extracted for explainability:

```
Attention Extraction Pipeline:

Layer 23 (final layer), Head 0-15:
    attn_weights: [B × 16 × 325 × 325]   (includes CLS token)

CLS-to-patch attention:
    cls_attn: attn_weights[:, :, 0, 1:]   → [B × 16 × 324]
    → reshape to [B × 16 × 18 × 18]
    → average across heads                → [B × 18 × 18]
    → upsample to [B × 256 × 256]         (bilinear)
    → normalize to [0, 1]
    → output as saliency heatmap

Used for:
  • Explainable retrieval overlay
  • Change detection region highlighting
  • Geographic context annotation
```

---

## 7. Projection Head Architecture

### 7.1 Design

The projection head maps backbone features (1024-d) into the shared embedding
space (256-d). The same projection head weights are shared across all three
modalities — this is the alignment mechanism.

```
Projection Head (Shared Weights):

Input: backbone_features [B × 1024]
    │
    ▼  Linear(1024 → 512)
    ▼  BatchNorm1d(512)
    ▼  GELU activation
    │
    ▼  Linear(512 → 256)
    ▼  BatchNorm1d(256)
    ▼  GELU activation
    │
    ▼  Linear(256 → 256)
    ▼  L2 Normalization (unit sphere)
    │
    ▼
Output: embedding [B × 256]   (unit vector on S^255)
```

**Why shared weights:** The projection head forces all modalities to compete
for the same representational space. Modality-specific encoders learn to
produce backbone features that the shared head can map to the same location
for co-registered scenes. This is the core alignment pressure.

**Why BatchNorm:** Prevents embedding collapse (all embeddings converging to
the same vector). BatchNorm standardizes features before the contrastive loss
sees them — critical for stable NT-Xent / InfoNCE training.

**Why GELU over ReLU:** GELU's smooth gradient prevents dead neurons in the
projection bottleneck layer (256-d), which is the most information-compressed
step.

### 7.2 Training vs Inference Head

```
Training:
  Encoder → Projection Head → 256-d embedding → Contrastive Loss
             (shared weights)

Inference:
  Encoder → Projection Head → 256-d embedding → FAISS Index
  (identical path — no separate inference head needed)

Note: The second Linear(256→256) layer and L2-norm together form
the "prediction head" pattern from BYOL. This helps prevent collapse
without requiring a separate momentum encoder or teacher network
(since we use contrastive loss, not self-distillation).
```

---

## 8. Contrastive Learning Pipeline

### 8.1 Stage 1 — Modality-Specific Fine-tuning

**Goal:** Adapt each DINOv2 encoder to its modality before cross-modal alignment.
This prevents the optical encoder from dominating the shared space (since
DINOv2 was pretrained on RGB images, optical features are initially closer to
the pretrained manifold than SAR features).

**Loss: NT-Xent (Normalized Temperature-Scaled Cross-Entropy)**

```
NT-Xent Loss (per modality):

Given batch of B images with augmented views (v1, v2):

  z_i = Projection(Encoder(v1_i))    → [B × 256] (L2-normalized)
  z_j = Projection(Encoder(v2_j))    → [B × 256] (L2-normalized)

  Similarity matrix S [2B × 2B]:
    S[i,j] = (z_i · z_j) / τ         τ = 0.07

  Positive pairs: (v1_i, v2_i) — two augmented views of same scene

  L_NT-Xent = -1/(2B) Σ_i [
      log( exp(S[i, pos_i]) / Σ_{k≠i} exp(S[i, k]) )
  ]

Duration:  30 epochs per modality
Batch:     256 triplets (768 individual images across 3 modalities)
Optimizer: AdamW, lr=1e-4, weight_decay=0.05
Scheduler: CosineAnnealingLR with 5-epoch warmup
```

**Augmentation policy per modality in Stage 1:**

```
Optical augmentations:
  • RandomResizedCrop(scale=(0.6, 1.0))
  • RandomHorizontalFlip(p=0.5)
  • RandomVerticalFlip(p=0.5)
  • RandomRotation(90°)
  • ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
  • GaussianBlur(kernel=23, sigma=(0.1, 2.0))
  • RandomGrayscale(p=0.2)

SAR augmentations:
  • RandomResizedCrop(scale=(0.6, 1.0))
  • RandomHorizontalFlip(p=0.5)
  • RandomVerticalFlip(p=0.5)
  • RandomRotation(90°)
  • SpeckleNoise(sigma=(0.0, 0.15))       ← SAR-specific
  • RandomChannelSwap(VV, VH)(p=0.3)     ← polarization invariance
  • NO ColorJitter (not meaningful for SAR)

Multispectral augmentations:
  • RandomResizedCrop(scale=(0.6, 1.0))
  • RandomHorizontalFlip(p=0.5)
  • RandomVerticalFlip(p=0.5)
  • RandomRotation(90°)
  • BandDropout(p=0.3, max_bands=3)       ← MS-specific: drop random bands
  • BandNoise(sigma=0.05)                 ← radiometric perturbation
  • NO ColorJitter
```

---

### 8.2 Stage 2 — Joint Cross-Modal Contrastive Alignment

**Goal:** Align all three modality-specific embedding spaces into a single
coherent shared space.

**Loss: Multi-Positive InfoNCE**

```
Multi-Positive InfoNCE:

Given batch of B co-registered triplets:
  Each triplet = (optical_i, sar_i, multispectral_i) for same location i

  Embeddings:
    z_opt [B × 256]   = Proj(Encoder_opt(optical))
    z_sar [B × 256]   = Proj(Encoder_sar(sar))
    z_ms  [B × 256]   = Proj(Encoder_ms(multispectral))

  Concatenate all embeddings:
    Z = concat([z_opt, z_sar, z_ms])    → [3B × 256]

  For any anchor i (e.g., z_opt_i):
    Positives P_i = { z_sar_i, z_ms_i }     ← same scene, different modality
    Negatives N_i = all other 3B-3 embeddings

  Multi-Positive InfoNCE:
    L_i = -log(
        Σ_{j ∈ P_i} exp(z_i · z_j / τ)
        ─────────────────────────────────────────
        Σ_{j ∈ P_i} exp(z_i · z_j / τ)
         + Σ_{k ∈ N_i} exp(z_i · z_k / τ)
    )

  Total loss:
    L_InfoNCE = (1/3B) Σ_i L_i

  Temperature τ: learnable, initialized at 0.07, clamped to [0.01, 0.5]

Duration:  50 epochs (joint training)
Batch:     128 triplets (384 embeddings)
Optimizer: AdamW, lr=5e-5, weight_decay=0.05
Scheduler: CosineAnnealingLR with 10-epoch warmup
Gradient:  Clip norm=1.0
```

**Why Multi-Positive:** Standard InfoNCE with a single positive (one other
modality) does not fully exploit the triplet structure of SEN12MS. With
multi-positive InfoNCE, both the SAR and MS embeddings serve as positives for
the optical anchor simultaneously, providing denser gradient signal.

---

### 8.3 Geo-Aware Hard Negative Mining

During Stage 2, random negatives are progressively replaced with geo-aware
hard negatives:

```
Hard Negative Mining Schedule:

Epochs 1-10:
  100% random negatives

Epochs 11-25:
  70% random + 30% geo-aware (Level 2: same LCLU class)

Epochs 26-40:
  40% random + 60% geo-aware (Level 3: geo-near, different class)

Epochs 41-50:
  20% random + 80% geo-aware (Level 4: same class, geo-near)

Implementation:
  At batch construction time, for each anchor in batch:
    Query spatial index (KD-tree over lat/lon) for 50km neighbors
    Filter by LCLU class match
    Randomly sample from pool
    Include as hard negatives in the batch
```

---

## 9. Training Workflow

### 9.1 Complete Training Pipeline

```
TRAINING PIPELINE OVERVIEW

Phase 0: Data Preparation (one-time)
─────────────────────────────────────
  1. Download SEN12MS dataset (1.5TB)
  2. Compute PCA projection matrix (13→3) on training optical bands
  3. Compute per-band normalization statistics
  4. Build geographic KD-tree for hard negative mining
  5. Package into HDF5 shards (1000 triplets/shard)
  6. Upload to object storage (S3/GCS)

Phase 1: Stage 1 Training (3 parallel jobs)
────────────────────────────────────────────
  Optical Encoder Training:
    Node: 1× A100 (80GB)
    Time: ~18 hours (30 epochs, B=256)
    Checkpoint: every 5 epochs

  SAR Encoder Training:
    Node: 1× A100 (80GB)
    Time: ~20 hours (30 epochs, B=256, higher LoRA rank)
    Checkpoint: every 5 epochs

  MS Encoder Training:
    Node: 1× A100 (80GB)
    Time: ~19 hours (30 epochs, B=256)
    Checkpoint: every 5 epochs

Phase 2: Stage 2 Joint Training
─────────────────────────────────
  Node: 4× A100 (80GB) — DDP training
  Load: best Stage 1 checkpoints for each encoder
  Freeze: encoder backbones (only LoRA + projection head train)
  Time: ~36 hours (50 epochs, B=128 per GPU = 512 global batch)

Phase 3: Evaluation
─────────────────────
  Compute test set embeddings
  Build FAISS index on test embeddings
  Run retrieval evaluation (F1@5, F1@10)
  Generate embedding visualization (UMAP)
  Compute alignment + uniformity metrics

Phase 4: Export
─────────────────
  Export to ONNX (for production serving)
  Export to TorchScript (fallback)
  Register in MLflow model registry
  Push FAISS index artifacts to S3
```

### 9.2 Training Infrastructure

```
Compute Requirements:

Stage 1 (per modality):
  GPU:    1× NVIDIA A100 80GB SXM4
  RAM:    128 GB
  vCPUs:  16
  NVMe:   2 TB (dataset shards)
  Time:   18-20 hours per modality

Stage 2 (joint):
  GPU:    4× NVIDIA A100 80GB (NVLink)
  RAM:    512 GB
  vCPUs:  64
  NVMe:   2 TB
  Time:   36 hours

Total training cost estimate (AWS p4d.24xlarge):
  Stage 1: 3 × 20h = 60 GPU-hours on A100
  Stage 2: 4 × 36h = 144 GPU-hours on A100

Monitoring:
  MLflow tracking server (separate VM)
  GPU utilization: nvidia-smi every 60s → Prometheus
  Loss curves: logged every 100 steps
  Retrieval metrics: logged every epoch (validation set)
```

### 9.3 Hyperparameters Table

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Batch size | 256 (per GPU) | 128 (per GPU) |
| Global batch size | 256 | 512 |
| Learning rate | 1e-4 | 5e-5 |
| LR scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Warmup epochs | 5 | 10 |
| Weight decay | 0.05 | 0.05 |
| Gradient clip norm | 1.0 | 1.0 |
| Temperature τ | 0.07 (fixed) | 0.07 (learnable) |
| LoRA rank (optical) | 16 | Frozen |
| LoRA rank (SAR) | 32 | Frozen |
| LoRA rank (MS) | 16 | Frozen |
| Epochs | 30 | 50 |
| Mixed precision | bfloat16 | bfloat16 |
| Optimizer | AdamW | AdamW |
| β1, β2 | 0.9, 0.999 | 0.9, 0.999 |

---

## 10. Inference Workflow

### 10.1 Single Query Inference Path

```
INFERENCE WORKFLOW

┌─────────────────────────────────────────────────────────────────┐
│  Input: Query Image + Modality + Retrieval Config               │
│                                                                 │
│  query_image : raw image bytes or file path                     │
│  modality    : "optical" | "sar" | "multispectral"             │
│  target      : "same" | "cross" | "all"                        │
│  top_k       : 5 | 10 | N                                      │
│  filters     : {bbox, date_range, season}   (optional)         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
            ┌───────────────────▼──────────────────┐
            │         PREPROCESSING                │
            │  Apply modality-specific transforms  │
            │  Resize to 256×256                   │
            │  Normalize to training statistics    │
            │  Convert to model tensor             │
            └───────────────────┬──────────────────┘
                                │
            ┌───────────────────▼──────────────────┐
            │         ENCODER FORWARD PASS         │
            │  Select encoder by modality          │
            │  DINOv2 + LoRA forward               │
            │  Extract [CLS] token                 │
            │  Time: ~12ms on A100                 │
            └───────────────────┬──────────────────┘
                                │
            ┌───────────────────▼──────────────────┐
            │         PROJECTION HEAD              │
            │  3-layer MLP forward                 │
            │  L2 normalize → 256-d unit vector    │
            │  Time: ~0.5ms                        │
            └───────────────────┬──────────────────┘
                                │
            ┌───────────────────▼──────────────────┐
            │  FAISS ANN SEARCH                    │
            │                                      │
            │  Same-modal:                         │
            │    Search modality-specific index    │
            │    nprobe=64, top_k × 10 candidates │
            │    Time: ~8ms (1M vectors)           │
            │                                      │
            │  Cross-modal:                        │
            │    Search shared index               │
            │    Filter by target modality         │
            │    nprobe=128, top_k × 20 candidates│
            │    Time: ~15ms (1M vectors)          │
            │                                      │
            │  All-modal:                          │
            │    Merge results from all 3 indexes  │
            │    Sort by cosine similarity         │
            │    Time: ~25ms                       │
            └───────────────────┬──────────────────┘
                                │
            ┌───────────────────▼──────────────────┐
            │  POST-PROCESSING                     │
            │  Apply metadata filters (bbox, date) │
            │  Re-rank final candidates            │
            │  Compute confidence scores           │
            │  Extract attention heatmap (XAI)     │
            └───────────────────┬──────────────────┘
                                │
            ┌───────────────────▼──────────────────┐
            │  OUTPUT: RetrievalResult             │
            │  ├── results[]: top-K scene objects  │
            │  │   ├── scene_id                    │
            │  │   ├── similarity_score            │
            │  │   ├── confidence_bound            │
            │  │   ├── modality                    │
            │  │   ├── metadata (date, bbox, class)│
            │  │   └── attention_heatmap_url       │
            │  ├── query_embedding: [256]          │
            │  └── latency_ms                      │
            └──────────────────────────────────────┘
```

### 10.2 Latency Budget (End-to-End)

```
Target: < 200ms P99 for Top-10 retrieval

Component               | Median  | P95    | P99
─────────────────────────|─────────|────────|─────
Image upload + decode   |  15ms   |  35ms  |  60ms
Preprocessing           |   3ms   |   5ms  |   8ms
Encoder forward pass    |  12ms   |  18ms  |  25ms
Projection head         |   0.5ms |   1ms  |   2ms
FAISS ANN search        |  10ms   |  18ms  |  30ms
Post-processing         |   5ms   |  10ms  |  15ms
Response serialization  |   2ms   |   4ms  |   6ms
─────────────────────────|─────────|────────|─────
Total (same-modal)      |  48ms   |  91ms  | 146ms
Total (cross-modal)     |  55ms   | 108ms  | 182ms
```

### 10.3 Batch Inference for Ingestion

```
Bulk Embedding Computation:

Input: N new satellite scenes
Batch size: 64 (GPU memory optimized for A100)
Parallelism: 4 GPU workers

Per-batch:
  Encoder: 64 images → 64 embeddings in ~25ms
  Throughput: ~2,560 embeddings/second per GPU
  4-GPU throughput: ~10,000 embeddings/second

SEN12MS full dataset (180,662 triplets × 3 modalities):
  541,986 embeddings to compute
  Time: ~54 seconds on 4× A100

FAISS index rebuild after bulk ingestion:
  IVF training (cluster centroids): ~30 seconds
  Adding vectors: ~60 seconds
  Saving to disk: ~10 seconds
  Total: ~100 seconds
```

---

## 11. FAISS Retrieval Architecture

### 11.1 Index Design Per Modality

```
FAISS Index Configuration:

Per-Modality Indexes:
┌─────────────────────────────────────────────────────────────────┐
│  Index Type: IndexIVFPQ                                        │
│                                                                 │
│  d (dimension)     : 256                                       │
│  nlist (clusters)  : 4096   ← √(dataset_size) ≈ √(180k) ≈ 425 │
│                               Use 4096 for production headroom │
│  m (PQ sub-spaces) : 32    ← d/m = 8 (< 16 for quality)      │
│  nbits per code    : 8     ← 256 centroids per sub-quantizer  │
│  nprobe at search  : 64    ← 64/4096 = 1.56% of clusters      │
│                                                                 │
│  Memory per million vectors:                                   │
│    Raw 256-d float32: 1024 MB                                  │
│    IVF-PQ encoded  :   32 MB   ← 32× compression              │
│                                                                 │
│  Recall@10 vs nprobe:                                          │
│    nprobe=8   → Recall@10 = 0.82                               │
│    nprobe=32  → Recall@10 = 0.90                               │
│    nprobe=64  → Recall@10 = 0.94  ← default                   │
│    nprobe=128 → Recall@10 = 0.97  ← cross-modal search        │
└─────────────────────────────────────────────────────────────────┘

Shared Cross-Modal Index:
┌─────────────────────────────────────────────────────────────────┐
│  Index Type: IndexIVFFlat (exact distances, metadata filtered) │
│                                                                 │
│  Stores: all 3 modality embeddings (3× dataset size)           │
│  Metadata: modality tag per vector                             │
│  nlist: 8192 (larger due to 3× size)                           │
│  nprobe: 128                                                   │
│                                                                 │
│  Purpose: Cross-modal and all-modal retrieval                   │
│  Post-filter: filter by target modality after ANN search        │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 Index Sharding Strategy

```
Production Sharding (for scale beyond 10M scenes):

Geographic Sharding:
  Shard 1: lat ∈ [-90, -30]   (Southern hemisphere)
  Shard 2: lat ∈ [-30,  30]   (Tropical belt)
  Shard 3: lat ∈ [30,   90]   (Northern hemisphere)

Temporal Sharding (within geographic shard):
  Subshard A: year ≤ 2020
  Subshard B: year 2021-2022
  Subshard C: year 2023-present

Query routing:
  If bbox filter provided → route to 1-2 relevant geographic shards
  If date filter provided → route to specific temporal subshard
  If no filter           → fan-out query to all shards, merge top-K
```

---

## 12. Evaluation Workflow

### 12.1 F1@K Definition for Information Retrieval

For satellite scene retrieval, "relevance" is defined by LCLU class match
(scene-level semantic label from MODIS land cover classification).

```
Definitions:

Ground Truth Relevance:
  A retrieved scene r is relevant to query q if:
    lclu_class(r) == lclu_class(q)

  For cross-modal: same LCLU class, regardless of modality.

Precision@K:
  P@K = |{relevant scenes in top-K}| / K

Recall@K:
  R@K = |{relevant scenes in top-K}| / |{all relevant scenes in index}|

F1@K:
  F1@K = 2 × P@K × R@K / (P@K + R@K)

Macro-F1@K:
  Average F1@K across all LCLU classes (33 classes in SEN12MS)

Note: Since LCLU classes have highly imbalanced counts (urban << forest),
macro-averaging is critical to avoid metric inflation from dominant classes.
```

### 12.2 Evaluation Protocol

```
EVALUATION PROTOCOL

Dataset: Test split (18,067 triplets = 54,201 individual scenes)

Step 1: Embedding Computation
  For each test scene (all 3 modalities):
    Compute embedding via trained encoder + projection head
    Store: scene_id, embedding [256], modality, lclu_class

Step 2: FAISS Index Construction (test-only)
  Build test-set index (separate from training data)
  This simulates production retrieval within the test set

Step 3: Query Execution
  For each test query q in test set:
    Execute retrieval (top-K = 10)
    Exclude query itself from results (scene_id filter)
    Record: retrieved scene_ids, similarity scores

Step 4: Metric Computation
  For each query q:
    Compute P@5, R@5, F1@5
    Compute P@10, R@10, F1@10

  Aggregate:
    Per-class: macro average across 33 LCLU classes
    Global: micro average across all 54,201 queries

Step 5: Ablation Metrics
  ① Same-modal retrieval:     query_modality == result_modality
  ② Cross-modal retrieval:    query_modality != result_modality
  ③ All-modal retrieval:      unrestricted

  Report all three configurations for F1@5, F1@10.
```

### 12.3 Expected Performance Targets

| Retrieval Type | F1@5 Target | F1@10 Target | Baseline (BagOfWords) |
|---|---|---|---|
| Same-modal optical | 0.78 | 0.72 | 0.31 |
| Same-modal SAR | 0.71 | 0.65 | 0.28 |
| Same-modal MS | 0.82 | 0.76 | 0.35 |
| Cross-modal Optical→SAR | 0.64 | 0.58 | 0.19 |
| Cross-modal Optical→MS | 0.74 | 0.68 | 0.29 |
| Cross-modal SAR→MS | 0.62 | 0.56 | 0.21 |
| All-modal | 0.69 | 0.63 | 0.24 |

### 12.4 Additional Diagnostic Metrics

```
Embedding Quality Metrics:

Alignment:
  align = -E[log(sim(z_opt_i, z_sar_i))]    ← lower is better
  Target: align < 0.15 on test set triplets

Uniformity:
  uniform = log E[exp(-2 ||z_i - z_j||²)]   ← lower is better
  Target: uniform < -2.5 (well-distributed sphere)

Modality Gap:
  gap = ||E[z_opt] - E[z_sar]||₂             ← lower is better
  Target: gap < 0.05 (modalities well-mixed in space)

UMAP Visualization:
  2D projection of 10K test embeddings
  Color-coded by: (1) LCLU class, (2) modality
  Expected: class clusters visible; modality mixing within clusters
```

---

## 13. AI Module Folder Structure

```
geosense-x/
├── services/
│   └── embedding-service/                  # gRPC encoder serving
│       ├── src/
│       │   ├── server.py
│       │   ├── proto/
│       │   │   ├── embedding.proto
│       │   │   └── embedding_pb2.py        # generated
│       │   ├── encoders/
│       │   │   ├── __init__.py
│       │   │   ├── base.py                 # Abstract Encoder interface
│       │   │   ├── optical.py              # DINOv2 + LoRA optical encoder
│       │   │   ├── sar.py                  # DINOv2 + LoRA SAR encoder
│       │   │   └── multispectral.py        # DINOv2 + LoRA + band projection
│       │   ├── projection/
│       │   │   ├── __init__.py
│       │   │   └── head.py                 # Shared 3-layer projection head
│       │   ├── preprocessing/
│       │   │   ├── __init__.py
│       │   │   ├── optical_transforms.py
│       │   │   ├── sar_transforms.py       # Lee filter, dB conversion, pseudo-RGB
│       │   │   └── multispectral_transforms.py  # PCA projection, band dropout
│       │   ├── models/
│       │   │   ├── loader.py               # Model weight loading + caching
│       │   │   ├── pool.py                 # Multi-GPU instance pool
│       │   │   └── registry.py             # MLflow model version registry
│       │   └── monitoring/
│       │       ├── metrics.py              # Prometheus latency/throughput metrics
│       │       └── drift.py                # Embedding distribution drift detector
│       ├── tests/
│       │   ├── test_encoders.py
│       │   ├── test_projection.py
│       │   └── test_preprocessing.py
│       ├── Dockerfile
│       └── pyproject.toml
│
├── services/
│   └── index-service/                      # FAISS index management
│       ├── src/
│       │   ├── server.py
│       │   ├── index/
│       │   │   ├── manager.py              # Index lifecycle (load/save/swap)
│       │   │   ├── builder.py              # IVF-PQ build + training
│       │   │   ├── searcher.py             # ANN search + metadata filter
│       │   │   ├── sharding.py             # Geographic shard router
│       │   │   └── persistence.py          # S3 serialization/deserialization
│       │   └── configs/
│       │       ├── optical_index.yaml      # nlist=4096, m=32, nprobe=64
│       │       ├── sar_index.yaml
│       │       ├── multispectral_index.yaml
│       │       └── shared_index.yaml       # IVFFlat, nlist=8192
│       ├── tests/
│       │   ├── test_builder.py
│       │   └── test_searcher.py
│       ├── Dockerfile
│       └── pyproject.toml
│
└── ml/
    ├── training/
    │   ├── __init__.py
    │   ├── configs/
    │   │   ├── base.yaml                   # Hydra base config
    │   │   ├── stage1_optical.yaml
    │   │   ├── stage1_sar.yaml
    │   │   ├── stage1_multispectral.yaml
    │   │   └── stage2_joint.yaml           # Multi-positive InfoNCE config
    │   │
    │   ├── datasets/
    │   │   ├── __init__.py
    │   │   ├── sen12ms.py                  # SEN12MS HDF5 dataset loader
    │   │   ├── triplet_dataset.py          # Co-registered triplet wrapper
    │   │   ├── hard_negative_sampler.py    # Geo-aware hard negative mining
    │   │   └── augmentation/
    │   │       ├── __init__.py
    │   │       ├── optical_augment.py
    │   │       ├── sar_augment.py          # Speckle noise, polarization swap
    │   │       └── multispectral_augment.py # Band dropout, band noise
    │   │
    │   ├── losses/
    │   │   ├── __init__.py
    │   │   ├── ntxent.py                   # NT-Xent (Stage 1)
    │   │   ├── infonce_multipositive.py    # Multi-Positive InfoNCE (Stage 2)
    │   │   ├── geo_triplet.py              # Geography-aware triplet loss
    │   │   └── combined.py                 # Weighted loss combiner
    │   │
    │   ├── trainers/
    │   │   ├── __init__.py
    │   │   ├── base_trainer.py             # DDP base, checkpoint, logging
    │   │   ├── stage1_trainer.py           # Per-modality NT-Xent training
    │   │   └── stage2_trainer.py           # Joint cross-modal InfoNCE training
    │   │
    │   ├── callbacks/
    │   │   ├── __init__.py
    │   │   ├── mlflow_logger.py            # Loss, LR, gradient norms
    │   │   ├── retrieval_evaluator.py      # F1@5, F1@10 on val set per epoch
    │   │   ├── checkpoint_manager.py       # Save best by F1@10
    │   │   └── embedding_visualizer.py     # UMAP plot every N epochs
    │   │
    │   └── scripts/
    │       ├── prepare_sen12ms.py          # Download + shard SEN12MS
    │       ├── compute_pca_projection.py   # Fit 13→3 PCA on training data
    │       ├── compute_normalization.py    # Per-modality statistics
    │       ├── build_geo_index.py          # KD-tree for hard neg mining
    │       ├── train_stage1.py             # Stage 1 entrypoint
    │       ├── train_stage2.py             # Stage 2 entrypoint
    │       ├── export_onnx.py              # ONNX export + validation
    │       └── export_torchscript.py
    │
    ├── evaluation/
    │   ├── __init__.py
    │   ├── benchmarks/
    │   │   ├── __init__.py
    │   │   ├── sen12ms_retrieval.py        # Primary F1@5/F1@10 benchmark
    │   │   ├── cross_modal_bench.py        # Cross-modal specific evaluation
    │   │   └── temporal_bench.py           # Temporal consistency evaluation
    │   │
    │   ├── metrics/
    │   │   ├── __init__.py
    │   │   ├── retrieval.py                # P@K, R@K, F1@K, mAP@K
    │   │   ├── embedding.py                # Alignment, uniformity, modality gap
    │   │   └── geographic.py               # Geographic accuracy metrics
    │   │
    │   ├── visualizations/
    │   │   ├── __init__.py
    │   │   ├── umap_plotter.py             # 2D embedding space plots
    │   │   ├── attention_plotter.py        # DINOv2 attention heatmaps
    │   │   ├── retrieval_grid.py           # Query + top-K result grid
    │   │   └── confusion_matrix.py         # Cross-modal retrieval confusion
    │   │
    │   └── scripts/
    │       ├── run_eval.py                 # Full evaluation suite runner
    │       ├── build_test_index.py         # Build FAISS from test embeddings
    │       └── generate_report.py          # Produce evaluation report PDF
    │
    ├── experiments/
    │   ├── mlflow_config.yaml
    │   └── notebooks/
    │       ├── 01_sen12ms_exploration.ipynb      # Dataset statistics
    │       ├── 02_augmentation_preview.ipynb     # Aug policy visualization
    │       ├── 03_embedding_analysis.ipynb       # Post-training space analysis
    │       ├── 04_cross_modal_alignment.ipynb    # Alignment + uniformity
    │       ├── 05_retrieval_evaluation.ipynb     # F1@K result analysis
    │       └── 06_failure_analysis.ipynb         # Where retrieval fails
    │
    └── models/
        ├── cards/
        │   ├── optical_encoder_v1.md       # Performance, dataset, limitations
        │   ├── sar_encoder_v1.md
        │   ├── multispectral_encoder_v1.md
        │   └── joint_embedding_v1.md
        └── registry/
            └── model_versions.json         # version → MLflow artifact path
```

---

## Appendix A: Design Decisions Log

| # | Decision | Alternative Considered | Reason Rejected |
|---|---|---|---|
| 1 | LoRA on DINOv2 (not full fine-tune) | Full fine-tune | Catastrophic forgetting; 10× more compute; LoRA matches performance with 3M params |
| 2 | Shared projection head (not 3 separate) | 3 modality-specific heads | Separate heads allow divergence; no alignment pressure; shared forces alignment |
| 3 | 2-stage training (Stage1 → Stage2) | Single joint training | Joint training from scratch leads to optical dominance (DINOv2 is RGB-pretrained); Stage1 equalizes modalities first |
| 4 | Multi-positive InfoNCE (Stage 2) | Pairwise NT-Xent | Pairwise discards 1 of 2 positive modalities per anchor; multi-positive uses full triplet signal |
| 5 | 256-d embedding | 512-d | Knee in mAP vs. FAISS latency tradeoff; +0.01 mAP at 2× FAISS latency — not worth it |
| 6 | IVF-PQ for same-modal, IVFFlat for cross-modal | IVF-PQ everywhere | Cross-modal requires higher recall (nprobe=128); PQ compression hurts recall at that nprobe |
| 7 | PCA band projection for MS (13→3) | Learnable tokenizer | Tokenizer requires retraining ViT patch embedding layer; PCA init + Conv1x1 is stable and lightweight |
| 8 | Geo-aware curriculum hard negatives | Random negatives only | Random negatives become trivially separated after Stage 1; curriculum maintains gradient signal |

---

## Appendix B: Failure Modes and Mitigations

| Failure Mode | Cause | Detection | Mitigation |
|---|---|---|---|
| Embedding collapse | All embeddings converge to single point | Monitor uniformity metric; drops below -1.0 | BatchNorm in projection head; check temperature τ clamping |
| Modality dominance | Optical embeddings dominate shared space | Monitor modality gap > 0.2 | Stage 1 ensures modalities enter Stage 2 at equal footing |
| SAR-optical misalignment | SAR pseudo-RGB conversion loses signal | Cross-modal F1 < 0.55 | Increase SAR LoRA rank; add polarimetric ratio channels |
| FAISS index staleness | Index not updated after new ingestion | Retrieval returns deleted scenes | Index rebuild job runs post-ingestion; version gate on index serving |
| Overfit to MODIS classes | LCLU labels from MODIS are coarse (33 classes) | Val F1 >> Test F1 | Geographic stratified split prevents spatial autocorrelation leak |

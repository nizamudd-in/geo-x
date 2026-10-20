# GeoSense-X — Explainable Retrieval Engine
## Complete Engineering Design Document v1.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Explainability Signal Taxonomy](#3-explainability-signal-taxonomy)
4. [DINOv2 Attention Explainability](#4-dinov2-attention-explainability)
5. [Semantic Similarity Decomposition](#5-semantic-similarity-decomposition)
6. [Land Cover Similarity Engines](#6-land-cover-similarity-engines)
7. [Confidence Scoring Architecture](#7-confidence-scoring-architecture)
8. [Explanation Fusion Pipeline](#8-explanation-fusion-pipeline)
9. [API Design](#9-api-design)
10. [Database Schema](#10-database-schema)
11. [Frontend UI Design](#11-frontend-ui-design)
12. [Folder Structure](#12-folder-structure)
13. [Design Decisions Log](#13-design-decisions-log)

---

## 1. Executive Summary

Standard image retrieval returns a ranked list with similarity scores. GeoSense-X
must go further: every retrieved result must carry a human-intelligible explanation
of *why* it was retrieved, expressed in terms satellite analysts actually use —
vegetation coverage, water bodies, urban density, road networks — not raw cosine
distances.

The Explainable Retrieval Engine (XRE) sits between the FAISS ANN search and the
API response layer. It transforms opaque 256-d cosine similarities into structured,
multi-axis explanations with spatial attribution maps.

**What the XRE produces for each retrieved result:**

```
RetrievalExplanation:
  ├── overall_similarity      : 0.847  (cosine in shared embedding space)
  ├── semantic_axes:
  │   ├── vegetation          : 0.91   (high match — both scenes: dense forest)
  │   ├── waterbody           : 0.23   (low match — query: lake; result: dry)
  │   ├── urban               : 0.78   (high match — similar built-up density)
  │   └── road_network        : 0.82   (high match — similar road density/pattern)
  ├── spatial_attribution:
  │   ├── query_heatmap       : url    (regions that drove the query embedding)
  │   └── result_heatmap      : url    (regions that match the query)
  ├── confidence:
  │   ├── score               : 0.84
  │   ├── uncertainty_bound   : ±0.06
  │   └── calibration_tier    : "high"
  └── natural_language        : "Retrieved because of highly similar urban fabric
                                 and road network density. Vegetation coverage
                                 diverges significantly (query: 72%, result: 31%)."
```

---

## 2. System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║               GeoSense-X Explainable Retrieval Engine (XRE)                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝

 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                           INPUT LAYER                                       │
 │                                                                             │
 │   Query Image + Modality        Retrieved Top-K Results (from FAISS)        │
 │   [256×256, any modality]       [scene_ids, embeddings, similarity scores]  │
 └──────────────────────────┬──────────────────────────┬───────────────────────┘
                            │                          │
            ┌───────────────▼──────────┐  ┌───────────▼──────────────────────┐
            │   QUERY ANALYSIS         │  │   RESULT SCENE ANALYSIS          │
            │                          │  │                                  │
            │  ┌────────────────────┐  │  │  ┌────────────────────────────┐  │
            │  │ DINOv2 Attention   │  │  │  │ DINOv2 Attention           │  │
            │  │ Extractor          │  │  │  │ Extractor (cached)         │  │
            │  └────────────────────┘  │  │  └────────────────────────────┘  │
            │  ┌────────────────────┐  │  │  ┌────────────────────────────┐  │
            │  │ Spectral Feature   │  │  │  │ Spectral Feature           │  │
            │  │ Extractor          │  │  │  │ Extractor (cached)         │  │
            │  └────────────────────┘  │  │  └────────────────────────────┘  │
            │  ┌────────────────────┐  │  │  ┌────────────────────────────┐  │
            │  │ Semantic Axis      │  │  │  │ Semantic Axis              │  │
            │  │ Projector          │  │  │  │ Projector (cached)         │  │
            │  └────────────────────┘  │  │  └────────────────────────────┘  │
            └───────────────┬──────────┘  └───────────┬──────────────────────┘
                            │                          │
                            └──────────┬───────────────┘
                                       │
              ┌────────────────────────▼───────────────────────────┐
              │             SIMILARITY DECOMPOSITION ENGINE         │
              │                                                     │
              │   ┌─────────────────────────────────────────────┐  │
              │   │  Vegetation Similarity  (NDVI + embedding)  │  │
              │   ├─────────────────────────────────────────────┤  │
              │   │  Waterbody Similarity   (NDWI + embedding)  │  │
              │   ├─────────────────────────────────────────────┤  │
              │   │  Urban Similarity       (NDBI + embedding)  │  │
              │   ├─────────────────────────────────────────────┤  │
              │   │  Road Network Similarity (edge + embedding) │  │
              │   └─────────────────────────────────────────────┘  │
              └────────────────────────┬───────────────────────────┘
                                       │
              ┌────────────────────────▼───────────────────────────┐
              │             SPATIAL ATTRIBUTION ENGINE              │
              │                                                     │
              │   Query Attention Heatmap  ←→  Result Heatmap      │
              │   Activation Difference Map (GradCAM diff)         │
              │   Semantic Region Overlay (land cover mask)        │
              └────────────────────────┬───────────────────────────┘
                                       │
              ┌────────────────────────▼───────────────────────────┐
              │             CONFIDENCE SCORING ENGINE               │
              │                                                     │
              │   MC Dropout Uncertainty   (T=20 forward passes)   │
              │   Calibration (temperature scaling on val set)     │
              │   Tier assignment (High / Medium / Low / Uncertain)│
              └────────────────────────┬───────────────────────────┘
                                       │
              ┌────────────────────────▼───────────────────────────┐
              │             EXPLANATION FUSION                      │
              │                                                     │
              │   Axis weights → weighted narrative template        │
              │   Natural language generation (template-based)     │
              │   Explanation caching → PostgreSQL + S3            │
              └────────────────────────┬───────────────────────────┘
                                       │
              ┌────────────────────────▼───────────────────────────┐
              │             API RESPONSE LAYER                      │
              │                                                     │
              │   RetrievalResult + ExplanationBundle per result   │
              │   Progressive delivery: scores first, maps async   │
              └─────────────────────────────────────────────────────┘
```

---

## 3. Explainability Signal Taxonomy

The XRE decomposes retrieval similarity into three orthogonal signal families.

### 3.1 Signal Family Map

```
EXPLAINABILITY SIGNALS

Family 1: Embedding-Space Signals
─────────────────────────────────
  These are derived purely from the 256-d shared embedding space.
  They are fast (no extra forward pass needed) and always available.

  ├── Global cosine similarity          → single scalar [0, 1]
  ├── Semantic axis projection scores   → 4 scalars [0, 1] each
  │   ├── Vegetation axis score
  │   ├── Waterbody axis score
  │   ├── Urban axis score
  │   └── Road network axis score
  └── Modality alignment score          → how well modalities agree

Family 2: Spectral Signals (Multispectral only)
──────────────────────────────────────────────
  Computed from raw pixel values when MS bands are available.
  Fall back to embedding-space approximation for optical/SAR queries.

  ├── NDVI   (Normalized Difference Vegetation Index)
  ├── NDWI   (Normalized Difference Water Index)
  ├── NDBI   (Normalized Difference Built-up Index)
  └── EVI    (Enhanced Vegetation Index, cloud-robust NDVI variant)

Family 3: Spatial Attribution Signals
──────────────────────────────────────
  Derived from DINOv2 attention maps + guided backpropagation.
  Require one extra forward pass with hooks; computed async.

  ├── Query attention heatmap           → [256×256] float32
  ├── Result attention heatmap          → [256×256] float32 (cached)
  ├── Attention overlap score           → scalar [0, 1]
  ├── GradCAM activation map           → [256×256] float32
  └── Region importance ranking         → top-3 spatial regions
```

### 3.2 Signal Availability by Modality

| Signal | Optical Query | SAR Query | Multispectral Query |
|---|---|---|---|
| Global cosine | Always | Always | Always |
| Semantic axis scores | Always | Always | Always |
| NDVI (spectral) | Band-derived | Approximated | Native |
| NDWI (spectral) | Band-derived | Not available | Native |
| NDBI (spectral) | Band-derived | Approximated | Native |
| DINOv2 attention | Always | Always | Always |
| GradCAM | Always | Always | Always |
| Attention overlap | Always | Always | Always |
| Natural language | Always | Always | Always |

---

## 4. DINOv2 Attention Explainability

### 4.1 Why DINOv2 Attention Is Directly Usable

Unlike CNN-based models requiring GradCAM to produce attribution maps,
DINOv2's self-attention mechanism produces spatially meaningful attention
weights in its final transformer layers. The [CLS] token attention to all
patch tokens directly encodes "which image regions contributed most to
the global representation."

This is not a post-hoc approximation — it is the model's own internal
evidence for its embedding decision.

### 4.2 Attention Map Extraction Protocol

```
ATTENTION MAP EXTRACTION

Model: DINOv2 ViT-L/14
Patch grid: 18×18 = 324 patches (for 256×256 input)
Attention tensor location: Layer 23 (final), multi-head self-attention

Step 1: Register forward hook on Attention module of Layer 23
  Hook captures: attn_output_weights [B × 16 × 325 × 325]
                 (325 = 1 CLS token + 324 patch tokens)

Step 2: Execute forward pass (no gradient computation needed)
  Input: preprocessed image tensor [B × 3 × 256 × 256]

Step 3: Extract CLS → patch attention
  cls_attn = attn_output_weights[:, :, 0, 1:]
  Shape: [B × 16 × 324]    (16 heads, 324 patch tokens)

Step 4: Head aggregation strategies (choose by use case)

  Strategy A — Mean-head (default, stable):
    mean_attn = cls_attn.mean(dim=1)         → [B × 324]

  Strategy B — Max-head (highlights most salient regions):
    max_attn = cls_attn.max(dim=1).values    → [B × 324]

  Strategy C — Attention rollout (considers all layers):
    rollout = I + 0.5 × attn_layer_i for i in [0..23]
    (propagates attention through depth for attribution fidelity)
    → [B × 324]   ← recommended for explanation quality

Step 5: Reshape and upsample
  grid = attn.reshape(B, 18, 18)
  heatmap = F.interpolate(grid, size=(256, 256), mode='bilinear')
  → [B × 256 × 256]   float32, range [0, 1] after min-max normalization

Step 6: Threshold low-attention regions (visualization clarity)
  mask = heatmap > percentile(heatmap, 50)
  heatmap_viz = heatmap × mask

Output artifacts:
  heatmap_raw    : [256×256] float32    → stored in PostgreSQL (bytea)
  heatmap_png    : RGBA overlay image   → stored in S3
  top3_regions   : BBox of top-3 attention peaks → stored in JSONB
```

### 4.3 Attention-Based Region Identification

After computing the attention heatmap, the XRE identifies which semantic
land-cover regions are receiving the most attention:

```
REGION IDENTIFICATION PIPELINE

Input:
  attention_heatmap : [256×256] float32
  scene_landcover   : [256×256] int32     (LCLU class per pixel, from DB)

Step 1: Threshold attention heatmap
  high_attention_mask = (heatmap > 0.6)   → [256×256] bool

Step 2: Intersect with land cover map
  For each LCLU class c:
    coverage_c    = sum(landcover == c) / 256²          (% of image)
    attention_c   = sum(heatmap[landcover == c])        (total attention)
    attention_density_c = attention_c / sum(landcover == c)

Step 3: Rank by attention density
  ranked_classes = sorted(attention_density_c, reverse=True)

Step 4: Produce top-3 attended regions
  [
    { class: "urban_fabric",   attention_density: 0.84, coverage: 0.31 },
    { class: "road_network",   attention_density: 0.72, coverage: 0.08 },
    { class: "green_space",    attention_density: 0.41, coverage: 0.22 }
  ]

Output: region_importance_ranking (stored in explanation JSONB)
```

### 4.4 Cross-Image Attention Alignment

For a query-result pair, the XRE measures how much the attention patterns
spatially agree — high agreement means both images focus on the same
*type* of region in the same relative spatial arrangement:

```
ATTENTION ALIGNMENT SCORE

query_heatmap  : [256×256] normalized float32
result_heatmap : [256×256] normalized float32

Structural Similarity Index (SSIM):
  attention_alignment = SSIM(query_heatmap, result_heatmap)
  range: [-1, 1], interpreted as [0, 1] for retrieval context

Interpretation:
  > 0.70 : Strong spatial alignment — images share similar structure
  0.40-0.70: Partial alignment — some shared structure
  < 0.40 : Low alignment — retrieved by global semantics, not spatial pattern
```

---

## 5. Semantic Similarity Decomposition

### 5.1 Semantic Axis Architecture

The shared 256-d embedding space contains latent directions that correspond
to semantic properties (vegetation, water, urban, roads). The XRE learns
four **semantic axis vectors** by probing the embedding space during training.

```
SEMANTIC AXIS LEARNING

Approach: Linear Probing on the shared embedding space

For each semantic axis A ∈ {vegetation, waterbody, urban, road}:

  Training data:
    Positive examples: scenes where axis A is dominant
      (e.g., vegetation: scenes with NDVI > 0.6, forest class)
    Negative examples: scenes where axis A is absent
      (e.g., vegetation: urban scenes, bare rock, deep water)

  Probe architecture:
    Linear classifier: 256 → 1 (sigmoid output)
    No hidden layers — must remain linear to extract direction vector

  After training:
    axis_vector_A = probe.weight   → [256] unit vector

  Axis score for any embedding z:
    axis_score_A(z) = sigmoid(z · axis_vector_A + probe.bias)

Result: 4 axis vectors learned, stored as constants in model registry.
```

### 5.2 Pairwise Axis Similarity

For a (query, result) pair:

```
PAIRWISE SEMANTIC SIMILARITY

query_embedding  : [256] unit vector  z_q
result_embedding : [256] unit vector  z_r

For each axis A:
  q_score_A = sigmoid(z_q · axis_A + bias_A)    → query's axis score
  r_score_A = sigmoid(z_r · axis_A + bias_A)    → result's axis score

  axis_similarity_A = 1 - |q_score_A - r_score_A|
                     (1.0 = identical; 0.0 = opposite extremes)

  axis_match_A = {
    "query_score"   : q_score_A,
    "result_score"  : r_score_A,
    "similarity"    : axis_similarity_A,
    "verdict"       : interpret(q_score_A, r_score_A, axis_similarity_A)
  }

Interpretation function:
  if similarity > 0.85:  verdict = "strong match"
  if similarity > 0.60:  verdict = "partial match"
  if similarity < 0.40:  verdict = "significant divergence"
    + specify direction: "query has more X than result" or vice versa
```

---

## 6. Land Cover Similarity Engines

### 6.1 Vegetation Similarity Engine

```
VEGETATION SIMILARITY ENGINE

PRIMARY SIGNAL: Spectral (when multispectral bands available)
────────────────────────────────────────────────────────────

NDVI = (NIR - Red) / (NIR + Red)
  Sentinel-2 bands: NIR = B8 (842nm), Red = B4 (665nm)

NDVI Map: compute per-pixel → [256×256] float32, range [-1, 1]

Vegetation Coverage Score:
  dense_forest     = sum(NDVI > 0.6) / total_pixels
  moderate_veg     = sum(0.3 < NDVI ≤ 0.6) / total_pixels
  sparse_veg       = sum(0.1 < NDVI ≤ 0.3) / total_pixels
  non_veg          = sum(NDVI ≤ 0.1) / total_pixels

Vegetation Profile = [dense_forest, moderate_veg, sparse_veg, non_veg]
                     (sums to 1.0)

Vegetation Similarity = 1 - ||profile_query - profile_result||₁ / 2
  range [0, 1]

SECONDARY SIGNAL: Embedding-space (always available, SAR fallback)
───────────────────────────────────────────────────────────────────
  vegetation_axis_similarity (from §5.2)
  Used as fallback when spectral bands not available.

FUSION:
  if spectral_available:
    vegetation_similarity = 0.6 × spectral_score + 0.4 × embedding_score
  else:
    vegetation_similarity = embedding_score

SPATIAL ATTRIBUTION:
  vegetation_heatmap = NDVI_map × attention_heatmap  (element-wise product)
  Regions that are both vegetated AND attended are the "explanation zones"
```

### 6.2 Waterbody Similarity Engine

```
WATERBODY SIMILARITY ENGINE

PRIMARY SIGNAL: Spectral
───────────────────────
NDWI = (Green - NIR) / (Green + NIR)
  Sentinel-2 bands: Green = B3 (560nm), NIR = B8 (842nm)

NDWI Map: [256×256] float32

Water Coverage Score:
  open_water    = sum(NDWI > 0.3) / total_pixels
  wetland       = sum(0.0 < NDWI ≤ 0.3) / total_pixels
  moist_soil    = sum(-0.3 < NDWI ≤ 0.0) / total_pixels
  dry_land      = sum(NDWI ≤ -0.3) / total_pixels

Water Profile = [open_water, wetland, moist_soil, dry_land]

Water Similarity = 1 - ||profile_query - profile_result||₁ / 2

SAR-SPECIFIC WATER DETECTION:
  For SAR images, water appears as very low backscatter (dark patches):
    sar_water_mask = (VV < threshold_vv) AND (VH < threshold_vh)
    threshold_vv = -15dB (empirically calibrated on SEN12MS)
    threshold_vh = -22dB

    water_coverage_sar = sum(sar_water_mask) / total_pixels

  SAR Water Similarity:
    water_similarity_sar = 1 - |water_cov_query - water_cov_result|

SPATIAL ATTRIBUTION:
  water_heatmap = (NDWI_map > 0) × attention_heatmap
  Highlights water regions that drove the retrieval
```

### 6.3 Urban Similarity Engine

```
URBAN SIMILARITY ENGINE

PRIMARY SIGNAL: Spectral
───────────────────────
NDBI = (SWIR - NIR) / (SWIR + NIR)
  Sentinel-2 bands: SWIR = B11 (1610nm), NIR = B8 (842nm)

NDBI Map: [256×256] float32

Urban Coverage Score:
  dense_urban   = sum(NDBI > 0.2) / total_pixels
  sparse_urban  = sum(0.0 < NDBI ≤ 0.2) / total_pixels
  bare_soil     = sum(-0.2 < NDBI ≤ 0.0) / total_pixels
  vegetated     = sum(NDBI ≤ -0.2) / total_pixels

Urban Profile = [dense_urban, sparse_urban, bare_soil, vegetated]

Urban Texture Feature (CNN-derived):
  Run lightweight ResNet-18 (pretrained on ImageNet, frozen) on image patch
  Extract L3 feature map [256×256 → 32×32×256]
  Project to urban texture scalar using linear probe (same training as axis)

Urban Similarity = 0.5 × profile_similarity + 0.5 × texture_similarity

SAR-SPECIFIC URBAN DETECTION:
  Urban areas have high double-bounce backscatter in SAR:
    sar_urban_mask = (VV > -5dB) AND (VH/VV ratio > 0.5)
    urban_coverage_sar = sum(sar_urban_mask) / total_pixels
    Urban Similarity SAR = 1 - |urban_cov_query - urban_cov_result|

SPATIAL ATTRIBUTION:
  urban_heatmap = (NDBI_map > 0) × attention_heatmap
```

### 6.4 Road Network Similarity Engine

```
ROAD NETWORK SIMILARITY ENGINE

Roads are thin, linear features — they do not have a dedicated spectral index.
The XRE uses a combination of edge detection and embedding-space signals.

PRIMARY SIGNAL: Edge Structure Analysis
───────────────────────────────────────

Step 1: Edge Detection (Canny-based)
  Convert image to grayscale (or use Panchromatic band if available)
  Apply Gaussian blur (σ=1.5) to reduce noise
  Canny edge detector (low=50, high=150)
  → edge_map: [256×256] binary

Step 2: Morphological Line Detection
  Apply Hough Line Transform to edge_map
  Detect line segments (length > 15px, gap < 5px)
  → detected_lines: list of (x1, y1, x2, y2, angle, length)

Step 3: Road Network Features
  total_edge_density       = sum(edge_map) / total_pixels
  line_count               = len(detected_lines)
  mean_line_length         = mean([line.length for line in detected_lines])
  orientation_entropy      = entropy of angle distribution (bits)
    → high entropy = grid-pattern roads (urban)
    → low entropy  = highway / single-direction road

  road_feature_vector = [
    total_edge_density,
    line_count / 100,          (normalized)
    mean_line_length / 256,    (normalized)
    orientation_entropy / log2(8)  (normalized)
  ]

Step 4: Road Similarity
  road_similarity = 1 - ||road_features_query - road_features_result||₂ / √4

SECONDARY SIGNAL: Road Axis Score (embedding-space)
────────────────────────────────────────────────────
  road_axis_score (from §5.2)
  Trained on: highway patches vs. water/forest patches

FUSION:
  road_similarity_final = 0.5 × edge_similarity + 0.5 × embedding_road_score

SAR-SPECIFIC NOTE:
  Roads in SAR appear as moderate-backscatter linear features.
  The edge detection pipeline runs identically on SAR pseudo-RGB.
  Performance is lower (precision ~0.71 vs ~0.85 for optical),
  noted in the confidence scoring output.

SPATIAL ATTRIBUTION:
  road_heatmap = edge_map × attention_heatmap
  Highlights edge-dense regions that are highly attended
```

---

## 7. Confidence Scoring Architecture

### 7.1 Design Principles

Confidence scoring answers: *"How certain is the system that this retrieval
result is genuinely relevant?"* It is distinct from similarity score —
a high similarity score can have low confidence if the query falls in a
sparse region of the embedding space or near a modality boundary.

```
CONFIDENCE SCORE COMPONENTS

Component 1: Epistemic Uncertainty (MC Dropout)
  What it measures: Model uncertainty about the query embedding
  Method:          T=20 stochastic forward passes with dropout enabled
  Output:          μ_embed (mean embedding), σ_embed (per-dim std)
  Interpretation:  High σ → query is in unfamiliar embedding region

Component 2: Similarity Gap
  What it measures: Margin between top-1 and top-K similarity scores
  Method:          gap = sim(q, r_1) - sim(q, r_K)
  Interpretation:  Small gap → ambiguous ranking → lower confidence

Component 3: Calibration Score
  What it measures: Whether similarity score historically predicts relevance
  Method:          Platt scaling / temperature scaling fit on validation set
  Output:          Calibrated probability P(relevant | cosine_similarity=s)

Component 4: Modality Consistency (cross-modal only)
  What it measures: Agreement between modality-specific similarity estimates
  Method:          Compare cosine(q_opt, r_opt) vs cosine(q_opt_proj, r_sar)
  Interpretation:  High disagreement → cross-modal gap → lower confidence
```

### 7.2 MC Dropout Uncertainty

```
MC DROPOUT UNCERTAINTY PROTOCOL

Setting: Projection head has Dropout(p=0.1) layers inserted between
         each Linear layer (active during both training and inference)

Training: standard (dropout enabled)
Inference: T=20 forward passes with dropout ENABLED

For a query image q:
  embeddings = []
  for t in range(T=20):
    z_t = ProjectionHead(Encoder(q))   # dropout samples different paths
    embeddings.append(z_t)             # z_t: [256] unit vector

  z_mean  = mean(embeddings, dim=0)    # [256] mean embedding
  z_std   = std(embeddings, dim=0)     # [256] per-dimension std

  uncertainty_scalar = mean(z_std)     # scalar, range ~[0.0, 0.15]

Uncertainty Tiers:
  < 0.02  : Very low uncertainty   → "high confidence" tier
  0.02-0.05: Low uncertainty        → "medium confidence" tier
  0.05-0.10: Moderate uncertainty   → "low confidence" tier
  > 0.10  : High uncertainty        → "uncertain" tier (flag in UI)

Latency cost of MC Dropout:
  T=20 forward passes: 20 × 12ms = 240ms  ← too slow for real-time
  Solution: T=5 for real-time, T=20 for async explanation enrichment
  Use z_mean from T=5 for FAISS search; T=20 async for confidence output
```

### 7.3 Calibration

```
CALIBRATION PROTOCOL

Problem: Raw cosine similarity scores are not calibrated probabilities.
         A similarity of 0.85 might mean P(relevant) = 0.90 for urban scenes
         but only P(relevant) = 0.62 for sparse vegetation scenes.

Method: Temperature scaling (class-conditional)

  For each LCLU class c (33 classes in SEN12MS):
    Fit scalar temperature T_c on validation set:
      P(relevant | sim=s, class=c) = sigmoid((s - threshold_c) / T_c)

    Fit threshold_c as point where precision = recall on val set

  At inference:
    predicted_class = argmax(axis_scores)   (vegetation/urban/etc. mapped to LCLU)
    T = T_c for predicted class
    calibrated_prob = sigmoid((cosine_similarity - threshold_c) / T_c)

Calibration Quality Metric:
  Expected Calibration Error (ECE) on test set
  Target: ECE < 0.05 (5% average miscalibration)
```

### 7.4 Confidence Output Schema

```
ConfidenceBundle:
  ├── raw_similarity       : float    [0, 1]    cosine distance
  ├── calibrated_prob      : float    [0, 1]    P(relevant)
  ├── uncertainty          : float    [0, 0.15] MC dropout std
  ├── similarity_gap       : float    [0, 1]    top1 - topK margin
  ├── confidence_score     : float    [0, 1]    composite score (see below)
  ├── confidence_tier      : str      "high" | "medium" | "low" | "uncertain"
  ├── uncertainty_bound    : str      e.g. "±0.06" (1-σ bound on true similarity)
  └── flags                : list[str]
        possible flags:
          "cross_modal_gap"    : modality alignment low
          "sparse_region"      : query in low-density embedding region
          "near_class_boundary": query near two LCLU class centroids
          "seasonal_mismatch"  : query and result from different seasons

Composite Confidence Score:
  confidence_score =
      0.40 × calibrated_prob
    + 0.30 × (1 - normalized_uncertainty)
    + 0.20 × similarity_gap_score
    + 0.10 × modality_consistency_score
```

---

## 8. Explanation Fusion Pipeline

### 8.1 Explanation Assembly

```
EXPLANATION FUSION PIPELINE

Inputs:
  global_similarity      : float (cosine)
  semantic_axes          : dict[axis_name → axis_match_object]
  spatial_attribution    : {query_heatmap, result_heatmap, alignment_score}
  confidence_bundle      : ConfidenceBundle
  region_importance      : list[RegionImportance]  (top-3 attended regions)

Step 1: Determine dominant explanation axes
  sorted_axes = sort(semantic_axes, key=lambda a: abs(a.query_score - a.result_score))
  divergent_axes  = [a for a if a.similarity < 0.4]   (significant divergence)
  convergent_axes = [a for a if a.similarity > 0.75]  (strong match)

Step 2: Determine explanation narrative category

  CASE A — High global similarity, high axis convergence:
    Category: "Strong Semantic Match"
    Template: "Retrieved because of highly similar {convergent_axes_list}.
               Overall embedding similarity: {global_sim:.2f}."

  CASE B — High global similarity, mixed axes:
    Category: "Partial Match with Notable Divergence"
    Template: "Retrieved due to shared {convergent_axes} characteristics.
               However, {divergent_axis} differs significantly:
               query shows {q_score:.0%} coverage vs {r_score:.0%} in result."

  CASE C — Moderate global similarity:
    Category: "Moderate Semantic Match"
    Template: "Moderate overall similarity ({global_sim:.2f}).
               Best-matching features: {top_2_convergent_axes}."

  CASE D — Low confidence:
    Category: "Uncertain Match"
    Template: "Retrieved with moderate similarity but high uncertainty
               (confidence: {confidence_score:.0%}). The query may fall
               near a class boundary. Recommended: verify manually."

Step 3: Attach spatial anchor to narrative
  top_region = region_importance[0]
  narrative += f" Primary attention region: {top_region.class_name}
                 ({top_region.coverage:.0%} of scene)."

Step 4: Produce structured explanation object
  ExplanationBundle:
    ├── query_id             : UUID
    ├── result_scene_id      : str
    ├── explanation_version  : "1.0"
    ├── global_similarity    : float
    ├── semantic_axes        : SemanticAxisBundle
    ├── spatial_attribution  : SpatialAttributionBundle
    ├── confidence           : ConfidenceBundle
    ├── region_importance    : list[RegionImportance]
    ├── narrative_category   : str (CASE A–D)
    ├── natural_language     : str (generated narrative)
    └── generated_at         : timestamp
```

### 8.2 Caching Strategy

```
EXPLANATION CACHING

Two-layer cache:

Layer 1: Per-scene precomputed signals (permanent)
  What:  Attention heatmap, spectral indices, semantic axis scores
  When:  Computed once at ingestion time for every scene in catalog
  Where: PostgreSQL (scene_explanations table) + S3 (heatmap PNGs)
  Key:   scene_id
  TTL:   Never expires (recompute only on model update)

Layer 2: Per-(query, result) pair explanations (transient)
  What:  Full ExplanationBundle including pairwise axis similarities
  When:  Computed on first retrieval of this (query_embedding, result_scene) pair
  Where: Redis (7-day TTL) + PostgreSQL (permanent for logged sessions)
  Key:   hash(query_embedding) + result_scene_id
  TTL:   Redis: 7 days; PostgreSQL: permanent for saved sessions

Model-update invalidation:
  On model version bump → flush Redis + requeue all scene precomputation
  Incremental: only recompute scenes added since last model version
```

---

## 9. API Design

### 9.1 Endpoint Inventory

```
BASE URL: /api/v1

RETRIEVAL WITH EXPLANATION
──────────────────────────
POST   /retrieval/optical              Optical same-modal retrieval
POST   /retrieval/sar                  SAR same-modal retrieval
POST   /retrieval/multispectral        Multispectral same-modal retrieval
POST   /retrieval/cross-modal          Cross-modal retrieval
POST   /retrieval/batch                Batch retrieval (up to 50 queries)

EXPLANATION (standalone, for already-retrieved results)
───────────────────────────────────────────────────────
GET    /explain/{query_id}/{scene_id}  Full explanation for a pair
GET    /explain/{query_id}/{scene_id}/heatmap/query    PNG heatmap (query)
GET    /explain/{query_id}/{scene_id}/heatmap/result   PNG heatmap (result)
GET    /explain/{query_id}/{scene_id}/heatmap/diff     Attention diff overlay
GET    /explain/{query_id}/{scene_id}/axes             Semantic axes only (fast)
POST   /explain/batch                  Batch explanation for multiple pairs

INTERACTIVE / STREAMING
───────────────────────
WS     /search/interactive             WebSocket: progressive result + explanation
POST   /search/refine                  Refine by axis (e.g., "more urban")

CONFIDENCE
──────────
GET    /confidence/{query_id}          Confidence bundle for a query
POST   /confidence/calibrate           Admin: recalibrate on new val data
```

### 9.2 Core Request/Response Schemas

```
POST /retrieval/optical
────────────────────────

Request:
  Content-Type: multipart/form-data

  Fields:
    image              : file         JPEG/GeoTIFF/PNG, max 50MB
    top_k              : int          5 or 10 (default: 10)
    include_explanation: bool         true (default: true)
    explanation_depth  : str          "fast" | "full" (default: "fast")
      "fast"  → global similarity + semantic axes + confidence (< 200ms P99)
      "full"  → + attention heatmaps + region importance (< 800ms P99)
    filters            : JSON (optional)
      {
        "bbox"        : [minLon, minLat, maxLon, maxLat],
        "date_range"  : { "from": "2022-01-01", "to": "2023-12-31" },
        "season"      : "summer" | "winter" | "spring" | "fall",
        "modality"    : "optical" | "sar" | "multispectral" | "any"
      }

Response 200:
  {
    "query_id"       : "3f7a2d91-...",
    "query_embedding": [0.031, -0.142, ...],  // 256-d, omitted if not requested
    "latency_ms"     : 147,
    "results"        : [
      {
        "rank"            : 1,
        "scene_id"        : "S2_2022_08_13_T32UPB_patch_0041",
        "modality"        : "optical",
        "similarity"      : 0.912,
        "thumbnail_url"   : "https://cdn.geosense.io/thumbs/...",
        "metadata"        : {
          "date"          : "2022-08-13",
          "bbox"          : [13.21, 52.45, 13.34, 52.52],
          "season"        : "summer",
          "lclu_class"    : "urban_fabric",
          "cloud_cover_pct": 3.2
        },
        "explanation"     : {
          "semantic_axes" : {
            "vegetation"  : {
              "query_score"   : 0.14,
              "result_score"  : 0.11,
              "similarity"    : 0.97,
              "verdict"       : "strong match"
            },
            "waterbody"   : {
              "query_score"   : 0.08,
              "result_score"  : 0.31,
              "similarity"    : 0.77,
              "verdict"       : "partial match"
            },
            "urban"       : {
              "query_score"   : 0.89,
              "result_score"  : 0.84,
              "similarity"    : 0.95,
              "verdict"       : "strong match"
            },
            "road_network": {
              "query_score"   : 0.76,
              "result_score"  : 0.81,
              "similarity"    : 0.95,
              "verdict"       : "strong match"
            }
          },
          "confidence"    : {
            "score"           : 0.88,
            "tier"            : "high",
            "calibrated_prob" : 0.91,
            "uncertainty"     : 0.018,
            "uncertainty_bound": "±0.04",
            "flags"           : []
          },
          "natural_language": "Retrieved because of highly similar urban fabric
                               and road network density. Both scenes show dense
                               urban development (~85%) with minimal vegetation.
                               Minor divergence in water body presence
                               (result has more canal coverage).",
          "heatmap_urls"  : {
            "query"  : "/explain/3f7a.../S2_2022.../heatmap/query",
            "result" : "/explain/3f7a.../S2_2022.../heatmap/result",
            "diff"   : "/explain/3f7a.../S2_2022.../heatmap/diff"
          },
          "region_importance": [
            { "class": "urban_fabric",  "attention_density": 0.84,
              "coverage_pct": 68.2 },
            { "class": "road_network",  "attention_density": 0.71,
              "coverage_pct": 9.1  },
            { "class": "green_space",   "attention_density": 0.38,
              "coverage_pct": 14.7 }
          ],
          "attention_alignment": 0.79
        }
      }
      // ... up to top_k results
    ]
  }

Response 422:
  { "error": "INVALID_MODALITY", "detail": "Image appears to be SAR. Use /retrieval/sar." }

Response 413:
  { "error": "IMAGE_TOO_LARGE", "detail": "Max 50MB. Received 73MB." }
```

### 9.3 WebSocket Interactive Search Protocol

```
WS /search/interactive
────────────────────────

Connection:
  ws://api.geosense.io/api/v1/search/interactive
  Auth: Bearer token in query param ?token=...

Client → Server messages:

  Type: "QUERY"
  {
    "type"    : "QUERY",
    "image_b64": "<base64-encoded image>",
    "modality" : "optical",
    "top_k"    : 10,
    "explanation_depth": "full"
  }

  Type: "REFINE"
  {
    "type"         : "REFINE",
    "query_id"     : "3f7a2d91-...",
    "axis"         : "urban",
    "direction"    : "more",      // "more" | "less"
    "lock_results" : [1, 3]       // keep results at rank 1 and 3
  }

  Type: "EXPLORE_HOP"
  {
    "type"     : "EXPLORE_HOP",
    "query_id" : "3f7a2d91-...",
    "pivot_scene_id": "S2_2022_08_13_...",   // use result as new query
    "preserve_axes" : ["urban", "road_network"]
  }

Server → Client messages (streaming):

  Phase 1: EMBEDDING_READY (< 50ms)
  { "type": "EMBEDDING_READY", "query_id": "..." }

  Phase 2: TOP_K_RESULTS (< 150ms after EMBEDDING_READY)
  { "type": "TOP_K_RESULTS",
    "results": [{ rank, scene_id, similarity, thumbnail_url, metadata }] }

  Phase 3: AXES_READY (< 50ms after TOP_K_RESULTS)
  { "type": "AXES_READY",
    "results": [{ scene_id, explanation: { semantic_axes, confidence, natural_language } }] }

  Phase 4: HEATMAPS_READY (< 500ms after AXES_READY, per result)
  { "type": "HEATMAP_READY",
    "scene_id": "...",
    "heatmap_urls": { query, result, diff } }

  Error:
  { "type": "ERROR", "code": "QUERY_FAILED", "detail": "..." }
```

### 9.4 Axis Refinement API

```
POST /search/refine
────────────────────

Request:
  {
    "query_id"   : "3f7a2d91-...",
    "refinements": [
      { "axis": "urban",         "direction": "more",     "weight": 1.5 },
      { "axis": "vegetation",    "direction": "less",     "weight": 0.5 },
      { "axis": "road_network",  "direction": "maintain", "weight": 1.0 }
    ],
    "top_k": 10
  }

Mechanism:
  Modified query embedding:
    z_refined = z_query + Σ (weight_i × direction_i × axis_vector_i)
    z_refined = L2_normalize(z_refined)

  direction_i: +1.0 for "more", -1.0 for "less", 0.0 for "maintain"
  weight_i: 0.5 - 2.0 (user-controllable slider in UI)

Response:
  Same schema as POST /retrieval/* with updated results and explanations.
  Includes "refinement_applied": true and "refinement_delta_embedding": [256].
```

---

## 10. Database Schema

### 10.1 Core Tables

```sql
-- ─────────────────────────────────────────────────────────────────
-- TABLE: scene_explanations
-- Precomputed per-scene explanation signals.
-- Populated at ingestion time; invalidated on model update.
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE scene_explanations (
    scene_id              TEXT          PRIMARY KEY
                          REFERENCES satellite_scenes(scene_id)
                          ON DELETE CASCADE,

    -- Model version this explanation was computed with
    model_version         TEXT          NOT NULL,

    -- Semantic axis scores (0.0 to 1.0)
    axis_vegetation       REAL          NOT NULL,
    axis_waterbody        REAL          NOT NULL,
    axis_urban            REAL          NOT NULL,
    axis_road_network     REAL          NOT NULL,

    -- Spectral index summaries
    ndvi_mean             REAL,           -- null for SAR
    ndvi_std              REAL,
    ndwi_mean             REAL,           -- null for SAR
    ndbi_mean             REAL,           -- null for SAR

    -- Vegetation profile [dense, moderate, sparse, non_veg]
    vegetation_profile    REAL[4],

    -- Water profile [open_water, wetland, moist_soil, dry_land]
    water_profile         REAL[4],

    -- Urban profile [dense, sparse, bare_soil, vegetated]
    urban_profile         REAL[4],

    -- Road network features [edge_density, line_count_norm, mean_len_norm, orient_entropy]
    road_features         REAL[4],

    -- Attention heatmap (stored as compressed float32 array)
    -- Full resolution [256×256] stored in S3; this is the 18×18 grid
    attention_grid        REAL[18][18],
    attention_heatmap_s3  TEXT,           -- S3 URI: s3://bucket/heatmaps/{scene_id}.npy
    attention_heatmap_png TEXT,           -- S3 URI: s3://bucket/heatmaps/{scene_id}.png

    -- Top-3 attended regions
    region_importance     JSONB,
    -- [{ "class": "urban_fabric", "attention_density": 0.84, "coverage_pct": 68.2 }, ...]

    -- Uncertainty from MC Dropout at ingestion time
    embedding_uncertainty REAL,           -- scalar std across T=20 passes

    -- Metadata
    computed_at           TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    compute_duration_ms   INTEGER,

    CONSTRAINT valid_axis_scores CHECK (
        axis_vegetation  BETWEEN 0 AND 1 AND
        axis_waterbody   BETWEEN 0 AND 1 AND
        axis_urban       BETWEEN 0 AND 1 AND
        axis_road_network BETWEEN 0 AND 1
    )
);

CREATE INDEX idx_scene_expl_model ON scene_explanations(model_version);
CREATE INDEX idx_scene_expl_urban ON scene_explanations(axis_urban);
CREATE INDEX idx_scene_expl_veg   ON scene_explanations(axis_vegetation);


-- ─────────────────────────────────────────────────────────────────
-- TABLE: retrieval_sessions
-- One row per retrieval request.
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE retrieval_sessions (
    query_id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id               UUID          REFERENCES users(id) ON DELETE SET NULL,
    api_key_id            UUID          REFERENCES api_keys(id) ON DELETE SET NULL,

    -- Query input
    modality              TEXT          NOT NULL
                          CHECK (modality IN ('optical', 'sar', 'multispectral')),
    query_image_s3        TEXT,           -- S3 URI to original query image
    query_embedding       vector(256)   NOT NULL,     -- pgvector

    -- Query axis profile
    query_axis_vegetation  REAL,
    query_axis_waterbody   REAL,
    query_axis_urban       REAL,
    query_axis_road_network REAL,

    -- Request config
    top_k                 INTEGER       NOT NULL DEFAULT 10,
    explanation_depth     TEXT          NOT NULL DEFAULT 'fast',
    filters               JSONB,

    -- Confidence at query level
    query_uncertainty     REAL,
    confidence_tier       TEXT,

    -- Performance
    total_latency_ms      INTEGER,
    faiss_latency_ms      INTEGER,
    explanation_latency_ms INTEGER,

    -- Metadata
    model_version         TEXT          NOT NULL,
    created_at            TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sessions_user    ON retrieval_sessions(user_id, created_at DESC);
CREATE INDEX idx_sessions_created ON retrieval_sessions(created_at DESC);
CREATE INDEX idx_sessions_modality ON retrieval_sessions(modality, created_at DESC);


-- ─────────────────────────────────────────────────────────────────
-- TABLE: retrieval_results
-- One row per (session, result scene) pair.
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE retrieval_results (
    id                    BIGSERIAL     PRIMARY KEY,
    query_id              UUID          NOT NULL
                          REFERENCES retrieval_sessions(query_id) ON DELETE CASCADE,
    scene_id              TEXT          NOT NULL
                          REFERENCES satellite_scenes(scene_id),
    rank                  SMALLINT      NOT NULL,

    -- Raw similarity
    cosine_similarity     REAL          NOT NULL,

    -- Pairwise semantic axis similarities
    axis_sim_vegetation   REAL,
    axis_sim_waterbody    REAL,
    axis_sim_urban        REAL,
    axis_sim_road_network REAL,

    -- Per-axis verdicts (stored as text for readability)
    axis_verdicts         JSONB,
    -- { "vegetation": "strong match", "urban": "partial match", ... }

    -- Confidence bundle
    calibrated_prob       REAL,
    confidence_score      REAL,
    confidence_tier       TEXT,
    uncertainty_bound     TEXT,
    confidence_flags      TEXT[],

    -- Attention alignment
    attention_alignment   REAL,

    -- Natural language explanation
    narrative_category    TEXT,
    natural_language      TEXT,

    -- Heatmap artifacts (S3 URIs)
    heatmap_diff_s3       TEXT,

    -- User feedback (populated later if user rates the result)
    user_feedback         TEXT
                          CHECK (user_feedback IN ('relevant', 'not_relevant',
                                                   'partially_relevant', NULL)),
    feedback_at           TIMESTAMPTZ,

    created_at            TIMESTAMPTZ   NOT NULL DEFAULT NOW(),

    UNIQUE (query_id, scene_id),
    UNIQUE (query_id, rank)
);

CREATE INDEX idx_results_query    ON retrieval_results(query_id, rank);
CREATE INDEX idx_results_scene    ON retrieval_results(scene_id);
CREATE INDEX idx_results_feedback ON retrieval_results(user_feedback)
    WHERE user_feedback IS NOT NULL;


-- ─────────────────────────────────────────────────────────────────
-- TABLE: semantic_axis_vectors
-- Stores learned semantic axis direction vectors.
-- One row per axis per model version.
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE semantic_axis_vectors (
    id                    SERIAL        PRIMARY KEY,
    model_version         TEXT          NOT NULL,
    axis_name             TEXT          NOT NULL
                          CHECK (axis_name IN ('vegetation', 'waterbody',
                                               'urban', 'road_network')),
    axis_vector           vector(256)   NOT NULL,     -- pgvector unit vector
    bias                  REAL          NOT NULL DEFAULT 0.0,
    threshold             REAL          NOT NULL,     -- P(relevant)=0.5 crossover
    temperature           REAL          NOT NULL DEFAULT 1.0, -- Platt scaling T

    -- Probe training metadata
    probe_train_accuracy  REAL,
    probe_val_accuracy    REAL,
    probe_trained_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),

    UNIQUE (model_version, axis_name)
);


-- ─────────────────────────────────────────────────────────────────
-- TABLE: calibration_records
-- Temperature scaling parameters per LCLU class per model version.
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE calibration_records (
    id                    SERIAL        PRIMARY KEY,
    model_version         TEXT          NOT NULL,
    lclu_class            TEXT          NOT NULL,
    temperature           REAL          NOT NULL,
    threshold             REAL          NOT NULL,
    ece_before            REAL,         -- Expected Calibration Error before scaling
    ece_after             REAL,         -- Expected Calibration Error after scaling
    calibrated_at         TIMESTAMPTZ   NOT NULL DEFAULT NOW(),

    UNIQUE (model_version, lclu_class)
);


-- ─────────────────────────────────────────────────────────────────
-- TABLE: explanation_feedback
-- Captures analyst feedback on individual explanations.
-- Used for: calibration improvement, axis vector refinement.
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE explanation_feedback (
    id                    BIGSERIAL     PRIMARY KEY,
    result_id             BIGINT        NOT NULL
                          REFERENCES retrieval_results(id) ON DELETE CASCADE,
    user_id               UUID          REFERENCES users(id) ON DELETE SET NULL,

    -- Which axis was wrong (if any)
    incorrect_axes        TEXT[],       -- e.g., ['vegetation', 'waterbody']
    correct_vegetation    REAL,         -- analyst-corrected score (null if not changed)
    correct_waterbody     REAL,
    correct_urban         REAL,
    correct_road_network  REAL,

    -- Was natural language explanation accurate?
    narrative_accurate    BOOLEAN,
    narrative_correction  TEXT,

    -- Was the overall explanation helpful?
    helpfulness_rating    SMALLINT      CHECK (helpfulness_rating BETWEEN 1 AND 5),

    created_at            TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);
```

### 10.2 Views for Analytics

```sql
-- ─────────────────────────────────────────────────────────────────
-- VIEW: explanation_quality_metrics
-- Monitors explanation accuracy via user feedback.
-- ─────────────────────────────────────────────────────────────────
CREATE VIEW explanation_quality_metrics AS
SELECT
    DATE_TRUNC('week', ef.created_at)       AS week,
    COUNT(*)                                 AS total_feedback,
    AVG(ef.helpfulness_rating)               AS avg_helpfulness,
    COUNT(*) FILTER (WHERE ef.narrative_accurate = TRUE)
        / COUNT(*)::FLOAT                    AS narrative_accuracy_rate,
    UNNEST(ef.incorrect_axes)                AS incorrect_axis,
    COUNT(*) FILTER (WHERE 'vegetation' = ANY(ef.incorrect_axes))
        / COUNT(*)::FLOAT                    AS vegetation_error_rate,
    COUNT(*) FILTER (WHERE 'urban' = ANY(ef.incorrect_axes))
        / COUNT(*)::FLOAT                    AS urban_error_rate
FROM explanation_feedback ef
GROUP BY 1
ORDER BY 1 DESC;


-- ─────────────────────────────────────────────────────────────────
-- VIEW: retrieval_confidence_distribution
-- Monitors confidence tier distribution over time.
-- ─────────────────────────────────────────────────────────────────
CREATE VIEW retrieval_confidence_distribution AS
SELECT
    DATE_TRUNC('day', rs.created_at)         AS day,
    rs.modality,
    rr.confidence_tier,
    COUNT(*)                                  AS result_count,
    AVG(rr.calibrated_prob)                  AS avg_calibrated_prob,
    AVG(rr.cosine_similarity)                AS avg_similarity
FROM retrieval_results rr
JOIN retrieval_sessions rs ON rs.query_id = rr.query_id
GROUP BY 1, 2, 3
ORDER BY 1 DESC;
```

---

## 11. Frontend UI Design

### 11.1 Main Retrieval Results Panel

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  GeoSense-X  │  Explore  │  Catalog  │  Intelligence  │  Search    [User ▾] │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────┐  ┌────────────────────────────────────────────┐│
│  │   QUERY IMAGE           │  │  RESULTS  (10 found, 147ms)         [↓CSV] ││
│  │                         │  │                                            ││
│  │  ┌─────────────────┐   │  │  Modality: [Optical ▾]  Sort: [Relevance▾] ││
│  │  │                 │   │  │  Filter: [Vegetation ≥] [Urban ≥] [Date ▾] ││
│  │  │  [scene image]  │   │  ├────────────────────────────────────────────┤│
│  │  │                 │   │  │                                            ││
│  │  └─────────────────┘   │  │  ┌──────────────────────────────────────┐  ││
│  │                         │  │  │ #1  S2_2022_08_Berlin_patch_0041    │  ││
│  │  Modality: Optical      │  │  │ ████████████████████░░  92.1%       │  ││
│  │  256 × 256 px           │  │  │                                      │  ││
│  │                         │  │  │ ┌──────────┐  Semantic Match        │  ││
│  │  ─────────────────      │  │  │ │[thumbnail│  ●●●●○ Urban   0.95   │  ││
│  │  QUERY PROFILE          │  │  │ │          │  ●●●●○ Roads   0.95   │  ││
│  │                         │  │  │ │          │  ●○○○○ Water   0.77   │  ││
│  │  Vegetation  ████░ 14%  │  │  │ │          │  ●○○○○ Vegeta  0.97   │  ││
│  │  Waterbody   █░░░░  8%  │  │  │ └──────────┘                        │  ││
│  │  Urban       ████████89%│  │  │                                      │  ││
│  │  Road Net    ████████76%│  │  │  ◆ HIGH CONFIDENCE  0.88  ±0.04     │  ││
│  │                         │  │  │                                      │  ││
│  │  ─────────────────      │  │  │  "Highly similar urban fabric and   │  ││
│  │  ATTENTION MAP          │  │  │   road network. Minor divergence     │  ││
│  │  ┌─────────────────┐   │  │  │   in water body presence."          │  ││
│  │  │ [heatmap        │   │  │  │                                      │  ││
│  │  │  overlay on     │   │  │  │  [View Heatmaps] [Use as Query] [♡] │  ││
│  │  │  query image]   │   │  │  └──────────────────────────────────────┘  ││
│  │  │ warm=attended   │   │  │                                            ││
│  │  └─────────────────┘   │  │  ┌──────────────────────────────────────┐  ││
│  │                         │  │  │ #2  S2_2023_06_Paris_patch_0187     │  ││
│  │  [Change Query Image]   │  │  │ ██████████████████░░░  87.4%        │  ││
│  │  [Refine by Axis ▾]     │  │  │ ...                                  │  ││
│  └─────────────────────────┘  │  └──────────────────────────────────────┘  ││
│                               └────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────────┘
```

### 11.2 Expanded Result Explanation Panel

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ← Back to Results          RESULT EXPLANATION                    [Export ▾] │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Scene: S2_2022_08_13_Berlin_patch_0041   │  Similarity: 92.1%  ◆ HIGH CONF│
│                                                                              │
├───────────────────────────┬──────────────────────────────────────────────────┤
│  QUERY                    │  RETRIEVED RESULT                                │
│  ┌───────────────────┐    │  ┌───────────────────┐                          │
│  │                   │    │  │                   │                          │
│  │  [query image]    │    │  │  [result image]   │                          │
│  │                   │    │  │                   │                          │
│  └───────────────────┘    │  └───────────────────┘                          │
│  [RGB] [Attention] [NDVI] │  [RGB] [Attention] [NDVI] [Diff ▾]              │
├───────────────────────────┴──────────────────────────────────────────────────┤
│  SEMANTIC AXIS COMPARISON                                                    │
│                                                                              │
│  Vegetation                                                                  │
│  Query  ████░░░░░░░░░░░░░░░░  14%    Result ██░░░░░░░░░░░░░░░░░░  11%      │
│                                              ◉ Strong Match (0.97)           │
│                                                                              │
│  Waterbody                                                                   │
│  Query  █░░░░░░░░░░░░░░░░░░░   8%    Result ████░░░░░░░░░░░░░░░░  31%      │
│                                              ◎ Partial Match (0.77)          │
│                                              ⚠ Result has more water          │
│                                                                              │
│  Urban Density                                                               │
│  Query  ████████████████░░░░  89%    Result ████████████████░░░░  84%      │
│                                              ◉ Strong Match (0.95)           │
│                                                                              │
│  Road Network                                                                │
│  Query  ██████████████░░░░░░  76%    Result ████████████████░░░░  81%      │
│                                              ◉ Strong Match (0.95)           │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  ATTENTION COMPARISON                                                        │
│                                                                              │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐        │
│  │ Query Attention  │   │ Result Attention  │   │  Diff Overlay    │        │
│  │ ┌──────────────┐ │   │ ┌──────────────┐ │   │ ┌──────────────┐ │        │
│  │ │[heatmap on   │ │   │ │[heatmap on   │ │   │ │[red=query    │ │        │
│  │ │ query image] │ │   │ │ result image]│ │   │ │ blue=result  │ │        │
│  │ │              │ │   │ │              │ │   │ │ green=shared]│ │        │
│  │ └──────────────┘ │   │ └──────────────┘ │   │ └──────────────┘ │        │
│  │ Top region:      │   │ Top region:      │   │ Alignment: 0.79  │        │
│  │ Urban fabric     │   │ Urban fabric     │   │ (Strong)         │        │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘        │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  CONFIDENCE BREAKDOWN                                                        │
│                                                                              │
│  Overall Confidence         ████████████████████░░  88%  HIGH               │
│  Embedding Uncertainty      ██░░░░░░░░░░░░░░░░░░░░  ±0.04  (Very Low)      │
│  Calibrated Probability     ████████████████████░░  91%                     │
│  Modality Consistency       ████████████████░░░░░░  83%                     │
│                                                                              │
│  ✓  No confidence flags raised                                               │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  EXPLANATION                                                                 │
│                                                                              │
│  "Retrieved because of highly similar urban fabric and road network density. │
│   Both scenes show dense urban development (~85%) with minimal vegetation.   │
│   Minor divergence in water body presence: result scene contains canal       │
│   infrastructure (~31% water) vs query scene (~8%). Primary attention        │
│   focused on urban fabric (68% of scene, 0.84 attention density)."          │
│                                                                              │
│  Was this explanation helpful?   [👍 Yes]  [👎 No, correct it]              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 11.3 Axis Refinement UI

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  REFINE SEARCH BY SEMANTIC AXIS                           [Apply] [Reset]    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Adjust how much weight each land-cover axis has in your search.             │
│                                                                              │
│  Vegetation Coverage                                                         │
│  Less ◄────────────────────●────────────────────► More                      │
│       0.0       0.5       [0.6]      0.8       1.0                          │
│                                                                              │
│  Waterbody Presence                                                          │
│  Less ◄──────────●──────────────────────────────► More                      │
│       0.0      [0.2]     0.5       0.8       1.0                            │
│                                                                              │
│  Urban Density                                                               │
│  Less ◄────────────────────────────────●────────► More                      │
│       0.0       0.3       0.6      [0.9]      1.0                           │
│                                                                              │
│  Road Network                                                                │
│  Less ◄──────────────────────●──────────────────► More                      │
│       0.0       0.3      [0.7]      0.8       1.0                           │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────          │
│  LOCK RESULTS (keep these in results regardless of refinement):              │
│  ☑ #1  S2_2022_08_Berlin_patch_0041  (92.1%)                                │
│  ☐ #2  S2_2023_06_Paris_patch_0187   (87.4%)                                │
│  ☐ #3  S2_2021_09_Warsaw_patch_0093  (84.9%)                                │
│                                                                              │
│  [Preview Changes]                          [Apply Refinement]              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 11.4 Confidence Indicator Design System

```
CONFIDENCE TIER VISUAL LANGUAGE

◆ HIGH CONFIDENCE (score ≥ 0.80)
  Color: #16A34A (green-600)
  Icon:  filled diamond ◆
  Badge: "HIGH CONF  0.88 ±0.04"
  Behavior: no warning shown

◈ MEDIUM CONFIDENCE (score 0.60–0.79)
  Color: #D97706 (amber-600)
  Icon:  half-filled diamond ◈
  Badge: "MEDIUM  0.71 ±0.09"
  Behavior: show "Verify recommended" tooltip

◇ LOW CONFIDENCE (score 0.40–0.59)
  Color: #DC2626 (red-600)
  Icon:  outline diamond ◇
  Badge: "LOW CONF  0.52 ±0.14"
  Behavior: show warning banner under result card

⚠ UNCERTAIN (score < 0.40)
  Color: #7C3AED (violet-600)
  Icon:  triangle warning ⚠
  Badge: "UNCERTAIN  0.31 ±0.19"
  Behavior: prominent "manual review recommended" alert
            show confidence flags as chip list

CONFIDENCE FLAGS (chip display under confidence badge):
  [⚡ cross-modal gap]      gray chip
  [⚡ sparse region]        gray chip
  [⚡ near class boundary]  gray chip
  [⚡ seasonal mismatch]    gray chip
```

### 11.5 Heatmap Overlay Controls

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  HEATMAP OVERLAY CONTROLS                                   [Download PNG]   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Layer: [DINOv2 Attention ▾]    ← dropdown                                  │
│         ├── DINOv2 Attention (default)                                      │
│         ├── Vegetation (NDVI)                                               │
│         ├── Waterbody (NDWI)                                                │
│         ├── Urban (NDBI)                                                    │
│         └── Road Network (Edge Density)                                     │
│                                                                              │
│  Opacity:  ─────────────────●───────  65%                                   │
│                                                                              │
│  Colormap: [Viridis ▾]   ← dropdown: Viridis / Plasma / Jet / Grayscale    │
│                                                                              │
│  Threshold: ─────────●───────────────  40%  (hide below 40% intensity)      │
│                                                                              │
│  Show:  ☑ Query   ☑ Result   ☑ Difference                                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 11.6 Component Architecture (React)

```
FRONTEND COMPONENT TREE (explanation-related)

<ResultsPage>
  ├── <QueryPanel>
  │   ├── <ImageUploadZone>
  │   ├── <ModalitySelector>
  │   ├── <QueryAxisProfile>        ← live-updating axis bar chart
  │   └── <QueryAttentionOverlay>   ← heatmap on query image
  │
  ├── <RefinementDrawer>            ← slides in from right
  │   ├── <AxisSlider axis="vegetation">
  │   ├── <AxisSlider axis="waterbody">
  │   ├── <AxisSlider axis="urban">
  │   ├── <AxisSlider axis="road_network">
  │   └── <LockResultsSelector>
  │
  └── <ResultsGrid>
      └── <ResultCard>  × N
          ├── <Thumbnail>
          ├── <SimilarityBar>
          ├── <SemanticAxesMiniChart>   ← compact 4-axis comparison
          ├── <ConfidenceBadge>
          ├── <NarrativeSummary>        ← truncated to 2 lines
          └── <ExplanationModal>        ← opens on "View Details"
              ├── <ImageComparisonPanel>
              │   ├── <HeatmapViewer side="query">
              │   ├── <HeatmapViewer side="result">
              │   └── <AttentionDiffViewer>
              ├── <SemanticAxisComparison>
              │   └── <AxisComparisonRow> × 4
              ├── <ConfidenceBreakdown>
              ├── <FullNarrative>
              └── <FeedbackWidget>

STATE MANAGEMENT (Zustand stores):

  searchStore:
    ├── queryId
    ├── queryEmbedding
    ├── queryAxisScores
    ├── results: ResultWithExplanation[]
    ├── refinementAxes: AxisWeights
    └── lockedResults: Set<scene_id>

  heatmapStore:
    ├── activeLayer: "attention" | "ndvi" | "ndwi" | "ndbi" | "road"
    ├── opacity: number
    ├── colormap: string
    ├── threshold: number
    └── loadedHeatmaps: Map<scene_id, HeatmapData>

DATA FETCHING STRATEGY (TanStack Query):

  Phase 1: Fire retrieval query → get results + fast explanation
  Phase 2: useQuery per result for heatmap URLs (lazy, on expand)
  Phase 3: WebSocket for streaming (interactive mode)
  Cache: staleTime=5min for retrieved results; heatmaps=Infinity (immutable)
```

---

## 12. Folder Structure

```
geosense-x/
│
├── services/embedding-service/src/
│   └── explainability/
│       ├── __init__.py
│       ├── attention/
│       │   ├── __init__.py
│       │   ├── extractor.py          # DINOv2 hook-based attention extraction
│       │   ├── rollout.py            # Attention rollout across all layers
│       │   ├── heatmap.py            # Upsampling, normalization, PNG rendering
│       │   └── alignment.py          # SSIM-based attention alignment score
│       ├── axes/
│       │   ├── __init__.py
│       │   ├── probe.py              # Linear probe training for axis vectors
│       │   ├── scorer.py             # Axis score computation for any embedding
│       │   └── registry.py           # Load/save axis vectors from DB
│       └── region/
│           ├── __init__.py
│           └── importance.py         # Top-3 region importance from attn + LCLU
│
├── apps/api/src/domain/
│   └── explanation/
│       ├── __init__.py
│       ├── engine.py                 # ExplanationEngine: orchestrates all signals
│       ├── decomposer.py             # Pairwise semantic axis similarity
│       ├── confidence/
│       │   ├── __init__.py
│       │   ├── mc_dropout.py         # T=20 MC Dropout uncertainty
│       │   ├── calibration.py        # Temperature scaling per LCLU class
│       │   ├── gap_scorer.py         # Top-1 vs Top-K similarity gap
│       │   └── composite.py          # Weighted composite confidence score
│       ├── spectral/
│       │   ├── __init__.py
│       │   ├── ndvi.py               # NDVI computation + profile
│       │   ├── ndwi.py               # NDWI computation + profile
│       │   ├── ndbi.py               # NDBI computation + profile
│       │   └── sar_indices.py        # SAR-specific water + urban detection
│       ├── road/
│       │   ├── __init__.py
│       │   ├── edge_detector.py      # Canny edge detection
│       │   ├── line_detector.py      # Hough line transform features
│       │   └── similarity.py         # Road feature vector similarity
│       ├── narrative/
│       │   ├── __init__.py
│       │   ├── categorizer.py        # CASE A–D narrative category assignment
│       │   └── generator.py          # Template-based natural language generation
│       └── cache/
│           ├── __init__.py
│           ├── scene_cache.py        # Per-scene signal cache (Redis + Postgres)
│           └── pair_cache.py         # Per-(query, result) explanation cache
│
├── apps/api/src/
│   ├── api/v1/
│   │   └── explanation/
│   │       ├── router.py             # Explanation endpoint router
│   │       ├── retrieve.py           # GET /explain/{query_id}/{scene_id}
│   │       ├── heatmaps.py           # GET /explain/.../heatmap/*
│   │       ├── axes.py               # GET /explain/.../axes
│   │       └── batch.py              # POST /explain/batch
│   └── models/schemas/
│       └── explanation.py            # Pydantic schemas for all explanation types
│
├── data/migrations/versions/
│   ├── 0005_scene_explanations.py    # scene_explanations table
│   ├── 0006_retrieval_sessions.py    # retrieval_sessions + retrieval_results
│   ├── 0007_axis_vectors.py          # semantic_axis_vectors table
│   └── 0008_calibration.py           # calibration_records + feedback tables
│
├── ml/training/
│   └── axis_probes/
│       ├── __init__.py
│       ├── datasets/
│       │   └── probe_dataset.py      # Pos/neg sample builder per axis
│       ├── trainers/
│       │   └── probe_trainer.py      # Linear probe training loop
│       └── scripts/
│           ├── train_probes.py       # Train all 4 axis probes
│           └── evaluate_probes.py    # Probe accuracy + axis visualization
│
└── apps/web/src/components/
    ├── explanation/
    │   ├── ExplanationModal.tsx
    │   ├── SemanticAxisComparison.tsx
    │   ├── AxisComparisonRow.tsx
    │   ├── ConfidenceBreakdown.tsx
    │   ├── ConfidenceBadge.tsx
    │   ├── FullNarrative.tsx
    │   └── FeedbackWidget.tsx
    ├── heatmap/
    │   ├── HeatmapViewer.tsx
    │   ├── AttentionDiffViewer.tsx
    │   └── HeatmapControls.tsx
    └── retrieval/
        ├── SemanticAxesMiniChart.tsx
        └── QueryAxisProfile.tsx
```

---

## 13. Design Decisions Log

| # | Decision | Alternatives Considered | Reason Rejected |
|---|---|---|---|
| 1 | Shared embedding space axis probing (linear probes) | Non-linear classifier probes | Non-linear probes extract axis *direction vectors* that can be used for embedding-space arithmetic (refinement); non-linear probes cannot produce directions |
| 2 | Template-based NL generation | LLM-generated explanation | LLM adds 200–400ms latency, hallucination risk on specific numeric values (NDVI scores, coordinates); templates are deterministic and verifiable |
| 3 | MC Dropout (T=5 real-time, T=20 async) | Ensemble of N models | Ensemble requires N× memory; MC Dropout reuses single model; T=5 is sufficient for tier assignment |
| 4 | Attention rollout (all layers) for XAI | Final-layer attention only | Final-layer attention can be noisy if earlier layers process low-level features; rollout propagates through depth for higher-fidelity attribution |
| 5 | SSIM for attention alignment | L2 distance of heatmaps | L2 treats all spatial shifts equally; SSIM is invariant to global brightness shifts (important as different scenes have different absolute attention magnitudes) |
| 6 | Per-LCLU calibration | Global calibration | Forest scenes and urban scenes have fundamentally different retrieval difficulty; global calibration systematically under-calibrates one class family |
| 7 | Heatmap PNGs in S3 + raw float32 in Postgres | Everything in S3 | Raw float32 grids in Postgres enable server-side colormap conversion and threshold filtering without S3 round-trip; PNGs in S3 serve frontend directly via CDN |
| 8 | Road similarity via edge detection + axis score | End-to-end road segmentation model | Segmentation model adds 80ms latency and requires labeled road data; Hough + embedding axis achieves ~0.74 correlation with labeled road density at ~5ms |

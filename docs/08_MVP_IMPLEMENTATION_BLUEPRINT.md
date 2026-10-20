# GeoSense-X — Phase 1 MVP Implementation Blueprint
# ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 11
## 5-Day Execution Plan

---

## Quick Reference

| Item | Value |
|---|---|
| Hackathon | ISRO Bharatiya Antariksh Hackathon 2026 — PS11 |
| MVP Deadline | Day 5, 18:00 |
| Core Feature | Cross-modal satellite image retrieval (Optical ↔ SAR ↔ MS) |
| Dataset | SEN12MS (use 10K subset for MVP) |
| Embedding Dim | 128-d (smaller for faster MVP training) |
| Top-K | 10 |
| FAISS Index | IVF-Flat (simpler than IVF-PQ for MVP) |
| Training Time | ~4 hours on single GPU (Colab A100 or local RTX) |
| Retrieval Latency Target | < 300ms end-to-end |

---

## Table of Contents

1. [Day-by-Day Plan](#1-day-by-day-plan)
2. [Complete Folder Structure](#2-complete-folder-structure)
3. [Every File — Purpose Registry](#3-every-file--purpose-registry)
4. [Dependencies](#4-dependencies)
5. [Dataset Preparation Steps](#5-dataset-preparation-steps)
6. [ML Training Workflow](#6-ml-training-workflow)
7. [FAISS Indexing Workflow](#7-faiss-indexing-workflow)
8. [API Workflow](#8-api-workflow)
9. [Development Milestones](#9-development-milestones)
10. [Risk Register](#10-risk-register)

---

## 1. Day-by-Day Plan

```
═══════════════════════════════════════════════════════════════════
  DAY 1  |  Environment + Dataset + Project Skeleton
  Target : Working data pipeline by 18:00
═══════════════════════════════════════════════════════════════════

MORNING (09:00 – 13:00)

  09:00  Set up monorepo root
         ├── Create folder structure (see §2)
         ├── Initialize git repo
         ├── Create .env.example
         └── Create docker-compose.yml (PostgreSQL + Redis)

  10:00  Install all dependencies (see §4)
         ├── Backend: uv/pip install into venv
         ├── Frontend: pnpm install
         └── Verify PyTorch + CUDA/MPS accessible

  11:00  Start SEN12MS download (runs in background all day)
         ├── Download ROIs_spring subset first (~60GB)
         └── Script: scripts/download_sen12ms.sh

  11:30  Write dataset loader
         ├── ml/data/sen12ms_dataset.py
         ├── ml/data/transforms.py
         └── Test: load one batch of 8 triplets, print shapes

  13:00  LUNCH

AFTERNOON (14:00 – 18:00)

  14:00  Write preprocessing pipeline
         ├── Optical  : normalize RGB, resize 224×224
         ├── SAR      : Lee speckle filter, pseudo-RGB, normalize
         └── MS       : PCA 13→3, normalize

  15:30  Test full data pipeline
         ├── Load triplet (optical, SAR, MS)
         ├── Apply all transforms
         ├── Verify tensor shapes: [3×224×224] for all three
         └── Visualize 8 samples (save PNG grid)

  16:30  Set up PostgreSQL schema
         ├── Run docker-compose up -d
         ├── Run Alembic migrations (scenes table, embeddings table)
         └── Verify DB connection from FastAPI

  17:30  Day 1 checkpoint review
         └── Confirm: data loads ✓, DB runs ✓, download progressing ✓

  OVERNIGHT: SEN12MS download continues


═══════════════════════════════════════════════════════════════════
  DAY 2  |  ML Training Pipeline
  Target : Trained model checkpoint by 18:00
═══════════════════════════════════════════════════════════════════

MORNING (09:00 – 13:00)

  09:00  Build encoder architecture
         ├── ml/models/encoder.py        ← DINOv2 + LoRA wrapper
         ├── ml/models/projection.py     ← 3-layer MLP head (1024→512→128)
         └── ml/models/cross_modal.py    ← CrossModalModel: 3 encoders + head

  10:30  Build contrastive loss
         ├── ml/losses/infonce.py        ← Multi-positive InfoNCE
         └── ml/losses/ntxent.py         ← NT-Xent for Stage 1

  11:30  Build training loop
         ├── ml/training/trainer.py      ← training + validation loop
         ├── ml/training/stage1.py       ← per-modality warm-up (5 epochs)
         └── ml/training/stage2.py       ← joint cross-modal (20 epochs)

  13:00  LUNCH

AFTERNOON (14:00 – 18:00)

  14:00  Run Stage 1 training
         ├── Train optical encoder  (5 epochs, ~30 min)
         ├── Train SAR encoder      (5 epochs, ~35 min)
         └── Train MS encoder       (5 epochs, ~32 min)

         If no GPU available → use pretrained DINOv2 features
         directly without fine-tuning (zero-shot baseline)

  16:00  Run Stage 2 joint training
         ├── Load Stage 1 checkpoints
         ├── Joint InfoNCE for 20 epochs (~90 min on A100)
         └── Save best checkpoint (by val loss)

  17:45  Validate checkpoint
         ├── Run quick retrieval test: 100 query set
         ├── Compute Recall@10 (aim for > 0.50 for MVP demo)
         └── Save model to ml/checkpoints/best_model.pt

  FALLBACK (if training stalls):
         Use zero-shot DINOv2 [CLS] features directly.
         No fine-tuning. Still demonstrates cross-modal retrieval.
         Recall@10 will be lower (~0.35) but demo works.


═══════════════════════════════════════════════════════════════════
  DAY 3  |  FAISS Index + FastAPI Backend
  Target : Working /retrieve API endpoint by 18:00
═══════════════════════════════════════════════════════════════════

MORNING (09:00 – 13:00)

  09:00  Compute embeddings for full dataset
         ├── ml/scripts/compute_embeddings.py
         │   Batch size 64, all 3 modalities
         │   Save: embeddings/optical.npy  [N×128]
         │         embeddings/sar.npy      [N×128]
         │         embeddings/ms.npy       [N×128]
         │         embeddings/metadata.json (scene_id, lat, lon, class)
         └── Time: ~20 min for 10K scenes on GPU

  10:00  Build FAISS indexes
         ├── ml/scripts/build_faiss_index.py
         │   Build 4 indexes:
         │     indexes/optical.index    (IVFFlat, nlist=64)
         │     indexes/sar.index        (IVFFlat, nlist=64)
         │     indexes/ms.index         (IVFFlat, nlist=64)
         │     indexes/shared.index     (all 3 modalities, nlist=128)
         └── Verify: search returns correct top-10 neighbors

  11:00  Build FastAPI application
         ├── backend/main.py             ← app factory
         ├── backend/config.py           ← settings
         ├── backend/api/v1/router.py    ← route aggregator
         └── backend/api/v1/retrieve.py  ← /retrieve/* endpoints

  12:00  Build retrieval service layer
         ├── backend/services/encoder_service.py   ← load model, encode query
         ├── backend/services/faiss_service.py     ← search, return top-10
         └── backend/services/result_service.py    ← fetch metadata, compose response

  13:00  LUNCH

AFTERNOON (14:00 – 18:00)

  14:00  Wire up /retrieve endpoints
         ├── POST /api/v1/retrieve/optical
         ├── POST /api/v1/retrieve/sar
         ├── POST /api/v1/retrieve/multispectral
         └── POST /api/v1/retrieve/cross-modal

  15:30  Test each endpoint with curl / Postman
         ├── Upload sample optical image → get top-10 optical results ✓
         ├── Upload sample optical image → get top-10 SAR results ✓
         ├── Upload sample SAR image → get top-10 MS results ✓
         └── Verify response time < 300ms

  16:30  Add scene thumbnail serving
         ├── backend/api/v1/scenes.py    ← GET /scenes/{id}/thumbnail
         └── Store thumbnails: static/thumbnails/{scene_id}.jpg

  17:30  Health check + Swagger UI
         ├── GET /health → { status: "ok", model_version, index_size }
         └── Verify auto-generated Swagger at /docs


═══════════════════════════════════════════════════════════════════
  DAY 4  |  Frontend + Integration
  Target : Working end-to-end demo in browser by 18:00
═══════════════════════════════════════════════════════════════════

MORNING (09:00 – 13:00)

  09:00  Build Next.js page structure
         ├── app/page.tsx                ← landing/redirect to /explore
         ├── app/explore/page.tsx        ← main demo page
         └── app/layout.tsx              ← root layout + providers

  09:30  Build core UI components
         ├── components/upload/ImageDropzone.tsx
         ├── components/upload/ModalitySelector.tsx
         └── components/upload/TopKSelector.tsx

  10:30  Build results display
         ├── components/results/ResultsGrid.tsx
         ├── components/results/ResultCard.tsx
         └── components/results/SimilarityBadge.tsx

  11:30  Wire up API client
         ├── lib/api/retrieve.ts         ← typed fetch wrapper
         └── lib/hooks/useRetrieval.ts   ← TanStack Query hook

  12:30  Build main explore page layout
         └── Two-panel: left=query, right=results grid

  13:00  LUNCH

AFTERNOON (14:00 – 18:00)

  14:00  Integration testing
         ├── Upload optical image → see 10 results in grid ✓
         ├── Switch to cross-modal → results change to SAR ✓
         ├── Similarity scores display correctly ✓
         └── Loading states work ✓

  15:00  Add basic explanation UI
         ├── Modality tag on each result card
         ├── Similarity bar (visual score indicator)
         └── Scene metadata tooltip (date, coordinates)

  16:00  Add map view (simple)
         ├── components/map/ResultsMap.tsx    ← MapLibre GL map
         ├── Pin results on map by lat/lon
         └── Click pin → highlight result card

  17:00  Polish and bug fixes
         ├── Error states (upload fails, timeout)
         ├── Empty state (no results)
         └── Responsive layout check

  17:30  Full end-to-end test
         ├── Upload SAR → retrieve optical (cross-modal) ✓
         ├── Upload optical → retrieve MS ✓
         └── Top-10 results shown with similarity scores ✓


═══════════════════════════════════════════════════════════════════
  DAY 5  |  Demo Polish + Evaluation + Submission
  Target : Demo-ready by 14:00; Submitted by 18:00
═══════════════════════════════════════════════════════════════════

MORNING (09:00 – 13:00)

  09:00  Run evaluation suite
         ├── ml/evaluation/evaluate_retrieval.py
         │   Compute on 1000-scene test set:
         │   ├── Recall@5, Recall@10
         │   ├── F1@5, F1@10
         │   ├── Same-modal (optical→optical, SAR→SAR, MS→MS)
         │   └── Cross-modal (all 6 direction pairs)
         └── Record results for presentation slides

  10:00  Demo data preparation
         ├── Select 5 compelling query images (diverse scenes)
         │   1. Dense urban (Berlin/Mumbai)
         │   2. Agriculture fields
         │   3. Coastline / water body
         │   4. Forest
         │   5. Mixed scene
         └── Pre-cache their results for fast demo

  11:00  Prepare presentation artifacts
         ├── Architecture diagram screenshot
         ├── Embedding space UMAP visualization
         │   ml/evaluation/visualize_embeddings.py
         │   → Save PNG: results/embedding_space.png
         ├── Sample retrieval result grids (screenshot)
         └── Metric table (F1@5, F1@10 per modality pair)

  12:00  Final bug fixes and polish
         ├── Ensure demo works on fresh browser
         ├── Pre-load demo images in UI
         └── Test on slow network (add loading skeleton)

  13:00  LUNCH

AFTERNOON (14:00 – 18:00)

  14:00  Rehearse demo
         ├── Run through all 5 query scenarios
         ├── Time each scenario (target: 2 min per scenario)
         └── Prepare fallback: pre-recorded screen capture

  15:00  Package for submission
         ├── Write DEMO_GUIDE.md
         ├── Write SETUP.md (one-command setup)
         ├── Create demo video (optional but recommended)
         └── Prepare GitHub repository

  16:00  Final submission
         ├── Push to GitHub
         ├── Upload submission artifacts
         └── Submit before 18:00

  CONTINGENCY TIME: 16:00–18:00 buffer for submission issues
```

---

## 2. Complete Folder Structure

```
geosense-x/
│
├── .env.example                          # Environment variable template
├── .gitignore
├── docker-compose.yml                    # PostgreSQL + Redis for local dev
├── Makefile                              # make setup / make train / make run
├── README.md
├── DEMO_GUIDE.md                         # Step-by-step demo instructions
├── SETUP.md                              # One-command setup guide
│
├── backend/                              # FastAPI application
│   ├── main.py
│   ├── config.py
│   ├── requirements.txt
│   │
│   ├── api/
│   │   └── v1/
│   │       ├── router.py
│   │       ├── retrieve.py
│   │       ├── scenes.py
│   │       └── health.py
│   │
│   ├── services/
│   │   ├── encoder_service.py
│   │   ├── faiss_service.py
│   │   └── result_service.py
│   │
│   ├── models/
│   │   ├── schemas.py
│   │   └── orm.py
│   │
│   ├── db/
│   │   ├── database.py
│   │   └── migrations/
│   │       ├── env.py
│   │       └── versions/
│   │           ├── 001_create_scenes.py
│   │           └── 002_create_embeddings.py
│   │
│   └── static/
│       └── thumbnails/                   # Precomputed scene thumbnail JPEGs
│
├── ml/                                   # All ML code
│   │
│   ├── data/
│   │   ├── sen12ms_dataset.py            # PyTorch Dataset for SEN12MS
│   │   ├── transforms.py                 # Per-modality preprocessing
│   │   ├── samplers.py                   # Triplet batch sampler
│   │   └── pca.py                        # PCA 13→3 for multispectral
│   │
│   ├── models/
│   │   ├── encoder.py                    # DINOv2 + LoRA wrapper
│   │   ├── projection.py                 # MLP projection head (1024→128)
│   │   └── cross_modal.py                # Combined model: 3 encoders + head
│   │
│   ├── losses/
│   │   ├── infonce.py                    # Multi-positive InfoNCE loss
│   │   └── ntxent.py                     # NT-Xent for Stage 1
│   │
│   ├── training/
│   │   ├── config.py                     # TrainingConfig dataclass
│   │   ├── trainer.py                    # Base training loop
│   │   ├── stage1.py                     # Modality-specific warm-up
│   │   └── stage2.py                     # Joint cross-modal training
│   │
│   ├── evaluation/
│   │   ├── evaluate_retrieval.py         # Recall@K, F1@K computation
│   │   └── visualize_embeddings.py       # UMAP plot of embedding space
│   │
│   ├── scripts/
│   │   ├── compute_embeddings.py         # Run inference on full dataset
│   │   ├── build_faiss_index.py          # Build and save all FAISS indexes
│   │   └── generate_thumbnails.py        # Create 128×128 JPEG thumbnails
│   │
│   └── checkpoints/                      # Saved model checkpoints
│       └── .gitkeep
│
├── indexes/                              # FAISS index files (binary)
│   ├── optical.index
│   ├── sar.index
│   ├── ms.index
│   └── shared.index
│
├── embeddings/                           # Precomputed numpy arrays
│   ├── optical.npy                       # [N × 128] float32
│   ├── sar.npy                           # [N × 128] float32
│   ├── ms.npy                            # [N × 128] float32
│   └── metadata.json                     # scene_id, lat, lon, lclu_class, date
│
├── data/
│   ├── sen12ms_raw/                      # Downloaded SEN12MS files
│   │   └── ROIs1158_spring/              # (gitignored, ~60GB)
│   └── sen12ms_processed/                # Preprocessed HDF5 shards
│       └── .gitkeep
│
├── scripts/
│   ├── download_sen12ms.sh               # wget/curl download script
│   ├── prepare_subset.py                 # Extract 10K triplet subset
│   └── setup_env.sh                      # One-command environment setup
│
├── frontend/                             # Next.js application
│   ├── package.json
│   ├── next.config.ts
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   │
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx
│   │   │   └── explore/
│   │   │       └── page.tsx
│   │   │
│   │   ├── components/
│   │   │   ├── upload/
│   │   │   │   ├── ImageDropzone.tsx
│   │   │   │   ├── ModalitySelector.tsx
│   │   │   │   └── TopKSelector.tsx
│   │   │   │
│   │   │   ├── results/
│   │   │   │   ├── ResultsGrid.tsx
│   │   │   │   ├── ResultCard.tsx
│   │   │   │   └── SimilarityBadge.tsx
│   │   │   │
│   │   │   ├── map/
│   │   │   │   └── ResultsMap.tsx
│   │   │   │
│   │   │   └── shared/
│   │   │       ├── LoadingSkeleton.tsx
│   │   │       └── ErrorMessage.tsx
│   │   │
│   │   ├── lib/
│   │   │   ├── api/
│   │   │   │   └── retrieve.ts
│   │   │   └── hooks/
│   │   │       └── useRetrieval.ts
│   │   │
│   │   └── types/
│   │       └── retrieval.ts
│   │
│   └── public/
│       └── demo-images/                  # Pre-loaded demo query images
│           ├── urban_optical.jpg
│           ├── forest_sar.jpg
│           ├── agriculture_ms.jpg
│           ├── coastal_optical.jpg
│           └── mixed_sar.jpg
│
└── results/                              # Evaluation outputs
    ├── metrics.json                      # F1@K, Recall@K results
    ├── embedding_space.png               # UMAP visualization
    └── retrieval_examples/               # Example query→result images
```

---

## 3. Every File — Purpose Registry

### Root Level

| File | Purpose |
|---|---|
| `.env.example` | Template for all environment variables; copy to `.env` |
| `docker-compose.yml` | Spins up PostgreSQL (port 5432) and Redis (port 6379) locally |
| `Makefile` | Shortcut commands: `make setup`, `make train`, `make index`, `make run` |
| `scripts/download_sen12ms.sh` | Downloads SEN12MS ROIs_spring subset using wget; supports resume |
| `scripts/prepare_subset.py` | Selects 10,000 balanced triplets from raw download; saves as HDF5 shards |
| `scripts/setup_env.sh` | Creates virtualenv, installs deps, initializes DB in one shot |

---

### `backend/` — FastAPI Application

| File | Purpose |
|---|---|
| `main.py` | Creates FastAPI app instance; registers routers; configures CORS; mounts static files |
| `config.py` | Pydantic Settings class; reads all env vars; exposes DB_URL, FAISS_DIR, MODEL_PATH |
| `requirements.txt` | Pinned Python dependencies for backend only |
| `api/v1/router.py` | Aggregates all v1 route modules into one APIRouter; mounted at `/api/v1` |
| `api/v1/retrieve.py` | Four POST endpoints: `/retrieve/optical`, `/sar`, `/multispectral`, `/cross-modal`; handles image upload, calls services, returns results |
| `api/v1/scenes.py` | GET `/scenes/{scene_id}/thumbnail` — serves JPEG thumbnail from static folder |
| `api/v1/health.py` | GET `/health` — returns model version, index sizes, DB status |
| `services/encoder_service.py` | Singleton: loads trained model on startup; `encode(image_bytes, modality) → np.ndarray [128]`; handles preprocessing per modality |
| `services/faiss_service.py` | Singleton: loads all 4 FAISS indexes on startup; `search(embedding, source_modality, target_modality, k) → list[SceneMatch]`; routes to correct index |
| `services/result_service.py` | Takes FAISS matches; fetches metadata from DB or metadata.json; composes RetrievalResult response objects |
| `models/schemas.py` | Pydantic request/response models: RetrievalRequest, RetrievalResult, SceneMatch, HealthResponse |
| `models/orm.py` | SQLAlchemy ORM models: Scene, EmbeddingRecord |
| `db/database.py` | Async SQLAlchemy engine; session factory; `get_db()` dependency |
| `db/migrations/env.py` | Alembic migration environment config |
| `db/migrations/versions/001_create_scenes.py` | Creates `scenes` table: scene_id, lat, lon, date, season, lclu_class, modality, thumbnail_path |
| `db/migrations/versions/002_create_embeddings.py` | Creates `embeddings` table: scene_id, modality, embedding (bytea), model_version |

---

### `ml/data/` — Data Pipeline

| File | Purpose |
|---|---|
| `sen12ms_dataset.py` | PyTorch Dataset; reads HDF5 shards; returns `(optical, sar, ms, metadata)` tuple per sample; handles missing data gracefully |
| `transforms.py` | Three transform pipelines: `OpticalTransform` (resize, normalize), `SARTransform` (speckle filter, pseudo-RGB), `MSTransform` (PCA projection, normalize); both train and inference modes |
| `samplers.py` | `TripletBatchSampler`: yields batch indices guaranteeing complete (optical, SAR, MS) triplets per scene; prevents partial-modality batches |
| `pca.py` | `PCAProjector`: fits 13→3 PCA on training set; saves coefficients to `ml/checkpoints/pca.npy`; `transform(x)` applies projection at runtime |

---

### `ml/models/` — Model Architecture

| File | Purpose |
|---|---|
| `encoder.py` | `SatelliteEncoder`: wraps `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')`; registers LoRA adapters on Q/K/V projections; exposes `encode(x) → [B×1024]` |
| `projection.py` | `ProjectionHead`: `Linear(1024→512) → BN → GELU → Linear(512→128) → BN → GELU → Linear(128→128) → L2Norm`; shared weights across modalities |
| `cross_modal.py` | `CrossModalModel`: holds three `SatelliteEncoder` instances (optical, sar, ms) + one shared `ProjectionHead`; `forward(optical, sar, ms) → (z_opt, z_sar, z_ms)` all [B×128]; `encode_single(x, modality) → [B×128]` for inference |

---

### `ml/losses/` — Training Objectives

| File | Purpose |
|---|---|
| `infonce.py` | `MultiPositiveInfoNCE`: given batch of triplet embeddings [3B×128], treats same-scene cross-modal pairs as positives; temperature τ=0.07; returns scalar loss |
| `ntxent.py` | `NTXentLoss`: standard NT-Xent for Stage 1 per-modality training; two augmented views per image; temperature τ=0.07 |

---

### `ml/training/` — Training Loop

| File | Purpose |
|---|---|
| `config.py` | `TrainingConfig` dataclass: batch_size=64, lr=1e-4, epochs_stage1=5, epochs_stage2=20, device, checkpoint_dir, data_dir |
| `trainer.py` | `BaseTrainer`: train loop, validation loop, checkpoint save/load, metric logging to console; used by both Stage1 and Stage2 |
| `stage1.py` | `Stage1Trainer(BaseTrainer)`: trains each encoder independently with NT-Xent; runs three sequential jobs (optical, SAR, MS); saves per-modality checkpoints |
| `stage2.py` | `Stage2Trainer(BaseTrainer)`: loads Stage 1 checkpoints; trains joint model with `MultiPositiveInfoNCE`; saves best model by validation loss |

---

### `ml/evaluation/`

| File | Purpose |
|---|---|
| `evaluate_retrieval.py` | Loads test embeddings + FAISS indexes; computes Recall@5, Recall@10, F1@5, F1@10 for all 9 modality-pair combinations (3 same-modal + 6 cross-modal); saves `results/metrics.json` |
| `visualize_embeddings.py` | Loads embeddings for 2000 test scenes; runs UMAP to 2D; color-codes by modality (●=optical, ■=SAR, ▲=MS) and by LCLU class; saves `results/embedding_space.png` |

---

### `ml/scripts/` — One-Shot Pipeline Scripts

| File | Purpose |
|---|---|
| `compute_embeddings.py` | Loads trained `CrossModalModel`; runs inference over full dataset in batches of 64; saves `embeddings/optical.npy`, `sar.npy`, `ms.npy`, `metadata.json`; logs progress |
| `build_faiss_index.py` | Loads all `.npy` files; builds 4 FAISS IVFFlat indexes (optical, SAR, MS, shared); trains each index; saves to `indexes/`; prints index stats |
| `generate_thumbnails.py` | For each scene, renders 128×128 JPEG composite; saves to `backend/static/thumbnails/{scene_id}.jpg`; used by `/scenes/{id}/thumbnail` endpoint |

---

### `frontend/` — Next.js UI

| File | Purpose |
|---|---|
| `app/layout.tsx` | Root layout; sets font, background; wraps in `QueryClientProvider` (TanStack Query) |
| `app/page.tsx` | Redirects to `/explore` |
| `app/explore/page.tsx` | Main demo page; two-column layout: QueryPanel (left) + ResultsGrid (right) |
| `components/upload/ImageDropzone.tsx` | Drag-and-drop or click-to-upload; shows preview; validates file type and size; calls `onImageSelected(file)` |
| `components/upload/ModalitySelector.tsx` | Tab/button group: Optical / SAR / Multispectral; controls source modality of query |
| `components/upload/TopKSelector.tsx` | Dropdown: Top-5 / Top-10; defaults to Top-10 |
| `components/results/ResultsGrid.tsx` | CSS grid of `ResultCard` components; handles loading and empty states |
| `components/results/ResultCard.tsx` | Shows thumbnail, rank badge, similarity score bar, modality tag, date, LCLU class; links to scene detail |
| `components/results/SimilarityBadge.tsx` | Color-coded badge: green >0.85, yellow 0.65–0.85, orange <0.65 |
| `components/map/ResultsMap.tsx` | MapLibre GL map; renders result scene footprint pins by lat/lon; clicking pin highlights result card |
| `components/shared/LoadingSkeleton.tsx` | Animated skeleton matching `ResultCard` shape; shown during API call |
| `components/shared/ErrorMessage.tsx` | Error state with message, icon, retry button |
| `lib/api/retrieve.ts` | Typed `fetch` wrapper; `retrieveImages(file, sourceModality, targetModality, topK) → RetrievalResponse`; handles FormData encoding |
| `lib/hooks/useRetrieval.ts` | TanStack Query mutation hook; `mutate({file, sourceModality, targetModality, topK})`; exposes `data`, `isLoading`, `error` |
| `types/retrieval.ts` | TypeScript interfaces: `RetrievalResult`, `SceneMatch`, `SemanticAxes`, `ConfidenceInfo` |

---

## 4. Dependencies

### Backend (`backend/requirements.txt`)

```
# Web Framework
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9       # file upload support
httpx==0.27.0                 # async HTTP client

# Database
sqlalchemy[asyncio]==2.0.31
asyncpg==0.29.0               # async PostgreSQL driver
alembic==1.13.2

# ML / Inference (backend uses the model at serve time)
torch==2.4.0
torchvision==0.19.0
numpy==1.26.4
Pillow==10.4.0
faiss-cpu==1.8.0              # use faiss-gpu if CUDA available
einops==0.8.0                 # tensor operations

# Config
pydantic-settings==2.4.0
python-dotenv==1.0.1

# Monitoring
loguru==0.7.2
```

### ML Training (`ml/requirements.txt`)

```
# Core
torch==2.4.0
torchvision==0.19.0
torchaudio==2.4.0             # not strictly needed but matches torch version

# DINOv2 (load via torch.hub, no pip install needed)
# LoRA implementation
loralib==0.1.2

# Data
numpy==1.26.4
h5py==3.11.0                  # HDF5 dataset format
rasterio==1.3.10               # GeoTIFF reading
Pillow==10.4.0
scipy==1.14.0                  # Lee speckle filter, PCA

# FAISS
faiss-cpu==1.8.0               # faiss-gpu==1.8.0 if CUDA

# Visualization / Evaluation
umap-learn==0.5.6
matplotlib==3.9.2
seaborn==0.13.2

# Training utilities
tqdm==4.66.5
loguru==0.7.2
```

### Frontend (`frontend/package.json` deps)

```json
{
  "dependencies": {
    "next": "15.0.0",
    "react": "19.0.0",
    "react-dom": "19.0.0",
    "typescript": "5.5.4",
    "@tanstack/react-query": "5.56.0",
    "tailwindcss": "3.4.10",
    "maplibre-gl": "4.7.1",
    "react-map-gl": "7.1.7",
    "react-dropzone": "14.2.3",
    "clsx": "2.1.1",
    "lucide-react": "0.441.0"
  },
  "devDependencies": {
    "@types/react": "19.0.0",
    "@types/node": "22.5.0",
    "eslint": "9.9.1",
    "eslint-config-next": "15.0.0"
  }
}
```

### System Requirements

```
Python    3.12+
Node.js   20+
pnpm      9+
Docker    25+ (for PostgreSQL + Redis)
CUDA      12.1+ (optional, training faster)
RAM       16GB minimum, 32GB recommended
Disk      120GB free (SEN12MS subset + processed data)
GPU       NVIDIA RTX 3080+ or A100 (or Google Colab A100 for training)
```

---

## 5. Dataset Preparation Steps

### Step 1 — Download SEN12MS

```
SOURCE: https://mediatum.ub.tum.de/1554803
        (TU Munich MediaTUM repository)

FILES TO DOWNLOAD (Spring season subset, ~60GB):
  ROIs1158_spring_s1.tar.gz    ← Sentinel-1 SAR (~15GB)
  ROIs1158_spring_s2.tar.gz    ← Sentinel-2 Multispectral (~40GB)
  ROIs0000_test_s1.tar.gz      ← Test SAR
  ROIs0000_test_s2.tar.gz      ← Test Multispectral

DOWNLOAD SCRIPT (scripts/download_sen12ms.sh):
  Uses wget with --continue flag (resume on interruption)
  Saves to data/sen12ms_raw/
  Verifies MD5 checksums after download
  Extracts archives automatically

ALTERNATIVE if bandwidth is limited:
  Use the Google Earth Engine export of SEN12MS (10K scenes)
  Or use the openly available torrent (check dataset page)
  Or: Download only 2 ROIs (~6GB) for demo purposes

DIRECTORY STRUCTURE AFTER EXTRACTION:
  data/sen12ms_raw/
  └── ROIs1158_spring/
      ├── s1_spring/
      │   └── ROIs1158_spring_s1_p0001.tif  ← [2 × 256 × 256] SAR GeoTIFF
      │   └── ...
      └── s2_spring/
          └── ROIs1158_spring_s2_p0001.tif  ← [13 × 256 × 256] MS GeoTIFF
```

### Step 2 — Verify Download

```
VERIFICATION:
  Check file count: ROIs1158_spring/s1_spring/ → expect ~180,000 files
  Check file size:  each .tif is ~1.3MB (SAR) or ~8MB (MS)
  Test load: rasterio.open(first_file).read() should return array
  Check co-registration: s1_p0001 and s2_p0001 are same scene

SCENE PAIRING:
  Each scene ID (p0001, p0002, ...) has exactly 2 files:
    s1_spring/ROIs1158_spring_s1_pNNNN.tif  ← SAR
    s2_spring/ROIs1158_spring_s2_pNNNN.tif  ← MS (13 bands, RGB subset = bands 3,2,1)
  Optical RGB = bands 3 (Red), 2 (Green), 1 (Blue) from s2 file
  Multispectral = all 13 bands from s2 file
```

### Step 3 — Extract 10K Subset

```
SCRIPT: scripts/prepare_subset.py

PURPOSE:
  Full SEN12MS = 180,662 triplets → too large for 5-day MVP
  We select 10,000 balanced triplets for training + evaluation

SELECTION STRATEGY:
  1. Parse all scene IDs from s1_spring/ directory
  2. Load MODIS land cover labels (from SEN12MS metadata CSV)
  3. Stratified sample: ~300 scenes per LCLU class
     (33 classes × 300 = 9,900, rounded to 10,000)
  4. Split:
     Train : 8,000 scenes (indices 0–7999)
     Val   : 1,000 scenes (indices 8000–8999)
     Test  : 1,000 scenes (indices 9000–9999)
  5. Save split manifest: data/splits.json
     { "train": [...scene_ids], "val": [...], "test": [...] }

OUTPUT:
  data/sen12ms_processed/
  ├── train.h5    ← 8,000 triplets in HDF5
  │   Keys per scene:
  │     /{scene_id}/optical    [3  × 256 × 256] float32
  │     /{scene_id}/sar        [2  × 256 × 256] float32
  │     /{scene_id}/ms         [13 × 256 × 256] float32
  │     /{scene_id}/metadata   JSON string (lat, lon, season, class)
  ├── val.h5      ← 1,000 triplets
  └── test.h5     ← 1,000 triplets

PROCESSING PER SCENE:
  SAR: convert from linear scale to dB: 10 × log10(x + 1e-10)
       clip to [-30, 0] dB range
       no spatial resize (keep 256×256)
  MS:  read all 13 bands
       clip 2nd–98th percentile per band (remove outliers)
       no spatial resize
  Optical: extract bands 3,2,1 from MS (= RGB)
           same clip as MS

TIME: ~2 hours for 10K scenes on a standard laptop
```

### Step 4 — Compute PCA Projection

```
SCRIPT: ml/scripts/compute_pca.py (called from prepare_subset.py)

PURPOSE:
  Learn 13→3 linear projection for multispectral bands
  Applied at training and inference time

METHOD:
  Load 8,000 MS training patches
  Reshape each: [13 × 256 × 256] → [13 × 65536] → [65536 × 13]
  Stack all: [8000 × 65536 × 13] → flatten to [8000×65536 × 13]
  Fit sklearn.decomposition.PCA(n_components=3)
  Save: ml/checkpoints/pca.npy  → stores components_ and mean_

VERIFY: PCA explains > 85% variance across 3 components
```

### Step 5 — Generate Thumbnails

```
SCRIPT: ml/scripts/generate_thumbnails.py

PURPOSE:
  Pre-render 128×128 JPEG thumbnails for all 10K scenes
  Served directly by FastAPI without compute at request time

PER SCENE:
  Load optical bands (3×256×256)
  Normalize to [0,255] uint8
  Resize to 128×128 (PIL LANCZOS)
  Save as JPEG (quality=85) to:
    backend/static/thumbnails/{scene_id}.jpg

TIME: ~5 minutes for 10K scenes on CPU

FALLBACK (if no optical):
  For SAR: pseudo-RGB [VV | VH | VV-VH], same normalization
  For MS-only: use first 3 bands (Blue, Green, Red → reorder to RGB)
```

---

## 6. ML Training Workflow

```
COMPLETE TRAINING EXECUTION ORDER

─────────────────────────────────────────────
STEP 0: Pre-flight checks
─────────────────────────────────────────────

  Check: CUDA available (or MPS for Apple Silicon)
  Check: Training HDF5 files exist
  Check: PCA checkpoint exists
  Check: ml/checkpoints/ directory exists and is writable

  If no GPU:
    Switch to zero-shot mode:
    Skip Stage 1 and Stage 2 entirely.
    Use DINOv2 [CLS] features directly (no projection head).
    Write zero-shot embedding extraction script.
    This gives Recall@10 ~0.35 but MVP works.

─────────────────────────────────────────────
STEP 1: Stage 1 — Optical Encoder (5 epochs)
─────────────────────────────────────────────

  Command: python ml/training/stage1.py --modality optical
  Config:
    model: dinov2_vitl14 (or vitb14 for faster training)
    lora_rank: 8 (small for MVP speed)
    batch_size: 32
    lr: 1e-4
    epochs: 5
    augmentation: RandomCrop, HFlip, VFlip, ColorJitter
    loss: NTXentLoss (τ=0.07)
    optimizer: AdamW (weight_decay=0.05)
    scheduler: CosineAnnealingLR

  Output: ml/checkpoints/stage1_optical.pt
  Expected time: 25 min on A100 / 90 min on RTX 3080

  Monitor:
    Train loss should drop from ~7.0 → ~4.5 over 5 epochs
    If loss stays > 6.5 after epoch 2: check data loading, try smaller τ

─────────────────────────────────────────────
STEP 2: Stage 1 — SAR Encoder (5 epochs)
─────────────────────────────────────────────

  Command: python ml/training/stage1.py --modality sar
  Config: same as optical, but:
    lora_rank: 16 (higher rank for larger domain gap)
    augmentation: RandomCrop, HFlip, VFlip, SpeckleNoise, ChannelSwap

  Output: ml/checkpoints/stage1_sar.pt
  Expected time: 30 min on A100 / 100 min on RTX 3080

─────────────────────────────────────────────
STEP 3: Stage 1 — Multispectral Encoder (5 epochs)
─────────────────────────────────────────────

  Command: python ml/training/stage1.py --modality ms
  Config: same as optical, but:
    use PCA projection (13→3) before encoder
    augmentation: RandomCrop, HFlip, VFlip, BandDropout(p=0.3)

  Output: ml/checkpoints/stage1_ms.pt
  Expected time: 28 min on A100 / 95 min on RTX 3080

─────────────────────────────────────────────
STEP 4: Stage 2 — Joint Cross-Modal Training (20 epochs)
─────────────────────────────────────────────

  Command: python ml/training/stage2.py
  Config:
    load: stage1_optical.pt, stage1_sar.pt, stage1_ms.pt
    freeze: encoder backbone (only LoRA + projection head)
    batch_size: 32 (triplets → 96 images per batch)
    lr: 5e-5
    epochs: 20
    loss: MultiPositiveInfoNCE (τ=0.07, learnable)
    optimizer: AdamW
    scheduler: CosineAnnealingLR with 2-epoch warmup

  Output: ml/checkpoints/best_model.pt (saved on best val loss)
  Expected time: 60 min on A100 / 200 min on RTX 3080

  Monitor every epoch:
    Val loss: should decrease from ~9.0 → ~6.0
    Quick Recall@10 on 200-scene val subset: aim for > 0.45

  Early stop condition:
    If val loss does not decrease for 5 epochs → stop, use best checkpoint

─────────────────────────────────────────────
STEP 5: Embedding Computation (all 10K scenes)
─────────────────────────────────────────────

  Command: python ml/scripts/compute_embeddings.py \
           --checkpoint ml/checkpoints/best_model.pt \
           --data data/sen12ms_processed/ \
           --output embeddings/

  Process:
    Load model from checkpoint
    Set model.eval(), torch.no_grad()
    Iterate dataset in batches of 64
    For each batch:
      z_optical = model.encode_single(optical_batch, 'optical')  [64×128]
      z_sar     = model.encode_single(sar_batch,     'sar')      [64×128]
      z_ms      = model.encode_single(ms_batch,      'ms')       [64×128]
    Accumulate into lists
    Save:
      np.save('embeddings/optical.npy', optical_embeddings)  # [10000 × 128]
      np.save('embeddings/sar.npy', sar_embeddings)          # [10000 × 128]
      np.save('embeddings/ms.npy', ms_embeddings)            # [10000 × 128]
      json.dump(metadata_list, 'embeddings/metadata.json')
        metadata per scene: {scene_id, lat, lon, lclu_class, date, season}

  Expected time: 8 min on A100 / 20 min on CPU (with batch=64)

─────────────────────────────────────────────
COMPLETE TRAINING TIMELINE
─────────────────────────────────────────────

  GPU available (A100):
    Stage 1 × 3: 25+30+28 = 83 min
    Stage 2:                 60 min
    Embedding compute:        8 min
    Total:                ~2.5 hours

  GPU available (RTX 3080):
    Stage 1 × 3: 90+100+95 = 285 min
    Stage 2:                200 min
    Embedding compute:       20 min
    Total:                ~8.4 hours  ← start on Day 2 morning

  No GPU (CPU only):
    Skip training → zero-shot DINOv2 features
    Embedding compute (batch=8):  ~90 min on 8-core CPU
    Total:                ~1.5 hours
```

---

## 7. FAISS Indexing Workflow

```
FAISS INDEX BUILD WORKFLOW
Script: ml/scripts/build_faiss_index.py

─────────────────────────────────────────────
INPUT
─────────────────────────────────────────────

  embeddings/optical.npy   → shape: [N × 128]  float32
  embeddings/sar.npy       → shape: [N × 128]  float32
  embeddings/ms.npy        → shape: [N × 128]  float32
  embeddings/metadata.json → list of N metadata dicts

  N = 10,000 for MVP

─────────────────────────────────────────────
STEP 1: L2 Normalize All Embeddings
─────────────────────────────────────────────

  All embeddings must be L2-normalized before adding to FAISS.
  With L2-normalized vectors:
    cosine_similarity(a, b) = inner_product(a, b)
  This lets us use IndexFlatIP (inner product) which equals cosine.

  optical_norm = optical / ||optical||₂   element-wise row normalization
  sar_norm     = sar     / ||sar||₂
  ms_norm      = ms      / ||ms||₂

─────────────────────────────────────────────
STEP 2: Build Per-Modality Indexes
─────────────────────────────────────────────

  For each modality in [optical, sar, ms]:

    Index type: IndexIVFFlat (exact distances within cluster)
    Metric:     METRIC_INNER_PRODUCT (= cosine for L2-normed vectors)
    Dimension:  128
    nlist:      64  (√10000 ≈ 100, but 64 works well for 10K)

    Build process:
      quantizer = faiss.IndexFlatIP(128)
      index = faiss.IndexIVFFlat(quantizer, 128, 64, faiss.METRIC_INNER_PRODUCT)
      index.train(embeddings_norm)     ← clusters 10K vecs into 64 centroids
      index.add_with_ids(embeddings_norm, ids)
        ids = np.arange(N)             ← integer IDs 0..9999
      index.nprobe = 8                 ← search 8/64 clusters per query
      faiss.write_index(index, 'indexes/optical.index')

    Stats per index:
      Training time:  < 2 seconds
      Search time:    < 5ms for top-10 on 10K vecs
      File size:      ~5MB

─────────────────────────────────────────────
STEP 3: Build Shared Cross-Modal Index
─────────────────────────────────────────────

  Combines all 3 modalities into one index.
  Critical: tag each vector with modality so we can filter post-search.

  Concatenate all embeddings:
    all_embeddings = stack([optical_norm, sar_norm, ms_norm])  [30000 × 128]
    all_ids        = [0..9999, 10000..19999, 20000..29999]
      ID encoding: id = scene_index + modality_offset
        optical offset = 0
        sar offset     = 10,000
        ms  offset     = 20,000
    Store mapping: {id: {scene_id, modality, array_index}}

  Index type: IndexIVFFlat
  nlist: 128  (larger for 3× data)
  nprobe: 16

  faiss.write_index(index, 'indexes/shared.index')

  File size: ~15MB for 30K vectors at 128-d

─────────────────────────────────────────────
STEP 4: Build ID Lookup Table
─────────────────────────────────────────────

  Save JSON: indexes/id_map.json
  {
    "0":     { "scene_id": "ROIs1158_spring_p0001", "modality": "optical", "idx": 0 },
    "1":     { "scene_id": "ROIs1158_spring_p0002", "modality": "optical", "idx": 1 },
    ...
    "10000": { "scene_id": "ROIs1158_spring_p0001", "modality": "sar", "idx": 0 },
    ...
    "20000": { "scene_id": "ROIs1158_spring_p0001", "modality": "ms",  "idx": 0 }
  }

  This is loaded at backend startup for O(1) metadata lookup by FAISS integer ID.

─────────────────────────────────────────────
STEP 5: Verify Indexes
─────────────────────────────────────────────

  Sanity check (printed by build script):
    Load each index
    Query with first embedding of each modality
    Expected: top-1 result = itself (similarity ≈ 1.0)
    Expected: top-10 includes at least 3 same-LCLU-class scenes

  Print index stats:
    indexes/optical.index: 10000 vectors, d=128, nlist=64
    indexes/sar.index:     10000 vectors, d=128, nlist=64
    indexes/ms.index:      10000 vectors, d=128, nlist=64
    indexes/shared.index:  30000 vectors, d=128, nlist=128

─────────────────────────────────────────────
SEARCH ROUTING LOGIC (used in faiss_service.py)
─────────────────────────────────────────────

  Given: source_modality, target_modality, query_embedding [128]

  Case 1: Same-modal (source == target)
    Search: indexes/{source_modality}.index
    Returns: top-10 IDs from that modality

  Case 2: Cross-modal (source != target)
    Search: indexes/shared.index
    Returns: top-50 candidates from all modalities
    Filter: keep only results where id_map[id].modality == target_modality
    Return: top-10 after filter

  Case 3: All-modal (target = "all")
    Search: indexes/shared.index
    Returns: top-10 regardless of modality
    Tag each result with its modality from id_map

  nprobe override for cross-modal:
    Use nprobe=16 for shared index (higher recall needed)
    Use nprobe=8  for per-modality indexes

─────────────────────────────────────────────
FAISS SERVICE STARTUP (backend/services/faiss_service.py)
─────────────────────────────────────────────

  On application startup (FastAPI lifespan):
    Load all 4 indexes into memory at once
    Load id_map.json into dict
    Load metadata.json into dict keyed by scene_id
    Total memory: ~80MB (4 indexes + mappings)

  Search function signature:
    search(
      query_embedding: np.ndarray,   # [128] float32, L2-normalized
      source_modality: str,
      target_modality: str,          # "optical" | "sar" | "ms" | "all"
      k: int = 10
    ) → list[SearchResult]

  SearchResult:
    scene_id: str
    modality: str
    similarity: float
    metadata: SceneMetadata
```

---

## 8. API Workflow

```
COMPLETE API REQUEST LIFECYCLE

─────────────────────────────────────────────
REQUEST: POST /api/v1/retrieve/cross-modal
─────────────────────────────────────────────

CLIENT SENDS:
  Content-Type: multipart/form-data
  Fields:
    image            : <binary JPEG/PNG/GeoTIFF>
    source_modality  : "optical"
    target_modality  : "sar"
    top_k            : 10

─────────────────────────────────────────────
STEP 1: Request Validation (FastAPI)
─────────────────────────────────────────────

  Validate:
    image size ≤ 50MB
    source_modality ∈ {"optical", "sar", "multispectral"}
    target_modality ∈ {"optical", "sar", "multispectral", "all"}
    top_k ∈ {5, 10}
    file content-type is image/*

  If invalid → return 422 with detail

─────────────────────────────────────────────
STEP 2: Image Preprocessing (encoder_service.py)
─────────────────────────────────────────────

  Read image bytes → PIL Image or rasterio array
  Apply modality-specific transform:
    Optical:  resize 224×224, normalize per ImageNet stats
    SAR:      speckle filter, clip dB range, pseudo-RGB, normalize
    MS:       apply PCA projection (13→3), normalize

  Result: torch.Tensor [1 × 3 × 224 × 224]

─────────────────────────────────────────────
STEP 3: Encode Query (encoder_service.py)
─────────────────────────────────────────────

  model.eval()
  with torch.no_grad():
    z = model.encode_single(image_tensor, source_modality)  [1 × 128]

  z_norm = z / ||z||₂    ← L2 normalize
  z_np = z_norm.cpu().numpy().squeeze()   → [128] float32

  Timing target: < 50ms on GPU, < 200ms on CPU

─────────────────────────────────────────────
STEP 4: FAISS Search (faiss_service.py)
─────────────────────────────────────────────

  results = faiss_service.search(
    query_embedding = z_np,
    source_modality = "optical",
    target_modality = "sar",
    k = 10
  )

  Internal:
    Search shared.index (cross-modal case)
    Get top-50 candidates → [50 IDs, 50 similarities]
    Filter: keep IDs where id_map[id].modality == "sar"
    Take top-10 by similarity after filter

  Return: list[{faiss_id, similarity_score, modality}]
  Timing target: < 10ms

─────────────────────────────────────────────
STEP 5: Metadata Enrichment (result_service.py)
─────────────────────────────────────────────

  For each FAISS result:
    scene_id = id_map[faiss_id]["scene_id"]
    metadata = metadata_dict[scene_id]
      { lat, lon, date, season, lclu_class }
    thumbnail_url = f"/static/thumbnails/{scene_id}.jpg"

  Compose SceneMatch:
    {
      rank:          1..10
      scene_id:      "ROIs1158_spring_p0041"
      modality:      "sar"
      similarity:    0.847
      thumbnail_url: "/static/thumbnails/ROIs1158_spring_p0041.jpg"
      metadata: {
        lat:         52.45
        lon:         13.21
        date:        "2016-04-12"
        season:      "spring"
        lclu_class:  "urban_fabric"
      }
    }

─────────────────────────────────────────────
STEP 6: Build Response
─────────────────────────────────────────────

  HTTP 200 Response:
  {
    "query_id"       : "3f7a2d91-c4b1-4e89-b123-abc123def456",
    "source_modality": "optical",
    "target_modality": "sar",
    "top_k"          : 10,
    "latency_ms"     : 87,
    "results"        : [
      {
        "rank"         : 1,
        "scene_id"     : "ROIs1158_spring_p0041",
        "modality"     : "sar",
        "similarity"   : 0.847,
        "thumbnail_url": "/static/thumbnails/ROIs1158_spring_p0041.jpg",
        "metadata": {
          "lat"        : 52.45,
          "lon"        : 13.21,
          "date"       : "2016-04-12",
          "season"     : "spring",
          "lclu_class" : "urban_fabric"
        }
      },
      // ... 9 more results
    ]
  }

  Total end-to-end latency: < 300ms P99

─────────────────────────────────────────────
ALL ENDPOINTS SUMMARY
─────────────────────────────────────────────

  GET  /health
    Returns: { status, model_version, index_sizes, db_status, uptime_s }
    Use: frontend startup check, demo health indicator

  POST /api/v1/retrieve/optical
    source_modality = "optical", target_modality = "optical"
    Use case: find visually similar optical scenes

  POST /api/v1/retrieve/sar
    source_modality = "sar", target_modality = "sar"
    Use case: find similar SAR backscatter patterns

  POST /api/v1/retrieve/multispectral
    source_modality = "multispectral", target_modality = "ms"
    Use case: spectral similarity search

  POST /api/v1/retrieve/cross-modal
    source_modality: any, target_modality: any (different)
    Use case: THE CORE DEMO — optical→SAR, SAR→MS, etc.
    This is what makes GeoSense-X unique

  GET  /api/v1/scenes/{scene_id}/thumbnail
    Returns: JPEG 128×128 thumbnail
    Use: frontend ResultCard image src

  GET  /docs
    FastAPI auto-generated Swagger UI (enable in dev mode)
```

---

## 9. Development Milestones

```
MILESTONE TRACKER

M1: Data Pipeline Ready
  Day: 1 — Evening
  Criteria:
    ✓ SEN12MS subset downloaded and verified
    ✓ HDF5 files created (train/val/test split)
    ✓ PCA projection computed and saved
    ✓ One complete triplet loads in < 100ms
    ✓ Batch of 32 triplets fits in GPU memory
  Blockers if missed: Training cannot start → cascade delay

M2: Model Training Complete
  Day: 2 — Evening
  Criteria:
    ✓ Stage 1 checkpoints saved for all 3 modalities
    ✓ Stage 2 best_model.pt saved
    ✓ Val loss for Stage 2 < 7.0
    ✓ Quick Recall@10 on 100-scene probe > 0.40
  Fallback: Zero-shot DINOv2 features (no training)
  Blockers if missed: No model for embedding computation

M3: FAISS Indexes Ready
  Day: 3 — Noon
  Criteria:
    ✓ embeddings/*.npy files exist for all 3 modalities
    ✓ All 4 FAISS indexes built and saved
    ✓ Sanity check passes (top-1 = self)
    ✓ Cross-modal search returns correct modality results
    ✓ Search latency < 10ms on 10K vectors
  Blockers if missed: API cannot search

M4: Backend API Working
  Day: 3 — Evening
  Criteria:
    ✓ FastAPI server starts without errors
    ✓ All 4 /retrieve/* endpoints return 200
    ✓ curl test: upload optical image → get 10 SAR results ✓
    ✓ Response includes thumbnail URLs
    ✓ End-to-end latency < 300ms
  Blockers if missed: Frontend has nothing to connect to

M5: Frontend Integration
  Day: 4 — Evening
  Criteria:
    ✓ Upload image in browser → results grid appears ✓
    ✓ Modality selector switches source/target ✓
    ✓ All 4 retrieval modes work from UI ✓
    ✓ Map shows result pins ✓
    ✓ Error states handled gracefully ✓
  Blockers if missed: No demo UI

M6: Evaluation Complete
  Day: 5 — 11:00
  Criteria:
    ✓ F1@5 and F1@10 computed for all modality pairs
    ✓ UMAP embedding visualization saved
    ✓ Results documented in presentation
  Blockers if missed: Weak submission (no metrics)

M7: Demo Ready
  Day: 5 — 14:00
  Criteria:
    ✓ All 5 demo scenarios work smoothly
    ✓ Pre-loaded demo images in UI
    ✓ Architecture diagram ready
    ✓ SETUP.md tested on clean environment
  Blockers if missed: Cannot demo live

M8: Submitted
  Day: 5 — 18:00 HARD DEADLINE
  Criteria:
    ✓ GitHub repo pushed
    ✓ Submission form completed
    ✓ Demo video uploaded (if required)
```

---

## 10. Risk Register

```
RISK #1: SEN12MS Download Too Slow
Probability: HIGH (dataset is hosted in Germany)
Impact: HIGH (blocks Day 2 training)
Mitigation:
  → Start download immediately (Day 1, 11:00)
  → Use aria2c with 16 parallel connections for speed
  → Have backup: smaller 2-ROI subset (~6GB) that downloads in 2 hours
  → Second backup: EuroSAT dataset (much smaller, publicly available)
     EuroSAT has optical only → demonstrates same-modal retrieval
     Add synthetic cross-modal via data augmentation if needed

RISK #2: No GPU Available for Training
Probability: MEDIUM
Impact: HIGH (training takes 8+ hours on CPU)
Mitigation:
  → Use Google Colab Pro (A100, ~$10/day, available immediately)
  → Use Kaggle free T4 GPU (20 hrs/week free)
  → FALLBACK: Zero-shot DINOv2 features
     DINOv2 ViT-L/14 [CLS] tokens already encode rich visual semantics
     Cross-modal retrieval still works — just less accurate
     Build index from zero-shot features in <2 hours on CPU
     Recall@10 ~0.30-0.35 (lower but demo-able)

RISK #3: FAISS Library Installation Issues
Probability: LOW-MEDIUM (especially on Windows)
Impact: MEDIUM
Mitigation:
  → Install faiss-cpu via conda: conda install -c conda-forge faiss-cpu
  → Or use Docker: docker run with faiss pre-installed
  → Fallback: use sklearn NearestNeighbors (exact search, slower but no install issues)
     np.dot(query, embeddings.T) → argsort → top-10
     Latency ~50ms for 10K vecs — acceptable for MVP

RISK #4: Cross-Modal Alignment Fails (Poor Recall@10)
Probability: MEDIUM
Impact: MEDIUM
Mitigation:
  → Increase Stage 2 epochs if time permits
  → Lower temperature τ: try 0.05 instead of 0.07
  → Check: are embeddings L2-normalized before FAISS?
  → Fallback: frame as "semantic similarity" rather than strict recall
     Demo with hand-picked query images that have good recall

RISK #5: Frontend-Backend CORS Issues
Probability: LOW
Impact: LOW (easy to fix)
Mitigation:
  → Configure CORSMiddleware in main.py on Day 3
  → Allow origins: ["http://localhost:3000"]
  → If HTTPS needed: use Nginx reverse proxy via docker-compose

RISK #6: SEN12MS SAR Bands Not Aligning with Optical
Probability: LOW
Impact: MEDIUM (cross-modal retrieval degrades)
Mitigation:
  → Verify co-registration on Day 1 (plot overlay of s1+s2 for same patch ID)
  → SEN12MS is published as co-registered — but verify with 5 spot checks
  → Acceptable error: < 1 pixel shift between modalities

CONTINGENCY EXECUTION ORDER
  If Day 3 12:00 and model is not trained:
    → Switch immediately to zero-shot pipeline
    → python ml/scripts/compute_embeddings_zeroshot.py
    → Continue with FAISS build and API as planned
    → You lose ~0.15 Recall@10 but the demo works

  If Day 4 17:00 and frontend not connecting:
    → Switch to Swagger UI as demo interface
    → Upload image in /docs Swagger page
    → Shows JSON results (less pretty but functional)

  If Day 5 14:00 and server crashes in demo:
    → Switch to pre-recorded screen capture video
    → Keep video ready from Day 4 evening
```

---

## Appendix A: `.env.example`

```
# Backend
DATABASE_URL=postgresql+asyncpg://geosense:password@localhost:5432/geosense
REDIS_URL=redis://localhost:6379/0

# Paths (relative to project root)
FAISS_INDEX_DIR=./indexes
EMBEDDINGS_DIR=./embeddings
MODEL_CHECKPOINT=./ml/checkpoints/best_model.pt
PCA_CHECKPOINT=./ml/checkpoints/pca.npy
THUMBNAILS_DIR=./backend/static/thumbnails

# Model
EMBEDDING_DIM=128
DINOV2_MODEL=dinov2_vitl14        # or dinov2_vitb14 for faster inference

# Server
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Appendix B: `Makefile` Commands

```
make setup       → run scripts/setup_env.sh (install all deps, start Docker)
make download    → run scripts/download_sen12ms.sh
make prepare     → run scripts/prepare_subset.py (create HDF5 shards)
make train       → run ml/training/stage1.py × 3 then stage2.py
make embed       → run ml/scripts/compute_embeddings.py
make index       → run ml/scripts/build_faiss_index.py
make thumbnails  → run ml/scripts/generate_thumbnails.py
make migrate     → run alembic upgrade head
make backend     → uvicorn backend.main:app --reload --port 8000
make frontend    → cd frontend && pnpm dev
make run         → make backend (bg) && make frontend (fg)
make eval        → python ml/evaluation/evaluate_retrieval.py
make demo        → open http://localhost:3000/explore in browser
```

## Appendix C: One-Command Setup (SETUP.md summary)

```
REQUIREMENTS: Python 3.12, Node 20, pnpm 9, Docker

1. Clone repo
2. cp .env.example .env
3. make setup          ← installs everything, starts DB
4. make download       ← starts SEN12MS download (runs overnight)
5. make prepare        ← after download completes
6. make train          ← starts training (~4 hours on A100)
7. make embed          ← after training completes
8. make index          ← builds FAISS indexes (~2 min)
9. make thumbnails     ← generates scene thumbnails (~5 min)
10. make migrate       ← sets up database schema
11. make run           ← starts backend + frontend
12. Open http://localhost:3000
```

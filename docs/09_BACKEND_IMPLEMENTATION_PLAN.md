# GeoSense-X — Backend Implementation Plan
# Phase 1 MVP — Complete Engineering Specification
## Principal AI Engineer Document v1.0

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Complete Folder Structure](#2-complete-folder-structure)
3. [File Registry — Every File, Every Responsibility](#3-file-registry--every-file-every-responsibility)
4. [Database Models](#4-database-models)
5. [API Routes Specification](#5-api-routes-specification)
6. [Services Layer](#6-services-layer)
7. [Retrieval Pipeline](#7-retrieval-pipeline)
8. [Explanation Pipeline](#8-explanation-pipeline)
9. [Confidence Pipeline](#9-confidence-pipeline)
10. [Caching Strategy](#10-caching-strategy)
11. [Development Order](#11-development-order)

---

## 1. Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║               GeoSense-X Backend — Layered Architecture                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

 ┌─────────────────────────────────────────────────────────────────────────┐
 │                        TRANSPORT LAYER                                  │
 │  FastAPI (HTTP REST)              FastAPI (WebSocket)                   │
 │  POST /retrieve/*                 WS /search/interactive                │
 │  GET  /explain/*                  Streaming progressive delivery        │
 │  GET  /scenes/*                                                         │
 │  GET  /health                                                           │
 └──────────────────────────────┬──────────────────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────────────────┐
 │                        API LAYER  (api/v1/)                             │
 │  Route handlers — validate input, call services, serialize response     │
 │  Pydantic schema validation on every request and response               │
 │  File upload handling (UploadFile), multipart parsing                   │
 │  Dependency injection: DB session, Redis client, rate limiter           │
 └──────────────────────────────┬──────────────────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────────────────┐
 │                       DOMAIN LAYER  (domain/)                           │
 │                                                                         │
 │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
 │  │  Retrieval       │  │  Explanation     │  │  Confidence          │  │
 │  │  Pipeline        │  │  Pipeline        │  │  Pipeline            │  │
 │  │                  │  │                  │  │                      │  │
 │  │ • encode query   │  │ • attn heatmap   │  │ • mc dropout         │  │
 │  │ • FAISS search   │  │ • axis scores    │  │ • calibration        │  │
 │  │ • rank results   │  │ • spectral index │  │ • gap score          │  │
 │  │ • route modality │  │ • road features  │  │ • composite score    │  │
 │  │                  │  │ • narrative gen  │  │ • tier assignment    │  │
 │  └──────────────────┘  └──────────────────┘  └──────────────────────┘  │
 └──────────────────────────────┬──────────────────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────────────────┐
 │                       SERVICES LAYER  (services/)                       │
 │                                                                         │
 │  EncoderService     FAISSService      ExplanationService                │
 │  (model singleton)  (index singletons)(heatmap + axes + narrative)      │
 │                                                                         │
 │  ConfidenceService  CacheService      ThumbnailService                  │
 │  (uncertainty +     (Redis + Postgres (scene image serving)             │
 │   calibration)       two-level cache)                                   │
 └──────────────────────────────┬──────────────────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────────────────┐
 │                    INFRASTRUCTURE LAYER  (core/)                        │
 │                                                                         │
 │  PostgreSQL (SQLAlchemy async)     Redis (aioredis)                     │
 │  FAISS indexes (in-memory)         S3 / local FS (heatmap storage)      │
 │  MLflow model registry             Prometheus metrics                   │
 └─────────────────────────────────────────────────────────────────────────┘

Request Flow (Cross-Modal Retrieval with Explanation):
  1. Client uploads optical image → POST /api/v1/retrieve/cross-modal
  2. Router validates request, reads image bytes
  3. EncoderService.encode(bytes, "optical") → embedding [128]
  4. FAISSService.search(embedding, "optical", "sar", k=10) → candidates
  5. ExplanationService.explain(query_emb, candidates) → explanations
  6. ConfidenceService.score(query_emb, candidates) → confidence bundles
  7. ResultService.compose(candidates, explanations, confidence) → response
  8. CacheService.store(query_id, response)
  9. Return RetrievalResponse to client
```

---

## 2. Complete Folder Structure

```
backend/
│
├── main.py                              # FastAPI app factory + lifespan
├── config.py                            # Pydantic Settings, env vars
├── requirements.txt                     # Pinned production deps
├── requirements-dev.txt                 # Dev/test deps (pytest, httpx)
├── alembic.ini                          # Alembic migration config
├── pyproject.toml                       # Ruff, mypy, pytest config
│
├── api/
│   ├── __init__.py
│   ├── dependencies.py                  # Shared FastAPI dependencies
│   └── v1/
│       ├── __init__.py
│       ├── router.py                    # Aggregates all v1 sub-routers
│       ├── retrieve.py                  # /retrieve/* endpoints
│       ├── explain.py                   # /explain/* endpoints
│       ├── scenes.py                    # /scenes/* endpoints
│       ├── search.py                    # /search/* (WebSocket)
│       ├── admin.py                     # /admin/* (index mgmt, protected)
│       └── health.py                    # /health, /ready, /metrics
│
├── domain/
│   ├── __init__.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── pipeline.py                  # RetrievalPipeline orchestrator
│   │   ├── router.py                    # Modality-aware search routing
│   │   ├── preprocessor.py             # Per-modality image preprocessing
│   │   └── ranker.py                   # Post-FAISS result re-ranking
│   │
│   ├── explanation/
│   │   ├── __init__.py
│   │   ├── pipeline.py                  # ExplanationPipeline orchestrator
│   │   ├── attention/
│   │   │   ├── __init__.py
│   │   │   ├── extractor.py            # DINOv2 attention hook extraction
│   │   │   ├── rollout.py              # Attention rollout (all layers)
│   │   │   ├── heatmap.py              # Upsampling, normalization, PNG
│   │   │   └── alignment.py            # SSIM attention alignment score
│   │   ├── axes/
│   │   │   ├── __init__.py
│   │   │   ├── scorer.py               # Semantic axis score computation
│   │   │   └── comparator.py           # Pairwise axis similarity
│   │   ├── spectral/
│   │   │   ├── __init__.py
│   │   │   ├── ndvi.py                 # NDVI + vegetation profile
│   │   │   ├── ndwi.py                 # NDWI + water profile
│   │   │   ├── ndbi.py                 # NDBI + urban profile
│   │   │   └── sar_indices.py          # SAR water/urban detection
│   │   ├── road/
│   │   │   ├── __init__.py
│   │   │   ├── edge_detector.py        # Canny edge detection
│   │   │   ├── line_detector.py        # Hough line transform
│   │   │   └── similarity.py           # Road feature similarity
│   │   └── narrative/
│   │       ├── __init__.py
│   │       ├── categorizer.py          # CASE A-D assignment
│   │       └── generator.py            # Template-based NL generation
│   │
│   └── confidence/
│       ├── __init__.py
│       ├── pipeline.py                  # ConfidencePipeline orchestrator
│       ├── mc_dropout.py               # Monte Carlo Dropout uncertainty
│       ├── calibration.py              # Temperature scaling per class
│       ├── gap_scorer.py               # Top-1 vs Top-K margin
│       └── composite.py                # Weighted composite scorer
│
├── services/
│   ├── __init__.py
│   ├── encoder_service.py              # Model singleton + encode()
│   ├── faiss_service.py                # Index singletons + search()
│   ├── explanation_service.py          # Explanation orchestration
│   ├── confidence_service.py           # Confidence orchestration
│   ├── result_service.py               # Compose final response
│   ├── cache_service.py                # Redis + DB two-level cache
│   ├── thumbnail_service.py            # Scene image serving
│   └── storage_service.py             # S3 / local FS abstraction
│
├── models/
│   ├── __init__.py
│   ├── orm/
│   │   ├── __init__.py
│   │   ├── base.py                     # SQLAlchemy declarative base
│   │   ├── scene.py                    # SatelliteScene ORM
│   │   ├── embedding.py                # EmbeddingRecord ORM
│   │   ├── session.py                  # RetrievalSession ORM
│   │   ├── result.py                   # RetrievalResult ORM
│   │   ├── explanation.py              # SceneExplanation ORM
│   │   └── axis_vector.py              # SemanticAxisVector ORM
│   └── schemas/
│       ├── __init__.py
│       ├── retrieval.py                # Request/response Pydantic schemas
│       ├── explanation.py              # Explanation Pydantic schemas
│       ├── confidence.py               # Confidence Pydantic schemas
│       ├── scene.py                    # Scene Pydantic schemas
│       └── common.py                   # Shared types (pagination, geo)
│
├── repositories/
│   ├── __init__.py
│   ├── scene_repo.py                   # Scene DB queries
│   ├── embedding_repo.py               # Embedding persistence
│   ├── session_repo.py                 # Session history queries
│   ├── result_repo.py                  # Result persistence
│   └── explanation_repo.py             # Explanation cache queries
│
├── core/
│   ├── __init__.py
│   ├── database.py                     # Async SQLAlchemy engine
│   ├── redis_client.py                 # aioredis connection pool
│   ├── storage.py                      # S3 / local FS adapter
│   ├── security.py                     # API key validation
│   ├── rate_limiter.py                 # Sliding window rate limit
│   ├── exceptions.py                   # Custom exception hierarchy
│   ├── logging.py                      # Structured JSON logging
│   ├── metrics.py                      # Prometheus counters/histograms
│   └── events.py                       # App startup/shutdown handlers
│
├── workers/
│   ├── __init__.py
│   ├── celery_app.py                   # Celery app factory
│   ├── embedding_tasks.py              # Async embedding computation
│   ├── heatmap_tasks.py                # Async heatmap generation
│   └── ingestion_tasks.py              # Scene ingestion jobs
│
├── migrations/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       ├── 001_create_scenes.py
│       ├── 002_create_embeddings.py
│       ├── 003_create_sessions_results.py
│       ├── 004_create_explanations.py
│       └── 005_create_axis_vectors.py
│
├── static/
│   └── thumbnails/                     # Precomputed scene thumbnails
│
└── tests/
    ├── conftest.py                     # Fixtures, async test DB
    ├── unit/
    │   ├── domain/
    │   │   ├── test_retrieval_pipeline.py
    │   │   ├── test_explanation_pipeline.py
    │   │   └── test_confidence_pipeline.py
    │   └── services/
    │       ├── test_encoder_service.py
    │       └── test_faiss_service.py
    └── integration/
        ├── test_retrieve_endpoint.py
        ├── test_explain_endpoint.py
        └── test_health_endpoint.py
```

---

## 3. File Registry — Every File, Every Responsibility

### Root Level

```
FILE: main.py
RESPONSIBILITY:
  FastAPI application factory function create_app().
  Registers all routers under /api/v1.
  Configures lifespan context manager:
    startup:  load ML model, load FAISS indexes, connect DB, connect Redis
    shutdown: flush Redis pipeline, close DB sessions, release GPU memory
  Mounts static file directory for thumbnails.
  Configures CORSMiddleware with allowed origins from config.
  Registers custom exception handlers for all custom exception types.
  Adds request logging middleware (log every request: method, path, latency).
  Exposes Prometheus metrics endpoint at /metrics.

FILE: config.py
RESPONSIBILITY:
  Single Pydantic BaseSettings class that reads ALL environment variables.
  Validates types and constraints at startup (fail fast if misconfigured).
  Exposes typed attributes used across the codebase:
    DATABASE_URL, REDIS_URL, FAISS_INDEX_DIR, MODEL_CHECKPOINT,
    PCA_CHECKPOINT, EMBEDDING_DIM (128), DINOV2_MODEL, THUMBNAILS_DIR,
    S3_BUCKET, S3_PREFIX, CORS_ORIGINS, API_KEY_HASH, LOG_LEVEL,
    RATE_LIMIT_PER_MINUTE, MC_DROPOUT_PASSES (5 real-time / 20 async),
    CELERY_BROKER_URL, CELERY_RESULT_BACKEND.
  Singleton: instantiated once, imported everywhere as `settings`.
```

---

### `api/` — Transport Layer

```
FILE: api/dependencies.py
RESPONSIBILITY:
  FastAPI dependency functions injected into route handlers.
  get_db()           → yields AsyncSession; commits on success, rolls back on error
  get_redis()        → yields aioredis client from connection pool
  get_current_user() → validates API key from Authorization header; returns user dict
  get_rate_limiter() → returns RateLimiter instance configured for current user tier
  get_encoder()      → returns EncoderService singleton (cached in app state)
  get_faiss()        → returns FAISSService singleton (cached in app state)
  These are the ONLY way route handlers access shared resources.
  No direct database or service instantiation in route files.

FILE: api/v1/router.py
RESPONSIBILITY:
  Aggregates all sub-routers into one APIRouter mounted at /api/v1.
  Import and include_router for: retrieve, explain, scenes, search, admin, health.
  Sets global prefix /api/v1 and common response headers.
  No business logic — pure routing aggregation.

FILE: api/v1/retrieve.py
RESPONSIBILITY:
  Four POST endpoints for image retrieval:
    POST /retrieve/optical        → same-modal optical search
    POST /retrieve/sar            → same-modal SAR search
    POST /retrieve/multispectral  → same-modal multispectral search
    POST /retrieve/cross-modal    → cross-modal search (any→any)
  Each endpoint:
    1. Accepts UploadFile image + form parameters (source_modality,
       target_modality, top_k, explanation_depth, filters JSON)
    2. Validates file size (≤50MB), validates modality enum values
    3. Reads image bytes
    4. Calls RetrievalPipeline.run(bytes, source, target, top_k, filters)
    5. Returns RetrievalResponse Pydantic model
  No ML logic here. Pure HTTP boundary.

FILE: api/v1/explain.py
RESPONSIBILITY:
  Endpoints to fetch explanations for already-retrieved results:
    GET /explain/{query_id}/{scene_id}           → full ExplanationBundle
    GET /explain/{query_id}/{scene_id}/axes      → semantic axes only (fast)
    GET /explain/{query_id}/{scene_id}/confidence → confidence bundle only
    GET /explain/{query_id}/{scene_id}/heatmap/query   → 200 PNG response
    GET /explain/{query_id}/{scene_id}/heatmap/result  → 200 PNG response
    GET /explain/{query_id}/{scene_id}/heatmap/diff    → 200 PNG response
    POST /explain/batch                          → batch explanations (≤20 pairs)
  Heatmap endpoints return StreamingResponse with Content-Type: image/png.
  All non-heatmap endpoints return JSON.
  Check cache first; if miss, trigger async explanation computation.

FILE: api/v1/scenes.py
RESPONSIBILITY:
  Scene catalog endpoints:
    GET /scenes/{scene_id}                → scene metadata object
    GET /scenes/{scene_id}/thumbnail      → JPEG StreamingResponse (128×128)
    GET /scenes/{scene_id}/bands          → available band info
    GET /scenes/                          → paginated scene list with filters
      Query params: modality, bbox, date_from, date_to, lclu_class, page, limit
  Delegates to scene_repo for DB queries and thumbnail_service for image serving.

FILE: api/v1/search.py
RESPONSIBILITY:
  Interactive WebSocket search endpoint:
    WS /search/interactive
  Manages WebSocket connection lifecycle.
  Accepts JSON messages from client (QUERY, REFINE, EXPLORE_HOP).
  Streams phased responses back:
    Phase 1: EMBEDDING_READY (after encode, ~50ms)
    Phase 2: TOP_K_RESULTS (after FAISS search, ~150ms total)
    Phase 3: AXES_READY (after axis scoring, ~200ms total)
    Phase 4: HEATMAP_READY per result (async, ~500ms per result)
  Handles connection errors, client disconnect, timeout (30s).
  POST /search/refine → axis-weighted query refinement, returns new top-K.

FILE: api/v1/admin.py
RESPONSIBILITY:
  Protected admin endpoints (require admin API key):
    POST /admin/index/rebuild   → trigger full FAISS index rebuild (async)
    GET  /admin/index/status    → index build progress, last updated time
    POST /admin/model/reload    → hot-reload model checkpoint without restart
    GET  /admin/cache/stats     → Redis memory usage, hit/miss ratio
    POST /admin/cache/clear     → flush Redis explanation cache
    GET  /admin/jobs            → list running Celery tasks
  All operations are async (Celery tasks). Endpoints return job IDs.

FILE: api/v1/health.py
RESPONSIBILITY:
  GET /health  → liveness check: returns 200 if app is running
  GET /ready   → readiness check: returns 200 only if:
                   DB connected, Redis connected, FAISS indexes loaded,
                   ML model loaded. Returns 503 if any dependency unhealthy.
  GET /metrics → Prometheus exposition format (text/plain)
  Used by Kubernetes liveness and readiness probes.
```

---

### `domain/retrieval/` — Retrieval Pipeline

```
FILE: domain/retrieval/pipeline.py
RESPONSIBILITY:
  RetrievalPipeline — the main orchestrator for all retrieval requests.
  Single public method: run(image_bytes, source_modality, target_modality,
                            top_k, filters, explanation_depth) → RetrievalResponse
  Internal execution order:
    1. Preprocessor.preprocess(bytes, source_modality) → tensor
    2. EncoderService.encode(tensor, source_modality) → embedding [128]
    3. ModalityRouter.route(embedding, source, target, k, filters) → candidates
    4. Ranker.rerank(embedding, candidates) → ranked candidates
    5. ExplanationPipeline.explain(embedding, candidates, depth) → explanations
    6. ConfidencePipeline.score(embedding, candidates) → confidence bundles
    7. ResultService.compose(candidates, explanations, confidence) → response
    8. SessionRepo.save(session_record)
  This file has no ML code — only orchestration logic.
  Instruments latency for every step via metrics.

FILE: domain/retrieval/preprocessor.py
RESPONSIBILITY:
  Converts raw image bytes into a normalized tensor ready for DINOv2.
  Three modality-specific code paths, selected at runtime:
    Optical path:
      PIL.Image.open → RGB → resize 224×224 → ToTensor → Normalize(ImageNet)
    SAR path:
      rasterio.open → read 2 bands (VV, VH) → Lee speckle filter →
      dB conversion (10×log10) → clip [-30, 0] → stack [VV, VH, VV-VH] →
      normalize per SAR statistics → tensor [3×224×224]
    Multispectral path:
      rasterio.open → read 13 bands → apply PCA projection (13→3) →
      normalize per MS statistics → tensor [3×224×224]
  Also used by batch embedding computation (ml/scripts/).
  PCA matrix loaded from checkpoint at first call, cached as class attribute.
  Raises PreprocessingError with modality-specific message if input is invalid.

FILE: domain/retrieval/router.py
RESPONSIBILITY:
  ModalityRouter — routes the query to the correct FAISS index(es).
  Encapsulates the search routing logic described in §7.
  Methods:
    route_same_modal(embedding, modality, k, filters) → candidates
    route_cross_modal(embedding, source, target, k, filters) → candidates
    route_all_modal(embedding, k, filters) → candidates
  Applies metadata filters (bbox spatial filter, date range filter) via
  PostGIS spatial query or metadata JSON scan after FAISS search.
  Returns raw candidate list before re-ranking.

FILE: domain/retrieval/ranker.py
RESPONSIBILITY:
  Post-processes raw FAISS results to produce the final ranked list.
  Re-ranks by cosine similarity (in case FAISS approximate search reordered).
  Deduplicates: removes results where scene_id already appears at higher rank.
  Applies diversity: if top-3 results are identical LCLU class, injects one
  result from the next most-similar different class (optional for demo quality).
  Assigns final rank (1..top_k) and normalizes similarity to [0,1] range.
  Returns List[RankedCandidate] with final rank order.
```

---

### `domain/explanation/` — Explanation Pipeline

```
FILE: domain/explanation/pipeline.py
RESPONSIBILITY:
  ExplanationPipeline — orchestrates all explanation signals for a result set.
  Two execution modes controlled by explanation_depth param:
    "fast" mode (< 80ms overhead):
      Run only: axis scoring, confidence, natural language generation.
      Skip: attention heatmap computation (deferred to async).
    "full" mode (< 500ms overhead):
      Run all signals including heatmap extraction.
  Public method: explain(query_embedding, query_tensor, candidates,
                          depth) → List[ExplanationBundle]
  For "fast" mode: returns bundles with heatmap_urls=None (filled later async).
  Dispatches async Celery task for heatmap generation in both modes.
  Checks cache: if scene's explanation precomputed at ingestion → use cached values.

FILE: domain/explanation/attention/extractor.py
RESPONSIBILITY:
  Registers a forward hook on DINOv2 transformer Layer 23 self-attention module.
  Captures attn_output_weights tensor [B × 16 × 325 × 325] during forward pass.
  Exposes method extract(model, image_tensor) → raw_attention [B×16×324].
  Hook is registered and de-registered per call (thread-safe for concurrent requests).
  Stores raw attention for downstream rollout or direct aggregation.
  Never runs gradient computation — always torch.no_grad() context.

FILE: domain/explanation/attention/rollout.py
RESPONSIBILITY:
  Implements attention rollout algorithm (Abnar & Zuidema 2020).
  Takes per-layer attention weights (all 24 layers) and computes the
  effective attention from input tokens through the full transformer depth.
  Algorithm: A_rollout = Π_{l=0}^{L} (I + 0.5 × A_l) for each layer l.
  Requires running the model with hooks on ALL layers (not just final layer).
  Result: [B × 324] attention vector from [CLS] to all patch tokens.
  Used instead of final-layer attention when explanation quality > speed.
  In "fast" mode: use final-layer mean-head attention (1 hook, faster).
  In "full" mode: use rollout (24 hooks, slower, better attribution).

FILE: domain/explanation/attention/heatmap.py
RESPONSIBILITY:
  Converts raw attention vector [324] into visualization artifacts.
  Steps:
    1. Reshape [324] → [18×18] patch attention grid
    2. Bilinear upsample to [256×256]
    3. Min-max normalize to [0,1]
    4. Apply threshold mask (zero out values below 40th percentile)
    5. Apply viridis colormap → RGBA [256×256×4] uint8
    6. Composite onto original image with configured opacity (65%)
    7. Encode as PNG bytes
  Output methods:
    to_png_bytes() → bytes (for HTTP response or S3 upload)
    to_array() → np.ndarray [256×256] float32 (for DB storage)
    to_top3_regions(lclu_mask) → List[RegionImportance] (for metadata)
  Stateless — all inputs passed at call time.

FILE: domain/explanation/attention/alignment.py
RESPONSIBILITY:
  Computes structural similarity between query and result attention heatmaps.
  Uses SSIM (structural similarity index) from skimage.metrics.
  Method: score(query_heatmap, result_heatmap) → float [0,1].
  Interprets score: > 0.70 strong spatial alignment, 0.40-0.70 partial, < 0.40 low.
  Returns both scalar score and interpretation string.
  Used to populate attention_alignment field in ExplanationBundle.

FILE: domain/explanation/axes/scorer.py
RESPONSIBILITY:
  Computes semantic axis scores for any embedding vector.
  Loads 4 axis vectors (vegetation, waterbody, urban, road_network) from DB
  on first call; caches in memory as module-level constants.
  Method: score(embedding [128]) → AxisScores {vegetation, waterbody, urban, road}
  Each score = sigmoid(embedding · axis_vector + bias), range [0,1].
  Higher score = stronger presence of that semantic in the scene.
  Also provides score_batch(embeddings [N×128]) → [N×4] for efficient bulk scoring.

FILE: domain/explanation/axes/comparator.py
RESPONSIBILITY:
  Computes pairwise axis similarity between a (query, result) embedding pair.
  Uses AxisScorer to get per-axis scores for both embeddings.
  For each axis:
    query_score_A, result_score_A from scorer
    similarity_A = 1 - |query_score_A - result_score_A|
    verdict = "strong match" | "partial match" | "significant divergence"
    divergence_direction = "query has more X" | "result has more X" if diverging
  Returns SemanticAxisBundle with full breakdown per axis.
  Determines which axes are convergent (sim > 0.75) vs divergent (sim < 0.40).

FILE: domain/explanation/spectral/ndvi.py
RESPONSIBILITY:
  Computes NDVI (Normalized Difference Vegetation Index) from image data.
  Method signature: compute(image_array: np.ndarray, modality: str) → NDVIResult
  If modality == "multispectral":
    bands = rasterio-read 13-band array; NIR=B8(idx 7), Red=B4(idx 3)
    ndvi = (NIR - Red) / (NIR + Red + 1e-8)  per-pixel [256×256]
  If modality == "optical":
    Approximate NDVI from RGB: uses G channel as NIR proxy (weaker signal)
    Flag result as "approximated=True"
  If modality == "sar":
    Return None (NDVI not computable from SAR)
  Computes coverage profile: [dense_forest, moderate_veg, sparse_veg, non_veg]
  Returns NDVIResult {ndvi_mean, ndvi_std, coverage_profile, approximated, map_array}

FILE: domain/explanation/spectral/ndwi.py
RESPONSIBILITY:
  Same structure as ndvi.py but for NDWI (Normalized Difference Water Index).
  NDWI = (Green - NIR) / (Green + NIR + 1e-8)
  Sentinel-2: Green=B3(idx 2), NIR=B8(idx 7)
  Computes water coverage profile: [open_water, wetland, moist_soil, dry_land]
  For SAR: computes SAR water proxy from low-backscatter thresholding.
    water_mask = (VV < -15dB) AND (VH < -22dB)
    Returns sar_water_coverage scalar instead of NDWI profile.

FILE: domain/explanation/spectral/ndbi.py
RESPONSIBILITY:
  Computes NDBI (Normalized Difference Built-up Index).
  NDBI = (SWIR - NIR) / (SWIR + NIR + 1e-8)
  Sentinel-2: SWIR=B11(idx 10), NIR=B8(idx 7)
  Computes urban coverage profile: [dense_urban, sparse_urban, bare_soil, vegetated]
  Urban texture feature: thin wrapper calling a frozen ResNet-18 feature extractor
  (loaded once as singleton) to compute texture similarity scalar.
  For SAR: high double-bounce proxy: (VV > -5dB) AND (VH/VV > 0.5)

FILE: domain/explanation/spectral/sar_indices.py
RESPONSIBILITY:
  SAR-specific land cover detection methods used when source_modality is "sar".
  Methods:
    detect_water(sar_array) → water_coverage_pct
    detect_urban(sar_array) → urban_coverage_pct
    detect_vegetation(sar_array) → vegetation_proxy_pct (VV/VH ratio method)
  All methods work on raw dB SAR array [2×H×W] (VV, VH channels).
  Results flagged as "sar_derived=True" in explanation output.
  Used as fallback when MS bands not available.

FILE: domain/explanation/road/edge_detector.py
RESPONSIBILITY:
  Extracts edge features from a satellite image patch.
  Converts input tensor/array to grayscale.
  Applies Gaussian blur (σ=1.5) then Canny edge detection (low=50, high=150).
  Returns edge_map: binary [256×256] np.ndarray.
  Also computes total_edge_density = sum(edge_map) / total_pixels.
  Works on all three modalities — SAR images processed identically.

FILE: domain/explanation/road/line_detector.py
RESPONSIBILITY:
  Detects linear road-like features from an edge map.
  Applies probabilistic Hough Line Transform (cv2.HoughLinesP).
  Extracts: line_count, mean_line_length, orientation_entropy.
  orientation_entropy: entropy of angle distribution (bits).
    High entropy (>2.5) → grid road pattern (urban grid).
    Low entropy (<1.0) → highway or single directional road.
  Returns RoadFeatureVector [4]: [edge_density, line_count_norm,
                                   mean_length_norm, orient_entropy_norm].

FILE: domain/explanation/road/similarity.py
RESPONSIBILITY:
  Computes road network similarity between a (query, result) pair.
  Combines edge-based signal with embedding road axis score.
  Method: compute(query_road_features, result_road_features,
                   query_embedding, result_embedding) → RoadSimilarity
  edge_similarity = 1 - ||query_features - result_features||₂ / √4
  axis_similarity = road_axis_comparator.compare(query_emb, result_emb)
  road_similarity = 0.5 × edge_similarity + 0.5 × axis_similarity
  Returns RoadSimilarity {edge_sim, axis_sim, combined_sim, feature_vectors}.

FILE: domain/explanation/narrative/categorizer.py
RESPONSIBILITY:
  Classifies a (query, result) explanation bundle into one of four narrative categories.
  Inputs: global_similarity, convergent_axes (list), divergent_axes (list),
          confidence_score.
  CASE A: high_sim AND len(divergent_axes) == 0    → "Strong Semantic Match"
  CASE B: high_sim AND len(divergent_axes) > 0     → "Partial Match with Notable Divergence"
  CASE C: medium_sim                               → "Moderate Semantic Match"
  CASE D: any_sim AND confidence < 0.50            → "Uncertain Match"
  Thresholds: high_sim > 0.80, medium_sim 0.60–0.80.
  Returns NarrativeCategory enum value.

FILE: domain/explanation/narrative/generator.py
RESPONSIBILITY:
  Generates natural language explanation strings from structured data.
  Template-based — no LLM calls. All output is deterministic.
  Uses string templates per NarrativeCategory with slot filling.
  Templates embed:
    - Convergent axis names (Oxford-comma joined list)
    - Global similarity as percentage
    - Divergent axis name + query_score vs result_score percentages
    - Top-attended region name + coverage percentage
    - Confidence tier word
  Method: generate(axis_bundle, confidence, region_importance,
                    category) → str (natural language paragraph)
  Maximum output: 3 sentences. No markdown formatting.
```

---

### `domain/confidence/` — Confidence Pipeline

```
FILE: domain/confidence/pipeline.py
RESPONSIBILITY:
  ConfidencePipeline — orchestrates all confidence signals into final score.
  Public method: score(query_embedding, query_tensor, candidates,
                        source_modality) → List[ConfidenceBundle]
  Execution:
    1. MCDropout.estimate(query_tensor, source_modality) → uncertainty scalar
    2. GapScorer.compute(similarities) → gap score
    3. Calibration.calibrate(similarity, predicted_class) → calibrated_prob
    4. Composite.compute(calibrated_prob, uncertainty, gap) → confidence_score
    5. TierAssigner.assign(confidence_score) → tier + flags
  Returns one ConfidenceBundle per candidate in candidates list.

FILE: domain/confidence/mc_dropout.py
RESPONSIBILITY:
  Monte Carlo Dropout uncertainty estimation.
  Runs T forward passes through the encoder with dropout ENABLED at inference.
  Method: estimate(image_tensor, modality, T=5) → MCDropoutResult
  Execution:
    Set model.train() (enables dropout), wrap in torch.no_grad()
    Run T=5 passes: [z_1, z_2, ..., z_5] each [128] unit vector
    Restore model.eval()
    Compute: z_mean = mean(z_i), z_std = std(z_i), uncertainty = mean(z_std)
  Returns MCDropoutResult {z_mean [128], z_std [128], uncertainty scalar}.
  z_mean is used as the FAISS query vector (more robust than single-pass).
  T=5 for real-time path; T=20 via async Celery task for full explanation.
  Requires Dropout layers in ProjectionHead (inserted during model definition).

FILE: domain/confidence/calibration.py
RESPONSIBILITY:
  Temperature scaling for probability calibration per LCLU class.
  Loads calibration parameters (temperature, threshold per class) from DB
  on first call; caches in memory.
  Method: calibrate(cosine_similarity, predicted_lclu_class) → float [0,1]
  Formula: P = sigmoid((similarity - threshold_c) / temperature_c)
  predicted_lclu_class: inferred from query axis scores (highest axis → class).
  If class not in calibration table: use global fallback temperature=1.0.
  Returns calibrated probability P(relevant | similarity, class).

FILE: domain/confidence/gap_scorer.py
RESPONSIBILITY:
  Computes similarity gap between top-1 and bottom of top-K results.
  Method: compute(similarities: List[float], k: int) → GapScore
  gap = similarities[0] - similarities[-1]   (top-1 minus top-K)
  normalized_gap = gap / (similarities[0] + 1e-8)
  Large gap (> 0.3): top result is clearly best → higher confidence.
  Small gap (< 0.05): results are nearly equally similar → ambiguous ranking.
  Returns GapScore {gap, normalized_gap, gap_interpretation}.
  Applied to each result: per-result gap = sim_1 - sim_k (relative to that result's rank).

FILE: domain/confidence/composite.py
RESPONSIBILITY:
  Computes weighted composite confidence score from all signals.
  Method: compute(calibrated_prob, uncertainty, gap_score,
                   modality_consistency=None) → CompositeScore
  Formula:
    score = 0.40 × calibrated_prob
          + 0.30 × (1 - normalized_uncertainty)
          + 0.20 × gap_score_normalized
          + 0.10 × modality_consistency (or 0 if same-modal)
  Assigns confidence tier:
    ≥ 0.80 → "high"    (◆ green badge)
    0.60-0.79 → "medium"  (◈ amber badge)
    0.40-0.59 → "low"     (◇ red badge)
    < 0.40 → "uncertain"  (⚠ violet badge)
  Computes uncertainty_bound string: f"±{1.96 × z_std.mean():.2f}"
  Detects and sets confidence flags:
    "cross_modal_gap" if source != target and modality_consistency < 0.5
    "sparse_region" if uncertainty > 0.08
    "near_class_boundary" if two axis scores within 0.1 of each other
    "seasonal_mismatch" if query.season != result.season
```

---

### `services/` — Service Singletons

```
FILE: services/encoder_service.py
RESPONSIBILITY:
  EncoderService — manages the ML model singleton.
  Initialized once at app startup via lifespan hook.
  Holds: CrossModalModel instance loaded from checkpoint.
  Methods:
    encode(image_bytes, modality) → np.ndarray [128] L2-normalized
      Calls preprocessor.preprocess() then model.encode_single()
      Returns float32 numpy array ready for FAISS
    encode_tensor(tensor, modality) → np.ndarray [128]
      For when tensor is already preprocessed (MC Dropout path)
    encode_batch(tensors, modality) → np.ndarray [N×128]
      For bulk embedding computation
  Thread safety: model.eval() + torch.no_grad() throughout.
  GPU memory: model loaded to CUDA if available, else CPU.
  Exposes model_version property (read from checkpoint metadata).

FILE: services/faiss_service.py
RESPONSIBILITY:
  FAISSService — manages all FAISS indexes as singletons.
  Initialized once at app startup.
  Loads from disk: optical.index, sar.index, ms.index, shared.index.
  Loads from disk: id_map.json (FAISS integer ID → scene_id + modality).
  Methods:
    search(embedding, source_modality, target_modality, k,
            filters=None) → List[SearchResult]
      Routes to correct index per ModalityRouter logic
      Applies nprobe: 8 (per-modal), 16 (shared cross-modal)
      Decodes FAISS integer IDs to scene_ids via id_map
    search_by_scene_id(scene_id, modality, k) → List[SearchResult]
      Looks up stored embedding for scene_id, then searches
    get_index_stats() → IndexStats {size, last_updated, nlist, nprobe}
  Total memory footprint: ~80MB for 4 indexes at 10K scenes, 128-d.

FILE: services/explanation_service.py
RESPONSIBILITY:
  ExplanationService — high-level interface for the explanation subsystem.
  Thin wrapper that calls ExplanationPipeline.explain() and handles:
    Cache lookup before pipeline invocation.
    Cache write after pipeline completes.
    Async task dispatch for deferred heatmap generation.
  Methods:
    explain(query_id, query_embedding, query_tensor, candidates,
             depth) → List[ExplanationBundle]
    get_heatmap_png(query_id, scene_id, view) → bytes | None
    trigger_heatmap_async(query_id, scene_id) → task_id
  The "service" layer is intentionally thin here — business logic lives in domain/.

FILE: services/confidence_service.py
RESPONSIBILITY:
  ConfidenceService — high-level interface for the confidence subsystem.
  Thin wrapper around ConfidencePipeline.score().
  Loads calibration table from DB on first call; refreshes every 1 hour via TTL.
  Methods:
    score(query_embedding, query_tensor, candidates,
           source_modality) → List[ConfidenceBundle]
    get_calibration_params(lclu_class) → CalibrationParams
    refresh_calibration() → None  (called by admin endpoint)

FILE: services/result_service.py
RESPONSIBILITY:
  ResultService — composes the final API response from raw pipeline outputs.
  Fetches scene metadata from metadata.json (in-memory dict, loaded at startup).
  Constructs thumbnail URLs from scene_id template.
  Merges: ranked_candidates + explanations + confidence_bundles → RetrievalResponse.
  Assigns final rank numbers (1..top_k).
  Trims natural language to 3 sentences for "fast" mode.
  Method: compose(query_id, source, target, candidates, explanations,
                   confidence, latency_ms) → RetrievalResponse

FILE: services/cache_service.py
RESPONSIBILITY:
  Two-level cache: Redis (hot, 7-day TTL) + PostgreSQL (permanent for sessions).
  Redis key patterns:
    "embed:{hash(image_bytes)}"             → embedding [128] as bytes (TTL 24h)
    "expl:{query_id}:{scene_id}"            → ExplanationBundle JSON (TTL 7d)
    "heatmap:{query_id}:{scene_id}:{view}"  → PNG bytes (TTL 7d)
    "scene_expl:{scene_id}"                 → precomputed scene signals (TTL 30d)
    "conf:{query_id}:{scene_id}"            → ConfidenceBundle JSON (TTL 7d)
    "calib:{model_version}"                 → calibration params JSON (TTL 1h)
  Methods:
    get(key) → bytes | None
    set(key, value, ttl) → None
    get_json(key) → dict | None
    set_json(key, value, ttl) → None
    delete_pattern(pattern) → int  (returns deleted count)
  Handles Redis connection errors gracefully — falls back to cache miss.

FILE: services/thumbnail_service.py
RESPONSIBILITY:
  Serves precomputed JPEG thumbnails for scene IDs.
  Method: get_thumbnail(scene_id) → bytes | None
  Lookup order:
    1. Check local filesystem: static/thumbnails/{scene_id}.jpg
    2. Check S3 if local miss: s3://bucket/thumbnails/{scene_id}.jpg
    3. If S3 miss: generate on-demand from raw scene data (slow path)
  On-demand generation: reads optical/pseudo-RGB from HDF5, renders 128×128 JPEG.
  Caches on-demand result to local filesystem for future requests.
  Returns None (HTTP 404) if scene not found anywhere.

FILE: services/storage_service.py
RESPONSIBILITY:
  Abstraction over file storage backend (S3 vs local filesystem).
  Configured by STORAGE_BACKEND env var: "local" | "s3".
  Methods:
    upload(key, data_bytes, content_type) → str (URL or path)
    download(key) → bytes | None
    exists(key) → bool
    get_url(key) → str (presigned S3 URL or local path)
  Used by heatmap pipeline to store PNG outputs.
  Ensures the rest of the codebase never references S3 or FS directly.
```

---

### `models/orm/` — Database ORM Models

(See §4 for full SQL schema — this section lists ORM file responsibilities.)

```
FILE: models/orm/base.py
RESPONSIBILITY:
  SQLAlchemy DeclarativeBase and a TimestampMixin.
  TimestampMixin adds created_at, updated_at to any model.
  All ORM models inherit from Base (not DeclarativeBase directly).
  Configures naming conventions for constraints (important for Alembic).

FILE: models/orm/scene.py
RESPONSIBILITY:
  SatelliteScene ORM model.
  Maps to `scenes` table.
  Fields: scene_id (PK), lat, lon, date, season, lclu_class, modality,
          thumbnail_path, bbox (PostGIS Geometry), source_dataset.
  Relationships: one-to-many with EmbeddingRecord.
  PostGIS column uses geoalchemy2.Geometry("POINT", srid=4326).

FILE: models/orm/embedding.py
RESPONSIBILITY:
  EmbeddingRecord ORM model.
  Maps to `embeddings` table.
  Fields: id (PK), scene_id (FK→scenes), modality, embedding (pgvector 128-d),
          model_version, computed_at.
  pgvector column: Vector(128) from pgvector SQLAlchemy extension.
  UNIQUE constraint: (scene_id, modality, model_version).

FILE: models/orm/session.py
RESPONSIBILITY:
  RetrievalSession ORM model.
  Maps to `retrieval_sessions` table.
  Fields: query_id (UUID PK), user_id, source_modality, target_modality,
          query_embedding (pgvector), query_axis_scores (JSONB),
          top_k, explanation_depth, filters (JSONB), model_version,
          total_latency_ms, created_at.

FILE: models/orm/result.py
RESPONSIBILITY:
  RetrievalResult ORM model.
  Maps to `retrieval_results` table.
  Fields: id (BIGSERIAL PK), query_id (FK→sessions), scene_id (FK→scenes),
          rank, cosine_similarity, axis_similarities (JSONB), axis_verdicts (JSONB),
          confidence_score, confidence_tier, calibrated_prob, uncertainty_bound,
          confidence_flags (TEXT[]), attention_alignment, narrative_category,
          natural_language, heatmap_diff_s3, user_feedback, feedback_at.

FILE: models/orm/explanation.py
RESPONSIBILITY:
  SceneExplanation ORM model.
  Maps to `scene_explanations` table.
  Stores precomputed per-scene signals (computed once at ingestion).
  Fields: scene_id (PK), model_version, axis_vegetation/waterbody/urban/road,
          ndvi_mean, ndvi_std, ndwi_mean, ndbi_mean,
          vegetation_profile (ARRAY), water_profile (ARRAY), urban_profile (ARRAY),
          road_features (ARRAY), attention_grid (ARRAY 18×18),
          attention_heatmap_s3, attention_heatmap_png (S3 URI),
          region_importance (JSONB), embedding_uncertainty, computed_at.

FILE: models/orm/axis_vector.py
RESPONSIBILITY:
  SemanticAxisVector ORM model.
  Maps to `semantic_axis_vectors` table.
  Fields: id, model_version, axis_name (vegetation|waterbody|urban|road_network),
          axis_vector (pgvector 128-d), bias, threshold, temperature,
          probe_train_accuracy, probe_val_accuracy, probe_trained_at.
  UNIQUE (model_version, axis_name).
```

---

### `models/schemas/` — Pydantic Schemas

```
FILE: models/schemas/retrieval.py
RESPONSIBILITY:
  All request and response Pydantic models for /retrieve/* endpoints.
  RetrievalRequest: image (UploadFile), source_modality, target_modality,
                     top_k (int, 5|10), explanation_depth ("fast"|"full"),
                     filters (FilterParams | None).
  FilterParams: bbox (list[float] | None), date_from, date_to, season, modality.
  SceneMetadata: scene_id, lat, lon, date, season, lclu_class.
  HeatmapURLs: query, result, diff (all str | None).
  SceneMatch: rank, scene_id, modality, similarity, thumbnail_url,
              metadata, explanation (ExplanationBundle), confidence (ConfidenceBundle).
  RetrievalResponse: query_id, source_modality, target_modality, top_k,
                      latency_ms, results (List[SceneMatch]).

FILE: models/schemas/explanation.py
RESPONSIBILITY:
  Pydantic models for explanation structures.
  AxisScore: query_score, result_score, similarity, verdict, divergence_direction.
  SemanticAxisBundle: vegetation, waterbody, urban, road_network (all AxisScore).
  RegionImportance: class_name, attention_density, coverage_pct.
  SpatialAttributionBundle: heatmap_urls, attention_alignment, alignment_interpretation.
  ExplanationBundle: semantic_axes, spatial_attribution, region_importance (list),
                      narrative_category, natural_language, explanation_version,
                      generated_at, from_cache (bool).

FILE: models/schemas/confidence.py
RESPONSIBILITY:
  Pydantic models for confidence structures.
  ConfidenceBundle: raw_similarity, calibrated_prob, uncertainty, similarity_gap,
                     confidence_score, confidence_tier, uncertainty_bound,
                     flags (list[str]).

FILE: models/schemas/scene.py
RESPONSIBILITY:
  Pydantic models for scene catalog operations.
  SceneDetail: full scene metadata including bbox, cloud_cover_pct, source_dataset.
  SceneList: items (List[SceneDetail]), total, page, limit.
  ThumbnailResponse: scene_id, url, width, height.

FILE: models/schemas/common.py
RESPONSIBILITY:
  Shared Pydantic types used across multiple schema files.
  Pagination: page (int ≥ 1), limit (int 1-100 default 20).
  GeoPoint: lat (float), lon (float).
  BBoxParam: min_lon, min_lat, max_lon, max_lat (all float).
  ModalityEnum: "optical" | "sar" | "multispectral" | "all".
  SeasonEnum: "spring" | "summer" | "fall" | "winter".
  ErrorResponse: error (str), detail (str), request_id (str | None).
```

---

### `repositories/` — Data Access Layer

```
FILE: repositories/scene_repo.py
RESPONSIBILITY:
  All database queries involving the `scenes` table.
  Methods:
    get_by_id(scene_id) → SatelliteScene | None
    get_batch(scene_ids) → List[SatelliteScene]
    list_scenes(filters, pagination) → Tuple[List[SatelliteScene], int]
      Uses PostGIS ST_Within for bbox spatial filter
    search_by_class(lclu_class, limit) → List[SatelliteScene]
    count_by_modality() → Dict[str, int]
  All methods are async (await db.execute()).

FILE: repositories/embedding_repo.py
RESPONSIBILITY:
  All database queries involving the `embeddings` table.
  Methods:
    get_embedding(scene_id, modality) → np.ndarray [128] | None
    save_embedding(scene_id, modality, embedding, model_version) → None
    get_batch_embeddings(scene_ids, modality) → Dict[str, np.ndarray]
    delete_stale(model_version) → int  (returns deleted count)
    mark_all_stale(old_version) → None  (for model update flows)

FILE: repositories/session_repo.py
RESPONSIBILITY:
  All database queries for retrieval session history.
  Methods:
    save_session(session_record) → str  (returns query_id)
    get_session(query_id) → RetrievalSession | None
    list_sessions(user_id, pagination) → List[RetrievalSession]
    delete_old_sessions(before_date) → int  (data retention)

FILE: repositories/result_repo.py
RESPONSIBILITY:
  All database queries for retrieval results.
  Methods:
    save_results(query_id, results: List[RetrievalResult]) → None
      Bulk insert all results for a session
    get_results(query_id) → List[RetrievalResult]
    update_feedback(result_id, feedback) → None
    get_feedback_stats(model_version) → FeedbackStats

FILE: repositories/explanation_repo.py
RESPONSIBILITY:
  All database queries for the `scene_explanations` table.
  Methods:
    get_scene_explanation(scene_id, model_version) → SceneExplanation | None
    save_scene_explanation(explanation: SceneExplanation) → None
    batch_get_explanations(scene_ids, model_version) → Dict[str, SceneExplanation]
    is_stale(scene_id, current_model_version) → bool
    mark_all_stale(old_model_version) → int
```

---

### `core/` — Infrastructure

```
FILE: core/database.py
RESPONSIBILITY:
  Async SQLAlchemy engine and session factory.
  Creates engine from DATABASE_URL with connection pool (pool_size=10, max_overflow=20).
  AsyncSessionMaker factory for creating per-request sessions.
  Exposes get_db() async generator for FastAPI dependency injection.
  Exposes init_db() called at startup to verify connection.

FILE: core/redis_client.py
RESPONSIBILITY:
  aioredis connection pool management.
  Creates pool on startup: max_connections=50, decode_responses=False (binary safe).
  Exposes get_redis() async generator for dependency injection.
  Provides health_check() method returning bool.
  Handles ConnectionRefusedError gracefully (returns None from get_redis).

FILE: core/storage.py
RESPONSIBILITY:
  StorageBackend abstract class with LocalStorage and S3Storage implementations.
  Reads STORAGE_BACKEND env var to select implementation at startup.
  Both implementations satisfy the same interface (upload, download, exists, get_url).
  LocalStorage: reads/writes to THUMBNAILS_DIR and HEATMAPS_DIR on filesystem.
  S3Storage: uses aioboto3 for async S3 operations.
  Singleton pattern: one instance created at startup.

FILE: core/exceptions.py
RESPONSIBILITY:
  Custom exception hierarchy used throughout the backend.
  Base: GeoSenseException(HTTPException) with request_id field.
  Sub-classes:
    PreprocessingError(400)   → invalid image format or corrupted data
    ModelNotLoadedError(503)  → model not ready yet
    IndexNotLoadedError(503)  → FAISS index not loaded
    SceneNotFoundError(404)   → scene_id does not exist
    EmbeddingNotFound(404)    → embedding not computed for scene
    ExplanationUnavailable(202) → explanation queued, not yet ready
    RateLimitExceeded(429)    → too many requests
  Registered handlers in main.py convert these to JSON error responses.

FILE: core/logging.py
RESPONSIBILITY:
  Structured JSON logging configuration using loguru.
  Configures output format: timestamp, level, request_id, module, message, extras.
  Binds request_id (from X-Request-ID header) to each request's log context.
  Provides log_retrieval_request(query_id, latency_ms, result_count) helper.
  Log level from LOG_LEVEL env var (DEBUG in dev, INFO in prod).

FILE: core/metrics.py
RESPONSIBILITY:
  Prometheus metrics definitions.
  Counters:
    retrieval_requests_total{modality, target_modality, status}
    explanation_requests_total{depth, cache_hit}
    confidence_tier_total{tier, modality}
  Histograms:
    retrieval_latency_seconds{step}  (step: encode, search, explain, total)
    embedding_dimension_norm         (for detecting embedding drift)
  Gauges:
    faiss_index_size{modality}
    model_loaded{version}
  Exposes metrics via /metrics endpoint in Prometheus text format.

FILE: core/events.py
RESPONSIBILITY:
  FastAPI lifespan async context manager (startup/shutdown).
  Startup sequence (order matters):
    1. Validate config (fail fast on missing required env vars)
    2. Connect PostgreSQL (raise if unavailable)
    3. Connect Redis (log warning if unavailable, continue)
    4. Load ML model checkpoint into EncoderService
    5. Load FAISS indexes into FAISSService
    6. Load metadata.json into memory dict
    7. Load axis vectors from DB into AxisScorer
    8. Load calibration params from DB into Calibration
    9. Initialize Celery connection
    10. Log startup complete + model version
  Shutdown sequence:
    1. Close DB sessions
    2. Close Redis connection pool
    3. Release GPU memory (model.cpu(), del model, torch.cuda.empty_cache())
    4. Log shutdown complete

FILE: core/rate_limiter.py
RESPONSIBILITY:
  Sliding window rate limiter using Redis.
  Uses Redis ZADD + ZREMRANGEBYSCORE (sorted set per user/IP).
  Checks: requests in last 60 seconds against per-tier limit.
  Tiers: anonymous=10/min, api_key=60/min, premium=600/min.
  Raises RateLimitExceeded with Retry-After header if limit exceeded.
  Degrades gracefully if Redis unavailable (skips rate limiting with warning).

FILE: core/security.py
RESPONSIBILITY:
  API key validation for admin and authenticated endpoints.
  Reads API key from Authorization: Bearer <key> header.
  Hashes key with SHA-256, compares against API_KEY_HASH in config.
  For MVP: single API key. Production: DB-backed key management.
  Returns user dict {user_id, tier} if valid, raises 401 if invalid.
  Admin endpoints additionally check user.tier == "admin".
```

---

### `workers/` — Async Background Jobs

```
FILE: workers/celery_app.py
RESPONSIBILITY:
  Celery application factory.
  Broker: Redis (CELERY_BROKER_URL from config).
  Backend: Redis (CELERY_RESULT_BACKEND from config).
  Task queues:
    "embedding"  → priority queue for embedding computation jobs
    "heatmap"    → deferred heatmap generation jobs
    "ingestion"  → bulk scene ingestion jobs
  Configures task serializer (json), result expiry (24h), acks_late=True.

FILE: workers/embedding_tasks.py
RESPONSIBILITY:
  Celery tasks for async embedding computation.
  compute_scene_embedding(scene_id, modality):
    Loads scene from HDF5, computes embedding, saves to DB.
  compute_batch_embeddings(scene_ids, modality):
    Bulk version, processes in mini-batches of 64.
  recompute_all_embeddings(model_version):
    Triggered on model update. Full dataset re-embedding.

FILE: workers/heatmap_tasks.py
RESPONSIBILITY:
  Celery tasks for deferred attention heatmap generation.
  generate_heatmap_pair(query_id, scene_id, query_image_bytes):
    Runs full attention extraction (rollout, T=20 MC dropout).
    Generates query heatmap, result heatmap, and diff overlay PNGs.
    Uploads PNGs to storage (S3 or local).
    Updates Redis cache with heatmap URLs.
    Updates DB retrieval_results row with heatmap_diff_s3.
  Called by ExplanationService after returning "fast" mode response.

FILE: workers/ingestion_tasks.py
RESPONSIBILITY:
  Celery tasks for scene catalog ingestion.
  ingest_scene(scene_path, modality):
    Parse metadata → save to scenes table.
    Compute thumbnail → save to static/thumbnails/.
    Compute embedding → save to embeddings table.
    Compute scene explanation → save to scene_explanations table.
  ingest_bulk(scene_paths):
    Queues individual ingest_scene tasks for each scene.
```

---

## 4. Database Models

```
TABLE: scenes
Purpose: Master catalog of all satellite scenes.
Columns:
  scene_id          TEXT         PK   "ROIs1158_spring_p0001"
  lat               DOUBLE       NOT NULL   center latitude
  lon               DOUBLE       NOT NULL   center longitude
  geometry          GEOMETRY(Point, 4326)   PostGIS spatial column (indexed)
  bbox_geom         GEOMETRY(Polygon, 4326) Scene footprint (indexed)
  date              DATE         NOT NULL
  season            TEXT         CHECK (spring|summer|fall|winter)
  lclu_class        TEXT         NOT NULL   MODIS land cover class name
  lclu_class_id     SMALLINT     NOT NULL   MODIS class integer
  modality          TEXT[]       NOT NULL   ["optical","sar","ms"] (all available)
  source_dataset    TEXT         DEFAULT 'sen12ms'
  cloud_cover_pct   REAL                    null for SAR
  thumbnail_path    TEXT                    local path or S3 URI
  hdf5_path         TEXT         NOT NULL   path to raw data shard
  hdf5_index        INTEGER      NOT NULL   row index within HDF5 file
  created_at        TIMESTAMPTZ  DEFAULT NOW()
Indexes:
  GIST(geometry)             → bbox spatial queries
  BTREE(lclu_class_id)       → class filtering
  BTREE(date)                → temporal filtering
  GIN(modality)              → modality array filtering

TABLE: embeddings
Purpose: Stores computed embeddings per (scene, modality, model_version).
Columns:
  id               BIGSERIAL    PK
  scene_id         TEXT         FK → scenes(scene_id) ON DELETE CASCADE
  modality         TEXT         CHECK (optical|sar|multispectral)
  embedding        vector(128)  NOT NULL   pgvector column (L2-normalized)
  model_version    TEXT         NOT NULL
  computed_at      TIMESTAMPTZ  DEFAULT NOW()
  compute_ms       INTEGER                  computation time for monitoring
Constraints:
  UNIQUE (scene_id, modality, model_version)
Indexes:
  HNSW index on embedding for pgvector approximate search (fallback to FAISS)
  BTREE(model_version, modality)

TABLE: retrieval_sessions
Purpose: Log of every retrieval request for analytics and replay.
Columns:
  query_id         UUID         PK DEFAULT gen_random_uuid()
  user_id          UUID         nullable (anonymous allowed)
  source_modality  TEXT         NOT NULL
  target_modality  TEXT         NOT NULL
  query_embedding  vector(128)  NOT NULL   pgvector
  query_axis_veg   REAL
  query_axis_water REAL
  query_axis_urban REAL
  query_axis_road  REAL
  top_k            SMALLINT     NOT NULL
  explanation_depth TEXT        NOT NULL
  filters          JSONB
  model_version    TEXT         NOT NULL
  total_latency_ms INTEGER
  encode_latency_ms INTEGER
  search_latency_ms INTEGER
  explain_latency_ms INTEGER
  created_at       TIMESTAMPTZ  DEFAULT NOW()
Indexes:
  BTREE(created_at DESC)         → recent sessions
  BTREE(user_id, created_at)     → per-user history

TABLE: retrieval_results
Purpose: Individual results for each retrieval session.
Columns:
  id               BIGSERIAL    PK
  query_id         UUID         FK → retrieval_sessions ON DELETE CASCADE
  scene_id         TEXT         FK → scenes
  rank             SMALLINT     NOT NULL   1..10
  cosine_similarity REAL        NOT NULL
  axis_sim_veg     REAL
  axis_sim_water   REAL
  axis_sim_urban   REAL
  axis_sim_road    REAL
  axis_verdicts    JSONB                   {vegetation: "strong match", ...}
  calibrated_prob  REAL
  confidence_score REAL
  confidence_tier  TEXT
  uncertainty_bound TEXT                   "±0.04"
  confidence_flags TEXT[]
  attention_alignment REAL
  narrative_category TEXT
  natural_language TEXT
  heatmap_diff_s3  TEXT
  user_feedback    TEXT         CHECK (relevant|not_relevant|partially_relevant)
  feedback_at      TIMESTAMPTZ
  created_at       TIMESTAMPTZ  DEFAULT NOW()
Constraints:
  UNIQUE (query_id, rank)
  UNIQUE (query_id, scene_id)
Indexes:
  BTREE(query_id, rank)
  BTREE(scene_id)
  BTREE(user_feedback) WHERE user_feedback IS NOT NULL

TABLE: scene_explanations
Purpose: Precomputed per-scene explanation signals (computed at ingestion).
Columns:
  scene_id         TEXT         PK FK → scenes(scene_id)
  model_version    TEXT         NOT NULL
  axis_veg         REAL         NOT NULL
  axis_water       REAL         NOT NULL
  axis_urban       REAL         NOT NULL
  axis_road        REAL         NOT NULL
  ndvi_mean        REAL
  ndvi_std         REAL
  ndwi_mean        REAL
  ndbi_mean        REAL
  veg_profile      REAL[4]                 [dense, moderate, sparse, non_veg]
  water_profile    REAL[4]                 [open_water, wetland, moist_soil, dry]
  urban_profile    REAL[4]                 [dense, sparse, bare_soil, vegetated]
  road_features    REAL[4]                 [edge_density, line_ct, mean_len, entropy]
  attention_grid   REAL[18][18]            18×18 compressed attention map
  attention_s3     TEXT                    S3 URI to full-res heatmap PNG
  region_importance JSONB                  [{class, attention_density, coverage_pct}]
  embedding_uncertainty REAL
  computed_at      TIMESTAMPTZ  DEFAULT NOW()
Indexes:
  BTREE(model_version)
  BTREE(axis_urban DESC)         → "find most urban scenes" admin queries

TABLE: semantic_axis_vectors
Purpose: Learned semantic axis direction vectors per model version.
Columns:
  id               SERIAL       PK
  model_version    TEXT         NOT NULL
  axis_name        TEXT         CHECK (vegetation|waterbody|urban|road_network)
  axis_vector      vector(128)  NOT NULL
  bias             REAL         NOT NULL DEFAULT 0.0
  threshold        REAL         NOT NULL
  temperature      REAL         NOT NULL DEFAULT 1.0
  probe_train_acc  REAL
  probe_val_acc    REAL
  probe_trained_at TIMESTAMPTZ  DEFAULT NOW()
Constraints:
  UNIQUE (model_version, axis_name)

TABLE: calibration_records
Purpose: Temperature scaling params per LCLU class per model version.
Columns:
  id               SERIAL       PK
  model_version    TEXT         NOT NULL
  lclu_class       TEXT         NOT NULL
  temperature      REAL         NOT NULL
  threshold        REAL         NOT NULL
  ece_before       REAL
  ece_after        REAL
  calibrated_at    TIMESTAMPTZ  DEFAULT NOW()
Constraints:
  UNIQUE (model_version, lclu_class)
```

---

## 5. API Routes Specification

```
BASE: /api/v1
AUTH: Optional Bearer token (required for admin endpoints)
RATE: 60 req/min per API key, 10 req/min anonymous

─────────────────────────────────────────────────────────────
RETRIEVAL ENDPOINTS
─────────────────────────────────────────────────────────────

POST /retrieve/optical
  Input:  multipart/form-data
          image (file, required)
          top_k (int, 5|10, default 10)
          explanation_depth ("fast"|"full", default "fast")
          filters (JSON string, optional)
  Output: RetrievalResponse
  Notes:  source_modality="optical", target_modality="optical"

POST /retrieve/sar
  Input:  same structure
  Output: RetrievalResponse
  Notes:  source_modality="sar", target_modality="sar"

POST /retrieve/multispectral
  Input:  same structure
  Output: RetrievalResponse
  Notes:  source_modality="multispectral", target_modality="ms"

POST /retrieve/cross-modal
  Input:  multipart/form-data
          image (file, required)
          source_modality (required, "optical"|"sar"|"multispectral")
          target_modality (required, "optical"|"sar"|"multispectral"|"all")
          top_k (int, 5|10, default 10)
          explanation_depth ("fast"|"full", default "fast")
          filters (JSON string, optional)
  Output: RetrievalResponse
  Notes:  Core MVP endpoint. source != target for cross-modal.
          target="all" returns mixed-modality results with tags.

─────────────────────────────────────────────────────────────
EXPLANATION ENDPOINTS
─────────────────────────────────────────────────────────────

GET /explain/{query_id}/{scene_id}
  Output: ExplanationBundle (full, cached or computed)
  Notes:  Returns 202 Accepted + task_id if heatmaps not yet ready.
          Returns 200 with complete bundle if already cached.

GET /explain/{query_id}/{scene_id}/axes
  Output: { semantic_axes: SemanticAxisBundle, generated_at }
  Notes:  Always fast (<10ms from cache). No heatmaps.

GET /explain/{query_id}/{scene_id}/confidence
  Output: ConfidenceBundle
  Notes:  Always fast (<10ms from cache).

GET /explain/{query_id}/{scene_id}/heatmap/query
  Output: StreamingResponse (image/png)
  Notes:  Returns 404 if not yet computed. Client polls or uses WS.

GET /explain/{query_id}/{scene_id}/heatmap/result
  Output: StreamingResponse (image/png)

GET /explain/{query_id}/{scene_id}/heatmap/diff
  Output: StreamingResponse (image/png)
  Notes:  Red=query-only attention, Blue=result-only, Green=shared.

POST /explain/batch
  Input:  JSON body: { pairs: [{query_id, scene_id}], depth: "fast" }
          Max 20 pairs per request.
  Output: { explanations: List[ExplanationBundle] }

─────────────────────────────────────────────────────────────
SCENE ENDPOINTS
─────────────────────────────────────────────────────────────

GET /scenes/{scene_id}
  Output: SceneDetail (metadata + available modalities)

GET /scenes/{scene_id}/thumbnail
  Output: StreamingResponse (image/jpeg, 128×128)

GET /scenes/
  Query params:
    modality (optional, "optical"|"sar"|"multispectral")
    lclu_class (optional, string)
    bbox (optional, "minLon,minLat,maxLon,maxLat")
    date_from, date_to (optional, "YYYY-MM-DD")
    season (optional)
    page (default 1), limit (default 20, max 100)
  Output: SceneList { items, total, page, limit }

─────────────────────────────────────────────────────────────
INTERACTIVE SEARCH ENDPOINT
─────────────────────────────────────────────────────────────

WS /search/interactive
  Protocol: JSON messages over WebSocket
  Auth: ?token=<api_key> query param
  Timeout: 30s inactivity disconnects
  Messages: QUERY → phased responses (EMBEDDING_READY, TOP_K_RESULTS,
                                      AXES_READY, HEATMAP_READY)
            REFINE → re-ranked results with axis weights applied
            EXPLORE_HOP → use result scene as new query anchor

POST /search/refine
  Input:  { query_id, refinements: [{axis, direction, weight}], top_k }
  Output: RetrievalResponse with updated results
  Notes:  Modifies query embedding in-place using axis vectors.
          Does not create new DB session (annotated as "refined").

─────────────────────────────────────────────────────────────
ADMIN ENDPOINTS (Bearer admin-api-key required)
─────────────────────────────────────────────────────────────

POST /admin/index/rebuild
  Output: { task_id, status: "queued" }
  Notes:  Triggers full FAISS rebuild Celery task.

GET  /admin/index/status
  Output: { optical: IndexInfo, sar: IndexInfo, ms: IndexInfo, shared: IndexInfo }
  IndexInfo: { size, last_built, is_building, build_progress_pct }

POST /admin/model/reload
  Input:  { checkpoint_path }
  Output: { status: "reloading", previous_version, new_version }

GET  /admin/cache/stats
  Output: { redis_memory_mb, hit_rate_pct, miss_rate_pct, key_counts_by_prefix }

POST /admin/cache/clear
  Input:  { prefix: "expl"|"heatmap"|"scene_expl"|"all" }
  Output: { deleted_keys: int }

GET  /admin/jobs
  Output: { active: [], scheduled: [], reserved: [] }

─────────────────────────────────────────────────────────────
HEALTH ENDPOINTS
─────────────────────────────────────────────────────────────

GET /health
  Output: { status: "ok", timestamp }
  Notes:  Always 200 if process is running.

GET /ready
  Output: { status: "ready"|"degraded", checks: { db, redis, faiss, model } }
  Notes:  503 if any check fails. Used by K8s readiness probe.

GET /metrics
  Output: text/plain (Prometheus exposition format)
  Notes:  Counters, histograms, gauges from core/metrics.py.
```

---

## 6. Services Layer

```
SERVICE INTERACTION MAP

  API Route
    │
    ├── EncoderService.encode()
    │     │ loads: CrossModalModel (from ml/)
    │     │ calls: Preprocessor.preprocess()
    │     └── returns: np.ndarray [128]
    │
    ├── FAISSService.search()
    │     │ loads: 4 FAISS indexes (from indexes/)
    │     │ loads: id_map.json (from indexes/)
    │     └── returns: List[SearchResult]
    │
    ├── ExplanationService.explain()
    │     │ checks: CacheService.get_json("expl:...")
    │     │ calls: ExplanationPipeline.explain()
    │     │   │ calls: AttentionExtractor.extract()
    │     │   │ calls: AxisScorer.score() × 2 (query + result)
    │     │   │ calls: AxisComparator.compare()
    │     │   │ calls: NarrativeCategorizer + NarrativeGenerator
    │     │   └── returns: ExplanationBundle (sans heatmaps)
    │     │ calls: CacheService.set_json("expl:...", bundle)
    │     │ dispatches: heatmap_tasks.generate_heatmap_pair.delay()
    │     └── returns: ExplanationBundle
    │
    ├── ConfidenceService.score()
    │     │ checks: CacheService.get_json("conf:...")
    │     │ calls: ConfidencePipeline.score()
    │     │   │ calls: MCDropout.estimate()
    │     │   │ calls: GapScorer.compute()
    │     │   │ calls: Calibration.calibrate()
    │     │   └── returns: ConfidenceBundle
    │     │ calls: CacheService.set_json("conf:...", bundle)
    │     └── returns: ConfidenceBundle
    │
    └── ResultService.compose()
          │ reads: metadata_dict (in-memory, loaded at startup)
          │ constructs thumbnail_url per scene_id
          └── returns: RetrievalResponse

SERVICE SINGLETON LIFECYCLE

  All singletons stored in FastAPI app.state (set in lifespan):
    app.state.encoder   = EncoderService()
    app.state.faiss     = FAISSService()
    app.state.cache     = CacheService()
    app.state.result    = ResultService()
    app.state.thumbnail = ThumbnailService()
    app.state.storage   = StorageService()

  Explanation and Confidence are stateless (no internal state) →
  instantiated per-request by domain pipeline (not stored in app.state).

  AxisScorer loads axis vectors once at module level →
  all requests share the same numpy arrays (read-only, thread-safe).
```

---

## 7. Retrieval Pipeline

```
RETRIEVAL PIPELINE — COMPLETE EXECUTION FLOW

Input: image_bytes (raw), source_modality, target_modality, top_k, filters

STEP 1: Embedding Cache Check (CacheService)
  key = "embed:{sha256(image_bytes)}"
  if Redis hit → skip Steps 2-3, use cached embedding
  if miss → proceed

STEP 2: Preprocessing (Preprocessor)
  Optical:  PIL decode → RGB → resize 224×224 → normalize
  SAR:      rasterio decode → dB convert → pseudo-RGB → normalize
  MS:       rasterio decode → PCA project (13→3) → normalize
  Output: torch.Tensor [1 × 3 × 224 × 224]

STEP 3: Encode (EncoderService)
  CrossModalModel.encode_single(tensor, source_modality)
  DINOv2 forward → [CLS] token → ProjectionHead → L2 normalize
  Output: np.ndarray [128] float32
  Side effect: save embedding to Redis cache (TTL 24h)

STEP 4: FAISS Search (FAISSService via ModalityRouter)

  CASE: same-modal (source == target)
    index = load index[source_modality]
    D, I = index.search(embedding.reshape(1,128), top_k)
    candidates = [decode_id(I[0][i], similarity=D[0][i]) for i in range(top_k)]

  CASE: cross-modal (source != target)
    index = shared_index
    D, I = index.search(embedding.reshape(1,128), top_k × 5)  # over-fetch
    candidates = filter(I, target_modality via id_map)[:top_k]

  CASE: all-modal (target == "all")
    index = shared_index
    D, I = index.search(embedding.reshape(1,128), top_k)
    candidates = [decode_id(I[0][i]) for i in range(top_k)]

  Metadata filter (if filters.bbox provided):
    candidates = [c for c if scene_in_bbox(metadata[c.scene_id], filters.bbox)]
    if len(candidates) < top_k → re-query with increased over-fetch

  Output: List[Candidate] ordered by similarity descending

STEP 5: Re-ranking (Ranker)
  Re-sort by exact cosine similarity (in case IVFFlat approximation reordered).
  Remove duplicates by scene_id.
  Optional diversity injection (demo quality): ensure ≥ 2 distinct LCLU classes.
  Assign final rank 1..top_k.
  Output: List[RankedCandidate] with final rank + similarity + scene_id + modality

STEP 6: Parallel Execution (asyncio.gather)
  Start simultaneously:
    Task A: ExplanationPipeline.explain(embedding, candidates, depth)
    Task B: ConfidencePipeline.score(embedding, candidates)
  Await both.

STEP 7: Compose Response (ResultService)
  Merge: candidates + explanations (from A) + confidence bundles (from B)
  Fetch metadata from in-memory dict for each scene_id
  Build thumbnail URL for each scene_id
  Assign final rank numbers
  Output: RetrievalResponse

STEP 8: Persist Session (async, non-blocking)
  asyncio.create_task(session_repo.save_session(...))
  asyncio.create_task(result_repo.save_results(...))
  Does not block the HTTP response.

STEP 9: Return Response
  Total pipeline target: < 300ms P99

LATENCY BUDGET BREAKDOWN
  Embedding cache check:    1ms
  Preprocessing:            3ms
  Encode (GPU):            12ms  (CPU: 150ms)
  FAISS search:             8ms  (cross-modal: 15ms)
  Explanation (fast mode): 30ms
  Confidence:              10ms
  Compose response:         3ms
  ─────────────────────────────
  Total (GPU + fast):      67ms  (well within 300ms budget)
  Total (CPU + fast):     230ms  (within budget)
  Total (GPU + full):     180ms  (within budget with rollout)
```

---

## 8. Explanation Pipeline

```
EXPLANATION PIPELINE — COMPLETE EXECUTION FLOW

Input: query_embedding [128], query_tensor [1×3×224×224],
       ranked_candidates: List[RankedCandidate], depth: "fast"|"full"

For each candidate in ranked_candidates:

STEP 1: Cache Lookup (per-scene precomputed signals)
  Check scene_explanations table for this scene_id + model_version.
  If found:
    axis_scores = cached (from scene_explanations)
    ndvi/ndwi/ndbi = cached
    road_features = cached
    attention_grid = cached (18×18 compressed)
    result_tensor = None (not re-run through model)
  If miss:
    Load raw scene from HDF5 → compute all signals fresh.

STEP 2: Semantic Axis Scoring
  query_axis_scores = AxisScorer.score(query_embedding)   ← from cache if set
  result_axis_scores = AxisScorer.score(candidate.embedding) ← from scene_expl

  AxisComparator.compare(query_axis_scores, result_axis_scores):
    For each axis {veg, water, urban, road}:
      query_score, result_score  ← from respective AxisScorer calls
      similarity = 1 - |q_score - r_score|
      verdict = categorize(similarity)
      direction = "query has more X" if q > r else "result has more X"
    Identify convergent_axes (sim > 0.75) and divergent_axes (sim < 0.40)

STEP 3: Attention Heatmap (depth-dependent)

  "fast" mode:
    Use cached attention_grid [18×18] from scene_explanations if available.
    Upsample to [256×256].
    Skip fresh attention extraction.
    heatmap_urls = None (will be filled by async Celery task).
    Dispatch: heatmap_tasks.generate_heatmap_pair.delay(query_id, scene_id, query_bytes)

  "full" mode:
    Register hooks on all 24 DINOv2 layers.
    Run query image through model → capture per-layer attentions.
    Run result image through model → capture per-layer attentions.
    Compute attention rollout for both → [324] vectors.
    Upsample both to [256×256] float32 heatmaps.
    Compute diff overlay: red=query_only, blue=result_only, green=shared.
    Encode all 3 to PNG bytes.
    Upload to storage.
    heatmap_urls = {query: url, result: url, diff: url}.

STEP 4: Attention Alignment Score
  SSIM(query_heatmap_18x18, result_heatmap_18x18)
  Works in "fast" mode using cached 18×18 grids (no need for full 256×256).
  interpretation = categorize(ssim_score)

STEP 5: Region Importance
  top_3_regions from scene_explanations.region_importance (cached).
  If not cached: compute from attention_grid × lclu_mask (from DB).

STEP 6: Narrative Generation
  category = NarrativeCategorizer.categorize(global_sim, convergent, divergent, confidence)
  narrative = NarrativeGenerator.generate(axis_bundle, confidence, top_3_regions, category)

STEP 7: Assemble ExplanationBundle
  {
    semantic_axes:         SemanticAxisBundle (from Step 2)
    spatial_attribution:   { heatmap_urls, attention_alignment, interpretation }
    region_importance:     List[RegionImportance] (from Step 5)
    narrative_category:    NarrativeCategory (from Step 6)
    natural_language:      str (from Step 6)
    explanation_version:   "1.0"
    generated_at:          now()
    from_cache:            True if scene_explanations hit
  }

STEP 8: Cache Result
  CacheService.set_json(f"expl:{query_id}:{scene_id}", bundle, TTL=7days)

Output: ExplanationBundle per candidate

TOTAL EXPLANATION OVERHEAD (fast mode):
  Axis scoring (cached axis vectors):      2ms
  SSIM attention alignment (18×18):        1ms
  Narrative generation (template):         0.5ms
  Cache read/write:                        2ms
  Total per result:                       ~6ms
  For top-10 results:                    ~60ms   ← fits in latency budget
```

---

## 9. Confidence Pipeline

```
CONFIDENCE PIPELINE — COMPLETE EXECUTION FLOW

Input: query_tensor [1×3×224×224], source_modality, ranked_candidates

STEP 1: MC Dropout Uncertainty Estimation (shared across all candidates)
  Runs T=5 forward passes through encoder with dropout enabled.
  All T=5 passes use same query image.
  Output: z_mean [128], z_std [128], uncertainty_scalar (mean of z_std)
  Note: z_mean replaces single-pass embedding for confidence computation.
        The FAISS search already ran with single-pass embedding (faster).
        MC Dropout result used only for confidence scoring, not re-ranking.
  Cache: store uncertainty_scalar in Redis with key "unc:{query_id}"
         TTL = session lifetime (24h)

STEP 2: Calibration (per candidate)
  For each candidate:
    predicted_class = argmax(AxisScorer.score(candidate.embedding))
      maps axis → LCLU class group
      vegetation → forest/agriculture classes
      urban → urban fabric classes
      waterbody → water body classes
      road → transport classes
    T_c, threshold_c = CalibrationTable.get(predicted_class, model_version)
    calibrated_prob = sigmoid((candidate.similarity - threshold_c) / T_c)

STEP 3: Similarity Gap Score (computed across all candidates)
  similarities = [c.similarity for c in ranked_candidates]
  top1_sim = similarities[0]
  For candidate at rank k:
    gap = top1_sim - similarities[k-1]  (0 for rank 1, positive for others)
    gap_norm = gap / (top1_sim + 1e-8)
    gap_score = 1.0 - gap_norm  (high = small gap = less confident in ranking)

STEP 4: Modality Consistency Score (cross-modal only)
  Only computed when source_modality != target_modality.
  Measures: does the query embedding "fit" in the target modality's subspace?
  Method: compute cosine similarity between query_embedding and
          centroid of target modality embeddings (loaded from index stats).
  modality_consistency = cosine(query_emb, target_modality_centroid)
  If same-modal → set to 1.0 (no cross-modal gap possible).

STEP 5: Composite Score
  For each candidate:
    uncertainty_component = 1 - normalize_uncertainty(uncertainty_scalar)
      normalize: map [0, 0.15] → [0, 1]
    confidence_score =
        0.40 × calibrated_prob
      + 0.30 × uncertainty_component
      + 0.20 × gap_score
      + 0.10 × modality_consistency

STEP 6: Tier Assignment + Flag Detection
  tier = assign_tier(confidence_score)
  flags = []
    if source != target AND modality_consistency < 0.5:
      flags.append("cross_modal_gap")
    if uncertainty_scalar > 0.08:
      flags.append("sparse_region")
    if max axis score - second_max axis score < 0.10:
      flags.append("near_class_boundary")
    if query.season != result.season:
      flags.append("seasonal_mismatch")

STEP 7: Assemble ConfidenceBundle
  {
    raw_similarity:       candidate.similarity
    calibrated_prob:      calibrated_prob
    uncertainty:          uncertainty_scalar
    similarity_gap:       gap (absolute)
    confidence_score:     composite_score
    confidence_tier:      tier
    uncertainty_bound:    f"±{1.96 × uncertainty_scalar:.2f}"
    flags:                flags
  }

STEP 8: Cache
  CacheService.set_json(f"conf:{query_id}:{scene_id}", bundle, TTL=7days)

TOTAL CONFIDENCE OVERHEAD:
  MC Dropout T=5:        60ms  (5 × 12ms GPU passes)
  Calibration (in-mem):   0.5ms per candidate
  Gap scoring:            0.1ms per candidate
  Composite + flags:      0.2ms per candidate
  For top-10:            ~65ms total (MC Dropout dominates)
  Combined explanation + confidence: ~125ms → well within budget
```

---

## 10. Caching Strategy

```
CACHE ARCHITECTURE: Two-Level (Redis + PostgreSQL)

─────────────────────────────────────────────────────────────
LEVEL 1: REDIS (Hot Cache, In-Memory, TTL-Based)
─────────────────────────────────────────────────────────────

Key Pattern              │ Value                │ TTL    │ Purpose
─────────────────────────┼──────────────────────┼────────┼─────────────────────
embed:{sha256(img)}      │ np.ndarray [128]      │ 24h    │ Skip re-encoding same image
                         │ as raw bytes (struct) │        │
expl:{qid}:{sid}         │ ExplanationBundle     │ 7d     │ Pair explanation cache
                         │ JSON                  │        │
conf:{qid}:{sid}         │ ConfidenceBundle      │ 7d     │ Pair confidence cache
                         │ JSON                  │        │
heatmap:{qid}:{sid}:q    │ PNG bytes             │ 7d     │ Query heatmap image
heatmap:{qid}:{sid}:r    │ PNG bytes             │ 7d     │ Result heatmap image
heatmap:{qid}:{sid}:d    │ PNG bytes             │ 7d     │ Diff heatmap image
scene_expl:{sid}:{mv}    │ SceneExplanation      │ 30d    │ Per-scene precomputed
                         │ JSON (no heatmap)     │        │ signals
calib:{model_version}    │ CalibrationTable      │ 1h     │ Temperature scaling params
                         │ JSON                  │        │ (refreshed hourly)
axis:{model_version}     │ AxisVectors dict      │ 1h     │ Semantic axis vectors
                         │ JSON                  │        │
unc:{query_id}           │ uncertainty scalar    │ 24h    │ MC Dropout result for
                         │ float as bytes        │        │ session replay

REDIS CONFIG FOR MVP:
  maxmemory: 2GB
  maxmemory-policy: allkeys-lru  (evict least-recently-used when full)
  Connection pool: 50 connections
  Serialization: msgpack for binary data (embeddings, PNG bytes)
                 JSON for structured objects

─────────────────────────────────────────────────────────────
LEVEL 2: POSTGRESQL (Cold Cache, Permanent, Query-Based)
─────────────────────────────────────────────────────────────

What is permanently stored:
  scene_explanations table:  precomputed signals for every scene
    → axis scores, NDVI/NDWI/NDBI, road features, 18×18 attention grid
    → Never expires. Invalidated only on model version change.
  retrieval_results table:   every query's results with explanations
    → Enables session replay without re-running pipeline
    → User feedback stored here for future calibration

Cache Miss → DB Fallback order:
  1. Redis miss on "expl:{qid}:{sid}" →
     2. Check retrieval_results table (already computed in past session)
        → if found: re-populate Redis, return
     3. Check scene_explanations table (precomputed at ingestion)
        → if found: compose pairwise explanation, cache in Redis, return
     4. Run ExplanationPipeline fresh, save all levels.

─────────────────────────────────────────────────────────────
LEVEL 3: S3 / LOCAL FS (Object Storage, Immutable)
─────────────────────────────────────────────────────────────

Stored permanently:
  thumbnails/{scene_id}.jpg        → 128×128 JPEG per scene
  heatmaps/{scene_id}.png          → Full-res attention heatmap per scene
  heatmaps/{qid}_{sid}_diff.png    → Diff heatmap per (query, result) pair

Retrieval: via thumbnail_service.get_thumbnail() and storage_service.get_url()
No TTL — immutable once written.

─────────────────────────────────────────────────────────────
CACHE INVALIDATION STRATEGY
─────────────────────────────────────────────────────────────

Trigger: Model version update (new checkpoint loaded)

Actions:
  1. Flush all Redis keys matching "expl:*", "conf:*", "scene_expl:*"
     CacheService.delete_pattern("expl:*")
     CacheService.delete_pattern("scene_expl:*")
     DO NOT flush "embed:*" (embeddings still valid if dim unchanged)
     DO NOT flush "heatmap:*" (visual output unchanged by fine-tuning)

  2. Mark all scene_explanations rows as stale:
     UPDATE scene_explanations SET model_version='STALE' WHERE model_version=old

  3. Refresh axis vectors and calibration from DB:
     Redis TTL on axis:{mv} and calib:{mv} will expire naturally in 1h

  4. Queue Celery task to recompute all scene_explanations:
     workers/embedding_tasks.recompute_all_embeddings.delay(new_version)
     This takes ~20 min for 10K scenes on GPU

─────────────────────────────────────────────────────────────
CACHE HIT RATE TARGETS (MVP)
─────────────────────────────────────────────────────────────

  embed:{hash}         Target: 30% hit rate
    (same image re-uploaded in demo scenarios)
  scene_expl:{sid}     Target: 95% hit rate
    (precomputed at ingestion for all 10K scenes)
  expl:{qid}:{sid}     Target: 20% hit rate
    (demo sessions re-retrieve similar results)
  heatmap:{qid}:{sid}  Target: 40% hit rate
    (demo repeatedly shows same pairs)

Monitoring: Redis INFO stats → hit_rate exported to Prometheus.
Alert if scene_expl hit rate drops below 80% (indicates missing ingestion).
```

---

## 11. Development Order

```
═══════════════════════════════════════════════════════════════════
PHASE 0: Foundation (implement first, everything depends on this)
═══════════════════════════════════════════════════════════════════

File order:
  1. config.py                    ← all settings in one place
  2. core/exceptions.py           ← exception types needed everywhere
  3. core/logging.py              ← set up before any other code runs
  4. core/database.py             ← DB engine needed for models
  5. models/orm/base.py           ← ORM base before ORM models
  6. models/orm/scene.py          ← scenes table
  7. models/orm/embedding.py      ← embeddings table
  8. models/orm/session.py        ← retrieval sessions
  9. models/orm/result.py         ← retrieval results
  10. models/orm/explanation.py   ← scene explanations
  11. models/orm/axis_vector.py   ← semantic axis vectors
  12. migrations/versions/001-005 ← create all tables
  13. core/redis_client.py        ← Redis connection
  14. core/storage.py             ← storage abstraction
  15. core/security.py            ← API key auth
  16. core/rate_limiter.py        ← rate limiting
  17. core/metrics.py             ← Prometheus metrics
  18. core/events.py              ← startup/shutdown (stub first)

Verify: alembic upgrade head → all tables created ✓
        fastapi app starts, /health returns 200 ✓

═══════════════════════════════════════════════════════════════════
PHASE 1: Data Access Layer
═══════════════════════════════════════════════════════════════════

  19. repositories/scene_repo.py
  20. repositories/embedding_repo.py
  21. repositories/session_repo.py
  22. repositories/result_repo.py
  23. repositories/explanation_repo.py

Verify: unit tests for each repo against test DB ✓

═══════════════════════════════════════════════════════════════════
PHASE 2: ML Services (the core)
═══════════════════════════════════════════════════════════════════

  24. services/storage_service.py     ← needed by encoder + heatmap
  25. services/encoder_service.py     ← model loading + encode()
  26. services/faiss_service.py       ← index loading + search()
  27. services/thumbnail_service.py   ← thumbnail serving

Verify:
  encoder_service.encode(sample_bytes, "optical") returns [128] array ✓
  faiss_service.search(embedding, "optical", "sar", 10) returns 10 results ✓

═══════════════════════════════════════════════════════════════════
PHASE 3: Retrieval Pipeline
═══════════════════════════════════════════════════════════════════

  28. domain/retrieval/preprocessor.py
  29. domain/retrieval/router.py
  30. domain/retrieval/ranker.py
  31. domain/retrieval/pipeline.py   ← orchestrates 28-30 + services

Verify:
  RetrievalPipeline.run(bytes, "optical", "sar", 10) returns RetrievalResponse
  (without explanations yet — pass explanation_depth=None for now)

═══════════════════════════════════════════════════════════════════
PHASE 4: Pydantic Schemas + API Routes (retrieval only)
═══════════════════════════════════════════════════════════════════

  32. models/schemas/common.py
  33. models/schemas/scene.py
  34. models/schemas/retrieval.py    ← explanation/confidence fields nullable
  35. api/dependencies.py
  36. api/v1/health.py
  37. api/v1/scenes.py
  38. api/v1/retrieve.py
  39. api/v1/router.py
  40. main.py                        ← complete with all routers

Verify:
  POST /api/v1/retrieve/cross-modal with sample image returns 200 ✓
  GET /scenes/{id} returns scene metadata ✓
  GET /scenes/{id}/thumbnail returns JPEG ✓
  GET /ready returns { status: "ready" } ✓

═══════════════════════════════════════════════════════════════════
PHASE 5: Explanation Pipeline
═══════════════════════════════════════════════════════════════════

  41. domain/explanation/attention/extractor.py
  42. domain/explanation/attention/rollout.py
  43. domain/explanation/attention/heatmap.py
  44. domain/explanation/attention/alignment.py
  45. domain/explanation/axes/scorer.py
  46. domain/explanation/axes/comparator.py
  47. domain/explanation/spectral/ndvi.py
  48. domain/explanation/spectral/ndwi.py
  49. domain/explanation/spectral/ndbi.py
  50. domain/explanation/spectral/sar_indices.py
  51. domain/explanation/road/edge_detector.py
  52. domain/explanation/road/line_detector.py
  53. domain/explanation/road/similarity.py
  54. domain/explanation/narrative/categorizer.py
  55. domain/explanation/narrative/generator.py
  56. domain/explanation/pipeline.py

Verify:
  ExplanationPipeline.explain(query_emb, query_tensor, candidates, "fast")
  returns ExplanationBundle with non-null axis scores and narrative ✓

═══════════════════════════════════════════════════════════════════
PHASE 6: Confidence Pipeline
═══════════════════════════════════════════════════════════════════

  57. domain/confidence/mc_dropout.py
  58. domain/confidence/calibration.py
  59. domain/confidence/gap_scorer.py
  60. domain/confidence/composite.py
  61. domain/confidence/pipeline.py

Verify:
  ConfidencePipeline.score(query_tensor, candidates, "optical")
  returns List[ConfidenceBundle] with valid tiers ✓

═══════════════════════════════════════════════════════════════════
PHASE 7: Services for Explanation + Confidence
═══════════════════════════════════════════════════════════════════

  62. services/cache_service.py
  63. services/explanation_service.py
  64. services/confidence_service.py
  65. services/result_service.py  ← complete version (with explanation + confidence)

Verify:
  Full pipeline end-to-end: POST /retrieve/cross-modal returns response
  with populated semantic_axes, confidence, and natural_language ✓

═══════════════════════════════════════════════════════════════════
PHASE 8: Explanation API + Schemas
═══════════════════════════════════════════════════════════════════

  66. models/schemas/explanation.py
  67. models/schemas/confidence.py
  68. api/v1/explain.py
  69. api/v1/search.py              ← WebSocket interactive search
  70. api/v1/admin.py
  71. api/v1/router.py              ← update to include new routers

Verify:
  GET /explain/{qid}/{sid} returns full ExplanationBundle ✓
  GET /explain/{qid}/{sid}/heatmap/query returns PNG ✓
  WS /search/interactive: QUERY message triggers phased response ✓

═══════════════════════════════════════════════════════════════════
PHASE 9: Async Workers
═══════════════════════════════════════════════════════════════════

  72. workers/celery_app.py
  73. workers/heatmap_tasks.py
  74. workers/embedding_tasks.py
  75. workers/ingestion_tasks.py

Verify:
  celery -A workers.celery_app worker --loglevel=info starts ✓
  heatmap_tasks.generate_heatmap_pair.delay() completes in < 2s ✓

═══════════════════════════════════════════════════════════════════
PHASE 10: Tests
═══════════════════════════════════════════════════════════════════

  76. tests/conftest.py
  77. tests/unit/domain/test_retrieval_pipeline.py
  78. tests/unit/domain/test_explanation_pipeline.py
  79. tests/unit/domain/test_confidence_pipeline.py
  80. tests/unit/services/test_encoder_service.py
  81. tests/unit/services/test_faiss_service.py
  82. tests/integration/test_retrieve_endpoint.py
  83. tests/integration/test_explain_endpoint.py
  84. tests/integration/test_health_endpoint.py

═══════════════════════════════════════════════════════════════════
TOTAL: 84 files, 10 implementation phases
MVP complete at end of Phase 8 (Phases 9-10 are polish)
═══════════════════════════════════════════════════════════════════

CRITICAL PATH (must not block):
  config → database → ORM models → migrations → encoder_service →
  faiss_service → retrieval_pipeline → retrieve_endpoint → DEMO WORKS

PARALLELIZABLE (can develop concurrently once critical path done):
  Explanation pipeline and Confidence pipeline are independent.
  Workers are independent of API routes.
  Tests can be written in parallel with implementation.
```

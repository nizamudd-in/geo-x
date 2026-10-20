"""
GeoSense-X — Prometheus Metrics Registry
==========================================
Defines all Prometheus counters, histograms, and gauges.

Metrics are registered module-level so they are shared across
all requests in the same process. Import and use these instances
directly — never create new metric objects per request.

Exposed at GET /metrics in Prometheus text format.

Usage:
    from core.metrics import RETRIEVAL_LATENCY, RETRIEVAL_REQUESTS
    RETRIEVAL_REQUESTS.labels(modality="optical", status="success").inc()
    with RETRIEVAL_LATENCY.labels(step="encode").time():
        embedding = encoder.encode(image)
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Summary

# --------------------------------------------------------------------------- #
# Retrieval Metrics
# --------------------------------------------------------------------------- #

RETRIEVAL_REQUESTS = Counter(
    name="geosense_retrieval_requests_total",
    documentation="Total number of retrieval requests",
    labelnames=["source_modality", "target_modality", "status"],
)

RETRIEVAL_LATENCY = Histogram(
    name="geosense_retrieval_latency_seconds",
    documentation="Retrieval request latency broken down by pipeline step",
    labelnames=["step"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0),
)
# Steps: "encode", "faiss_search", "explanation", "confidence", "compose", "total"

# --------------------------------------------------------------------------- #
# Explanation Metrics
# --------------------------------------------------------------------------- #

EXPLANATION_REQUESTS = Counter(
    name="geosense_explanation_requests_total",
    documentation="Total number of explanation requests",
    labelnames=["depth", "cache_hit"],
)

HEATMAP_GENERATION_LATENCY = Histogram(
    name="geosense_heatmap_generation_seconds",
    documentation="Time to generate attention heatmap",
    labelnames=["mode"],  # "fast" | "full"
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

# --------------------------------------------------------------------------- #
# Confidence Metrics
# --------------------------------------------------------------------------- #

CONFIDENCE_TIER_TOTAL = Counter(
    name="geosense_confidence_tier_total",
    documentation="Count of results by confidence tier",
    labelnames=["tier", "source_modality"],
)

MC_DROPOUT_LATENCY = Histogram(
    name="geosense_mc_dropout_latency_seconds",
    documentation="MC Dropout forward pass latency",
    labelnames=["passes"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0),
)

# --------------------------------------------------------------------------- #
# Infrastructure Metrics
# --------------------------------------------------------------------------- #

FAISS_INDEX_SIZE = Gauge(
    name="geosense_faiss_index_size",
    documentation="Number of vectors in each FAISS index",
    labelnames=["modality"],
)

MODEL_LOADED = Gauge(
    name="geosense_model_loaded",
    documentation="1 if ML model is loaded and ready, 0 otherwise",
    labelnames=["model_version"],
)

CACHE_HIT_TOTAL = Counter(
    name="geosense_cache_hits_total",
    documentation="Cache hit count by cache type",
    labelnames=["cache_type"],  # embedding, explanation, heatmap, scene_expl
)

CACHE_MISS_TOTAL = Counter(
    name="geosense_cache_misses_total",
    documentation="Cache miss count by cache type",
    labelnames=["cache_type"],
)

EMBEDDING_NORM = Histogram(
    name="geosense_embedding_norm",
    documentation="L2 norm of embeddings (drift detection — should be ~1.0)",
    labelnames=["modality"],
    buckets=(0.8, 0.9, 0.95, 0.98, 0.99, 1.0, 1.01, 1.02, 1.05, 1.1, 1.2),
)

# --------------------------------------------------------------------------- #
# WebSocket Metrics
# --------------------------------------------------------------------------- #

WS_CONNECTIONS_ACTIVE = Gauge(
    name="geosense_ws_connections_active",
    documentation="Number of active WebSocket connections",
)

WS_MESSAGES_TOTAL = Counter(
    name="geosense_ws_messages_total",
    documentation="Total WebSocket messages by type",
    labelnames=["message_type"],  # QUERY, REFINE, EXPLORE_HOP, ERROR
)

# --------------------------------------------------------------------------- #
# TODO: register custom Celery task metrics
# from celery.contrib.prometheus import setup_metrics
# setup_metrics(registry=REGISTRY)
# --------------------------------------------------------------------------- #

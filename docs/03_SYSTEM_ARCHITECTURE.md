# System Architecture

## Core Retrieval Engine

Optical Images
SAR Images
Multispectral Images

↓

Preprocessing Layer

↓

DINOv2 Backbone

↓

Projection Head

↓

Contrastive Learning

↓

Unified Shared Embedding Space

↓

FAISS Vector Index

↓

Similarity Search

↓

Top-5 Retrieval
Top-10 Retrieval

---

## Intelligence Layer

### Module 1 — Explainable Retrieval

Retrieved Result
↓
Feature Attribution
↓
Reason Generator

Output:

* Similar vegetation
* Similar road network
* Similar water bodies
* Similar urban structure

---

### Module 2 — Geographic Intelligence

Ranking Score =

Visual Similarity
+
Land Cover Similarity
+
Elevation Similarity
+
Location Similarity

---

### Module 3 — Temporal Retrieval

Supports:

* Seasonal Retrieval
* Multi-temporal Search
* Time-aware Ranking

---

### Module 4 — Historical Retrieval

Current Observation
↓
2024
↓
2022
↓
2020
↓
2018

---

### Module 5 — Change Detection

Detect:

* Forest → Urban
* Agriculture → Built-up
* Water → Dry Land

---

### Module 6 — Confidence Estimation

Outputs:

* Retrieval Score
* Confidence Score
* Sensor Agreement
* Temporal Stability

---

### Module 7 — Interactive Search

Examples:

Find similar forests.

Exclude urban regions.

Find agricultural areas near water bodies.

---

### Module 8 — Multi-Hop Search

Optical
↓
SAR
↓
Multispectral
↓
Elevation
↓
Historical Records

---

## Analyst Dashboard

* Upload Query Image
* View Retrieved Results
* View Similarity Scores
* View Confidence Scores
* View Explanations
* View Geographic Locations
* View Temporal Evolution
* View Change Analysis

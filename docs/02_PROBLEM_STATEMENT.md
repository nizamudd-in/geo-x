# Existing System Limitations

## 1. No Explainable Retrieval

Current Systems:

Query Image
↓
Top 10 Results

Problem:
No explanation for retrieval decisions.

Solution:

Similarity: 92%

Reason:

* Similar vegetation density
* Similar river geometry
* Similar urban density
* Similar road structure

---

## 2. Weak Optical ↔ SAR Understanding

Optical images contain colors.

SAR images contain radar reflections.

Current systems struggle to understand that both may represent the same location.

Goal:

# Optical Forest

SAR Forest

within a shared embedding space.

---

## 3. No Temporal Awareness

Current systems fail when the same location changes across seasons.

Example:

Summer Farm
↓
Harvested Farm
↓
Flooded Farm

Goal:

Retrieve semantically similar regions despite temporal changes.

---

## 4. No Geographic Context

Current systems rely only on visual similarity.

Goal:

Visual Similarity
+
Location
+
Elevation
+
Land Cover

for intelligent ranking.

---

## 5. No Multi-Hop Retrieval

Current:

Optical
↓
Retrieve SAR

Goal:

Optical
↓
SAR
↓
Multispectral
↓
Elevation
↓
Historical Data

---

## 6. No Interactive Search

Current:

Single Query
Single Result List

Goal:

Natural Language Queries

Examples:

Find similar forests near water bodies.

Find SAR matches with dense vegetation.

Exclude urban regions.

---

## 7. No Confidence Estimation

Current:

Similarity = 0.91

Goal:

Confidence = High

Reason:

* Seasonal mismatch low
* Sensor mismatch low
* Cloud contamination low

---

## 8. No Change Detection

Current systems ignore temporal evolution.

Goal:

2018 Forest
↓
2026 Urban

Detect and explain land-cover changes.

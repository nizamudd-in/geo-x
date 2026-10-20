"""
GeoSense-X — Natural Language Explanation Generator
=====================================================
Template-based natural language generation for retrieval explanations.
Deterministic — no LLM. All output is reproducible and verifiable.
Maximum output: 3 sentences.
"""

from __future__ import annotations

from models.schemas.explanation import (
    ExplanationBundle,
    NarrativeCategoryEnum,
    RegionImportance,
    SemanticAxisBundle,
)


class NarrativeGenerator:
    """Generates natural language explanations from structured explanation data."""

    @staticmethod
    def generate(
        global_similarity: float,
        axis_bundle: SemanticAxisBundle,
        confidence_score: float,
        region_importance: list[RegionImportance],
        category: NarrativeCategoryEnum,
    ) -> str:
        """
        Generate a natural language explanation string.

        Args:
            global_similarity: Cosine similarity [0, 1].
            axis_bundle: Pairwise axis comparison data.
            confidence_score: Composite confidence score [0, 1].
            region_importance: Top-3 attended regions.
            category: Narrative category from NarrativeCategorizer.

        Returns:
            String of 1–3 sentences. No markdown formatting.
        """
        convergent = axis_bundle.convergent_axes
        divergent = axis_bundle.divergent_axes

        if category == NarrativeCategoryEnum.STRONG_MATCH:
            return NarrativeGenerator._case_a(
                global_similarity, convergent, region_importance
            )
        elif category == NarrativeCategoryEnum.PARTIAL_MATCH:
            return NarrativeGenerator._case_b(
                global_similarity, convergent, divergent, axis_bundle, region_importance
            )
        elif category == NarrativeCategoryEnum.MODERATE_MATCH:
            return NarrativeGenerator._case_c(global_similarity, convergent)
        else:
            return NarrativeGenerator._case_d(confidence_score)

    @staticmethod
    def _case_a(
        sim: float, convergent: list[str], regions: list[RegionImportance]
    ) -> str:
        """CASE A: Strong semantic match, no divergence."""
        axes_str = _oxford_join([a.replace("_", " ") for a in convergent]) or "overall features"
        sentence1 = (
            f"Retrieved because of highly similar {axes_str} "
            f"(overall similarity: {sim:.0%})."
        )
        if regions:
            r = regions[0]
            sentence2 = (
                f"Primary attention focused on {r.class_name.replace('_', ' ')} "
                f"({r.coverage_pct:.0f}% of scene)."
            )
            return f"{sentence1} {sentence2}"
        return sentence1

    @staticmethod
    def _case_b(
        sim: float,
        convergent: list[str],
        divergent: list[str],
        axis_bundle: SemanticAxisBundle,
        regions: list[RegionImportance],
    ) -> str:
        """CASE B: High similarity with notable divergence."""
        conv_str = _oxford_join([a.replace("_", " ") for a in convergent]) or "several features"
        sentence1 = (
            f"Retrieved due to shared {conv_str} characteristics "
            f"(similarity: {sim:.0%})."
        )
        if divergent:
            div_axis = divergent[0]
            # TODO: access per-axis score from bundle
            sentence2 = (
                f"However, {div_axis.replace('_', ' ')} coverage diverges significantly "
                f"between query and result."
            )
            return f"{sentence1} {sentence2}"
        return sentence1

    @staticmethod
    def _case_c(sim: float, convergent: list[str]) -> str:
        """CASE C: Moderate match."""
        conv_str = (
            _oxford_join([a.replace("_", " ") for a in convergent[:2]])
            or "some shared features"
        )
        return (
            f"Moderate overall similarity ({sim:.0%}). "
            f"Best-matching features: {conv_str}."
        )

    @staticmethod
    def _case_d(confidence: float) -> str:
        """CASE D: Uncertain match."""
        return (
            f"Retrieved with moderate similarity but high uncertainty "
            f"(confidence: {confidence:.0%}). "
            f"The query may fall near a class boundary — manual verification recommended."
        )


def _oxford_join(items: list[str]) -> str:
    """Join a list with Oxford comma: 'a, b, and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"

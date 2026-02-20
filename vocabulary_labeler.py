"""
vocabulary_labeler.py

Drop-in module for centroid-based cluster labeling using the pre-computed
OpenAlex vocabulary embeddings. Designed to integrate cleanly with your
existing KeyBERT-based labeling pipeline.

Key design:
  - Loads once at startup, fast cosine lookup at label time
  - Can be disabled via USE_VOCABULARY_LABELS = False
  - Falls back gracefully to KeyBERT if vocab file not found
  - Returns top-N candidates with scores for debugging

Usage in your pipeline:
    from vocabulary_labeler import VocabularyLabeler

    # Initialize once (loads embeddings into memory)
    labeler = VocabularyLabeler("data/vocab_embeddings.npz")

    # Label a cluster given its centroid embedding (768-dim SPECTER2 vector)
    label, score, candidates = labeler.label_cluster(centroid_embedding)
    # label: "Reinforcement Learning"
    # score: 0.87 (cosine similarity)
    # candidates: [("Reinforcement Learning", 0.87), ("Multi-Agent Systems", 0.82), ...]
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────
# Feature flag — set to False to fall back to KeyBERT for all clusters
# Can also be overridden at runtime: labeler.enabled = False
# ─────────────────────────────────────────────────────────────────
USE_VOCABULARY_LABELS = True

# Minimum cosine similarity to accept a vocabulary match.
# Below this threshold, fall back to KeyBERT.
# Tuning guide:
#   0.85+ : Very confident match, nearly always correct
#   0.75  : Good match, occasional misses for niche topics (recommended default)
#   0.65  : Permissive, accepts weaker matches — may label oddly specific clusters
#   0.60- : Too low, produces generic-looking labels
MIN_CONFIDENCE = 0.75


@dataclass
class LabelResult:
    label: str
    score: float
    source: str  # "vocabulary" or "fallback"
    candidates: list  # [(label, score), ...] top-N matches


class VocabularyLabeler:
    """Fast vocabulary-based cluster labeler using pre-computed SPECTER2 embeddings."""

    def __init__(
        self,
        embeddings_path: str = "data/vocab_embeddings.npz",
        enabled: bool = USE_VOCABULARY_LABELS,
        min_confidence: float = MIN_CONFIDENCE,
        top_n: int = 5,
    ):
        self.enabled = enabled
        self.min_confidence = min_confidence
        self.top_n = top_n
        self._loaded = False

        if self.enabled:
            self._load(embeddings_path)

    def _load(self, path: str) -> None:
        """Load pre-computed vocabulary embeddings."""
        p = Path(path)
        if not p.exists():
            print(f"[VocabularyLabeler] WARNING: Embeddings not found at {path}")
            print("[VocabularyLabeler] Falling back to KeyBERT for all clusters.")
            print("[VocabularyLabeler] Run embed_vocabulary.py to generate embeddings.")
            self.enabled = False
            return

        data = np.load(p, allow_pickle=True)
        self.embeddings = data["embeddings"].astype(np.float32)   # (N, 768) already L2-normalized
        self.labels = data["labels"].tolist()
        self.subfields = data["subfields"].tolist()
        self._loaded = True

        print(f"[VocabularyLabeler] Loaded {len(self.labels)} vocabulary entries from {path}")

    def label_cluster(self, centroid: np.ndarray) -> LabelResult:
        """
        Label a cluster given its centroid embedding.

        Args:
            centroid: 768-dim SPECTER2 embedding of the cluster centroid
                      (mean of all paper embeddings in the cluster).
                      Should be L2-normalized for cosine similarity.

        Returns:
            LabelResult with label, confidence score, and top candidates.
        """
        if not self.enabled or not self._loaded:
            return LabelResult(
                label=None,
                score=0.0,
                source="fallback",
                candidates=[],
            )

        # Normalize centroid (defensive — in case caller didn't)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        # Cosine similarity = dot product (embeddings are pre-normalized)
        scores = self.embeddings @ centroid.astype(np.float32)

        # Top-N matches
        top_indices = np.argsort(scores)[::-1][: self.top_n]
        candidates = [(self.labels[i], float(scores[i])) for i in top_indices]

        best_label, best_score = candidates[0]

        # Confidence check
        if best_score < self.min_confidence:
            return LabelResult(
                label=None,
                score=best_score,
                source="fallback",
                candidates=candidates,
            )

        return LabelResult(
            label=best_label,
            score=best_score,
            source="vocabulary",
            candidates=candidates,
        )

    def label_clusters(self, centroids: dict) -> dict:
        """
        Label multiple clusters at once.

        Args:
            centroids: {cluster_id: centroid_embedding}

        Returns:
            {cluster_id: LabelResult}
        """
        return {cid: self.label_cluster(c) for cid, c in centroids.items()}

    def stats(self) -> dict:
        """Return summary stats about the loaded vocabulary."""
        if not self._loaded:
            return {"loaded": False}

        from collections import Counter
        sf_counts = Counter(self.subfields)
        return {
            "loaded": True,
            "total_entries": len(self.labels),
            "subfield_counts": dict(sf_counts.most_common()),
        }


# ─────────────────────────────────────────────────────────────────
# Integration helper: wraps your existing label_clusters function
# ─────────────────────────────────────────────────────────────────

def label_clusters_with_vocab_fallback(
    cluster_embeddings: dict,
    vocab_labeler: VocabularyLabeler,
    keybert_fallback_fn,
    verbose: bool = False,
) -> dict:
    """
    Label clusters using vocabulary first, falling back to KeyBERT.

    Args:
        cluster_embeddings: {cluster_id: (centroid_embedding, paper_texts)}
                            centroid_embedding: 768-dim SPECTER2 mean vector
                            paper_texts: list of title+abstract strings
        vocab_labeler: VocabularyLabeler instance
        keybert_fallback_fn: your existing KeyBERT labeling function
                             signature: fn(paper_texts) -> str
        verbose: print labeling decisions

    Returns:
        {cluster_id: {"label": str, "source": str, "score": float}}
    """
    results = {}
    vocab_count = 0
    fallback_count = 0

    for cluster_id, (centroid, paper_texts) in cluster_embeddings.items():
        result = vocab_labeler.label_cluster(centroid)

        if result.source == "vocabulary":
            label = result.label
            vocab_count += 1
            if verbose:
                runner_up = result.candidates[1] if len(result.candidates) > 1 else None
                print(f"  Cluster {cluster_id}: '{label}' (vocab, score={result.score:.3f})")
                if runner_up:
                    print(f"    Runner-up: '{runner_up[0]}' ({runner_up[1]:.3f})")
        else:
            # Fall back to KeyBERT
            label = keybert_fallback_fn(paper_texts)
            fallback_count += 1
            if verbose:
                print(
                    f"  Cluster {cluster_id}: '{label}' (keybert fallback, "
                    f"best_vocab_score={result.score:.3f})"
                )

        results[cluster_id] = {
            "label": label,
            "source": result.source,
            "score": result.score,
            "candidates": result.candidates,
        }

    if verbose:
        print(f"\nLabeling summary: {vocab_count} vocabulary, {fallback_count} KeyBERT fallback")

    return results

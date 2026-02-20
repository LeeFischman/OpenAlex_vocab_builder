# Vocabulary-Based Cluster Labeling (Option 2)

## Overview

Instead of extracting keywords from cluster papers, this approach maintains a vocabulary
of known research area phrases from OpenAlex and finds the closest match to each cluster
centroid. Produces cleaner, more consistent labels at the cost of a one-time setup step.

## Files

| File | Purpose |
|------|---------|
| `build_openalex_vocabulary.py` | Fetches CS/AI topics from OpenAlex API → `data/openalex_vocab.json` |
| `embed_vocabulary.py` | Pre-computes SPECTER2 embeddings → `data/vocab_embeddings.npz` |
| `vocabulary_labeler.py` | Drop-in module for centroid matching at pipeline runtime |

## Setup (one-time)

```bash
# Step 1: Fetch vocabulary from OpenAlex (~300 CS/AI topics, no API key needed)
python build_openalex_vocabulary.py --subfields cs_ai

# Step 2: Pre-compute SPECTER2 embeddings (runs once, ~2-3 min on CPU)
python embed_vocabulary.py --text-field combined

# Step 3: Verify it worked
python -c "
from vocabulary_labeler import VocabularyLabeler
vl = VocabularyLabeler()
import json; print(json.dumps(vl.stats(), indent=2))
"
```

## Integration into your pipeline

```python
from vocabulary_labeler import VocabularyLabeler, label_clusters_with_vocab_fallback

# Initialize once at startup
vocab_labeler = VocabularyLabeler(
    embeddings_path="data/vocab_embeddings.npz",
    enabled=True,          # Set False to disable and use KeyBERT for everything
    min_confidence=0.75,   # Fall back to KeyBERT below this cosine similarity
)

# Compute centroids (mean of paper embeddings per cluster)
centroids = {}
for cluster_id, paper_indices in cluster_assignments.items():
    cluster_vecs = high_dim_embeddings[paper_indices]  # shape: (n_papers, 768)
    centroid = cluster_vecs.mean(axis=0)
    centroids[cluster_id] = centroid

# Label using vocabulary, fall back to KeyBERT for low-confidence clusters
labels = label_clusters_with_vocab_fallback(
    cluster_embeddings={cid: (centroid, paper_texts[cid]) for cid, centroid in centroids.items()},
    vocab_labeler=vocab_labeler,
    keybert_fallback_fn=your_existing_keybert_function,
    verbose=True,
)
```

## Disabling

To disable entirely and revert to KeyBERT:

```python
# Option A: Environment variable (easy for A/B testing)
import os
os.environ["USE_VOCABULARY_LABELS"] = "false"

# Option B: At initialization
vocab_labeler = VocabularyLabeler(enabled=False)

# Option C: In vocabulary_labeler.py — change the flag at the top
USE_VOCABULARY_LABELS = False
```

## Subfield scopes

| Scope | Subfields | Approx. topics | Use when |
|-------|-----------|---------------|---------|
| `ai_only` | Artificial Intelligence, Computer Vision | ~100 | Pure cs.AI/cs.CV corpus |
| `cs_ai` (default) | AI + theory + networks + HCI + signal + software | ~300 | arXiv cs.* papers |
| `all_cs` | All 13 CS subfields | ~500 | Broader CS corpus |

## Tuning min_confidence

| Value | Behavior |
|-------|---------|
| `0.85` | Conservative — only very clear matches, many KeyBERT fallbacks |
| `0.75` | Recommended default — good accuracy, occasional fallbacks |
| `0.65` | Permissive — fewer fallbacks, may mislabel niche topics |

To find the right threshold for your corpus, run in verbose mode and inspect
clusters where the vocabulary score is between 0.65–0.80.

## GitHub Actions

Add to your workflow so the vocabulary gets built automatically on first run:

```yaml
- name: Build vocabulary embeddings (if not cached)
  run: |
    if [ ! -f data/vocab_embeddings.npz ]; then
      python build_openalex_vocabulary.py --subfields cs_ai
      python embed_vocabulary.py
    fi
```

Cache the `data/vocab_embeddings.npz` file in GitHub Actions to avoid rebuilding on every run:

```yaml
- uses: actions/cache@v3
  with:
    path: data/vocab_embeddings.npz
    key: vocab-embeddings-v1
```

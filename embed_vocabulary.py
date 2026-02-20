"""
embed_vocabulary.py

Pre-computes SPECTER2 embeddings for all vocabulary entries and saves them
as a numpy array. This runs once (or when the vocabulary is updated) and
produces the fast lookup index used at cluster-labeling time.

Usage:
    python embed_vocabulary.py
    python embed_vocabulary.py --vocab data/openalex_vocab.json --output data/vocab_embeddings.npz
    python embed_vocabulary.py --text-field display_name          # embed topic names only
    python embed_vocabulary.py --text-field keywords              # embed expanded keyword phrases
    python embed_vocabulary.py --text-field combined              # embed "name: kw1, kw2..." (default)
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

# ─────────────────────────────────────────────────────────────────
# Text construction strategy
#
# Three options for what text we embed per vocabulary entry:
#
#   display_name: Just the topic name, e.g. "Reinforcement Learning"
#       Pros: Clean, unambiguous. Cons: Short = less context for embedding.
#
#   keywords: Each keyword as a separate embedding, keep the best match.
#       Pros: Fine-grained. Cons: More embeddings to store and search.
#
#   combined (default): "Topic Name: kw1, kw2, kw3, ..."
#       Pros: Richer context improves embedding quality. One vector per topic.
#       Cons: Keywords can add noise for narrow topics.
#
# Recommendation: use "combined" — SPECTER2 is trained on scientific text
# so a title-like combined string works well and stays single-vector-per-topic.
# ─────────────────────────────────────────────────────────────────

def make_text(entry: dict, text_field: str) -> str:
    """Build the text string to embed for a vocabulary entry."""
    name = entry["display_name"]
    keywords = entry.get("keywords", [])

    if text_field == "display_name":
        return name

    elif text_field == "keywords":
        # Join keywords as a phrase; fall back to display_name if empty
        if keywords:
            return "; ".join(keywords)
        return name

    elif text_field == "combined":
        # "Topic Name: keyword1; keyword2; keyword3"
        if keywords:
            return f"{name}: {'; '.join(keywords)}"
        return name

    else:
        raise ValueError(f"Unknown text_field: {text_field}")


def embed_texts(texts: list[str], model, tokenizer, batch_size: int = 32, device: str = "cpu") -> np.ndarray:
    """Embed a list of texts using SPECTER2 in batches."""
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding vocabulary"):
        batch = texts[i : i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # SPECTER2: use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
            # L2-normalize for cosine similarity
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute SPECTER2 embeddings for vocabulary")
    parser.add_argument("--vocab", default="data/openalex_vocab.json", help="Input vocabulary JSON")
    parser.add_argument("--output", default="data/vocab_embeddings.npz", help="Output embeddings file")
    parser.add_argument(
        "--text-field",
        choices=["display_name", "keywords", "combined"],
        default="combined",
        help="Text representation strategy (default: combined)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding")
    parser.add_argument("--model", default="allenai/specter2_base", help="HuggingFace model to use")
    args = parser.parse_args()

    # Load vocabulary
    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}\nRun build_openalex_vocabulary.py first.")

    with open(vocab_path) as f:
        vocab = json.load(f)

    topics = vocab["topics"]
    print(f"Loaded {len(topics)} topics from {vocab_path}")

    # Build texts
    texts = [make_text(t, args.text_field) for t in topics]
    labels = [t["display_name"] for t in topics]
    subfields = [t["subfield"] for t in topics]

    print(f"\nText strategy: {args.text_field}")
    print(f"Example texts:")
    for t in texts[:3]:
        print(f"  {t[:100]}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nLoading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    # Embed
    embeddings = embed_texts(texts, model, tokenizer, batch_size=args.batch_size, device=device)
    print(f"\nEmbeddings shape: {embeddings.shape}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        labels=np.array(labels),
        subfields=np.array(subfields),
        text_field=np.array(args.text_field),
    )

    print(f"Saved embeddings to: {output_path}")
    print(f"\nFile size: {output_path.stat().st_size / 1024:.1f} KB")
    print("Next step: the main pipeline will load this file for fast centroid matching")


if __name__ == "__main__":
    main()

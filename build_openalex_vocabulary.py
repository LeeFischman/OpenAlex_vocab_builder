"""
build_openalex_vocabulary.py

One-time script to fetch CS/AI topics from OpenAlex and build a vocabulary
file for centroid-based cluster labeling. Run this once to generate
data/openalex_vocab.json, which is then used by the main pipeline.

Usage:
    python build_openalex_vocabulary.py
    python build_openalex_vocabulary.py --output data/openalex_vocab.json
    python build_openalex_vocabulary.py --subfields ai_only  # Narrowest scope
    python build_openalex_vocabulary.py --subfields cs_ai    # Default: CS + AI subfields
    python build_openalex_vocabulary.py --subfields all_cs   # All Computer Science
"""

import json
import time
import argparse
import requests
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Subfield configuration
# OpenAlex uses Scopus ASJC codes for CS subfields (field_id = 17)
# ─────────────────────────────────────────────────────────────────

# Core AI/ML subfields — tightly scoped for cs.AI papers
AI_ONLY_SUBFIELDS = {
    1702: "Artificial Intelligence",
    1707: "Computer Vision and Pattern Recognition",
}

# Broader CS + AI subfields — good default for arXiv cs.* papers
CS_AI_SUBFIELDS = {
    1702: "Artificial Intelligence",
    1703: "Computational Theory and Mathematics",
    1705: "Computer Networks and Communications",
    1707: "Computer Vision and Pattern Recognition",
    1708: "Hardware and Architecture",
    1709: "Human-Computer Interaction",
    1711: "Signal Processing",
    1712: "Software",
}

# All CS subfields — most inclusive
ALL_CS_SUBFIELDS = {
    1700: "General Computer Science",
    1701: "Computer Science Applications",
    1702: "Artificial Intelligence",
    1703: "Computational Theory and Mathematics",
    1704: "Computer Graphics and Computer-Aided Design",
    1705: "Computer Networks and Communications",
    1706: "Computer Science (miscellaneous)",
    1707: "Computer Vision and Pattern Recognition",
    1708: "Hardware and Architecture",
    1709: "Human-Computer Interaction",
    1710: "Information Systems",
    1711: "Signal Processing",
    1712: "Software",
}

SUBFIELD_PRESETS = {
    "ai_only": AI_ONLY_SUBFIELDS,
    "cs_ai": CS_AI_SUBFIELDS,
    "all_cs": ALL_CS_SUBFIELDS,
}

BASE_URL = "https://api.openalex.org/topics"
EMAIL = "lee.fischman@gmail.com"  # Polite pool: include your email for faster rate limits


def fetch_topics_for_subfield(subfield_id: int, subfield_name: str, email: str) -> list[dict]:
    """Fetch all topics for a given subfield ID from OpenAlex."""
    topics = []
    page = 1
    per_page = 200

    print(f"  Fetching subfield {subfield_id}: {subfield_name}...", end="", flush=True)

    while True:
        params = {
            "filter": f"subfield.id:{subfield_id}",
            "per_page": per_page,
            "page": page,
            "select": "id,display_name,subfield,field,keywords,works_count",
            "mailto": email,
        }

        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"\n    Error fetching page {page}: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        topics.extend(results)

        total = data["meta"]["count"]
        if len(topics) >= total:
            break

        page += 1
        time.sleep(0.1)  # Be polite to the API

    print(f" {len(topics)} topics")
    return topics


def build_vocabulary(subfields: dict, email: str) -> dict:
    """Fetch all topics and build the vocabulary structure."""
    all_topics = []
    seen_ids = set()

    for subfield_id, subfield_name in subfields.items():
        topics = fetch_topics_for_subfield(subfield_id, subfield_name, email)
        for t in topics:
            topic_id = t["id"]
            if topic_id not in seen_ids:
                seen_ids.add(topic_id)
                all_topics.append(t)

    print(f"\nTotal unique topics fetched: {len(all_topics)}")

    # Build clean vocabulary structure
    vocab_entries = []
    for t in all_topics:
        entry = {
            "topic_id": t["id"],
            "display_name": t["display_name"],
            "subfield": t.get("subfield", {}).get("display_name", ""),
            "subfield_id": t.get("subfield", {}).get("id", ""),
            "field": t.get("field", {}).get("display_name", ""),
            "keywords": t.get("keywords", []),
            "works_count": t.get("works_count", 0),
        }
        vocab_entries.append(entry)

    # Sort by works_count descending — most active topics first
    vocab_entries.sort(key=lambda x: x["works_count"], reverse=True)

    return {
        "metadata": {
            "source": "OpenAlex Topics API",
            "subfields_included": list(subfields.values()),
            "total_topics": len(vocab_entries),
            "description": "Vocabulary for centroid-based cluster labeling in AI Research Atlas",
        },
        "topics": vocab_entries,
    }


def print_preview(vocab: dict, n: int = 20) -> None:
    """Print a preview of the vocabulary."""
    print(f"\n{'─'*60}")
    print(f"VOCABULARY PREVIEW (top {n} by works_count)")
    print(f"{'─'*60}")
    for entry in vocab["topics"][:n]:
        kw_preview = "; ".join(entry["keywords"][:4]) if entry["keywords"] else "—"
        print(f"  [{entry['subfield'][:25]:<25}] {entry['display_name']}")
        print(f"    Keywords: {kw_preview}")
    print(f"{'─'*60}")
    print(f"Total: {vocab['metadata']['total_topics']} topics")


def main():
    parser = argparse.ArgumentParser(description="Build OpenAlex vocabulary for cluster labeling")
    parser.add_argument("--output", default="data/openalex_vocab.json", help="Output JSON file path")
    parser.add_argument(
        "--subfields",
        choices=["ai_only", "cs_ai", "all_cs"],
        default="cs_ai",
        help="Subfield scope: ai_only (~100 topics), cs_ai (~300, default), all_cs (~500)",
    )
    parser.add_argument("--email", default=EMAIL, help="Email for OpenAlex polite pool")
    parser.add_argument("--preview", action="store_true", default=True, help="Print vocabulary preview")
    args = parser.parse_args()

    subfields = SUBFIELD_PRESETS[args.subfields]
    print(f"Building vocabulary with scope: {args.subfields}")
    print(f"Including {len(subfields)} subfields:")
    for sid, sname in subfields.items():
        print(f"  {sid}: {sname}")
    print()

    vocab = build_vocabulary(subfields, args.email)

    if args.preview:
        print_preview(vocab)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"\nSaved vocabulary to: {output_path}")
    print("Next step: run embed_vocabulary.py to pre-compute SPECTER2 embeddings")


if __name__ == "__main__":
    main()

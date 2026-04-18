"""
DVC Stage: download_data
════════════════════════
Downloads the full PlantDoc dataset from Hugging Face and assigns a
binary label to every image:

  HEALTHY  (0) — label exactly matches a known healthy-leaf string
                 (e.g. "Tomato leaf", "Cherry leaf")
  DISEASED (1) — any other label (disease name, pest, blight, rust, …)

An image with multiple annotations is labelled DISEASED if ANY annotation
is a disease/pest label.

Outputs
───────
data/raw/
  images/          <N>.jpg  (all images from both train and test splits)
  metadata.json    per-image records + dataset-level statistics
"""

import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def assign_binary_label(categories: list[str], healthy_keywords: list[str]) -> int:
    """
    Return 0 (HEALTHY) only when every category in the image is a
    known healthy-leaf string.  Return 1 (DISEASED) otherwise.
    """
    for cat in categories:
        if cat not in healthy_keywords:
            return 1  # at least one diseased / unknown annotation
    return 0


def download(output_dir: str, params_path: str = "params.yaml") -> None:
    p = load_params(params_path)
    dp = p["data"]
    out = Path(output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    healthy_keywords: list[str] = dp["healthy_keywords"]

    # ── Load both splits ────────────────────────────────────────────────────
    print(f"Loading {dp['dataset_name']} from Hugging Face (train + test)...")
    ds_train = load_dataset(dp["dataset_name"], split="train")
    ds_test = load_dataset(dp["dataset_name"], split="test")

    # Merge — we handle our own splits in the next stage
    from datasets import concatenate_datasets

    dataset = concatenate_datasets([ds_train, ds_test])
    print(f"Total images: {len(dataset)}")

    # ── Discover all label strings ──────────────────────────────────────────
    all_labels: set[str] = set(
        lbl for row in dataset["objects"] for lbl in row["category"]
    )
    diseased_labels = sorted(l for l in all_labels if l not in healthy_keywords)
    print(f"All label classes ({len(all_labels)}): {sorted(all_labels)}")
    print(f"\nHealthy labels  ({len(healthy_keywords)}): {healthy_keywords}")
    print(f"Diseased labels ({len(diseased_labels)}): {diseased_labels}")

    # ── Save each image and compute binary label ────────────────────────────
    metadata: dict[str, dict] = {}
    label_counts = {0: 0, 1: 0}
    label_names = {0: "HEALTHY", 1: "DISEASED"}

    for i, example in enumerate(tqdm(dataset, desc="Saving images")):
        img: Image.Image = example["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        fname = f"img_{i:05d}.jpg"
        img.save(str(img_dir / fname), "JPEG", quality=95)

        cats = example["objects"]["category"]
        label_int = assign_binary_label(cats, healthy_keywords)
        label_counts[label_int] += 1

        metadata[fname] = {
            "label": label_int,
            "label_name": label_names[label_int],
            "original_labels": cats,
        }

    summary = {
        "total_images": len(dataset),
        "label_map": {"HEALTHY": 0, "DISEASED": 1},
        "class_counts": {
            "HEALTHY": label_counts[0],
            "DISEASED": label_counts[1],
        },
        "healthy_keywords": healthy_keywords,
        "all_labels": sorted(all_labels),
        "samples": metadata,
    }

    with open(out / "metadata.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'─'*50}")
    print(f"Saved {len(dataset)} images to {img_dir}")
    print(f"  HEALTHY  : {label_counts[0]}")
    print(f"  DISEASED : {label_counts[1]}")
    balance = label_counts[1] / len(dataset) * 100
    print(f"  Class balance: {balance:.1f}% diseased")
    print(f"Metadata → {out / 'metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    download(args.output_dir, args.params)

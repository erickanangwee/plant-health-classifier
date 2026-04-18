"""
DVC Stage: prepare_data
════════════════════════
Extracts EfficientNet-B0 features from every image and produces
stratified train / val / test numpy arrays.

Design decisions
────────────────
- EfficientNet-B0 global-average-pool layer → 1,280-dim float32 vector.
- Weights frozen (ImageNet pre-training only); no fine-tuning.
- StandardScaler is fit ONLY on the training set and saved to disk.
  It is then applied to val and test — preventing data leakage.
- Stratified splits preserve the HEALTHY/DISEASED ratio in each subset.

Outputs
───────
data/processed/
  X_train.npy  y_train.npy
  X_val.npy    y_val.npy
  X_test.npy   y_test.npy
  scaler.joblib
  split_stats.json
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_efficientnet_extractor(image_size: int, batch_size: int):
    """
    Returns a function that accepts a list of PIL Images and returns
    a (N, 1280) float32 numpy array of EfficientNet-B0 embeddings.

    Processing is done in batches to avoid OOM on CPU.
    """
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T

    weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = tvm.efficientnet_b0(weights=weights)

    # Remove the classifier head — keep global average pool output (1280-dim)
    model.classifier = torch.nn.Identity()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"EfficientNet-B0 running on: {device}")

    preprocess = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def extract_batch(images: list) -> np.ndarray:
        """Extract features for a list of PIL Images."""
        results = []
        for start in range(0, len(images), batch_size):
            batch_imgs = images[start : start + batch_size]
            tensors = torch.stack([preprocess(img) for img in batch_imgs])
            tensors = tensors.to(device)
            with torch.no_grad():
                feats = model(tensors)
            results.append(feats.cpu().numpy())
        return np.concatenate(results, axis=0).astype(np.float32)

    return extract_batch


def prepare(raw_dir: str, processed_dir: str, params_path: str = "params.yaml") -> None:
    p = load_params(params_path)
    dp = p["data"]
    fp = p["features"]
    raw = Path(raw_dir)
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(raw / "metadata.json") as f:
        meta = json.load(f)

    samples = meta["samples"]
    fnames = sorted(samples.keys())
    labels = np.array([samples[fn]["label"] for fn in fnames], dtype=np.int32)

    print(f"Total samples: {len(fnames)}")
    print(f"  HEALTHY : {int(np.sum(labels == 0))}")
    print(f"  DISEASED: {int(np.sum(labels == 1))}")

    # ── Load all images
    print("\nLoading images from disk...")
    images = [
        Image.open(raw / "images" / fn).convert("RGB")
        for fn in tqdm(fnames, desc="Loading")
    ]

    # ── Extract EfficientNet-B0 features
    print("\nExtracting EfficientNet-B0 features...")
    extract = build_efficientnet_extractor(fp["image_size"], fp["batch_size"])
    X = extract(images)
    print(f"Feature matrix: {X.shape}  dtype={X.dtype}")

    # ── Stratified splits
    seed = dp["seed"]
    t_r = dp["train_ratio"]
    v_r = dp["val_ratio"]
    rest_r = round(1.0 - t_r, 4)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        labels,
        test_size=rest_r,
        stratify=labels,
        random_state=seed,
    )
    # Split the remainder evenly into val and test
    val_frac = round(v_r / rest_r, 4)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=round(1.0 - val_frac, 4),
        stratify=y_tmp,
        random_state=seed,
    )

    print("\nSplit sizes:")
    print(
        f"  Train : {len(X_train):>5}  "
        f"({int(np.sum(y_train==0))} healthy / {int(np.sum(y_train==1))} diseased)"
    )
    print(
        f"  Val   : {len(X_val):>5}  "
        f"({int(np.sum(y_val==0))} healthy / {int(np.sum(y_val==1))} diseased)"
    )
    print(
        f"  Test  : {len(X_test):>5}  "
        f"({int(np.sum(y_test==0))} healthy / {int(np.sum(y_test==1))} diseased)"
    )

    # ── Fit scaler on TRAINING SET ONLY
    if fp["normalize"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        joblib.dump(scaler, out / "scaler.joblib")
        print("\nStandardScaler fitted on train, saved to processed/scaler.joblib")
    else:
        joblib.dump(None, out / "scaler.joblib")

    # ── Save arrays
    np.save(out / "X_train.npy", X_train)
    np.save(out / "y_train.npy", y_train)
    np.save(out / "X_val.npy", X_val)
    np.save(out / "y_val.npy", y_val)
    np.save(out / "X_test.npy", X_test)
    np.save(out / "y_test.npy", y_test)

    stats = {
        "backbone": fp["backbone"],
        "feature_dim": int(X.shape[1]),
        "total_samples": int(len(X)),
        "train": int(len(X_train)),
        "val": int(len(X_val)),
        "test": int(len(X_test)),
        "class_balance": {
            "train_healthy": int(np.sum(y_train == 0)),
            "train_diseased": int(np.sum(y_train == 1)),
        },
    }
    with open(out / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nAll processed arrays saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    prepare(args.raw_dir, args.processed_dir, args.params)

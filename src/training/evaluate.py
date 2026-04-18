"""
DVC Stage: evaluate
====================
Evaluates all trained models on the held-out TEST set (never seen during
training or tuning).  Selects the champion by highest test-set F1-weighted
score.  Also computes the leaf centroid vector used by the API's leaf guard.

Champion selection criteria
----------------------------
Primary   : f1_weighted on test set   (handles class imbalance)
Tiebreaker: roc_auc on test set

Outputs
--------
models/champion/
  best_model.joblib
  scaler.joblib         (copied from data/processed/)
  champion_info.json
  leaf_centroid.npy     (mean EfficientNet-B0 embedding of training images)
models/evaluation/
  test_metrics.json
  classification_report.txt
"""

import json
import shutil
import argparse
import yaml
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score, confusion_matrix
)


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_leaf_centroid(raw_dir: Path, params: dict) -> np.ndarray:
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T
    from PIL import Image
    from tqdm import tqdm

    fp       = params["features"]
    img_size = fp["image_size"]
    bsz      = fp["batch_size"]

    weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = tvm.efficientnet_b0(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_files  = sorted((raw_dir / "images").glob("*.jpg"))
    embeddings = []

    for start in tqdm(range(0, len(img_files), bsz), desc="Computing centroid"):
        batch = img_files[start : start + bsz]
        imgs  = [Image.open(fp).convert("RGB") for fp in batch]
        tensors = torch.stack([transform(im) for im in imgs]).to(device)
        with torch.no_grad():
            embs = model(tensors).cpu().numpy()
        embeddings.append(embs)

    centroid = np.mean(np.concatenate(embeddings, axis=0), axis=0)
    print(f"Centroid computed from {len(img_files)} images  shape: {centroid.shape}")
    return centroid.astype(np.float32)


def evaluate(
    models_dir: str, processed_dir: str, raw_dir: str,
    output_dir: str, params_path: str = "params.yaml",
) -> None:
    p    = load_params(params_path)
    mdir = Path(models_dir)
    prc  = Path(processed_dir)
    raw  = Path(raw_dir)
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    champ_dir = mdir / "champion"
    champ_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(p["mlflow"]["tracking_uri"])
    mlflow.set_experiment(p["mlflow"]["experiment_name"])

    X_test = np.load(prc / "X_test.npy")
    y_test = np.load(prc / "y_test.npy")
    print(f"Test set: {X_test.shape[0]} samples  "
          f"({int(np.sum(y_test==0))} healthy / {int(np.sum(y_test==1))} diseased)")

    # Evaluate every model on the test set
    model_paths = {
        "LogisticRegression": mdir / "logisticregression" / "best_model.joblib",
        "RandomForest":       mdir / "randomforest"       / "best_model.joblib",
        "XGBoost":            mdir / "xgboost"            / "best_model.joblib",
    }

    test_results: dict[str, dict] = {}
    for name, weight_path in model_paths.items():
        if not weight_path.exists():
            print(f"  Skipping {name} - not found at {weight_path}")
            continue
        model   = joblib.load(weight_path)
        preds   = model.predict(X_test)
        probas  = model.predict_proba(X_test)[:, 1]
        f1      = f1_score(y_test, preds, average="weighted", zero_division=0)
        auc     = roc_auc_score(y_test, probas)
        test_results[name] = {
            "f1_weighted": float(f1),
            "roc_auc":     float(auc),
            "report_dict": classification_report(
                y_test, preds,
                target_names=["HEALTHY", "DISEASED"],
                output_dict=True, zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }
        print(f"  {name:<25}  Test F1: {f1:.4f}  AUC: {auc:.4f}")

    if not test_results:
        raise RuntimeError("No trained models found. Run train_models stage first.")

    # Select champion
    champion_name = max(
        test_results,
        key=lambda n: (test_results[n]["f1_weighted"], test_results[n]["roc_auc"]),
    )
    champ_m = test_results[champion_name]
    print(f"\nChampion: {champion_name}  "
          f"(F1={champ_m['f1_weighted']:.4f}, AUC={champ_m['roc_auc']:.4f})")

    shutil.copy(model_paths[champion_name], champ_dir / "best_model.joblib")
    shutil.copy(prc / "scaler.joblib",      champ_dir / "scaler.joblib")

    champion_info = {
        "champion_model":    champion_name,
        "test_f1_weighted":  champ_m["f1_weighted"],
        "test_roc_auc":      champ_m["roc_auc"],
        "all_test_results": {
            n: {"f1_weighted": v["f1_weighted"], "roc_auc": v["roc_auc"]}
            for n, v in test_results.items()
        },
    }
    with open(champ_dir / "champion_info.json", "w") as f:
        json.dump(champion_info, f, indent=2)

    champ_model = joblib.load(champ_dir / "best_model.joblib")
    with mlflow.start_run(run_name=f"Champion-{champion_name}"):
        mlflow.log_params({"champion_model": champion_name})
        mlflow.log_metric("test_f1_weighted", champ_m["f1_weighted"])
        mlflow.log_metric("test_roc_auc",     champ_m["roc_auc"])
        mlflow.sklearn.log_model(
            champ_model,
            artifact_path="champion_model",
            registered_model_name=p["mlflow"]["model_name"],
        )

    print("\nComputing leaf centroid for the leaf guard...")
    centroid = compute_leaf_centroid(raw, p)
    np.save(champ_dir / "leaf_centroid.npy", centroid)

    with open(out / "test_metrics.json", "w") as f:
        json.dump(test_results, f, indent=2)

    report_txt = ""
    for name, res in test_results.items():
        model  = joblib.load(model_paths[name])
        preds  = model.predict(X_test)
        report_txt += (f"\n{'='*55}\n{name}\n"
                       f"F1={res['f1_weighted']:.4f}  AUC={res['roc_auc']:.4f}\n"
                       f"{'-'*55}\n")
        report_txt += classification_report(
            y_test, preds, target_names=["HEALTHY", "DISEASED"], zero_division=0
        )
    with open(out / "classification_report.txt", "w") as f:
        f.write(report_txt)

    print(f"\nEvaluation outputs saved to {out}")
    print(f"Champion info saved to {champ_dir / 'champion_info.json'}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all trained models and select the champion."
    )
    parser.add_argument("--models-dir",    default="models")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--raw-dir",       default="data/raw")
    parser.add_argument("--output-dir",    default="models/evaluation")
    parser.add_argument("--params",        default="params.yaml")
    args = parser.parse_args()

    evaluate(
        models_dir    = args.models_dir,
        processed_dir = args.processed_dir,
        raw_dir       = args.raw_dir,
        output_dir    = args.output_dir,
        params_path   = args.params,
    )


if __name__ == "__main__":
    main()
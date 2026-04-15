"""
DVC Stage: train_models
════════════════════════
Trains Logistic Regression, Random Forest, and XGBoost on EfficientNet-B0
features.  Each model is independently tuned with Optuna (TPE sampler,
stratified k-fold cross-validation on the TRAINING set only).

Every Optuna trial is logged to MLflow as a child run.  The best trial
per model is refitted on the full training set and saved.

Key design principles
──────────────────────
- Optuna uses the TRAINING set only for cross-validation.  The val set
  is used only once (at the end of this stage) to report held-out metrics.
  The TEST set is never touched here — it is reserved for final evaluation.
- class_weight="balanced" (or scale_pos_weight for XGBoost) corrects for
  the HEALTHY / DISEASED imbalance without oversampling.

Outputs
───────
models/
  logisticregression/  best_model.joblib  best_params.json
  randomforest/        best_model.joblib  best_params.json
  xgboost/             best_model.joblib  best_params.json
  training_summary.json
"""

import json
import argparse
import yaml
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import optuna
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_metrics(model, X: np.ndarray, y: np.ndarray,
                    prefix: str = "") -> dict[str, float]:
    preds  = model.predict(X)
    probas = model.predict_proba(X)[:, 1]
    key    = lambda k: f"{prefix}{k}" if prefix else k
    return {
        key("accuracy"):    float(accuracy_score(y, preds)),
        key("precision"):   float(precision_score(y, preds, zero_division=0)),
        key("recall"):      float(recall_score(y, preds, zero_division=0)),
        key("f1_weighted"): float(f1_score(y, preds, average="weighted",
                                           zero_division=0)),
        key("f1_binary"):   float(f1_score(y, preds, average="binary",
                                           zero_division=0)),
        key("roc_auc"):     float(roc_auc_score(y, probas)),
    }


# ── Objective factories 

def make_lr_objective(X_tr, y_tr, p, cv):
    lp = p["logistic_regression"]
    opt_metric = p["optuna"]["metric"]

    def objective(trial: optuna.Trial) -> float:
        solver = trial.suggest_categorical("solver", lp["solver"])
        penalty = trial.suggest_categorical("penalty", lp["penalty"])

        # --- FIX 1: Convert string "none" → None ---
        if penalty == "none":
            penalty = None

        # --- FIX 2: Enforce valid solver–penalty combinations ---
        if solver == "lbfgs" and penalty not in ("l2", None):
            raise optuna.TrialPruned()

        if solver == "liblinear" and penalty not in ("l1", "l2"):
            raise optuna.TrialPruned()

        if solver == "saga" and penalty not in ("l1", "l2", "elasticnet", None):
            raise optuna.TrialPruned()

        # --- FIX 3: Only tune C when relevant ---
        if penalty is None:
            C = 1.0
        else:
            C = trial.suggest_float("C", lp["C_low"], lp["C_high"], log=True)

        model = LogisticRegression(
            C=C,
            solver=solver,
            penalty=penalty,
            max_iter=lp["max_iter"],
            class_weight="balanced",
            random_state=p["data"]["seed"],
        )

        # --- FIX 4: Prevent pipeline crash ---
        try:
            scores = cross_val_score(
                model,
                X_tr,
                y_tr,
                cv=cv,
                scoring=opt_metric,
                n_jobs=-1,
                error_score=np.nan,
            )
            return float(np.nanmean(scores))
        except Exception:
            return float("nan")

    return objective


def make_rf_objective(X_tr, y_tr, p, cv):
    rp = p["random_forest"]
    opt_metric = p["optuna"]["metric"]

    def objective(trial: optuna.Trial) -> float:
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators",
                                           rp["n_estimators_low"],
                                           rp["n_estimators_high"]),
            max_depth=trial.suggest_int("max_depth",
                                        rp["max_depth_low"],
                                        rp["max_depth_high"]),
            min_samples_split=trial.suggest_int("min_samples_split",
                                                 rp["min_samples_split_low"],
                                                 rp["min_samples_split_high"]),
            max_features=trial.suggest_categorical("max_features",
                                                    rp["max_features"]),
            class_weight="balanced",
            random_state=p["data"]["seed"],
            n_jobs=-1,
        )
        scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring=opt_metric,
                                 n_jobs=1)  # n_jobs=1 here; model already uses -1
        return float(scores.mean())
    return objective


def make_xgb_objective(X_tr, y_tr, p, cv):
    xp = p["xgboost"]
    opt_metric = p["optuna"]["metric"]
    # Compute class imbalance ratio for XGBoost
    ratio = float(np.sum(y_tr == 0) / (np.sum(y_tr == 1) + 1e-8))

    def objective(trial: optuna.Trial) -> float:
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators",
                                            xp["n_estimators_low"],
                                            xp["n_estimators_high"]),
            max_depth=trial.suggest_int("max_depth",
                                         xp["max_depth_low"],
                                         xp["max_depth_high"]),
            learning_rate=trial.suggest_float("learning_rate",
                                               xp["learning_rate_low"],
                                               xp["learning_rate_high"], log=True),
            subsample=trial.suggest_float("subsample",
                                           xp["subsample_low"],
                                           xp["subsample_high"]),
            colsample_bytree=trial.suggest_float("colsample_bytree",
                                                  xp["colsample_bytree_low"],
                                                  xp["colsample_bytree_high"]),
            gamma=trial.suggest_float("gamma",
                                       xp["gamma_low"],
                                       xp["gamma_high"]),
            scale_pos_weight=ratio,   # handles class imbalance
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=p["data"]["seed"],
            n_jobs=-1,
        )
        scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring=opt_metric)
        return float(scores.mean())
    return objective


# ── Core tuning + logging function 

def tune_and_log(
    model_name: str,
    build_model_fn,        # best_params → fitted estimator (unfitted)
    objective_fn,
    X_train, y_train,
    X_val,   y_val,
    p: dict,
    output_dir: Path,
) -> dict:
    op   = p["optuna"]
    seed = p["data"]["seed"]

    print(f"\n{'═'*60}")
    print(f"  Tuning {model_name}  ({op['n_trials']} Optuna trials, "
          f"{op['cv_folds']}-fold CV)")
    print(f"{'═'*60}")

    study = optuna.create_study(
        direction=op["direction"],
        study_name=model_name,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(
        objective_fn,
        n_trials=op["n_trials"],
        timeout=op["timeout"],
        show_progress_bar=True,
        gc_after_trial=True,
    )

    best_params    = study.best_params
    best_cv_score  = study.best_value
    n_completed    = len([t for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE])

    print(f"\n  ✓ Best CV {op['metric']}: {best_cv_score:.4f}  "
          f"({n_completed}/{op['n_trials']} trials completed)")
    print(f"  Best params: {best_params}")

    # ── Refit on full training set 
    model = build_model_fn(best_params)
    model.fit(X_train, y_train)

    train_m = compute_metrics(model, X_train, y_train, prefix="train_")
    val_m   = compute_metrics(model, X_val,   y_val,   prefix="val_")

    # ── Log to MLflow 
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_type",      model_name)
        mlflow.log_param("n_optuna_trials", n_completed)
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_f1_weighted", best_cv_score)
        mlflow.log_metrics(train_m)
        mlflow.log_metrics(val_m)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"{p['mlflow']['model_name']}-{model_name}",
        )

    print(f"\n  Val F1(weighted): {val_m['val_f1_weighted']:.4f}  "
          f"| Val AUC: {val_m['val_roc_auc']:.4f}  "
          f"| Val Acc: {val_m['val_accuracy']:.4f}")

    # ── Save locally for DVC tracking 
    model_slug = model_name.lower().replace(" ", "_")
    model_dir  = output_dir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "best_model.joblib")

    record = {
        "model_name":   model_name,
        "best_params":  best_params,
        "cv_score":     best_cv_score,
        "val_metrics":  val_m,
        "train_metrics": train_m,
    }
    with open(model_dir / "best_params.json", "w") as f:
        json.dump(record, f, indent=2)

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir",    default="models")
    parser.add_argument("--params",        default="params.yaml")
    args = parser.parse_args()

    p   = load_params(args.params)
    out = Path(args.output_dir)
    prc = Path(args.processed_dir)

    mlflow.set_tracking_uri(p["mlflow"]["tracking_uri"])
    mlflow.set_experiment(p["mlflow"]["experiment_name"])

    # ── Load splits 
    X_train = np.load(prc / "X_train.npy")
    y_train = np.load(prc / "y_train.npy")
    X_val   = np.load(prc / "X_val.npy")
    y_val   = np.load(prc / "y_val.npy")
    print(f"Loaded  Train: {X_train.shape}  Val: {X_val.shape}")
    print(f"Class balance  Train: "
          f"{int(np.sum(y_train==0))} healthy / {int(np.sum(y_train==1))} diseased")

    cv   = StratifiedKFold(n_splits=p["optuna"]["cv_folds"],
                           shuffle=True, random_state=p["data"]["seed"])
    seed = p["data"]["seed"]
    results = []

    # ── 1. Logistic Regression 
    def build_lr(bp):
        penalty = bp.get("penalty", "l2")
        if penalty == "none":
            penalty = None
        C       = bp.get("C", 1.0) if penalty is not None else 1.0
        return LogisticRegression(
            C=C, solver=bp["solver"], penalty=penalty,
            max_iter=p["logistic_regression"]["max_iter"],
            class_weight="balanced", random_state=seed,
        )
    results.append(tune_and_log(
        "LogisticRegression", build_lr,
        make_lr_objective(X_train, y_train, p, cv),
        X_train, y_train, X_val, y_val, p, out,
    ))

    # ── 2. Random Forest 
    def build_rf(bp):
        return RandomForestClassifier(
            n_estimators=bp["n_estimators"],
            max_depth=bp["max_depth"],
            min_samples_split=bp["min_samples_split"],
            max_features=bp["max_features"],
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
    results.append(tune_and_log(
        "RandomForest", build_rf,
        make_rf_objective(X_train, y_train, p, cv),
        X_train, y_train, X_val, y_val, p, out,
    ))

    # ── 3. XGBoost 
    ratio = float(np.sum(y_train == 0) / (np.sum(y_train == 1) + 1e-8))
    def build_xgb(bp):
        return XGBClassifier(
            n_estimators=bp["n_estimators"],
            max_depth=bp["max_depth"],
            learning_rate=bp["learning_rate"],
            subsample=bp["subsample"],
            colsample_bytree=bp["colsample_bytree"],
            gamma=bp["gamma"],
            scale_pos_weight=ratio,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=seed, n_jobs=-1,
        )
    results.append(tune_and_log(
        "XGBoost", build_xgb,
        make_xgb_objective(X_train, y_train, p, cv),
        X_train, y_train, X_val, y_val, p, out,
    ))

    # ── Print summary 
    print(f"\n{'═'*60}")
    print("TRAINING SUMMARY")
    print(f"{'Model':<25} {'Val F1':>10} {'Val AUC':>10} {'Val Acc':>10}")
    print("─" * 58)
    for r in results:
        vm = r["val_metrics"]
        print(f"{r['model_name']:<25} "
              f"{vm['val_f1_weighted']:>10.4f} "
              f"{vm['val_roc_auc']:>10.4f} "
              f"{vm['val_accuracy']:>10.4f}")

    with open(out / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\ntraining_summary.json saved to {out}")


if __name__ == "__main__":
    main()
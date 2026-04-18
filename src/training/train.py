"""
DVC Stage: train_models
════════════════════════
Trains Logistic Regression, Random Forest, and XGBoost on EfficientNet-B0
features.  Each model is independently tuned with Optuna (TPE sampler,
stratified k-fold cross-validation on the TRAINING set only).

Every Optuna trial is logged to MLflow.  The best trial per model is
refitted on the full training set and saved.

Key design principles
──────────────────────
- Optuna uses the TRAINING set only for cross-validation.
- Val set is used once at the end to report held-out metrics.
- TEST set is never touched here — reserved for final evaluation.
- class_weight="balanced" / scale_pos_weight handles class imbalance.

Outputs
───────
models/
  logisticregression/  best_model.joblib  best_params.json
  randomforest/        best_model.joblib  best_params.json
  xgboost/             best_model.joblib  best_params.json
  training_summary.json
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
# Suppress the sklearn 1.8 FutureWarning about penalty deprecation
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn.linear_model._logistic"
)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_metrics(model, X: np.ndarray, y: np.ndarray, prefix: str = "") -> dict:
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]

    def k(name):
        return f"{prefix}{name}" if prefix else name

    return {
        k("accuracy"): float(accuracy_score(y, preds)),
        k("precision"): float(precision_score(y, preds, zero_division=0)),
        k("recall"): float(recall_score(y, preds, zero_division=0)),
        k("f1_weighted"): float(
            f1_score(y, preds, average="weighted", zero_division=0)
        ),
        k("f1_binary"): float(f1_score(y, preds, average="binary", zero_division=0)),
        k("roc_auc"): float(roc_auc_score(y, probas)),
    }


# ── Objective factories ──────────────────────────────────────────────────────


def make_lr_objective(X_tr, y_tr, p, cv):
    """
    Logistic Regression objective.

    Optuna requires every named parameter to have the SAME set of choices
    across ALL trials (CategoricalDistribution does not support dynamic
    value space).  Linking solver and penalty as two separate categoricals
    with different choice-lists depending on each other breaks this rule.

    Solution: encode valid (solver, penalty) pairs as a SINGLE categorical
    parameter 'solver_penalty'.  This gives Optuna a fixed, static choice
    list and avoids the dynamic-space error entirely.

    Valid pairs used:
      lbfgs__l2   — lbfgs only supports l2
      saga__l1    — saga supports l1 (sparse solutions)
      saga__l2    — saga with l2 (equivalent to lbfgs but scales better)
    """
    lp = p["logistic_regression"]
    opt_metric = p["optuna"]["metric"]
    seed = p["data"]["seed"]

    # All valid (solver, penalty) combinations as a single flat string
    COMBOS = ["lbfgs__l2", "saga__l1", "saga__l2"]

    def objective(trial: optuna.Trial) -> float:
        combo = trial.suggest_categorical("solver_penalty", COMBOS)
        solver, penalty = combo.split("__")  # e.g. "saga__l1" → ("saga", "l1")
        C = trial.suggest_float("C", lp["C_low"], lp["C_high"], log=True)

        model = LogisticRegression(
            C=C,
            solver=solver,
            penalty=penalty,
            max_iter=lp["max_iter"],
            class_weight="balanced",
            random_state=seed,
        )

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
            mean_score = float(np.nanmean(scores))
            if np.isnan(mean_score):
                raise optuna.TrialPruned()
            return mean_score
        except optuna.TrialPruned:
            raise
        except Exception:
            raise optuna.TrialPruned()

    return objective


def make_rf_objective(X_tr, y_tr, p, cv):
    rp = p["random_forest"]
    opt_metric = p["optuna"]["metric"]
    seed = p["data"]["seed"]

    def objective(trial: optuna.Trial) -> float:
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int(
                "n_estimators", rp["n_estimators_low"], rp["n_estimators_high"]
            ),
            max_depth=trial.suggest_int(
                "max_depth", rp["max_depth_low"], rp["max_depth_high"]
            ),
            min_samples_split=trial.suggest_int(
                "min_samples_split",
                rp["min_samples_split_low"],
                rp["min_samples_split_high"],
            ),
            max_features=trial.suggest_categorical("max_features", rp["max_features"]),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        # n_jobs=1 on cross_val_score — RandomForest already uses n_jobs=-1
        scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring=opt_metric, n_jobs=1)
        return float(scores.mean())

    return objective


def make_xgb_objective(X_tr, y_tr, p, cv):
    xp = p["xgboost"]
    opt_metric = p["optuna"]["metric"]
    seed = p["data"]["seed"]
    # Imbalance ratio: n_negative / n_positive
    ratio = float(np.sum(y_tr == 0) / (np.sum(y_tr == 1) + 1e-8))

    def objective(trial: optuna.Trial) -> float:
        model = XGBClassifier(
            n_estimators=trial.suggest_int(
                "n_estimators", xp["n_estimators_low"], xp["n_estimators_high"]
            ),
            max_depth=trial.suggest_int(
                "max_depth", xp["max_depth_low"], xp["max_depth_high"]
            ),
            learning_rate=trial.suggest_float(
                "learning_rate",
                xp["learning_rate_low"],
                xp["learning_rate_high"],
                log=True,
            ),
            subsample=trial.suggest_float(
                "subsample", xp["subsample_low"], xp["subsample_high"]
            ),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree",
                xp["colsample_bytree_low"],
                xp["colsample_bytree_high"],
            ),
            gamma=trial.suggest_float("gamma", xp["gamma_low"], xp["gamma_high"]),
            scale_pos_weight=ratio,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=seed,
            n_jobs=-1,
        )
        scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring=opt_metric)
        return float(scores.mean())

    return objective


# ── Core tuning + logging function ──────────────────────────────────────────


def tune_and_log(
    model_name: str,
    build_model_fn,
    objective_fn,
    X_train,
    y_train,
    X_val,
    y_val,
    p: dict,
    output_dir: Path,
) -> dict:
    op = p["optuna"]
    seed = p["data"]["seed"]

    print(f"\n{'═' * 60}")
    print(
        f"  Tuning {model_name}  "
        f"({op['n_trials']} Optuna trials, {op['cv_folds']}-fold CV)"
    )
    print(f"{'═' * 60}")

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

    best_params = study.best_params
    best_cv_score = study.best_value
    n_completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )

    print(
        f"\n  ✓ Best CV {op['metric']}: {best_cv_score:.4f}  "
        f"({n_completed}/{op['n_trials']} trials completed)"
    )
    print(f"  Best params: {best_params}")

    # Refit on full training set with the best hyperparameters
    model = build_model_fn(best_params)
    model.fit(X_train, y_train)

    train_m = compute_metrics(model, X_train, y_train, prefix="train_")
    val_m = compute_metrics(model, X_val, y_val, prefix="val_")

    # Log to MLflow
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_optuna_trials", n_completed)
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_f1_weighted", best_cv_score)
        mlflow.log_metrics(train_m)
        mlflow.log_metrics(val_m)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=(f"{p['mlflow']['model_name']}-{model_name}"),
        )

    print(
        f"\n  Val F1(weighted): {val_m['val_f1_weighted']:.4f}  "
        f"| Val AUC: {val_m['val_roc_auc']:.4f}  "
        f"| Val Acc: {val_m['val_accuracy']:.4f}"
    )

    # Save locally for DVC tracking
    model_slug = model_name.lower().replace(" ", "_")
    model_dir = output_dir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "best_model.joblib")

    record = {
        "model_name": model_name,
        "best_params": best_params,
        "cv_score": best_cv_score,
        "val_metrics": val_m,
        "train_metrics": train_m,
    }
    with open(model_dir / "best_params.json", "w") as f:
        json.dump(record, f, indent=2)

    return record


# ── main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    p = load_params(args.params)
    out = Path(args.output_dir)
    prc = Path(args.processed_dir)

    mlflow.set_tracking_uri(p["mlflow"]["tracking_uri"])
    mlflow.set_experiment(p["mlflow"]["experiment_name"])

    # Load processed splits
    X_train = np.load(prc / "X_train.npy")
    y_train = np.load(prc / "y_train.npy")
    X_val = np.load(prc / "X_val.npy")
    y_val = np.load(prc / "y_val.npy")

    print(f"Loaded  Train: {X_train.shape}  Val: {X_val.shape}")
    print(
        f"Class balance  Train: "
        f"{int(np.sum(y_train == 0))} healthy / "
        f"{int(np.sum(y_train == 1))} diseased"
    )

    cv = StratifiedKFold(
        n_splits=p["optuna"]["cv_folds"], shuffle=True, random_state=p["data"]["seed"]
    )
    seed = p["data"]["seed"]
    results = []

    # ── 1. Logistic Regression ───────────────────────────────────────────────
    def build_lr(bp):
        # best_params has 'solver_penalty' e.g. "saga__l1" — unpack it
        solver, penalty = bp["solver_penalty"].split("__")
        return LogisticRegression(
            C=bp["C"],
            solver=solver,
            penalty=penalty,
            max_iter=p["logistic_regression"]["max_iter"],
            class_weight="balanced",
            random_state=seed,
        )

    results.append(
        tune_and_log(
            "LogisticRegression",
            build_lr,
            make_lr_objective(X_train, y_train, p, cv),
            X_train,
            y_train,
            X_val,
            y_val,
            p,
            out,
        )
    )

    # ── 2. Random Forest ─────────────────────────────────────────────────────
    def build_rf(bp):
        return RandomForestClassifier(
            n_estimators=bp["n_estimators"],
            max_depth=bp["max_depth"],
            min_samples_split=bp["min_samples_split"],
            max_features=bp["max_features"],
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )

    results.append(
        tune_and_log(
            "RandomForest",
            build_rf,
            make_rf_objective(X_train, y_train, p, cv),
            X_train,
            y_train,
            X_val,
            y_val,
            p,
            out,
        )
    )

    # ── 3. XGBoost ───────────────────────────────────────────────────────────
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
            random_state=seed,
            n_jobs=-1,
        )

    results.append(
        tune_and_log(
            "XGBoost",
            build_xgb,
            make_xgb_objective(X_train, y_train, p, cv),
            X_train,
            y_train,
            X_val,
            y_val,
            p,
            out,
        )
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'Model':<25} {'Val F1':>10} {'Val AUC':>10} {'Val Acc':>10}")
    print("─" * 58)
    for r in results:
        vm = r["val_metrics"]
        print(
            f"{r['model_name']:<25} "
            f"{vm['val_f1_weighted']:>10.4f} "
            f"{vm['val_roc_auc']:>10.4f} "
            f"{vm['val_accuracy']:>10.4f}"
        )

    with open(out / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\ntraining_summary.json saved to {out}")


if __name__ == "__main__":
    main()

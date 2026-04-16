"""
Singleton loader for the champion classifier and scaler.
Loaded once at FastAPI startup; reused for every request.

Paths are anchored to this file's location (api/), then resolved
to the project root — safe regardless of where uvicorn is launched from.
"""

import joblib
from pathlib import Path

# Project root = one directory above api/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_model  = None
_scaler = None


def get_model():
    global _model
    if _model is None:
        path = _PROJECT_ROOT / "models" / "champion" / "best_model.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Champion model not found at {path}. "
                "Run `dvc repro` to train and export it."
            )
        _model = joblib.load(path)
        print(f"[model_loader] Loaded {type(_model).__name__} from {path}")
    return _model


def get_scaler():
    global _scaler
    if _scaler is None:
        path = _PROJECT_ROOT / "models" / "champion" / "scaler.joblib"
        if path.exists():
            obj = joblib.load(path)
            _scaler = obj
            if _scaler is not None:
                print(f"[model_loader] Scaler loaded from {path}")
        else:
            _scaler = None
    return _scaler
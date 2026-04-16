"""
Leaf Guard
══════════
Rejects non-leaf images before classification.

Method: cosine similarity between the query image's EfficientNet-B0
embedding and the mean ("centroid") embedding of the entire training set.

If similarity < params.leaf_guard.similarity_threshold → REJECTED.
"""

import numpy as np
import yaml
import torch
import torchvision.models as tvm
import torchvision.transforms as T
from pathlib import Path
from PIL import Image


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_params(path: str = None) -> dict:
    if path is None:
        path = _PROJECT_ROOT / "params.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


class LeafGuard:
    def __init__(self, params_path: str = None):
        p  = _load_params(params_path)
        gp = p["leaf_guard"]
        fp = p["features"]

        self.threshold = gp["similarity_threshold"]
        # Anchor centroid path to project root
        centroid_path  = _PROJECT_ROOT / gp["centroid_path"]

        if not centroid_path.exists():
            raise FileNotFoundError(
                f"Leaf centroid not found at '{centroid_path}'. "
                "Run `dvc repro` to generate it."
            )
        self.centroid = np.load(centroid_path).astype(np.float32)

        # Build EfficientNet-B0 embedder (same backbone used in training)
        weights        = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1
        model          = tvm.efficientnet_b0(weights=weights)
        model.classifier = torch.nn.Identity()
        model.eval()
        self._model    = model
        self._transform = T.Compose([
            T.Resize((fp["image_size"], fp["image_size"])),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def embed(self, image: Image.Image) -> np.ndarray:
        x   = self._transform(image).unsqueeze(0)
        emb = self._model(x).squeeze().numpy()
        return emb.astype(np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def check(self, image: Image.Image) -> tuple[bool, float]:
        """
        Returns (is_leaf: bool, similarity_score: float).
        True  → image is close enough to training distribution → proceed
        False → reject, return 422 to the user
        """
        sim = self._cosine(self.embed(image), self.centroid)
        return sim >= self.threshold, round(float(sim), 4)
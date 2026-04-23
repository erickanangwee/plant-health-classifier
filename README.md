# Plant Leaf Health Classifier - MLOps Pipeline

## Problem

Binary classification of plant leaf images as **HEALTHY** or **DISEASED**
across all species in the PlantDoc dataset (2,568 images, 38 classes).
Non-leaf images are automatically rejected.

## Architecture

PlantDoc (HuggingFace)
→ DVC Pipeline (download → features → train → evaluate)
→ MLflow Model Registry (champion selection)
→ FastAPI Microservice (EfficientNet-B0 + sklearn classifier)
→ Docker Container
→ Kubernetes (Minikube)
→ GitHub Actions CI/CD

## Quick Start

### Option 1 — Docker

```bash
docker pull <USERNAME>/plant-health-api:latest
docker run -p 8000:8000 anangwe/plant-health-api:latest

curl -X POST http://localhost:8000/predict -F "file=@leaf.jpg"
```

### Option 2 — Reproduce training from scratch

```bash
git clone <REPO_URL> && cd plant-health-classifier
pip install -r requirements-train.txt
mlflow server --port 5000 &
dvc repro
```

## API Reference

| Method | Path     | Description                      |
| ------ | -------- | -------------------------------- |
| GET    | /health  | Liveness + model status          |
| GET    | /docs    | Interactive Swagger UI           |
| GET    | /classes | Output class names               |
| POST   | /predict | Upload leaf → HEALTHY / DISEASED |

## Models

All three models use EfficientNet-B0 (frozen) features + Optuna tuning:

| Model               | Tuning           | Logged in |
| ------------------- | ---------------- | --------- |
| Logistic Regression | 30 Optuna trials | MLflow    |
| Random Forest       | 30 Optuna trials | MLflow    |
| XGBoost             | 30 Optuna trials | MLflow    |

Champion is selected by highest F1-weighted on the held-out test set.

## Frontend

| Environment | URL                                        | Purpose               |
| ----------- | ------------------------------------------ | --------------------- |
| Development | `http://localhost:5173`                    | Local Vite dev server |
| Replit      | `https://<repl>.<user>.repl.co`            | Team preview / demo   |
| Production  | `https://plant-health-frontend.vercel.app` | Public deployment     |

### Run Frontend Locally

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### Environment Variables

| Variable       | Description      | Example                 |
| -------------- | ---------------- | ----------------------- |
| `VITE_API_URL` | FastAPI base URL | `http://localhost:8000` |

Leave `VITE_API_URL` empty to use the Vite dev proxy (recommended for local dev).  
Use ngrok tunnelling to get a `VITE_API_URL` with `ngrok http 8000`

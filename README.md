# Plant Leaf Health Classifier

A machine learning microservice that classifies plant leaf images as **HEALTHY** or **DISEASED**. Upload any plant leaf photo and get an instant prediction. Non-leaf images are automatically rejected.

---

## What This Project Does

| Component      | Description                                                              |
| -------------- | ------------------------------------------------------------------------ |
| **ML Model**   | Logistic Regression (champion) trained on EfficientNet-B0 image features |
| **Dataset**    | PlantDoc — 2,568 images across 38 disease classes and 10+ plant species  |
| **API**        | FastAPI backend — upload an image, get a JSON prediction                 |
| **Frontend**   | React web app — drag-and-drop interface, live at Vercel                  |
| **Pipeline**   | DVC for data versioning, MLflow for experiment tracking                  |
| **Deployment** | Docker + Kubernetes (Minikube)                                           |
| **CI/CD**      | GitHub Actions — automatically retrains and redeploys on every push      |

---

## Training Results

Three classifiers were trained and evaluated on a held-out test set. All models used frozen **EfficientNet-B0** (ImageNet pre-trained) as the feature extractor, producing 1,280-dim embeddings per image, with **Optuna** (30 trials × 5-fold cross-validation) for hyperparameter tuning.

| Model                   | F1-Weighted (Test) | ROC-AUC (Test) | Status      |
| ----------------------- | ------------------ | -------------- | ----------- |
| **Logistic Regression** | **0.9378**         | **0.9810**     | 🏆 Champion |
| XGBoost                 | 0.9151             | 0.9752         | —           |
| Random Forest           | 0.8919             | 0.9486         | —           |

**Champion model:** Logistic Regression - selected by highest test-set F1-weighted score, with ROC-AUC as tiebreaker.

> Logistic Regression outperforming tree-based models here is expected: EfficientNet-B0 produces linearly separable features in high-dimensional space, making a linear classifier optimal. The 1,280-dim embeddings encode rich semantic structure, and logistic regression can exploit linear separability directly without the overfitting risk of deeper trees on a 1,797-sample training set.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Set Up Python Environment](#3-set-up-python-environment)
4. [Pull Data and Models with DVC](#4-pull-data-and-models-with-dvc)
5. [Run the API Locally](#5-run-the-api-locally)
6. [Test the API](#6-test-the-api)
7. [Run the Frontend Locally](#7-run-the-frontend-locally)
8. [Run with Docker](#8-run-with-docker)
9. [Run on Kubernetes](#9-run-on-kubernetes)
10. [Retrain the Models](#10-retrain-the-models)
11. [View Experiments in MLflow](#11-view-experiments-in-mlflow)
12. [Project Structure](#12-project-structure)
13. [Environment Variables](#13-environment-variables)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Prerequisites

Install the following tools before you begin. Click each link for installation instructions.

| Tool           | Minimum Version | Download                                                      |
| -------------- | --------------- | ------------------------------------------------------------- |
| Git            | 2.40+           | https://git-scm.com/downloads                                 |
| Python         | 3.10+           | https://www.python.org/downloads                              |
| Node.js        | 18+             | https://nodejs.org (for the frontend only)                    |
| Docker Desktop | latest          | https://www.docker.com/products/docker-desktop                |
| Minikube       | latest          | https://minikube.sigs.k8s.io/docs/start (for Kubernetes only) |

**Verify your installations:**

```bash
git --version
python3 --version
node --version
docker --version
```

> **Windows users:** All commands in this guide use **Git Bash**. Open Git Bash in your IDE or from the Start menu (it is installed with Git). Do not use Command Prompt or PowerShell unless stated otherwise.

---

## 2. Clone the Repository

```bash
git clone https://github.com/<USERNAME>/plant-health-classifier.git
cd plant-health-classifier
```

You should now be inside the project folder. Confirm with:

```bash
ls
```

You should see files like `dvc.yaml`, `params.yaml`, `Dockerfile`, `requirements.txt`, and folders like `api/`, `src/`, `k8s/`, and `frontend/`.

---

## 3. Set Up Python Environment

A virtual environment keeps the project's Python packages isolated from the rest of your computer. You only need to do this once.

**Create the virtual environment:**

```bash
python3 -m venv venv
```

**Activate it:**

```bash
# macOS / Linux / Git Bash on Windows:
source venv/Scripts/activate

# If that doesn't work on macOS/Linux, try:
source venv/bin/activate
```

Your terminal prompt should now start with `(venv)`. This tells you the environment is active.

> **Important:** You must activate the virtual environment every time you open a new terminal window before running any Python commands.

**Install the dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements-train.txt
```

This installs all packages needed to run the training pipeline and the API. It may take 3–5 minutes the first time.

**Expected output (last few lines):**

```
Successfully installed datasets-2.18.0 torch-2.1.0 torchvision-0.16.0
  xgboost-2.0.3 optuna-3.6.0 mlflow-2.12.0 dvc-3.49.0 ...
```

---

## 4. Pull Data and Models with DVC

DVC (Data Version Control) manages the dataset and trained model files. Instead of storing large files in Git, DVC stores them separately and lets you download them with one command.

**Set up the DVC remote (where data is stored):**

```bash
dvc remote add -d localremote /tmp/dvc-remote
```

**Pull the data and models:**

```bash
dvc pull
```

This downloads:

- `data/raw/` — 2,568 plant leaf images
- `data/processed/` — extracted EfficientNet-B0 feature arrays
- `models/champion/` — the best trained model and scaler
- `models/evaluation/` — test metrics and classification reports

**Expected output:**

```
Collecting                                  |8.00 [00:00, 1.53entry/s]
Fetching
Building workspace index                   |4.00 [00:00, 1.21entry/s]
Comparing indexes                          |5.00 [00:00, 1.31entry/s]
Applying changes                           |3.00 [00:00, 1.18file/s]
```

**Verify the files are present:**

```bash
ls models/champion/
```

You should see:

```
best_model.joblib   champion_info.json   leaf_centroid.npy   scaler.joblib
```

If this folder is empty or missing, see [Troubleshooting](#14-troubleshooting).

---

## 5. Run the API Locally

The API is a FastAPI server that receives image uploads and returns predictions.

```bash
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected startup output:**

```
=== Plant Health Classifier API — Starting up ===
[startup] Loading champion classifier and scaler...
[model_loader] Loaded LogisticRegression from models/champion/best_model.joblib
[model_loader] Scaler loaded from models/champion/scaler.joblib
[startup] Initialising leaf guard...
[startup] Building EfficientNet-B0 feature extractor...
=== API ready ===
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

> **Note:** The first time you run this, PyTorch will download the EfficientNet-B0 weights (~21 MB). This happens once and is cached automatically.

The API is now live at: **http://localhost:8000**

You can view the interactive API documentation at: **http://localhost:8000/docs**

**Leave this terminal running.** Open a new terminal for the next steps.

---

## 6. Test the API

Open a **new terminal**, activate the virtual environment, and run these tests.

```bash
cd plant-health-classifier
source venv/Scripts/activate   # or: source venv/bin/activate
```

**Test 1 — Health check (confirms the API is running):**

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "LogisticRegression",
  "version": "1.0.0"
}
```

**Test 2 — List output classes:**

```bash
curl http://localhost:8000/classes
```

Expected response:

```json
{
  "classes": ["HEALTHY", "DISEASED"],
  "label_map": { "HEALTHY": 0, "DISEASED": 1 },
  "description": "..."
}
```

**Test 3 — Upload a leaf image and get a prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@data/raw/images/img_00001.jpg" \ # or the path to your image
  | python3 -m json.tool
```

Expected response:

```json
{
  "filename": "img_00001.jpg",
  "prediction": "DISEASED",
  "confidence": 0.9734,
  "healthy_prob": 0.0266,
  "diseased_prob": 0.9734,
  "is_leaf": true,
  "leaf_similarity": 0.8214,
  "model_used": "LogisticRegression",
  "message": "Disease or pest infestation detected. Consider consulting an agronomist."
}
```

**Test 4 — Verify rejection of a non-leaf image:**

Download any non-plant image (for example, a photo of your desktop) and upload it:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/any/non_leaf_photo.jpg" \
  | python3 -m json.tool
```

Expected response (HTTP 422):

```json
{
  "detail": {
    "filename": "non_leaf_photo.jpg",
    "rejected": true,
    "reason": "Image does not appear to be a plant leaf (similarity=0.214, threshold=0.680)...",
    "leaf_similarity": 0.214,
    "threshold_used": 0.4
  }
}
```

**Test 5 — Use the interactive Swagger UI:**

Open your browser and go to **http://localhost:8000/docs**. Click **POST /predict → Try it out → Choose File**, upload any leaf image, and click **Execute** to see the full request and response.

---

## 7. Run the Frontend Locally

The frontend is a React web application that provides a drag-and-drop interface for the API.

> **Requirement:** The API must be running (Step 5) before starting the frontend.

Open a **new terminal** (keep the API terminal open):

```bash
cd plant-health-classifier/frontend
npm install
npm run dev
```

**Expected output:**

```
  VITE v5.0.0  ready in 312 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://0.0.0.0:5173/
```

Open your browser at **http://localhost:5173**. You will see the PlantGuard interface. Drag a plant leaf image onto the upload zone and wait 2–5 seconds for the result card to appear.

> **How the proxy works:** When running locally, the frontend sends API requests to `/predict` which Vite automatically forwards to `http://localhost:8000`. No environment variable is needed for local development.

---

## 8. Run with Docker

Docker packages the entire API (model included) into a portable container. You do not need Python installed to run it this way.

**Prerequisite:** Docker Desktop must be running.

**Step 1 — Build the Docker image:**

```bash
# Run from the project root (not inside frontend/)
docker build -t plant-health-api:latest .
```

This takes 3–8 minutes the first time. It installs all dependencies and copies the trained model into the image.

**Expected output (last line):**

```
Successfully tagged plant-health-api:latest
```

**Step 2 — Run the container:**

```bash
docker run -d \
  --name plant-api \
  -p 8000:8000 \
  plant-health-api:latest
```

**Step 3 — Check the container is running:**

```bash
docker ps
```

You should see `plant-api` listed with status `Up`.

**Step 4 — Wait ~30 seconds for the model to load, then test:**

```bash
docker logs plant-api
# Wait until you see: "=== API ready ==="

curl http://localhost:8000/health
```

**Step 5 — Stop the container when done:**

```bash
docker stop plant-api
docker rm plant-api
```

---

## 9. Run on Kubernetes

Kubernetes runs multiple copies (replicas) of the container for high availability.

**Prerequisite:** Docker Desktop must be running.

**Step 1 — Start Minikube:**

```bash
minikube start --memory=4096 --cpus=2
```

**Expected output:**

```
✅  minikube v1.32.0 on ...
✨  Using the docker driver based on existing profile
👍  Starting control plane node minikube in cluster minikube
🔥  Creating docker container (CPUs=2, Memory=4096MB) ...
🐳  Preparing Kubernetes v1.28.3 on Docker 24.0.7 ...
✅  Done! kubectl is now configured to use "minikube" cluster
```

**Step 2 — Deploy the API:**

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

**Step 3 — Watch the pods start up:**

```bash
kubectl get pods -w
```

Wait until both pods show `1/1 Running`. This takes ~3 minutes. Press `Ctrl+C` to stop watching.

```
NAME                                  READY   STATUS    RESTARTS   AGE
plant-health-api-7d4b9f8c6-ab123      1/1     Running   0          110s
plant-health-api-7d4b9f8c6-cd456      1/1     Running   0          110s
```

**Step 4 — Get the service URL:**

```bash
minikube service plant-health-service --url
```

This outputs something like `http://192.168.49.2:30080`. Use this URL to make requests:

```bash
SERVICE_URL=$(minikube service plant-health-service --url)

curl $SERVICE_URL/health

curl -X POST $SERVICE_URL/predict \
  -F "file=@data/raw/images/img_00025.jpg" \
  | python3 -m json.tool
```

**Step 5 — Stop Kubernetes when done:**

```bash
minikube stop
```

---

## 10. Retrain the Models

If you want to retrain from scratch (for example, after changing `params.yaml`), use the DVC pipeline. This runs all four stages in the correct order: download data → extract features → train models → evaluate and select champion.

> **Warning:** Full retraining takes **40–60 minutes** depending on your hardware. The EfficientNet-B0 feature extraction step is the slowest part (~8–12 minutes for 2,568 images on CPU).

**Step 1 — Start the MLflow tracking server** (in a separate terminal, leave it running):

```bash
source venv/Scripts/activate   # or: source venv/bin/activate
mlflow server --host 0.0.0.0 --port 5000
```

Open **http://localhost:5000** to see the MLflow UI.

**Step 2 — Run the full pipeline** (in your main terminal):

```bash
source venv/Scripts/activate
dvc repro
```

DVC is smart — it only re-runs stages whose inputs have changed. If only `params.yaml` changed under the `xgboost` section, it will skip data download and feature extraction and only re-run training and evaluation.

**Expected output:**

```
Running stage 'download_data':   ...  2568 images saved
Running stage 'prepare_data':    ...  Feature matrix: (2568, 1280)
Running stage 'train_models':    ...  XGBoost Val F1: 0.9351
Running stage 'evaluate':        ...  🏆 Champion: XGBoost (F1=0.9318)
```

**Step 3 — Save the pipeline state:**

```bash
dvc push
git add dvc.lock
git commit -m "feat: retrain models with updated parameters"
git push origin main
```

---

## 11. View Experiments in MLflow

MLflow records every training run with its hyperparameters, metrics, and model artifacts.

**Start the MLflow server** (if not already running):

```bash
source venv/Scripts/activate
mlflow server --host 0.0.0.0 --port 5000
```

Open **http://localhost:5000** in your browser.

Navigate to the **plant-health-classification** experiment. You will see:

- One run per model per training session (LogisticRegression, RandomForest, XGBoost)
- A Champion run showing the final selected model
- Metrics including `val_f1_weighted`, `val_roc_auc`, `test_f1_weighted`
- Logged model artifacts registered in the Model Registry

Click any run to drill into its parameters and metrics. Use the **Compare** button to compare runs side by side.

---

## 12. Project Structure

```
plant-health-classifier/
│
├── api/                        API source code
│   ├── main.py                 FastAPI application — all endpoints
│   ├── schemas.py              Request and response data models
│   ├── model_loader.py         Loads champion model and scaler once at startup
│   └── leaf_guard.py           Rejects non-leaf images using cosine similarity
│
├── src/                        Training pipeline source code
│   ├── data/
│   │   ├── download_dataset.py Downloads PlantDoc and assigns binary labels
│   │   └── prepare_dataset.py  Extracts EfficientNet-B0 features, splits data
│   └── training/
│       ├── train.py            Trains 3 classifiers with Optuna tuning + MLflow
│       └── evaluate.py         Evaluates on test set, selects champion model
│
├── frontend/                   React web application
│   ├── src/
│   │   ├── App.jsx             Root component
│   │   ├── components/         UploadZone, ResultCard, ConfidenceBar, StatusBadge
│   │   └── hooks/
│   │       └── usePrediction.js  API call logic and state management
│   ├── vite.config.js          Vite config (includes dev proxy to :8000)
│   └── vercel.json             Vercel deployment settings
│
├── k8s/
│   ├── deployment.yaml         Kubernetes Deployment — 2 replicas
│   └── service.yaml            Kubernetes NodePort Service on port 30080
│
├── data/                       DVC-managed (not in Git)
│   ├── raw/                    Original images and metadata.json
│   └── processed/              Feature arrays (X_train.npy etc.) and scaler
│
├── models/                     DVC-managed (not in Git)
│   ├── champion/               Best model: best_model.joblib, scaler.joblib, leaf_centroid.npy
│   └── evaluation/             Test metrics and classification report
│
├── .github/workflows/
│   └── mlops-pipeline.yml      CI/CD: lint → train → build Docker → smoke test → deploy frontend
│
├── dvc.yaml                    Pipeline stage definitions
├── dvc.lock                    Locked pipeline state (always commit this)
├── params.yaml                 All hyperparameters and configuration values
├── Dockerfile                  Multi-stage Docker build for the API
├── requirements.txt            API/inference Python dependencies
└── requirements-train.txt      Training pipeline Python dependencies
```

---

## 13. Environment Variables

### API (FastAPI backend)

| Variable           | Default | Description                                                                          |
| ------------------ | ------- | ------------------------------------------------------------------------------------ |
| `ALLOWED_ORIGINS`  | `*`     | Comma-separated list of allowed CORS origins. Set to the frontend URL in production. |
| `PYTHONUNBUFFERED` | `1`     | Ensures Python output is not buffered (set automatically in Docker).                 |

### Frontend (React / Vite)

| Variable       | Default   | Description                                                                                                                                      |
| -------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `VITE_API_URL` | _(empty)_ | Base URL of the FastAPI backend. Leave empty for local dev (Vite proxy handles it). Set to your public API URL for Vercel or Replit deployments. |

**Setting `VITE_API_URL` for a public deployment:**

If your API is exposed via ngrok:

```bash
# In frontend/.env.local (never commit this file)
VITE_API_URL=https://abc123.ngrok-free.app
```

Then rebuild the frontend:

```bash
cd frontend
npm run build
```

---

## 14. Troubleshooting

---

### `dvc pull` fails or `models/champion/` is empty

**Cause:** The DVC remote is not configured or the data has not been pushed yet.

**Solution A** — If you have access to the original machine, push the data first:

```bash
dvc push
```

**Solution B** — Retrain from scratch on your machine:

```bash
# This downloads the dataset directly from Hugging Face and retrains everything
mlflow server --port 5000 &   # start in background
dvc repro
```

---

### `ModuleNotFoundError: No module named 'api'`

**Cause:** Python cannot find the project's root package.

**Solution:** Always prefix commands with `PYTHONPATH=.`:

```bash
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### `FileNotFoundError: Champion model not found at models/champion/best_model.joblib`

**Cause:** The model files were not pulled from DVC or training has not been run yet.

**Solution:**

```bash
dvc pull
# If that fails, retrain:
dvc repro
```

---

### API starts but returns errors on `/predict`

**Cause:** EfficientNet-B0 weights failed to download (common on slow or restricted networks).

**Solution:** Pre-download the weights manually:

```bash
python3 -c "
import torchvision.models as tvm
tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
print('Weights downloaded successfully.')
"
```

---

### `npm install` fails in the `frontend/` directory

**Cause:** Node.js is not installed or the version is too old.

**Solution:** Install Node.js 18+ from https://nodejs.org, then retry:

```bash
node --version   # should show v18.x.x or higher
cd frontend
npm install
```

---

### Docker build fails with `COPY models/champion/ ./models/champion/`

**Cause:** The `models/champion/` directory does not exist locally. Docker cannot copy files that are not there.

**Solution:** Run DVC pull or the training pipeline before building Docker:

```bash
dvc pull
docker build -t plant-health-api:latest .
```

---

### Kubernetes pods stay in `Pending` or `ImagePullBackOff`

**Cause:** Either Minikube doesn't have enough resources, or the Docker image name in `k8s/deployment.yaml` is wrong.

**Check the error:**

```bash
kubectl describe pod <pod-name>
```

**Fix for ImagePullBackOff** — update the image name in `k8s/deployment.yaml`:

```yaml
image: <YOUR_DOCKERHUB_USERNAME>/plant-health-api:latest
```

**Fix for Pending (not enough memory):**

```bash
minikube stop
minikube start --memory=6144 --cpus=4
```

---

### Frontend shows "Cannot reach the API server"

**Cause:** The FastAPI API is not running, or `VITE_API_URL` points to a URL that is unreachable.

**Solutions:**

1. Confirm the API is running: `curl http://localhost:8000/health`
2. If using ngrok, confirm the ngrok tunnel is active: `curl https://your-url.ngrok-free.app/health`
3. If the ngrok URL changed (it changes on every restart), update `VITE_API_URL` in `frontend/.env.local` and rebuild

---

### `(venv)` is not showing in my terminal

**Cause:** The virtual environment is not activated.

**Solution:**

```bash
# From the project root:
source venv/Scripts/activate   # Git Bash / Windows
source venv/bin/activate       # macOS / Linux
```

---

### MLflow UI shows no experiments

**Cause:** The MLflow server was not running when training was executed, so runs were not logged.

**Solution:** Always start the MLflow server before running `dvc repro`:

```bash
mlflow server --host 0.0.0.0 --port 5000
# Then in a separate terminal:
dvc repro
```

---

## API Quick Reference

| Method | Endpoint   | Description                                             |
| ------ | ---------- | ------------------------------------------------------- |
| `GET`  | `/health`  | Check if the API and model are loaded                   |
| `GET`  | `/classes` | List output classes and their numeric labels            |
| `POST` | `/predict` | Upload a leaf image, get HEALTHY or DISEASED prediction |
| `GET`  | `/docs`    | Interactive Swagger UI — test the API in your browser   |

**Example prediction request (Python):**

```python
import requests

with open("leaf.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("leaf.jpg", f, "image/jpeg")},
    )

result = response.json()
print(f"Prediction : {result['prediction']}")
print(f"Confidence : {result['confidence']:.1%}")
print(f"Message    : {result['message']}")
```

---

## Model Performance Summary

All three models use frozen **EfficientNet-B0** (ImageNet pre-trained) for feature extraction, producing 1,280-dim embeddings per image. Hyperparameters were tuned with **Optuna** (30 trials × 5-fold stratified cross-validation on the training set). The champion is selected by highest **F1-weighted** on the held-out test set, with ROC-AUC as tiebreaker.

| Model                   | F1-Weighted (Test) | ROC-AUC (Test) | Result      |
| ----------------------- | ------------------ | -------------- | ----------- |
| **Logistic Regression** | **0.9378**         | **0.9810**     | 🏆 Champion |
| XGBoost                 | 0.9151             | 0.9752         | —           |
| Random Forest           | 0.8919             | 0.9486         | —           |

**Why Logistic Regression won:** EfficientNet-B0 embeddings are already linearly separable in 1,280-dimensional space. A linear classifier exploits this directly without the risk of overfitting that tree-based ensembles face on a ~1,800-sample training set.

---

## Live URLs

| Service         | URL                                               |
| --------------- | ------------------------------------------------- |
| Local API       | http://localhost:8000                             |
| Local API docs  | http://localhost:8000/docs                        |
| Local frontend  | http://localhost:5173                             |
| MLflow UI       | http://localhost:5000                             |
| Kubernetes API  | Run `minikube service plant-health-service --url` |
| Vercel frontend | https://plant-health-frontend-omega.vercel.app    |

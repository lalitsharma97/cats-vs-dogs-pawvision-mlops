# Cats vs Dogs MLOps Pipeline

End-to-end MLOps pipeline for binary image classification (Cats vs Dogs) for a pet adoption platform.

## Project Structure

```
cats-vs-dogs-pawvision-mlops/
├── .github/workflows/          # CI/CD pipeline configurations
├── data/                       # Dataset (raw and processed, tracked by DVC)
├── models/                     # Trained models and checkpoints
├── src/                        # Source code
│   ├── data/                   # Data preprocessing and loading
│   ├── models/                 # Model architecture and training
│   ├── inference/              # FastAPI inference service
│   └── utils/                  # Logging, metrics, utilities
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter notebooks for experimentation
├── deployment/                 # Deployment manifests (K8s, Docker Compose)
├── configs/                    # Configuration files
├── mlruns/                     # MLflow experiment tracking
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## MLOps Pipeline Stages

### M1: Model Development & Experiment Tracking
- Data versioning with DVC
- Baseline CNN model
- Experiment tracking with MLflow

### M2: Model Packaging & Containerization
- FastAPI inference service
- Docker containerization (using Podman commands)
- Reproducible environment specification

### M3: CI Pipeline
- Automated testing with pytest
- GitHub Actions CI/CD pipeline
- Docker image building and registry push (using Podman commands)

### M4: CD Pipeline & Deployment
- Kubernetes deployment
- GitOps-based continuous deployment
- Post-deployment smoke tests

### M5: Monitoring & Logging
- Request/response logging
- Performance metrics tracking
- Model performance monitoring

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cats-vs-dogs-pawvision-mlops
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Initialize DVC**
   ```bash
   dvc init
   ```

4. **Download dataset**
   ```bash
   # Place Kaggle dataset in data/raw/
   dvc add data/raw/
   ```

## Usage

### Training
```bash
python src/models/train.py --config configs/training_config.yaml
```

### Inference API
```bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

### Run Tests
```bash
pytest tests/
```

### Build Docker Image
```bash
podman build -t cats-dogs-classifier:latest .
```

## Assignment Deliverables

- Complete source code with all modules
- Configuration files (DVC, CI/CD, Docker/Podman, deployment manifests)
- Trained model artifacts
- Screen recording of complete MLOps workflow

## Podman Usage

This project uses Podman as the container runtime instead of Docker. All Docker files (Dockerfile, docker-compose.yml) are compatible with Podman.

### Local Development with Podman
```bash
# Build image
podman build -t cats-dogs-classifier:latest .

# Run container
podman run -p 8000:8000 cats-dogs-classifier:latest

# Use docker-compose with podman-compose
podman-compose up -d
```

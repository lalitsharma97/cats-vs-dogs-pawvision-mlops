# Data & Code Versioning Strategy

## Overview
This project uses a dual versioning strategy:
- **Git** for source code versioning (project structure, scripts, and notebooks)
- **DVC** for dataset versioning and model tracking

## Versioning Structure

### Git (Source Code)
- **Purpose:** Track source code, configuration, and project structure
- **Tracked:**
  - Source code (`src/`)
  - Configuration files (`configs/`)
  - Deployment manifests (`deployment/`)
  - Tests (`tests/`)
  - Documentation (`README.md`, `AGENTS.md`)
  - DVC metadata files (`.dvc`, `.dvcignore`, `*.dvc`)

### DVC (Data & Models)
- **Purpose:** Track large datasets and trained models
- **Tracked:**
  - Raw dataset (`data/raw/cat/`, `data/raw/dog/`)
  - Trained models (`models/saved_models/best_model.pt`)
  - Processed data (when generated)

## Setup Instructions

### Initial Setup
```bash
# Initialize DVC (already done)
python -m dvc init

# Add dataset to DVC tracking
python -m dvc add data/raw/cat
python -m dvc add data/raw/dog

# Add model to DVC tracking
python -m dvc add models/saved_models/best_model.pt

# Commit DVC metadata to Git
git add data/raw/.gitignore data/raw/cat.dvc data/raw/dog.dvc
git add models/saved_models/best_model.pt.dvc
git commit -m "Add dataset and model to DVC tracking"
```

### Working with DVC

#### Adding New Data
```bash
# Add new dataset to DVC
python -m dvc add data/raw/new_dataset

# Commit the DVC metadata file
git add data/raw/new_dataset.dvc
git commit -m "Add new dataset to DVC tracking"
```

#### Updating Data
```bash
# When data changes, DVC automatically detects it
python -m dvc status

# Commit the updated DVC metadata
git add data/raw/cat.dvc
git commit -m "Update cat dataset"
```

#### Checking Out Data
```bash
# Clone repository with DVC data
git clone https://github.com/lalitsharma97/cats-vs-dogs-pawvision-mlops.git
cd cats-vs-dogs-pawvision-mlops

# Download data from DVC remote (if configured)
python -m dvc pull

# Or use local DVC cache
python -m dvc checkout
```

#### Remote Storage (Optional)
```bash
# Configure remote storage (S3, GCS, Azure, etc.)
python -m dvc remote add -d myremote s3://my-bucket/dvc-storage

# Push data to remote
python -m dvc push

# Pull data from remote
python -m dvc pull
```

## File Structure

```
cats-vs-dogs-pawvision-mlops/
├── .dvc/                          # DVC configuration
│   ├── .gitignore                 # DVC cache ignore rules
│   ├── config                     # DVC configuration
│   └── tmp/                       # DVC temporary files
├── .dvcignore                     # DVC ignore patterns
├── .gitignore                     # Git ignore patterns
├── data/
│   ├── raw/
│   │   ├── .gitignore             # Git ignore for raw data
│   │   ├── .dvc/                  # DVC cache for raw data
│   │   ├── cat.dvc                # DVC metadata for cat images
│   │   ├── dog.dvc                # DVC metadata for dog images
│   │   ├── cat/                   # Cat images (tracked by DVC)
│   │   └── dog/                   # Dog images (tracked by DVC)
│   └── processed/
│       └── .gitkeep               # Placeholder for processed data
├── models/
│   ├── saved_models/
│   │   ├── best_model.pt.dvc      # DVC metadata for model
│   │   └── best_model.pt          # Trained model (tracked by DVC)
│   └── checkpoints/
│       └── .gitkeep               # Placeholder for checkpoints
└── src/                           # Source code (tracked by Git)
```

## Benefits

### Git + DVC Strategy
- **Git:** Fast, efficient for code and small files
- **DVC:** Handles large files without bloating Git repository
- **Separation:** Clear separation between code and data
- **Version Control:** Both code and data are versioned
- **Reproducibility:** Exact data versions can be reproduced
- **Collaboration:** Team can share code and data efficiently

### Advantages Over Git LFS
- **No Storage Limits:** DVC works with any storage backend
- **Better Performance:** DVC is optimized for ML workflows
- **Pipeline Integration:** DVC integrates with ML pipelines
- **Data Management:** Better data lifecycle management
- **Cost Control:** More flexible storage options

## Workflow Example

### Training New Model
```bash
# 1. Update dataset (if needed)
python -m dvc add data/raw/new_images
git add data/raw/new_images.dvc
git commit -m "Add new training images"

# 2. Train model
python -m src.models.train

# 3. Add new model to DVC
python -m dvc add models/saved_models/new_model.pt
git add models/saved_models/new_model.pt.dvc
git commit -m "Add new trained model"

# 4. Push changes
git push
python -m dvc push  # If remote storage configured
```

### Reproducing Results
```bash
# 1. Clone repository
git clone https://github.com/lalitsharma97/cats-vs-dogs-pawvision-mlops.git
cd cats-vs-dogs-pawvision-mlops

# 2. Checkout specific version
git checkout <commit-hash>

# 3. Get exact data version
python -m dvc checkout

# 4. Reproduce training
python -m src.models.train
```

## Current Status

### DVC Tracked Files
- ✅ `data/raw/cat/` - 200 cat images
- ✅ `data/raw/dog/` - 200 dog images  
- ✅ `models/saved_models/best_model.pt` - Trained model (65% accuracy)

### Git Tracked Files
- ✅ Source code with MLflow integration
- ✅ Configuration files
- ✅ DVC metadata files
- ✅ Deployment manifests
- ✅ Test suite

### MLflow Tracking
- ✅ Experiment tracking enabled
- ✅ 3 training runs logged locally
- ✅ Metrics and artifacts tracked
- ⚠️ MLflow data excluded from Git (use .gitignore)

## Best Practices

1. **Always commit DVC metadata files** (`.dvc` files) to Git
2. **Use meaningful commit messages** for data changes
3. **Configure remote storage** for team collaboration
4. **Regular DVC push/pull** to sync data with team
5. **Document data sources** and preprocessing steps
6. **Use DVC pipelines** for complex data workflows
7. **Monitor DVC cache size** and clean up when needed
8. **Test data checkout** before pushing to production

## Troubleshooting

### Data Not Found After Clone
```bash
# Solution: Checkout data from DVC
python -m dvc checkout
```

### DVC Cache Issues
```bash
# Clean DVC cache
python -m dvc cache clean

# Rebuild cache
python -m dvc checkout
```

### Large Repository Size
```bash
# Check DVC cache size
python -m dvc cache du

# Clean unused cache
python -m dvc gc
```

## References
- [DVC Documentation](https://dvc.org/doc)
- [DVC vs Git LFS](https://dvc.org/doc/user-guide/large-dataset-management)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
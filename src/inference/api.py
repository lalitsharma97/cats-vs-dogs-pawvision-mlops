"""
FastAPI inference service for Cats vs Dogs classification
Provides REST API endpoints for model predictions
"""

import os
import io
import logging
from typing import Dict, List, Optional
from datetime import datetime
import yaml
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time

from src.models.architecture import get_model
from src.utils.logging import setup_logger

# Set up logging
logger = setup_logger("inference_api")

# Initialize FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classification API",
    description="REST API for binary image classification using trained CNN model",
    version="1.0.0"
)

# Prometheus metrics
request_count = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
request_latency = Histogram('api_request_latency_seconds', 'API request latency')
prediction_count = Counter('predictions_total', 'Total predictions made', ['class_name'])

# Global variables for model
model = None
device = None
class_names = ['cat', 'dog']
config = None


def load_model():
    """Load the trained model"""
    global model, device, config
    
    try:
        # Load configuration
        with open("configs/model_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize model
        model = get_model("cnn", "configs/model_config.yaml")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Load trained weights
        model_path = "models/saved_models/best_model.pt"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded successfully from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}, using untrained model")
        
        logger.info("Model initialization completed")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocess uploaded image for model inference
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Preprocessed tensor ready for model
    """
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 224x224
        image = image.resize((224, 224), Image.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting up inference service...")
    load_model()
    logger.info("Inference service ready")


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Health status of the service
    """
    request_count.labels(endpoint='/health', method='GET').inc()
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }
    
    logger.info("Health check performed")
    return health_status


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint
    
    Returns:
        Prometheus metrics in text format
    """
    request_count.labels(endpoint='/metrics', method='GET').inc()
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Image file for classification")
):
    """
    Prediction endpoint for image classification
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results with class probabilities and label
    """
    start_time = time.time()
    request_count.labels(endpoint='/predict', method='POST').inc()
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        logger.info(f"Received image for prediction: {file.filename}, size: {len(image_bytes)} bytes")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload an image file."
            )
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        # Get class name and probabilities
        predicted_class = class_names[predicted_class_idx]
        class_probabilities = {
            class_names[i]: probabilities[0][i].item() 
            for i in range(len(class_names))
        }
        
        # Update metrics
        prediction_count.labels(class_name=predicted_class).inc()
        request_latency.observe(time.time() - start_time)
        
        # Log prediction
        logger.info(
            f"Prediction: {predicted_class} (confidence: {confidence:.4f}), "
            f"probabilities: {class_probabilities}"
        )
        
        # Return prediction result
        result = {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    request_count.labels(endpoint='/', method='GET').inc()
    
    return {
        "message": "Cats vs Dogs Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "src.inference.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

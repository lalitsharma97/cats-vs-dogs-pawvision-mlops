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
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response
import time
import json

from src.models.architecture import get_model
from src.utils.logging import setup_logger
from src.monitoring.performance_tracker import performance_tracker

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
error_count = Counter('api_errors_total', 'Total API errors', ['endpoint', 'error_type'])
active_requests = Gauge('api_active_requests', 'Number of active requests')
model_inference_time = Histogram('model_inference_time_seconds', 'Model inference time')
preprocessing_time = Histogram('preprocessing_time_seconds', 'Image preprocessing time')

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
    try:
        load_model()
        logger.info("Inference service ready")
    except Exception as e:
        logger.error(f"Failed to initialize model during startup: {e}")
        # Don't crash the app - continue without model if loading fails
        logger.warning("Starting service without loaded model - predictions may fail")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with detailed information"""
    start_time = time.time()
    active_requests.inc()
    
    # Log request details
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response details (excluding sensitive data)
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"Status: {response.status_code} Duration: {duration:.3f}s"
        )
        
        return response
        
    except Exception as e:
        # Log errors
        duration = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path} "
            f"Error: {str(e)} Duration: {duration:.3f}s"
        )
        error_count.labels(endpoint=request.url.path, error_type=type(e).__name__).inc()
        raise
    finally:
        active_requests.dec()


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
            error_count.labels(endpoint='/predict', error_type='invalid_file_type').inc()
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload an image file."
            )
        
        # Preprocess image with timing
        preprocess_start = time.time()
        image_tensor = preprocess_image(image_bytes)
        image_tensor = image_tensor.to(device)
        preprocess_duration = time.time() - preprocess_start
        preprocessing_time.observe(preprocess_duration)
        
        # Make prediction with timing
        inference_start = time.time()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        inference_duration = time.time() - inference_start
        model_inference_time.observe(inference_duration)
        
        # Get class name and probabilities
        predicted_class = class_names[predicted_class_idx]
        class_probabilities = {
            class_names[i]: probabilities[0][i].item() 
            for i in range(len(class_names))
        }
        
        # Update metrics
        prediction_count.labels(class_name=predicted_class).inc()
        request_latency.observe(time.time() - start_time)
        
        # Record prediction for performance tracking
        performance_tracker.record_prediction(
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probabilities,
            processing_time={
                "preprocessing": round(preprocess_duration, 3),
                "inference": round(inference_duration, 3),
                "total": round(time.time() - start_time, 3)
            },
            metadata={
                "filename": file.filename,
                "file_size": len(image_bytes)
            }
        )
        
        # Log prediction with detailed metrics
        logger.info(
            f"Prediction completed: {predicted_class} (confidence: {confidence:.4f}), "
            f"preprocessing: {preprocess_duration:.3f}s, inference: {inference_duration:.3f}s, "
            f"total: {time.time() - start_time:.3f}s"
        )
        
        # Log probabilities (safe for monitoring)
        logger.debug(f"Class probabilities: {class_probabilities}")
        
        # Return prediction result
        result = {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": {
                "preprocessing": round(preprocess_duration, 3),
                "inference": round(inference_duration, 3),
                "total": round(time.time() - start_time, 3)
            }
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        error_count.labels(endpoint='/predict', error_type=type(e).__name__).inc()
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
            "docs": "/docs",
            "performance": "/performance"
        }
    }


@app.get("/performance")
async def performance_stats():
    """
    Performance statistics endpoint
    Returns aggregated performance metrics
    """
    request_count.labels(endpoint='/performance', method='GET').inc()
    
    # Get performance metrics from tracker
    performance_summary = performance_tracker.get_summary()
    
    stats = {
        "endpoint": "/performance",
        "timestamp": datetime.utcnow().isoformat(),
        "performance_summary": performance_summary,
        "metrics_available": {
            "api_requests_total": "Total API requests by endpoint and method",
            "api_request_latency_seconds": "Request latency distribution",
            "predictions_total": "Total predictions by class",
            "api_errors_total": "Total errors by endpoint and type",
            "api_active_requests": "Currently active requests",
            "model_inference_time_seconds": "Model inference time distribution",
            "preprocessing_time_seconds": "Image preprocessing time distribution"
        },
        "monitoring": {
            "prometheus_metrics": "/metrics",
            "logs": "Available in application logs",
            "health_check": "/health",
            "performance_report": "Generated automatically in logs/performance_report.json"
        }
    }
    
    logger.info("Performance statistics requested")
    return stats


@app.get("/performance/report")
async def get_performance_report():
    """
    Generate and return a detailed performance report
    """
    request_count.labels(endpoint='/performance/report', method='GET').inc()
    
    try:
        report = performance_tracker.export_report()
        if report:
            logger.info("Performance report generated")
            return report
        else:
            raise HTTPException(status_code=500, detail="Could not generate performance report")
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


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

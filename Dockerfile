FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configurations
COPY src/ ./src/
COPY configs/ ./configs/

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create models directory and copy trained model
RUN mkdir -p models/saved_models
COPY models/saved_models/ ./models/saved_models/

# Create logs directory with proper permissions
RUN mkdir -p logs && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/saved_models/best_model.pt
ENV CONFIG_PATH=/app/configs/model_config.yaml
ENV LOG_LEVEL=INFO

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Run the API server
CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

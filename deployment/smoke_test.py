"""
Smoke tests for post-deployment validation
Tests health endpoint and prediction endpoint
"""

import requests
import sys
import time
import os

def test_health_endpoint(base_url="http://localhost:8000"):
    """Test the health check endpoint"""
    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Health check passed: {data}")
                return True
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    
    print("✗ Health check failed after retries")
    return False

def test_prediction_endpoint(base_url="http://localhost:8000"):
    """Test the prediction endpoint with sample data"""
    try:
        # Check if test image exists
        test_image_path = "test_image.jpg"
        if not os.path.exists(test_image_path):
            print(f"✗ Test image not found at {test_image_path}")
            return False
        
        # Send prediction request
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/predict", files=files, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Prediction endpoint passed: {result}")
            return True
        else:
            print(f"✗ Prediction endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Prediction endpoint failed: {e}")
        return False

def main():
    """Run all smoke tests"""
    # Allow custom base URL via environment variable
    base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    
    print(f"Running smoke tests against {base_url}...")
    
    health_passed = test_health_endpoint(base_url)
    
    if health_passed:
        prediction_passed = test_prediction_endpoint(base_url)
    else:
        prediction_passed = False
    
    if health_passed and prediction_passed:
        print("\n✓ All smoke tests passed")
        sys.exit(0)
    else:
        print("\n✗ Some smoke tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
Smoke tests for post-deployment validation
Tests health endpoint and prediction endpoint
"""

import requests
import sys
import time

def test_health_endpoint():
    """Test the health check endpoint"""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✓ Health check passed")
                return True
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    
    print("✗ Health check failed after retries")
    return False

def test_prediction_endpoint():
    """Test the prediction endpoint with sample data"""
    try:
        # Create a dummy image payload (in real scenario, send actual image)
        files = {'file': open('data/processed/test_sample.jpg', 'rb')}
        response = requests.post("http://localhost:8000/predict", files=files, timeout=10)
        
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
    print("Running smoke tests...")
    
    health_passed = test_health_endpoint()
    prediction_passed = test_prediction_endpoint()
    
    if health_passed and prediction_passed:
        print("\n✓ All smoke tests passed")
        sys.exit(0)
    else:
        print("\n✗ Some smoke tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

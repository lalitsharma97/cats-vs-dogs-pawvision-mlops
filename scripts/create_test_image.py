"""
Create test image for smoke testing
"""

from PIL import Image
import numpy as np
from pathlib import Path

def create_test_image():
    """Create a simple test image for smoke testing"""
    
    # Create a simple 224x224 RGB image
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    # Save test image
    test_image_path = Path("test_image.jpg")
    image.save(test_image_path)
    
    print(f"Test image created at {test_image_path}")

if __name__ == "__main__":
    create_test_image()
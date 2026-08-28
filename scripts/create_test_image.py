"""
Create a simple test image for API testing
"""

from PIL import Image
import numpy as np

# Create a simple test image (224x224 RGB)
image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
image = Image.fromarray(image_array)
image.save("test_image.jpg")
print("Test image created: test_image.jpg")

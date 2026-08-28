"""
Script to update Kubernetes deployment image
"""
import sys
import os

def update_deployment_image(image_tag):
    """Update the image in Kubernetes deployment YAML"""
    # Read the file
    with open('deployment/kubernetes/deployment.yaml', 'r') as f:
        content = f.read()
    
    # Replace the image placeholder
    content = content.replace('IMAGE_PLACEHOLDER', image_tag)
    
    # Write back
    with open('deployment/kubernetes/deployment.yaml', 'w') as f:
        f.write(content)
    
    print(f"Updated image to: {image_tag}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_tag = sys.argv[1]
    else:
        image_tag = "ghcr.io/lalitsharma97/cats-vs-dogs-pawvision-mlops:latest"
    
    update_deployment_image(image_tag)

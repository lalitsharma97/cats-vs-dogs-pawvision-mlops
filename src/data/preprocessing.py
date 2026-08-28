"""
Data preprocessing module for Cats vs Dogs classification
Handles image loading, resizing, train/val/test splitting, and augmentation
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import yaml


class DataPreprocessor:
    """Handles data preprocessing for cats vs dogs classification"""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """
        Initialize the preprocessor with configuration
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.target_size = tuple(self.config['data']['image_size'])
        self.train_split = self.config['data']['train_split']
        self.val_split = self.config['data']['val_split']
        self.test_split = self.config['data']['test_split']
        self.use_augmentation = self.config['data']['augmentation']
        
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image to 224x224 RGB
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array (224, 224, 3)
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to target size
            image = image.resize(self.target_size, Image.LANCZOS)
            
            # Convert to numpy array and normalize to [0, 1]
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            return image_array
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def organize_dataset_structure(self, raw_data_path: str, processed_data_path: str):
        """
        Create the processed dataset directory structure
        
        Args:
            raw_data_path: Path to raw dataset
            processed_data_path: Path to processed dataset
        """
        # Create directory structure
        splits = ['train', 'val', 'test']
        classes = ['cat', 'dog']
        
        for split in splits:
            for class_name in classes:
                os.makedirs(
                    os.path.join(processed_data_path, split, class_name),
                    exist_ok=True
                )
    
    def split_and_copy_images(self, raw_data_path: str, processed_data_path: str):
        """
        Split dataset into train/val/test and copy processed images
        
        Args:
            raw_data_path: Path to raw dataset
            processed_data_path: Path to processed dataset
        """
        # Collect all image paths and labels
        image_paths = []
        labels = []
        
        # Assuming raw data has structure like: data/raw/cats/ and data/raw/dogs/
        # Adjust based on actual Kaggle dataset structure
        for class_name in ['cat', 'dog']:
            class_path = os.path.join(raw_data_path, class_name)
            if os.path.exists(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(class_path, img_file))
                        labels.append(class_name)
        
        print(f"Found {len(image_paths)} images")
        
        # Split into train/val/test
        X_train, X_temp, y_train, y_temp = train_test_split(
            image_paths, labels, 
            test_size=(1 - self.train_split),
            stratify=labels,
            random_state=42
        )
        
        val_size_adjusted = self.val_split / (self.val_split + self.test_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size_adjusted),
            stratify=y_temp,
            random_state=42
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Process and copy images
        splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        for split_name, (paths, labels) in splits.items():
            for img_path, label in zip(paths, labels):
                # Preprocess image
                processed_img = self.load_and_preprocess_image(img_path)
                
                if processed_img is not None:
                    # Generate filename
                    filename = f"{Path(img_path).stem}.npy"
                    dest_path = os.path.join(
                        processed_data_path, split_name, label, filename
                    )
                    
                    # Save processed image
                    np.save(dest_path, processed_img)
    
    def get_data_statistics(self, processed_data_path: str) -> dict:
        """
        Get statistics about the processed dataset
        
        Args:
            processed_data_path: Path to processed dataset
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {'train': {}, 'val': {}, 'test': {}}
        
        for split in ['train', 'val', 'test']:
            for class_name in ['cat', 'dog']:
                class_path = os.path.join(processed_data_path, split, class_name)
                if os.path.exists(class_path):
                    count = len([f for f in os.listdir(class_path) if f.endswith('.npy')])
                    stats[split][class_name] = count
        
        return stats


def main():
    """Main function to run preprocessing"""
    preprocessor = DataPreprocessor()
    
    raw_data_path = "data/raw"
    processed_data_path = "data/processed"
    
    # Create directory structure
    preprocessor.organize_dataset_structure(raw_data_path, processed_data_path)
    
    # Split and process images
    preprocessor.split_and_copy_images(raw_data_path, processed_data_path)
    
    # Get statistics
    stats = preprocessor.get_data_statistics(processed_data_path)
    print("\nDataset Statistics:")
    for split, class_counts in stats.items():
        print(f"{split}: {class_counts}")


if __name__ == "__main__":
    main()

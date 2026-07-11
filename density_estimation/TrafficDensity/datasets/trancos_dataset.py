import os
import sys
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from typing import Tuple, Optional, List

# Add the directory containing TrafficDensity to sys.path to enable absolute imports with top-level package name
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from TrafficDensity.utils.density_map import generate_density_map

class TrancosDataset(Dataset):
    """
    PyTorch Dataset loader for the TRANCOS traffic dataset.
    Loads traffic scene images, reads vehicle point annotations, generates continuous 
    ground-truth density maps, and applies image normalization and optional augmentations.
    """
    def __init__(
        self, 
        root_dir: str, 
        target_shape: Optional[Tuple[int, int]] = (480, 640), 
        sigma: float = 4.0, 
        is_train: bool = True,
        image_paths: Optional[List[str]] = None
    ):
        """
        Args:
            root_dir: Path to the dataset split directory (e.g., 'TRANCOS - edited/train_data').
            target_shape: Tuple (height, width) specifying the image size for model training. 
                          Should be a multiple of 8. If None, original size is used.
            sigma: Standard deviation of the Gaussian filter used for density map smoothing.
            is_train: If True, applies data augmentations (e.g., random horizontal flips).
            image_paths: Optional pre-selected list of image file paths. If None, all images in root_dir are loaded.
        """
        self.root_dir = root_dir
        self.target_shape = target_shape
        self.sigma = sigma
        self.is_train = is_train
        
        self.image_dir = os.path.join(self.root_dir, "images")
        self.txt_dir = os.path.join(self.root_dir, "txt")
        
        # Load image paths
        if image_paths is not None:
            self.image_paths = image_paths
        else:
            self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
            
        if not self.image_paths:
            raise FileNotFoundError(f"No traffic images found in {self.image_dir}")
            
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image_tensor: Normalized float tensor of shape (3, H_new, W_new).
            density_tensor: Target density map float tensor of shape (1, H_new/8, W_new/8).
        """
        img_path = self.image_paths[idx]
        
        # Load image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]
        
        # Locate corresponding annotation file
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(self.txt_dir, f"{base_name}.txt")
        
        # Load points
        points = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            x, y = float(parts[0]), float(parts[1])
                            points.append((x, y))
                        except ValueError:
                            continue
        points = np.array(points, dtype=np.float32)
        
        # Step 1: Generate original resolution density map
        density = generate_density_map((h_orig, w_orig), points, sigma=self.sigma)
        
        # Step 2: Determine target shape (must be a multiple of 8 for CSRNet's max-pooling)
        if self.target_shape is not None:
            new_h, new_w = self.target_shape
        else:
            new_h = (h_orig // 8) * 8
            new_w = (w_orig // 8) * 8
            
        dens_h, dens_w = new_h // 8, new_w // 8
        
        # Step 3: Resize image to target shape
        if (new_h, new_w) != (h_orig, w_orig):
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img
            
        # Step 4: Resize density map to 1/8th of target shape and scale to preserve vehicle count sum
        density_resized = cv2.resize(density, (dens_w, dens_h), interpolation=cv2.INTER_LINEAR)
        
        orig_sum = density.sum()
        resized_sum = density_resized.sum()
        if resized_sum > 0:
            # Rescale the density map so its integral equals the number of vehicles
            density_resized = density_resized * (orig_sum / resized_sum)
            
        # Step 5: Convert to tensors and normalize
        # F.to_tensor converts image to float tensor of shape (3, H, W) scaled to [0.0, 1.0]
        img_tensor = F.to_tensor(img_resized)
        img_tensor = F.normalize(
            img_tensor, 
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        density_tensor = torch.from_numpy(density_resized).unsqueeze(0).float()
        
        # Step 6: Data augmentation (random horizontal flip during training)
        if self.is_train and np.random.random() > 0.5:
            img_tensor = torch.flip(img_tensor, dims=[2])
            density_tensor = torch.flip(density_tensor, dims=[2])
            
        return img_tensor, density_tensor

if __name__ == "__main__":
    # Test path
    test_train_path = os.path.join(project_root, "TRANCOS - edited", "train_data")
    
    if os.path.exists(test_train_path):
        print(f"Loading dataset from: {test_train_path}")
        dataset = TrancosDataset(root_dir=test_train_path, target_shape=(480, 640), sigma=4.0, is_train=True)
        print(f"Dataset size: {len(dataset)} samples")
        
        img_t, dens_t = dataset[0]
        print(f"Image tensor shape: {img_t.shape}")
        print(f"Density tensor shape: {dens_t.shape}")
        print(f"Sum of density tensor (predicted count): {dens_t.sum().item():.2f}")
        
        # Load the original text file count to compare
        img_name = os.path.splitext(os.path.basename(dataset.image_paths[0]))[0]
        txt_path = os.path.join(test_train_path, "txt", f"{img_name}.txt")
        
        with open(txt_path, 'r') as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        print(f"Original coordinates count: {len(lines)}")
        print("Dataset loading test passed successfully!")
    else:
        print(f"Path not found for manual dataset test: {test_train_path}")

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

# Add the directory containing TrafficDensity to sys.path to enable absolute imports with top-level package name
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from TrafficDensity.models.csrnet import CSRNet
from TrafficDensity.utils.visualize import save_and_show_prediction

def preprocess_image(image_path: str) -> tuple:
    """
    Load and preprocess the image for model inference.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (original_image_rgb, input_tensor).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    # Read using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"Failed to read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h_orig, w_orig = img_rgb.shape[:2]
    
    # CSRNet requires dimensions to be multiples of 8 due to max-pooling layers
    new_h = (h_orig // 8) * 8
    new_w = (w_orig // 8) * 8
    
    if (new_h, new_w) != (h_orig, w_orig):
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img_rgb

    # Normalize image
    # F.to_tensor scales to [0.0, 1.0] and transposes to (C, H, W)
    img_tensor = F.to_tensor(img_resized)
    img_tensor = F.normalize(
        img_tensor, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    # Add batch dimension: shape (1, 3, H, W)
    input_tensor = img_tensor.unsqueeze(0)
    
    return img_rgb, input_tensor

def main():
    parser = argparse.ArgumentParser(description="Run CSRNet inference on a single image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=os.path.join(project_root, "TrafficDensity", "checkpoints", "best_model.pth"),
        help="Path to the trained model checkpoint (.pth file)."
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default=os.path.join(project_root, "TrafficDensity", "prediction_result.png"),
        help="Path to save the resulting visualization plot."
    )
    parser.add_argument("--show", action="store_true", help="Display the visualization window using plt.show() (may block).")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use non-interactive backend if not displaying to avoid blocking
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    # Load model
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint path '{args.model_path}' does not exist.")
        sys.exit(1)
        
    print(f"Loading model checkpoint from {args.model_path}...")
    model = CSRNet(load_weights=False)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # Preprocess image
    print(f"Loading and preprocessing image: {args.image_path}...")
    original_img, input_tensor = preprocess_image(args.image_path)
    input_tensor = input_tensor.to(device)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(input_tensor)
        
    # Get predicted density map
    pred_density = output.squeeze().cpu().numpy()
    
    # Calculate vehicle count
    estimated_count = float(pred_density.sum())
    
    print("\n=========================================================")
    print(f"ESTIMATED VEHICLE COUNT: {estimated_count:.2f}")
    print("=========================================================")
    
    # Save and display visualization
    print(f"Generating visualization and saving to: {args.save_path}...")
    save_and_show_prediction(original_img, pred_density, estimated_count, args.save_path, show=args.show)
    print("Inference completed.")

if __name__ == "__main__":
    main()

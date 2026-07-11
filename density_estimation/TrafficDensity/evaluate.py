import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the directory containing TrafficDensity to sys.path to enable absolute imports with top-level package name
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from TrafficDensity.datasets.trancos_dataset import TrancosDataset
from TrafficDensity.models.csrnet import CSRNet
from TrafficDensity.utils.metrics import get_vehicle_count

def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> tuple:
    """
    Evaluate the model performance on the given dataloader.

    Args:
        model: PyTorch model.
        dataloader: PyTorch DataLoader for evaluation.
        device: Device to run evaluation on.

    Returns:
        A tuple of (mae, rmse, avg_count_error).
    """
    model.eval()
    
    total_abs_error = 0.0
    total_squared_error = 0.0
    total_count_error = 0.0
    num_samples = 0
    
    pbar = tqdm(dataloader, desc="Evaluating", leave=True)
    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate counts
            pred_count = get_vehicle_count(outputs)
            gt_count = get_vehicle_count(targets)
            
            # Compute errors
            error = pred_count - gt_count
            abs_error = abs(error)
            squared_error = error ** 2
            
            total_count_error += error
            total_abs_error += abs_error
            total_squared_error += squared_error
            num_samples += images.size(0)
            
            pbar.set_postfix(mae=total_abs_error / num_samples)
            
    if num_samples == 0:
        return 0.0, 0.0, 0.0
        
    mae = total_abs_error / num_samples
    rmse = (total_squared_error / num_samples) ** 0.5
    avg_count_error = total_count_error / num_samples
    
    return mae, rmse, avg_count_error

def main():
    parser = argparse.ArgumentParser(description="Evaluate CSRNet on the TRANCOS test dataset.")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default=os.path.join(project_root, "TRANCOS - edited", "test_data"),
        help="Path to the TRANCOS test_data directory."
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=os.path.join(project_root, "TrafficDensity", "checkpoints", "best_model.pth"),
        help="Path to the trained model checkpoint (.pth file)."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation (default 1 for varying sizes).")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick dry-run with 3 samples to verify the script.")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' does not exist.")
        sys.exit(1)
        
    import glob
    if args.dry_run:
        print("Dry-run mode activated: limiting evaluation to 3 samples.")
        img_dir = os.path.join(args.dataset_path, "images")
        all_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        test_paths = all_paths[:3]
        test_dataset = TrancosDataset(
            root_dir=args.dataset_path,
            target_shape=None,
            is_train=False,
            image_paths=test_paths
        )
    else:
        test_dataset = TrancosDataset(
            root_dir=args.dataset_path,
            target_shape=None,  # Use original size padded/resized to multiple of 8
            is_train=False
        )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    print(f"Loaded evaluation dataset with {len(test_dataset)} samples.")

    # Load model
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint path '{args.model_path}' does not exist.")
        print("Please train a model first using train.py.")
        sys.exit(1)

    print(f"Loading model checkpoint from {args.model_path}...")
    model = CSRNet(load_weights=False)  # Weights will be loaded from checkpoint, no need to download VGG
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle checkpoint dictionary or direct state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    print("Model loaded successfully.")

    # Run evaluation
    print("\nRunning evaluation on test set...")
    mae, rmse, avg_count_error = evaluate_model(model, test_loader, device)

    # Print results
    print("\n=========================================================")
    print("EVALUATION METRICS RESULTS")
    print("=========================================================")
    print(f"Average MAE (Mean Absolute Error):         {mae:.4f}")
    print(f"Average RMSE (Root Mean Squared Error):     {rmse:.4f}")
    print(f"Average Count Error (Mean Bias):            {avg_count_error:+.4f}")
    print("=========================================================")

if __name__ == "__main__":
    main()

import os
import sys
import argparse
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the directory containing TrafficDensity to sys.path to enable absolute imports with top-level package name
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from TrafficDensity.datasets.trancos_dataset import TrancosDataset
from TrafficDensity.models.csrnet import CSRNet
from TrafficDensity.utils.metrics import calculate_mae, calculate_rmse

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: PyTorch device.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    
    # Progress bar for training batches
    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return running_loss / len(dataloader)

def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> tuple:
    """
    Evaluate the model on the validation dataset.

    Args:
        model: PyTorch model.
        dataloader: Validation DataLoader.
        device: PyTorch device.

    Returns:
        Tuple of (avg_mae, avg_rmse).
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    # Progress bar for validation batches
    pbar = tqdm(dataloader, desc="  Validation", leave=False)
    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
            
    # Concatenate results across all batches if list is not empty
    if len(all_predictions) > 0:
        # Predictions and targets can have variable size if batch size is 1 or if resized.
        # But we can calculate MAE and RMSE per sample and average them.
        total_mae = 0.0
        total_rmse_sq = 0.0
        total_samples = 0
        
        for pred, target in zip(all_predictions, all_targets):
            # pred: shape (B, 1, H, W)
            # target: shape (B, 1, H, W)
            # Calculate counts by summing over density map pixels
            pred_counts = pred.sum(dim=(1, 2, 3))  # Shape (B,)
            target_counts = target.sum(dim=(1, 2, 3))  # Shape (B,)
            
            mae_batch = torch.abs(pred_counts - target_counts)
            rmse_sq_batch = (pred_counts - target_counts) ** 2
            
            total_mae += mae_batch.sum().item()
            total_rmse_sq += rmse_sq_batch.sum().item()
            total_samples += pred.size(0)
            
        avg_mae = total_mae / total_samples
        avg_rmse = (total_rmse_sq / total_samples) ** 0.5
    else:
        avg_mae = float('inf')
        avg_rmse = float('inf')
        
    return avg_mae, avg_rmse

def main():
    parser = argparse.ArgumentParser(description="Train CSRNet on the TRANCOS dataset.")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default=os.path.join(project_root, "TRANCOS - edited", "train_data"),
        help="Path to the TRANCOS train_data directory."
    )
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default=os.path.join(project_root, "TrafficDensity", "checkpoints"),
        help="Directory to save model checkpoints."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for Adam optimizer.")
    parser.add_argument("--sigma", type=float, default=4.0, help="Sigma for density map Gaussian filter.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from.")
    parser.add_argument("--no_pretrained", action="store_true", help="Do not load pretrained VGG-16 weights (useful for quick testing).")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick dry-run with 4 samples to verify the pipeline.")
    parser.add_argument("--colab", action="store_true", help="Mount Google Drive and save checkpoints there.")
    parser.add_argument(
        "--google_drive_dir", 
        type=str, 
        default="/content/drive/MyDrive/TrafficDensity/checkpoints",
        help="Directory on Google Drive to save checkpoints (if --colab is set)."
    )
    args = parser.parse_args()

    if args.dry_run:
        print("Dry-run mode activated: overriding epochs to 1, batch size to 2, and limiting dataset size.")
        args.epochs = 1
        args.batch_size = 2

    # Google Colab Drive Mount Support
    if args.colab:
        try:
            from google.colab import drive
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
            args.checkpoint_path = args.google_drive_dir
            print(f"Checkpoints will be saved to Google Drive at: {args.checkpoint_path}")
        except ImportError:
            print("Warning: Failed to import google.colab. Running in local mode instead.")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seeds for reproducibility of the train/val split
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Gather all image paths in the training split
    img_dir = os.path.join(args.dataset_path, "images")
    all_img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    
    if not all_img_paths:
        print(f"Error: No images found in {img_dir}. Please check your dataset path.")
        sys.exit(1)
        
    # Split into 90% train and 10% val
    random.shuffle(all_img_paths)
    if args.dry_run:
        train_paths = all_img_paths[:2]
        val_paths = all_img_paths[2:4]
    else:
        split_idx = int(0.9 * len(all_img_paths))
        train_paths = all_img_paths[:split_idx]
        val_paths = all_img_paths[split_idx:]
    
    print(f"Total training images: {len(train_paths)}")
    print(f"Total validation images: {len(val_paths)}")

    # Instantiate datasets
    train_dataset = TrancosDataset(
        root_dir=args.dataset_path,
        target_shape=(480, 640),
        sigma=args.sigma,
        is_train=True,
        image_paths=train_paths
    )
    
    val_dataset = TrancosDataset(
        root_dir=args.dataset_path,
        target_shape=(480, 640),
        sigma=args.sigma,
        is_train=False,
        image_paths=val_paths
    )

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Initialize model
    model = CSRNet(load_weights=not args.no_pretrained)
    model = model.to(device)

    # Criterion, Optimizer, Scheduler
    # Pixel-wise MSE Loss
    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Resuming variables
    start_epoch = 0
    best_mae = float('inf')
    best_rmse = float('inf')

    # Resume training if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict') and scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_mae = checkpoint.get('best_mae', float('inf'))
            best_rmse = checkpoint.get('best_rmse', float('inf'))
            print(f"Resumed from epoch {start_epoch - 1}. Best MAE so far: {best_mae:.4f}")
        else:
            print(f"Warning: No checkpoint found at '{args.resume}'. Starting from scratch.")

    print("\nStarting training loop...")
    for epoch in range(start_epoch, args.epochs):
        # Train one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate one epoch
        val_mae, val_rmse = validate_epoch(model, val_loader, device)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_mae': best_mae,
            'best_rmse': best_rmse
        }
        
        # Always save the last model checkpoint
        last_ckpt_path = os.path.join(args.checkpoint_path, "last_model.pth")
        torch.save(checkpoint, last_ckpt_path)
        
        # Check if this epoch yielded the best MAE
        is_best = val_mae < best_mae
        if is_best:
            best_mae = val_mae
            best_rmse = val_rmse
            best_ckpt_path = os.path.join(args.checkpoint_path, "best_model.pth")
            checkpoint['best_mae'] = best_mae
            checkpoint['best_rmse'] = best_rmse
            torch.save(checkpoint, best_ckpt_path)
            
        # Print metrics as required: Epoch, Loss, MAE, RMSE
        print(
            f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MAE: {val_mae:.4f} (Best: {best_mae:.4f}) | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"{'(*New Best Saved!)' if is_best else ''}"
        )
        
    print("\nTraining complete.")
    print(f"Best Validation MAE: {best_mae:.4f} | Best Validation RMSE: {best_rmse:.4f}")

if __name__ == "__main__":
    main()

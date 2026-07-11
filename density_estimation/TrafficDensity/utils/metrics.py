import torch
import numpy as np
from typing import Union

def calculate_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error (MAE) between predicted counts and ground truth counts.

    The total vehicle count is calculated by taking the sum over the density map.
    MAE = (1 / N) * sum(|C_pred - C_gt|)

    Args:
        predictions: PyTorch float tensor of shape (B, C, H, W) or (B, H, W).
        targets: PyTorch float tensor of shape (B, C, H, W) or (B, H, W).

    Returns:
        MAE value as a float.
    """
    # Calculate vehicle count per sample in the batch
    # Summing over all dimensions except batch dimension (dimension 0)
    sum_dims = list(range(1, predictions.ndim))
    
    pred_counts = predictions.sum(dim=sum_dims)
    target_counts = targets.sum(dim=sum_dims)
    
    mae = torch.abs(pred_counts - target_counts).mean().item()
    return float(mae)

def calculate_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Root Mean Squared Error (RMSE) between predicted counts and ground truth counts.

    The total vehicle count is calculated by taking the sum over the density map.
    RMSE = sqrt((1 / N) * sum((C_pred - C_gt)^2))

    Args:
        predictions: PyTorch float tensor of shape (B, C, H, W) or (B, H, W).
        targets: PyTorch float tensor of shape (B, C, H, W) or (B, H, W).

    Returns:
        RMSE value as a float.
    """
    # Calculate vehicle count per sample in the batch
    sum_dims = list(range(1, predictions.ndim))
    
    pred_counts = predictions.sum(dim=sum_dims)
    target_counts = targets.sum(dim=sum_dims)
    
    mse = ((pred_counts - target_counts) ** 2).mean().item()
    rmse = mse ** 0.5
    return float(rmse)

def get_vehicle_count(density_map: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Extract the total vehicle count from a density map.
    
    Args:
        density_map: A density map as a PyTorch tensor or a NumPy array.
        
    Returns:
        The total vehicle count (the sum of all density values).
    """
    if isinstance(density_map, torch.Tensor):
        return float(density_map.sum().item())
    elif isinstance(density_map, np.ndarray):
        return float(density_map.sum())
    else:
        raise TypeError("density_map must be a PyTorch Tensor or a NumPy ndarray")

if __name__ == "__main__":
    # Sanity checks
    # Let's create dummy tensors representing a batch of 3 samples
    dummy_pred = torch.tensor([
        [[[0.1, 0.2], [0.3, 0.4]]],  # Sum = 1.0 (1 vehicle predicted)
        [[[1.0, 1.0], [1.0, 1.0]]],  # Sum = 4.0 (4 vehicles predicted)
        [[[0.0, 0.0], [0.0, 0.0]]]   # Sum = 0.0 (0 vehicles predicted)
    ])
    
    dummy_target = torch.tensor([
        [[[0.2, 0.2], [0.3, 0.3]]],  # Sum = 1.0 (1 vehicle actual) -> Error = 0
        [[[1.0, 1.0], [1.0, 0.0]]],  # Sum = 3.0 (3 vehicles actual) -> Error = 1
        [[[0.5, 0.5], [0.5, 0.5]]]   # Sum = 2.0 (2 vehicles actual) -> Error = -2
    ])
    
    mae = calculate_mae(dummy_pred, dummy_target)
    rmse = calculate_rmse(dummy_pred, dummy_target)
    
    # Expected predictions: [1.0, 4.0, 0.0]
    # Expected targets: [1.0, 3.0, 2.0]
    # Errors: [0.0, 1.0, -2.0] -> Abs errors: [0.0, 1.0, 2.0] -> Mean = 1.0
    # Squared errors: [0.0, 1.0, 4.0] -> Mean = 5/3 = 1.666... -> Sqrt = 1.290994
    print(f"Computed MAE: {mae:.6f} (Expected: 1.0)")
    print(f"Computed RMSE: {rmse:.6f} (Expected: 1.290994)")
    
    assert abs(mae - 1.0) < 1e-5, "MAE calculation incorrect!"
    assert abs(rmse - (5.0/3.0)**0.5) < 1e-5, "RMSE calculation incorrect!"
    print("Metrics check passed!")

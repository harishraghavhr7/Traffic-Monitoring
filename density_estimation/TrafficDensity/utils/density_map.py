import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Union, List

def generate_density_map(
    image_shape: Tuple[int, int],
    points: Union[np.ndarray, List[Tuple[float, float]]],
    sigma: float = 4.0
) -> np.ndarray:
    """
    Generate a 2D density map from coordinate points using a Gaussian filter.

    Each coordinate point (representing a vehicle center) is treated as an impulse (1.0).
    The density map is convolved with a 2D Gaussian kernel of standard deviation `sigma`.
    The sum of the resulting density map will approximately equal the number of points.

    Args:
        image_shape: A tuple containing (height, width) of the target image.
        points: A numpy array or list of shape (N, 2) where each row represents (x, y) coordinates.
        sigma: Standard deviation of the Gaussian filter. Default is 4.0.

    Returns:
        A 2D numpy float32 array representing the density map of shape (height, width).
    """
    # Parse dimensions
    height, width = image_shape[:2]
    
    # Initialize an empty density map
    density_map = np.zeros((height, width), dtype=np.float32)
    
    if len(points) == 0:
        return density_map

    points_arr = np.array(points, dtype=np.float32)
    
    # Count points that fall inside the image boundary
    num_valid_points = 0
    for x, y in points_arr:
        px = int(np.round(x))
        py = int(np.round(y))
        if 0 <= px < width and 0 <= py < height:
            density_map[py, px] += 1.0
            num_valid_points += 1

    # Apply the Gaussian filter to smooth the impulse points
    # Using mode='constant' and cval=0.0 to prevent bleed-in from outside boundary conditions
    density_map = gaussian_filter(density_map, sigma=sigma, mode='constant', cval=0.0)
    
    # Normalize to preserve the exact sum of valid points
    if num_valid_points > 0:
        current_sum = density_map.sum()
        if current_sum > 0:
            density_map = density_map * (num_valid_points / current_sum)
    
    return density_map

if __name__ == "__main__":
    # Quick sanity check
    h, w = 100, 100
    test_points = np.array([
        [10.0, 10.0],
        [20.5, 30.2],
        [50.0, 50.0],
        [99.0, 99.0]  # Point on the border
    ])
    
    density = generate_density_map((h, w), test_points, sigma=4.0)
    
    print(f"Generated Density Map Shape: {density.shape}")
    print(f"Number of points: {len(test_points)}")
    print(f"Sum of density map: {density.sum():.6f}")
    
    # Verify that the sum is equal to the number of points
    assert abs(density.sum() - len(test_points)) < 1e-4, f"Sum {density.sum()} deviates from count {len(test_points)}!"
    print("Density map generation check passed!")

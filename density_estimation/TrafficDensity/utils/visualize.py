import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def generate_heatmap(density_map: np.ndarray) -> np.ndarray:
    """
    Convert a single-channel continuous density map into a 3-channel Jet color heatmap.

    Args:
        density_map: 2D numpy float32 array.

    Returns:
        3D numpy uint8 array (RGB) representing the Jet colormap.
    """
    # Clip any negative values that might be output by the network
    density_map_clipped = np.clip(density_map, a_min=0, a_max=None)
    
    # Normalize to [0.0, 1.0]
    map_min = density_map_clipped.min()
    map_max = density_map_clipped.max()
    denom = map_max - map_min
    
    if denom > 0:
        normalized_map = (density_map_clipped - map_min) / denom
    else:
        normalized_map = np.zeros_like(density_map_clipped)
        
    # Scale to 8-bit representation [0, 255]
    map_8bit = (normalized_map * 255).astype(np.uint8)
    
    # Apply OpenCV BGR JET Colormap
    heatmap_bgr = cv2.applyColorMap(map_8bit, cv2.COLORMAP_JET)
    
    # Convert BGR to RGB
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    return heatmap_rgb

def generate_overlay(
    image: np.ndarray, 
    density_map: np.ndarray, 
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay a Jet color density map heatmap onto the original image.

    Args:
        image: Original RGB image array (H, W, 3) in range [0, 255].
        density_map: 2D numpy float32 density map.
        alpha: Weighting factor for the overlay transparency (0 = only image, 1 = only heatmap).

    Returns:
        3D numpy uint8 array (RGB) representing the overlaid visualization.
    """
    # Generate the heatmap
    heatmap = generate_heatmap(density_map)
    
    # Resize heatmap to match original image dimensions if they differ
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    # Ensure image is in uint8 [0, 255] format
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
    else:
        image_uint8 = image
        
    # Blending original image with heatmap
    overlay = cv2.addWeighted(image_uint8, 1.0 - alpha, heatmap, alpha, 0)
    
    return overlay

def save_and_show_prediction(
    image: np.ndarray, 
    density_map: np.ndarray, 
    estimated_count: float, 
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Generate a 4-panel comparative plot, saving it to disk and displaying it.

    Panels:
        1. Original Image
        2. Greyscale Density Map
        3. Jet Color Heatmap
        4. Heatmap Overlaid on Original Image

    Args:
        image: Original RGB image array (H, W, 3) in range [0, 255].
        density_map: 2D numpy float32 density map.
        estimated_count: The summed prediction count to print on the plot title.
        save_path: File path to save the generated plot. If None, saves to 'prediction.png'.
        show: If True, calls plt.show() to display the plot (may block in interactive environments).
    """
    # Generate visualizations
    heatmap = generate_heatmap(density_map)
    overlay = generate_overlay(image, density_map, alpha=0.4)
    
    # Setup matplotlib figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Original Image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # 2. Greyscale Density Map
    # We display it using gray colormap to highlight raw pixel densities
    axes[1].imshow(density_map, cmap="gray")
    axes[1].set_title(f"Density Map (Sum: {density_map.sum():.2f})")
    axes[1].axis("off")
    
    # 3. Jet Heatmap
    axes[2].imshow(heatmap)
    axes[2].set_title("Jet Heatmap")
    axes[2].axis("off")
    
    # 4. Heatmap Overlay
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay (alpha=0.4)")
    axes[3].axis("off")
    
    # Global title
    plt.suptitle(
        f"Traffic Density Prediction | Estimated Count: {estimated_count:.2f}", 
        fontsize=16, 
        fontweight="bold", 
        y=0.98
    )
    plt.tight_layout()
    
    # Save the figure
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.savefig("prediction.png", bbox_inches="tight", dpi=150)
        
    # Show figure if requested
    if show:
        try:
            plt.show()
        except Exception:
            print("Running in headless environment. Skipping plt.show(). Plot saved successfully.")
    
    plt.close(fig)

if __name__ == "__main__":
    # Use non-interactive backend for testing
    import matplotlib
    matplotlib.use("Agg")
    
    # Sanity checks
    h, w = 120, 160
    dummy_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    dummy_density = np.zeros((h // 8, w // 8), dtype=np.float32)
    # Put three peaks
    dummy_density[5, 5] = 1.0
    dummy_density[10, 15] = 2.0
    
    # Smooth them to simulate a real density map
    from scipy.ndimage import gaussian_filter
    dummy_density = gaussian_filter(dummy_density, sigma=1.0)
    # Scale to count
    dummy_density = dummy_density * (3.0 / dummy_density.sum())
    
    heatmap = generate_heatmap(dummy_density)
    overlay = generate_overlay(dummy_img, dummy_density)
    
    print(f"Dummy Image Shape: {dummy_img.shape}")
    print(f"Dummy Density Shape: {dummy_density.shape}")
    print(f"Heatmap Shape: {heatmap.shape}")
    print(f"Overlay Shape: {overlay.shape}")
    
    assert heatmap.shape == (h // 8, w // 8, 3), "Heatmap shape mismatch!"
    assert overlay.shape == (h, w, 3), "Overlay shape mismatch!"
    
    # Test plotting without blocking
    save_and_show_prediction(dummy_img, dummy_density, 3.0, "dummy_test_plot.png", show=False)
    print("Visualization check passed successfully!")

# --- START OF FILE watermark_utils.py ---

import cv2
import numpy as np
import os
import torch

# === Helper Functions (Tensor <-> OpenCV) ===

def tensor_to_cv2(tensor_img):
    """Converts a PyTorch tensor (C, H, W) in range [-1, 1] to OpenCV image (H, W, C) in range [0, 255]."""
    img = tensor_img.add_(1).div_(2).mul_(255).clamp_(0, 255) # Denorm to [0, 255]
    img = img.permute(1, 2, 0).to('cpu', torch.uint8).numpy() # C,H,W -> H,W,C
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def cv2_to_tensor(cv2_img):
    """Converts an OpenCV image (H, W, C) BGR [0, 255] to PyTorch tensor (C, H, W) [-1, 1]."""
    # Convert BGR to RGB
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose(2, 0, 1)) # H,W,C -> C,H,W
    img = img.float().div_(255) # uint8 [0, 255] -> float [0, 1]
    img = img.mul_(2).add_(-1) # Normalize to [-1, 1]
    return img

# === Watermarking Logic ===

def get_watermark_position(image_shape, watermark_size, position):
    """Get (x, y) coordinates for placing watermark."""
    h, w = image_shape[:2] # Use H, W from image shape
    positions = {
        "top-left": (0, 0),
        "top-right": (w - watermark_size, 0),
        "bottom-left": (0, h - watermark_size),
        "bottom-right": (w - watermark_size, h - watermark_size),
        "center": ((w - watermark_size) // 2, (h - watermark_size) // 2)
    }
    return positions.get(position, (0, 0)) # Default to top-left

def generate_structured_noise(size, strength=10):
    """Creates structured noise (gradient + randomness)."""
    base_noise = np.random.normal(0, strength, (size, size, 3))
    gradient_x = np.linspace(0.8, 1.2, size).reshape((1, size, 1))
    gradient_y = np.linspace(0.8, 1.2, size).reshape((size, 1, 1))
    structured = base_noise * gradient_x * gradient_y # Apply gradient in both axes
    return np.clip(structured, -strength*3, strength*3).astype(np.int16) # Clip noise range slightly


def apply_robust_watermark_cv2(image_cv2, watermark_cv2, size_percent, position, save_noisy_path=None, noisy_strength=10):
    """
    Applies a robust watermark directly to an OpenCV image (numpy array).

    Args:
        image_cv2 (np.ndarray): Input image in OpenCV format (H, W, C), BGR, uint8 [0, 255].
        watermark_cv2 (np.ndarray): Watermark image in OpenCV format (H, W, 3 or 4), BGR/BGRA, uint8 [0, 255].
        size_percent (float): Watermark size as a percentage of image width.
        position (str): Position name (e.g., 'top-left').
        save_noisy_path (str, optional): Path to save the intermediate noisy image. Defaults to None.
        noisy_strength (int): Strength of the structured noise.

    Returns:
        np.ndarray: Watermarked image in OpenCV format (H, W, C), BGR, uint8 [0, 255].
    """
    if image_cv2 is None or watermark_cv2 is None:
        print("❌ Error: Received None for image or watermark in apply_robust_watermark_cv2.")
        return image_cv2 # Return original image on error

    h, w = image_cv2.shape[:2]
    watermark_size = int(w * size_percent / 100)

    # Ensure watermark size is valid
    if watermark_size <= 0 or watermark_size > min(h, w):
        print(f"⚠️ Warning: Calculated watermark size ({watermark_size}) is invalid for image size ({w}x{h}). Skipping watermark.")
        return image_cv2
    if watermark_size > watermark_cv2.shape[0] or watermark_size > watermark_cv2.shape[1]:
         print(f"⚠️ Warning: Resizing watermark ({watermark_cv2.shape[1]}x{watermark_cv2.shape[0]}) up to {watermark_size}x{watermark_size}. Quality might decrease.")


    x, y = get_watermark_position(image_cv2.shape, watermark_size, position)

    # Ensure coordinates are valid
    if x < 0 or y < 0 or x + watermark_size > w or y + watermark_size > h:
        print(f"⚠️ Warning: Watermark position ({x},{y}) size ({watermark_size}) exceeds image bounds ({w}x{h}). Skipping.")
        return image_cv2

    # Resize watermark
    watermark_resized = cv2.resize(watermark_cv2, (watermark_size, watermark_size), interpolation=cv2.INTER_AREA)

    # Extract RGB and alpha mask
    if watermark_resized.shape[2] == 4:
        watermark_rgb = watermark_resized[:, :, :3]
        # Ensure alpha mask is float and in [0, 1]
        alpha_mask = watermark_resized[:, :, 3].astype(np.float32) / 255.0
    else:
        watermark_rgb = watermark_resized
        alpha_mask = np.ones((watermark_size, watermark_size), dtype=np.float32)

    # Target region of interest (ROI)
    roi = image_cv2[y:y+watermark_size, x:x+watermark_size]

    # --- Adaptive Alpha based on ROI Complexity ---
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray_roi, cv2.CV_64F)
    edges = cv2.normalize(np.abs(edges), None, 0, 1, cv2.NORM_MINMAX) # Normalize edge intensity
    edges = cv2.GaussianBlur(edges, (5, 5), 0) # Smooth edges
    # Adaptive alpha: lower in smooth areas, higher in complex areas
    adaptive_alpha = np.clip(alpha_mask * (0.5 + 0.4 * edges), 0.1, 0.9) # Base: 0.5, Range: 0.4. Clip prevents full transparency/opacity
    adaptive_alpha = adaptive_alpha[..., np.newaxis] # Add channel dim for broadcasting
    # ---------------------------------------------

    # --- Add structured noise before watermarking ---
    noise = generate_structured_noise(watermark_size, strength=noisy_strength)
    # Add noise carefully to avoid wrap-around artifacts with uint8
    noisy_roi_int16 = roi.astype(np.int16) + noise
    noisy_roi = np.clip(noisy_roi_int16, 0, 255).astype(np.uint8)
    # ----------------------------------------------

    # Save intermediate noisy image if requested
    if save_noisy_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_noisy_path), exist_ok=True)
            # Create a copy of the *original* image with the *noisy ROI* placed back
            temp_image_for_noisy = image_cv2.copy()
            temp_image_for_noisy[y:y+watermark_size, x:x+watermark_size] = noisy_roi
            cv2.imwrite(save_noisy_path, temp_image_for_noisy)
            # print(f"💾 Saved intermediate noisy image to {save_noisy_path}")
        except Exception as e:
            print(f"❌ Error saving intermediate noisy image to {save_noisy_path}: {e}")


    # Blend watermark onto the *noisy* ROI using the adaptive alpha
    blended_roi = (
        (1 - adaptive_alpha) * noisy_roi.astype(np.float32) +
        adaptive_alpha * watermark_rgb.astype(np.float32)
    )
    blended_roi = np.clip(blended_roi, 0, 255).astype(np.uint8)

    # Create a copy of the original image to place the result
    output_image = image_cv2.copy()
    output_image[y:y+watermark_size, x:x+watermark_size] = blended_roi

    return output_image

# --- END OF FILE watermark_utils.py ---
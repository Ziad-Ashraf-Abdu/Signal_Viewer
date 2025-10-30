"""
File utility functions for image and general file processing
"""
import base64
import io
from PIL import Image
import numpy as np


def pil_from_base64(b64_string):
    """
    Convert base64 string to PIL Image - used by SAR app
    
    Args:
        b64_string: base64 encoded image string (content only, no header)
        
    Returns:
        PIL.Image: decoded image in RGBA format
    """
    return Image.open(io.BytesIO(base64.b64decode(b64_string))).convert('RGBA')


def image_to_base64_bytes(img, fmt='PNG'):
    """
    Convert PIL Image to base64 string - used by SAR app
    
    Args:
        img: PIL Image object
        fmt: image format ('PNG', 'JPEG', etc.)
        
    Returns:
        str: base64 encoded image string (content only)
    """
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def create_data_url_from_image(img, fmt='PNG'):
    """
    Create data URL from PIL Image for HTML display
    
    Args:
        img: PIL Image object
        fmt: image format
        
    Returns:
        str: data URL for HTML img src
    """
    b64_string = image_to_base64_bytes(img, fmt)
    return f"data:image/{fmt.lower()};base64,{b64_string}"


def numpy_to_base64(array, format='PNG'):
    """
    Convert numpy array to base64 image string
    
    Args:
        array: numpy array (H, W) or (H, W, 3/4)
        format: output image format
        
    Returns:
        str: base64 encoded image
    """
    if array.dtype != np.uint8:
        # Normalize to 0-255
        array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
    
    if len(array.shape) == 2:
        # Grayscale to RGB
        img = Image.fromarray(array, mode='L').convert('RGB')
    else:
        img = Image.fromarray(array)
    
    return image_to_base64_bytes(img, format)


def validate_file_size(content_string, max_size_mb=10):
    """
    Validate file size from base64 content
    
    Args:
        content_string: base64 encoded content
        max_size_mb: maximum file size in MB
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Calculate size in bytes (base64 adds ~33% overhead)
    size_bytes = len(content_string) * 3 // 4
    
    if size_bytes > max_size_mb * 1024 * 1024:
        return False, f"File size exceeds {max_size_mb}MB limit"
    
    return True, None
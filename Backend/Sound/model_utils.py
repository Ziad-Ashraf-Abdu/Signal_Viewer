import sys
import os

# Correct path setup - shared is in Backend/shared/
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_path = os.path.join(current_dir, '..', 'shared')

if shared_path not in sys.path:
    sys.path.insert(0, shared_path)

from shared.model_utils import model_manager, ensure_model_loaded

# Re-export for backward compatibility
def ensure_model():
    """Ensure the drone detection model is loaded."""
    return ensure_model_loaded('drone_detection')
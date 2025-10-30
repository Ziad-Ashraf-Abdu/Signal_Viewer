"""
Model management for Doppler application
"""
from shared.model_utils import model_manager

# Initialize models
VELOCITY_MODEL_LOADED = False

def load_models():
    """Load all models for Doppler application"""
    global VELOCITY_MODEL_LOADED
    
    # Load velocity prediction model
    velocity_model, velocity_error = model_manager.load_keras_model(
        'velocity_regressor_dense.h5', 'velocity_prediction'
    )
    VELOCITY_MODEL_LOADED = velocity_model is not None
    if VELOCITY_MODEL_LOADED:
        print("✅ Velocity prediction model loaded successfully!")
    else:
        print(f"⚠️ Could not load velocity prediction model: {velocity_error}")
    
    return VELOCITY_MODEL_LOADED
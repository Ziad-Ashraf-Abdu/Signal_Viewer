"""
Model management for Human application
"""
from shared.model_utils import model_manager

# Initialize models
MODEL_LOADED = False
GENDER_MODEL_LOADED = False

def load_models():
    """Load all models for Human application"""
    global MODEL_LOADED, GENDER_MODEL_LOADED
    
    # Load audio reconstruction model
    reconstruction_model, recon_error = model_manager.load_keras_model(
        './Anti-Aliasing.keras', 'audio_reconstruction'
    )
    MODEL_LOADED = reconstruction_model is not None
    if MODEL_LOADED:
        print("✅ AI model for audio reconstruction loaded successfully!")
    else:
        print(f"⚠️ Could not load reconstruction model: {recon_error}")
    
    # Load gender detection model with error handling
    try:
        gender_processor, gender_model, gender_device, gender_error = model_manager.load_transformers_model(
            "prithivMLmods/Common-Voice-Gender-Detection", 'gender_detection'
        )
        GENDER_MODEL_LOADED = gender_model is not None
        if GENDER_MODEL_LOADED:
            print("✅ Gender detection model loaded successfully!")
        else:
            print(f"⚠️ Could not load gender detection model: {gender_error}")
    except Exception as e:
        print(f"⚠️ Error loading gender detection model: {e}")
        GENDER_MODEL_LOADED = False
    
    return MODEL_LOADED, GENDER_MODEL_LOADED
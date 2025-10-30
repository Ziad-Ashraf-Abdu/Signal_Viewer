"""
Shared configuration and styling constants for all Dash applications
"""
import os

# ============================================
# UI Style Constants (Used by ALL applications)
# ============================================

# Color Palette
PRIMARY_COLOR = "#4A90E2"
SECONDARY_COLOR = "#6c757d"
BACKGROUND_COLOR = "#F7F9FC"
CARD_BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#333333"
SUBTLE_TEXT_COLOR = "#666666"
BORDER_COLOR = "#EAEAEA"
ERROR_COLOR = '#D32F2F'
SUCCESS_COLOR = '#388E3C'
WARNING_COLOR = "#F59E0B"
INFO_COLOR = "#2196F3"

# Typography
FONT_FAMILY = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
FONT_SIZE_SM = "14px"
FONT_SIZE_MD = "16px"
FONT_SIZE_LG = "18px"
FONT_SIZE_XL = "24px"

# Layout
BORDER_RADIUS = "12px"
BOX_SHADOW = "0 4px 12px rgba(0,0,0,0.05)"
SPACING_UNIT = "24px"

# ============================================
# Reusable Component Styles
# ============================================

# Card Style (Used by SAR, Sound, Human, Doppler)
CARD_STYLE = {
    "backgroundColor": CARD_BACKGROUND_COLOR,
    "borderRadius": BORDER_RADIUS,
    "padding": SPACING_UNIT,
    "boxShadow": BOX_SHADOW,
    "border": f"1px solid {BORDER_COLOR}",
    "marginBottom": SPACING_UNIT
}

# Button Base Style
BUTTON_STYLE_BASE = {
    "border": "none",
    "borderRadius": "8px",
    "padding": "14px 24px",
    "fontSize": FONT_SIZE_MD,
    "fontWeight": "600",
    "cursor": "pointer",
    'transition': 'all 0.2s ease',
    'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
    'textAlign': 'center',
    'display': 'inline-block',
    'lineHeight': '1.2',
    'width': '100%'
}

# Button Variants
BUTTON_STYLE_PRIMARY = {
    **BUTTON_STYLE_BASE,
    "backgroundColor": PRIMARY_COLOR,
    "color": "white"
}

BUTTON_STYLE_SECONDARY = {
    **BUTTON_STYLE_BASE,
    "backgroundColor": SECONDARY_COLOR,
    "color": "white"
}

BUTTON_STYLE_SUCCESS = {
    **BUTTON_STYLE_BASE,
    "backgroundColor": SUCCESS_COLOR,
    "color": "white"
}

BUTTON_STYLE_WARNING = {
    **BUTTON_STYLE_BASE,
    "backgroundColor": WARNING_COLOR,
    "color": "white"
}

BUTTON_STYLE_ERROR = {
    **BUTTON_STYLE_BASE,
    "backgroundColor": ERROR_COLOR,
    "color": "white"
}

BUTTON_STYLE_INFO = {
    **BUTTON_STYLE_BASE,
    "backgroundColor": INFO_COLOR,
    "color": "white"
}

BUTTON_DISABLED_STYLE = {
    'backgroundColor': '#cccccc',
    'cursor': 'not-allowed',
    'boxShadow': 'none'
}

# Upload Style (Used by ALL applications)
UPLOAD_STYLE = {
    "width": "100%",
    "height": "120px",
    "lineHeight": "120px",
    "borderWidth": "2px",
    "borderStyle": "dashed",
    "borderRadius": "10px",
    "textAlign": "center",
    "cursor": "pointer",
    "backgroundColor": "#fafafa",
    'transition': 'all 0.3s ease-in-out',
    "borderColor": BORDER_COLOR
}

# ============================================
# Application-Specific Configuration
# ============================================

# Model Configuration
MODEL_CONFIG = {
    "drone_detection": {
        "model_id": "preszzz/drone-audio-detection-05-12",
        "target_sr": 16000
    },
    "gender_detection": {
        "model_id": "prithivMLmods/Common-Voice-Gender-Detection", 
        "target_sr": 16000
    },
    "velocity_prediction": {
        "model_path": "velocity_regressor_dense.h5",
        "target_sr": 3000
    },
    "audio_reconstruction": {
        "model_path": "./Anti-Aliasing.keras",
        "target_sr": 16000
    }
}

# Audio Processing Constants
AUDIO_CONFIG = {
    "max_duration": 20,  # seconds
    "playback_sr": 3000,  # Hz - minimum for browser compatibility
    "chunk_duration": 2,  # seconds for model processing
}

# Physics Constants
PHYSICS_CONSTANTS = {
    "speed_of_sound": 343.0,  # m/s
    "nyquist_multiplier": 2.0,
}

# ============================================
# Helper Functions
# ============================================

def get_button_style(variant="primary", disabled=False):
    """Get button style by variant"""
    style_map = {
        "primary": BUTTON_STYLE_PRIMARY,
        "secondary": BUTTON_STYLE_SECONDARY,
        "success": BUTTON_STYLE_SUCCESS,
        "warning": BUTTON_STYLE_WARNING,
        "error": BUTTON_STYLE_ERROR,
        "info": BUTTON_STYLE_INFO,
    }
    
    style = style_map.get(variant, BUTTON_STYLE_PRIMARY).copy()
    if disabled:
        style.update(BUTTON_DISABLED_STYLE)
    
    return style

def get_model_config(model_name):
    """Get configuration for a specific model"""
    return MODEL_CONFIG.get(model_name, {})
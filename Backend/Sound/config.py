# --- SEARCHABLE COMMENT: Imports ---
import os

# ============================================
# --- SEARCHABLE COMMENT: Model Configuration ---
# ============================================
MODEL_ID = "preszzz/drone-audio-detection-05-12" # Model identifier on Hugging Face Hub
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN") # Optional: Hugging Face API token for private models

# ============================================
# --- SEARCHABLE COMMENT: UI Style Constants ---
# ============================================
PRIMARY_COLOR = "#4A90E2" # Blue
SECONDARY_COLOR = "#6c757d" # Greyish color for buttons/elements
BACKGROUND_COLOR = "#F7F9FC" # Light grey background
CARD_BACKGROUND_COLOR = "#FFFFFF" # White cards
TEXT_COLOR = "#333333" # Dark text
SUBTLE_TEXT_COLOR = "#666666" # Lighter grey text
BORDER_COLOR = "#EAEAEA" # Light border
FONT_FAMILY = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif' # Use Inter font stack
ERROR_COLOR = '#D32F2F' # Red for errors/warnings
SUCCESS_COLOR = '#388E3C' # Green for success messages
WARNING_COLOR = "#F59E0B" # Orange for warnings

# --- SEARCHABLE COMMENT: Card Style ---
CARD_STYLE = {
    "backgroundColor": CARD_BACKGROUND_COLOR, "borderRadius": "12px", "padding": "24px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.05)", "border": f"1px solid {BORDER_COLOR}",
    "marginBottom": "24px" # Added default bottom margin
}

# --- SEARCHABLE COMMENT: Button Styles ---
BUTTON_STYLE = {
    "backgroundColor": PRIMARY_COLOR, "color": "white", "border": "none",
    "borderRadius": "8px", "padding": "14px 24px", "fontSize": "16px",
    "fontWeight": "600", "cursor": "pointer", 'transition': 'all 0.2s ease',
    'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'textAlign': 'center', 'display':'inline-block',
    'lineHeight': '1.2', 'width': '100%' # Default buttons to full width
}
BUTTON_DISABLED_STYLE = { # Style for disabled buttons
    'backgroundColor': '#cccccc', 'cursor': 'not-allowed', 'boxShadow': 'none'
}
BUTTON_STYLE_PRIMARY = {**BUTTON_STYLE}
BUTTON_STYLE_SECONDARY = {**BUTTON_STYLE, "backgroundColor": SECONDARY_COLOR}
BUTTON_STYLE_SUCCESS = {**BUTTON_STYLE, "backgroundColor": SUCCESS_COLOR}
BUTTON_STYLE_WARNING = {**BUTTON_STYLE, "backgroundColor": WARNING_COLOR}
BUTTON_STYLE_ERROR = {**BUTTON_STYLE, "backgroundColor": ERROR_COLOR}

# --- SEARCHABLE COMMENT: Upload Style ---
UPLOAD_STYLE = {
    "width": "100%", "height": "120px", "lineHeight": "120px", "borderWidth": "2px",
    "borderStyle": "dashed", "borderRadius": "10px", "borderColor": "#d0d0d0",
    "textAlign": "center", "cursor": "pointer", "backgroundColor": "#fafafa",
    'transition': 'all 0.3s ease-in-out', "marginBottom": "20px"
}

# Add any other constants you might need here

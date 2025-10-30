"""
Shared utilities for Dash applications
"""
from .config import *
from .audio_processing import AudioProcessor
from .file_utils import pil_from_base64, image_to_base64_bytes
from .analysis_utils import AnalysisUtils
from .plotting_utils import PlottingUtils
from .model_utils import ModelManager
from .ui_components import create_upload_component, create_card, create_button

__version__ = "1.0.0"
__all__ = [
    'AudioProcessor',
    'AnalysisUtils', 
    'PlottingUtils',
    'ModelManager',
    'create_upload_component',
    'create_card',
    'create_button',
    'pil_from_base64',
    'image_to_base64_bytes',
]
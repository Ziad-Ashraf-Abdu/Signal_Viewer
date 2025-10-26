# --- SEARCHABLE COMMENT: Imports ---
import plotly.graph_objs as go
import numpy as np

# --- SEARCHABLE COMMENT: Import Config ---
# Import style constants
import config

# ============================================
# --- SEARCHABLE COMMENT: Plotting Helpers ---
# ============================================

# --- SEARCHABLE COMMENT: Initial Waveform Plot Function ---
def create_initial_figure(audio_data):
    """
    Generates a Plotly figure showing a preview of the uploaded audio waveform.
    Uses Scattergl for better performance with potentially large datasets.
    Downsamples the displayed data if the audio file is very long.

    Args:
        audio_data (np.ndarray): The audio signal data.

    Returns:
        plotly.graph_objs.Figure: The generated figure object.
    """
    fig = go.Figure()
    max_points = 50000 # Limit points on the plot for performance
    if len(audio_data) <= max_points:
        step = 1
        x_display = np.arange(len(audio_data)) # Use sample index for x-axis
        y_display = audio_data
    else:
        # Downsample data for plotting if too long
        step = max(1, len(audio_data) // max_points)
        y_display = audio_data[::step]
        x_display = np.arange(len(y_display)) * step # Adjust x-axis for downsampling

    # --- SEARCHABLE COMMENT: Initial Waveform Trace ---
    fig.add_trace(go.Scattergl(x=x_display, y=y_display, mode="lines", line=dict(color=config.PRIMARY_COLOR, width=1.5))) # Slightly thicker line

    # --- SEARCHABLE COMMENT: Initial Waveform Layout ---
    fig.update_layout(
        title="Waveform Preview",
        margin=dict(l=50, r=30, t=50, b=50), # Adjusted margins for better spacing
        height=280, # Slightly taller plot
        xaxis_title="Sample Index", yaxis_title="Amplitude",
        plot_bgcolor=config.CARD_BACKGROUND_COLOR, # Match card background for seamless look
        paper_bgcolor=config.CARD_BACKGROUND_COLOR,
        font=dict(family=config.FONT_FAMILY, color=config.TEXT_COLOR), # Use consistent font and color
        xaxis=dict(gridcolor=config.BORDER_COLOR), # Lighter grid lines
        yaxis=dict(gridcolor=config.BORDER_COLOR, zerolinecolor=config.BORDER_COLOR) # Lighter grid and zero lines
    )
    return fig

# --- SEARCHABLE COMMENT: Resampled Waveform Plot Function ---
# --- SEARCHABLE COMMENT: Aliasing Visualization ---
def create_resampled_figure(original_audio, original_sr, new_sr, max_freq):
    """
    Generates a Plotly figure comparing a segment of the original audio
    with the points that would be sampled at the `new_sr`.
    Visually demonstrates the effect of sampling and potential aliasing.

    Args:
        original_audio (np.ndarray): The original full audio signal.
        original_sr (int): The original sampling rate.
        new_sr (int): The target sampling rate from the slider.
        max_freq (float): Estimated maximum frequency in the original signal.

    Returns:
        plotly.graph_objs.Figure: The generated figure object.
    """
    fig = go.Figure()
    nyquist_rate = 2 * max_freq
    title_text = "Waveform Sampling (Zoomed View)"
    is_aliasing = new_sr < nyquist_rate # --- SEARCHABLE COMMENT: Aliasing Check ---

    # --- SEARCHABLE COMMENT: Aliasing Warning Title ---
    if is_aliasing:
        # Modify title to warn about aliasing
        title_text = f"⚠️ Aliasing Likely (Fs={new_sr} Hz < Nyquist={nyquist_rate:.0f} Hz)"
        title_font_color = config.ERROR_COLOR # Use error color for title
    else:
        title_font_color = config.TEXT_COLOR # Use default text color

    # Display a short segment (e.g., 50ms) for clarity
    display_duration_s = 0.05
    display_samples_orig = int(min(len(original_audio), original_sr * display_duration_s))

    # Handle cases where the segment is too short to plot
    if display_samples_orig < 2:
        fig.update_layout(title="Audio segment too short to display sampling visualization.")
        return fig

    original_audio_segment = original_audio[:display_samples_orig]
    # Time axis for the original segment
    time_original = np.linspace(0, display_samples_orig / original_sr, num=display_samples_orig)

    # --- SEARCHABLE COMMENT: Original Signal Trace (Resampled Plot) ---
    # Plot the original signal segment faintly in the background
    fig.add_trace(go.Scattergl(
        x=time_original, y=original_audio_segment, mode="lines",
        line=dict(color='rgba(150, 150, 150, 0.5)', width=2), name="Original Signal" # Thicker faint line
    ))

    # --- SEARCHABLE COMMENT: Sample Point Calculation ---
    # Calculate which points *would be* sampled at the new rate within this segment
    num_samples_new = int(display_duration_s * new_sr)
    if num_samples_new < 2: # Need at least 2 points to show sampling
        fig.update_layout(title=title_text + " - (Target SR too low for visualization)", title_font_color=title_font_color)
        return fig

    # Determine the indices in the *original* segment array corresponding to the new sample times
    max_index = display_samples_orig - 1
    sample_indices_orig = np.linspace(0, max_index, num=num_samples_new, dtype=int)
    # Ensure indices are within bounds (important due to potential floating point inaccuracies)
    sample_indices_orig = np.clip(sample_indices_orig, 0, max_index)

    # Get the amplitude values and times for the sampled points
    sampled_audio = original_audio_segment[sample_indices_orig]
    time_sampled = sample_indices_orig / original_sr # Calculate exact time of each sample

    # --- SEARCHABLE COMMENT: Sampled Signal Trace (Resampled Plot) ---
    # Plot the sampled points and connect them with lines
    line_color = config.ERROR_COLOR if is_aliasing else config.PRIMARY_COLOR # --- SEARCHABLE COMMENT: Aliasing Color Change ---
    fig.add_trace(go.Scattergl(
        x=time_sampled, y=sampled_audio, mode="lines+markers",
        line=dict(color=line_color, width=1.5), # Sampled line style
        marker=dict(color=line_color, size=7, symbol='circle-open'), # Marker style for sample points
        name=f"Sampled at {new_sr} Hz"
    ))

    # --- SEARCHABLE COMMENT: Resampled Plot Layout ---
    fig.update_layout(
        title=dict(text=title_text, font=dict(color=title_font_color, size=16)), # Styled title
        margin=dict(l=50, r=30, t=60, b=50), height=320, # Adjusted margins and height
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        plot_bgcolor=config.CARD_BACKGROUND_COLOR, paper_bgcolor=config.CARD_BACKGROUND_COLOR, # Match card background
        font=dict(family=config.FONT_FAMILY, color=config.TEXT_COLOR), # Consistent font
        xaxis=dict(gridcolor=config.BORDER_COLOR), # Lighter grid
        yaxis=dict(gridcolor=config.BORDER_COLOR, zerolinecolor=config.BORDER_COLOR),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255,255,255,0.7)') # Semi-transparent legend
    )
    return fig

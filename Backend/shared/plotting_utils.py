"""
Unified plotting utilities for creating consistent visualizations across all applications
"""
import plotly.graph_objs as go
import numpy as np
import pandas as pd

from .config import *


class PlottingUtils:
    """
    Create consistent, styled plots for SAR, Sound, Human, and Doppler applications
    """
    
    # ==================== BASE PLOT CONFIGURATION ====================
    
    @staticmethod
    def create_figure_layout(title, height=400, width=None, showlegend=True):
        """
        Create consistent figure layout with application styling
        
        Args:
            title: plot title
            height: plot height in pixels
            width: plot width in pixels (optional)
            showlegend: whether to show legend
            
        Returns:
            dict: Plotly layout configuration
        """
        layout = dict(
            title=dict(
                text=title,
                font=dict(family=FONT_FAMILY, size=16, color=TEXT_COLOR),
                x=0.5,
                xanchor='center'
            ),
            margin=dict(l=60, r=30, t=60, b=60),
            height=height,
            plot_bgcolor=CARD_BACKGROUND_COLOR,
            paper_bgcolor=CARD_BACKGROUND_COLOR,
            font=dict(family=FONT_FAMILY, color=TEXT_COLOR, size=12),
            xaxis=dict(
                gridcolor=BORDER_COLOR,
                zerolinecolor=BORDER_COLOR,
                showgrid=True
            ),
            yaxis=dict(
                gridcolor=BORDER_COLOR,
                zerolinecolor=BORDER_COLOR,
                showgrid=True
            ),
            showlegend=showlegend,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255,255,255,0.7)'
            )
        )
        
        if width:
            layout['width'] = width
            
        return layout
    
    # ==================== SAR APPLICATION PLOTS ====================
    
    @staticmethod
    def create_histogram_plot(hist_df, title="Intensity Distribution"):
        """
        Create histogram plot for SAR image analysis
        
        Args:
            hist_df: DataFrame with 'intensity' and 'count' columns
            title: plot title
            
        Returns:
            plotly.graph_objs.Figure: histogram figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=hist_df['intensity'],
            y=hist_df['count'],
            marker_color=PRIMARY_COLOR,
            opacity=0.8,
            name="Intensity Distribution"
        ))
        
        fig.update_layout(PlottingUtils.create_figure_layout(title, height=400))
        fig.update_xaxes(title_text="Intensity Value")
        fig.update_yaxes(title_text="Pixel Count")
        
        return fig
    
    @staticmethod
    def create_image_with_stats(pil_img, stats):
        """
        Create image display with statistics overlay (for SAR app)
        
        Args:
            pil_img: PIL Image object
            stats: image statistics dictionary
            
        Returns:
            plotly.graph_objs.Figure: image figure with annotations
        """
        from .file_utils import create_data_url_from_image
        
        # Convert image to data URL
        img_src = create_data_url_from_image(pil_img)
        
        # Create figure with image
        fig = go.Figure()
        
        # Add image as layout image
        fig.add_layout_image(
            dict(
                source=img_src,
                xref="paper", yref="paper",
                x=0, y=1,
                sizex=1, sizey=1,
                xanchor="left", yanchor="top",
                layer="below"
            )
        )
        
        # Add statistics as annotations
        annotations = []
        stats_text = "<br>".join([f"{k}: {v}" for k, v in stats.items() 
                                 if k in ['mean', 'stdDev', 'min', 'max', 'pixels']])
        
        annotations.append(dict(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=stats_text,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=BORDER_COLOR,
            borderwidth=1,
            font=dict(size=10, color=TEXT_COLOR)
        ))
        
        fig.update_layout(
            PlottingUtils.create_figure_layout("SAR Image with Statistics", height=500),
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    # ==================== AUDIO APPLICATION PLOTS ====================
    
    @staticmethod
    def create_waveform_plot(audio_data, sr, title="Waveform", max_points=50000):
        """
        Create waveform plot for audio applications
        
        Args:
            audio_data: audio signal array
            sr: sample rate
            title: plot title
            max_points: maximum points to display for performance
            
        Returns:
            plotly.graph_objs.Figure: waveform figure
        """
        fig = go.Figure()
        
        # Handle large datasets by downsampling
        if len(audio_data) <= max_points:
            time_axis = np.arange(len(audio_data)) / sr
            display_data = audio_data
        else:
            step = max(1, len(audio_data) // max_points)
            time_axis = np.arange(0, len(audio_data), step) / sr
            display_data = audio_data[::step]
        
        fig.add_trace(go.Scattergl(
            x=time_axis,
            y=display_data,
            mode="lines",
            line=dict(color=PRIMARY_COLOR, width=1.5),
            name="Waveform"
        ))
        
        fig.update_layout(PlottingUtils.create_figure_layout(title, height=300))
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Amplitude")
        
        return fig
    
    @staticmethod
    def create_spectrum_plot(frequencies, magnitude, title="Frequency Spectrum", 
                           nyquist_freq=None, show_nyquist=True):
        """
        Create frequency spectrum plot
        
        Args:
            frequencies: frequency array (Hz)
            magnitude: magnitude spectrum
            title: plot title
            nyquist_freq: Nyquist frequency for reference line
            show_nyquist: whether to show Nyquist line
            
        Returns:
            plotly.graph_objs.Figure: spectrum figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scattergl(
            x=frequencies,
            y=magnitude,
            mode="lines",
            line=dict(color=PRIMARY_COLOR, width=2),
            name="Spectrum"
        ))
        
        # Add Nyquist frequency line if provided
        if show_nyquist and nyquist_freq is not None:
            fig.add_vline(
                x=nyquist_freq,
                line_dash="dash",
                line_color=WARNING_COLOR,
                annotation_text=f"Nyquist: {nyquist_freq/1000:.1f} kHz",
                annotation_position="top right"
            )
        
        fig.update_layout(PlottingUtils.create_figure_layout(title, height=350))
        fig.update_xaxes(title_text="Frequency (Hz)")
        fig.update_yaxes(title_text="Normalized Magnitude")
        
        return fig
    
    @staticmethod
    def create_comparison_plot(signals_dict, title="Signal Comparison", view_mode='overlap'):
        """
        Create comparison plot for multiple signals
        
        Args:
            signals_dict: dictionary of {signal_name: {freq: [], mag: [], color: str}}
            title: plot title
            view_mode: 'overlap' or 'separate'
            
        Returns:
            plotly.graph_objs.Figure or list: figure(s) for comparison
        """
        if view_mode == 'overlap':
            return PlottingUtils._create_overlap_plot(signals_dict, title)
        else:
            return PlottingUtils._create_separate_plots(signals_dict, title)
    
    @staticmethod
    def _create_overlap_plot(signals_dict, title):
        """Create single plot with overlapping signals"""
        fig = go.Figure()
        
        color_palette = [PRIMARY_COLOR, ERROR_COLOR, SUCCESS_COLOR, WARNING_COLOR, INFO_COLOR]
        
        for i, (name, signal_data) in enumerate(signals_dict.items()):
            color = signal_data.get('color', color_palette[i % len(color_palette)])
            
            fig.add_trace(go.Scattergl(
                x=signal_data['frequencies'],
                y=signal_data['magnitude'],
                mode="lines",
                line=dict(color=color, width=2),
                name=name
            ))
        
        fig.update_layout(PlottingUtils.create_figure_layout(title, height=400))
        fig.update_xaxes(title_text="Frequency (Hz)")
        fig.update_yaxes(title_text="Normalized Magnitude")
        
        return fig
    
    @staticmethod
    def _create_separate_plots(signals_dict, title):
        """Create separate subplots for each signal"""
        from dash import dcc
        import dash_bootstrap_components as dbc
        
        figures = []
        color_palette = [PRIMARY_COLOR, ERROR_COLOR, SUCCESS_COLOR, WARNING_COLOR, INFO_COLOR]
        
        for i, (name, signal_data) in enumerate(signals_dict.items()):
            color = signal_data.get('color', color_palette[i % len(color_palette)])
            
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=signal_data['frequencies'],
                y=signal_data['magnitude'],
                mode="lines",
                line=dict(color=color, width=2),
                name=name
            ))
            
            fig.update_layout(PlottingUtils.create_figure_layout(
                f"{name} Spectrum", height=300, showlegend=False
            ))
            fig.update_xaxes(title_text="Frequency (Hz)")
            fig.update_yaxes(title_text="Normalized Magnitude")
            
            figures.append(dcc.Graph(figure=fig))
        
        # Return as Bootstrap columns
        num_signals = len(signals_dict)
        col_width = 12 // min(num_signals, 3)  # Max 3 columns per row
        
        rows = []
        current_row = []
        
        for i, fig in enumerate(figures):
            current_row.append(dbc.Col(fig, width=12, md=col_width, className="mb-3"))
            
            if (i + 1) % 3 == 0 or i == len(figures) - 1:
                rows.append(dbc.Row(current_row, className="g-3"))
                current_row = []
        
        return rows
    
    # ==================== DOPPLER APPLICATION PLOTS ====================
    
    @staticmethod
    def create_doppler_simulation_plot(source_pos, observer_pos, wave_fronts=None, 
                                     title="Doppler Effect Simulation"):
        """
        Create Doppler effect simulation visualization
        
        Args:
            source_pos: (x, y) tuple of source position
            observer_pos: (x, y) tuple of observer position
            wave_fronts: list of wave front radii
            title: plot title
            
        Returns:
            plotly.graph_objs.Figure: simulation figure
        """
        fig = go.Figure()
        
        # Add observer
        fig.add_trace(go.Scatter(
            x=[observer_pos[0]], y=[observer_pos[1]],
            mode='markers+text',
            marker=dict(size=16, color=INFO_COLOR, line=dict(color='white', width=2)),
            text=['👂 Observer'],
            textposition='top center',
            name='Observer'
        ))
        
        # Add wave fronts if provided
        if wave_fronts:
            for i, radius in enumerate(wave_fronts):
                if radius > 0:
                    opacity = 1 - (i / len(wave_fronts)) * 0.7
                    fig.add_shape(
                        type="circle",
                        x0=source_pos[0] - radius, y0=source_pos[1] - radius,
                        x1=source_pos[0] + radius, y1=source_pos[1] + radius,
                        line=dict(color=f"rgba(102, 126, 234, {opacity})", dash="dot", width=2)
                    )
        
        fig.update_layout(
            PlottingUtils.create_figure_layout(title, height=500, showlegend=False),
            xaxis=dict(
                title="X Position (meters)",
                range=[-300, 300],
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)'
            ),
            yaxis=dict(
                title="Y Position (meters)", 
                range=[-150, 150],
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)'
            )
        )
        
        return fig


# Global instance for convenience
plotting_utils = PlottingUtils()
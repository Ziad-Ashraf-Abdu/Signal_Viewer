"""
Callback functions for Doppler application
"""
import base64
import io
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback_context, no_update, html, dcc
import urllib.parse

# Import shared modules
from shared.audio_processing import audio_processor
from shared.analysis_utils import analysis_utils

# Import local modules - use absolute imports
from utils import load_audio_from_bytes, detect_dominant_frequency, predict_velocity, create_audio_spectrum_plot
from models import VELOCITY_MODEL_LOADED

# Import Doppler simulator
from doppler_simulator import DopplerSimulator

doppler_sim = DopplerSimulator()

def register_callbacks(app):
    """Register all callbacks for the Doppler application"""
    
    @app.callback(
        [
            Output('uploaded-audio-data', 'data'),
            Output('aliasing-audio', 'data'),
            Output('upload-status', 'children'),
            Output('fs-slider', 'disabled'),
            Output('fs-slider', 'max'),
            Output('fs-slider', 'value'),
            Output('fs-slider', 'marks'),
            Output('audio-orig', 'src'),
            Output('detected-freq-display', 'children'),
            Output('use-audio-freq-btn', 'disabled'),
            Output('audio-spectrum-graph', 'figure'),
            Output('predicted-velocity-display', 'children'),
            Output('predicted-velocity-store', 'data'),
            Output('use-audio-velocity-btn', 'disabled'),
        ],
        Input('upload-audio', 'contents'),
        State('upload-audio', 'filename')
    )
    def handle_upload_and_analyze(contents, filename):
        if contents is None:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No Audio Uploaded", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
            return [None] * 7 + ["", "Detected Frequency: — Hz", True, empty_fig, "Predicted Velocity: — m/s", None, True]

        try:
            _, b64 = contents.split(',')
            wav_bytes = base64.b64decode(b64)
            buffer = io.BytesIO(wav_bytes)

            # Load audio data
            data_float, sr = load_audio_from_bytes(buffer, filename)
            aliasing_audio = data_float.tolist()

            # Detect dominant frequency
            dominant_freq = detect_dominant_frequency(data_float, sr)

            # Predict velocity
            predicted_velocity = predict_velocity(data_float, sr)
            velocity_text = f"Predicted Velocity: {predicted_velocity:.2f} m/s" if predicted_velocity is not None else "Predicted Velocity: — m/s"
            velocity_button_disabled = predicted_velocity is None

            # Create spectrum plot
            fig = create_audio_spectrum_plot(data_float, sr, dominant_freq)

            # Slider marks
            marks = {i: f"{i // 1000}k" for i in range(1000, sr + 1, max(2000, sr // 8))}
            marks[sr] = f"{sr // 1000}k"

            # Original audio playable
            orig_src = audio_processor.make_playable_wav(data_float, sr)

            # Store for Doppler section
            doppler_data = {'y': data_float.tolist(), 'sr': int(sr), 'dominant_freq': dominant_freq}

            return (
                doppler_data, aliasing_audio, f"✅ Loaded: {filename} | Fs = {sr} Hz",
                False, sr, min(sr, 22050), marks, orig_src,
                f"Detected Frequency: {dominant_freq:.1f} Hz", False, fig,
                velocity_text, predicted_velocity, velocity_button_disabled
            )

        except Exception as e:
            empty_fig = go.Figure().update_layout(title="Error")
            return (
                None, None, f"❌ An unexpected error occurred: {str(e)}", True, 44100, 22050, {}, "",
                "Detected Frequency: — Hz", True, empty_fig,
                "Predicted Velocity: — m/s", None, True
            )

    @app.callback(
        Output('audio-alias', 'src'),
        Input('fs-slider', 'value'),
        State('aliasing-audio', 'data'),
        State('uploaded-audio-data', 'data')
    )
    def update_aliasing(target_fs, audio_list, doppler_data):
        if audio_list is None or doppler_data is None:
            return ""
        data = np.array(audio_list, dtype=np.float32)
        orig_sr = doppler_data['sr']
        
        # Use shared audio processor for downsampling
        aliased, aliased_sr = audio_processor.downsample_with_aliasing(data, orig_sr, target_fs)
        return audio_processor.make_playable_wav(aliased, aliased_sr)

    @app.callback(
        Output('slider-value-display', 'children'),
        Input('fs-slider', 'value')
    )
    def display_slider_value(value):
        return f"Aliasing Sampling Frequency: {value} Hz"

    @app.callback(
        [Output('source-speed', 'value'), Output('source-type', 'value')],
        Input('use-audio-velocity-btn', 'n_clicks'),
        State('predicted-velocity-store', 'data'),
        prevent_initial_call=True
    )
    def use_audio_velocity(n_clicks, velocity):
        if velocity is not None:
            return round(abs(velocity), 2), 'moving'
        return no_update, no_update

    @app.callback(
        Output('freq-input', 'value'),
        Input('use-audio-freq-btn', 'n_clicks'),
        State('uploaded-audio-data', 'data'),
        prevent_initial_call=True
    )
    def use_audio_freq(n_clicks, audio_data):
        if audio_data and 'dominant_freq' in audio_data:
            return int(audio_data['dominant_freq'])
        return 500

    @app.callback(
        Output('source-vel-inputs', 'style'),
        Input('source-type', 'value')
    )
    def toggle_source_vel(source_type):
        return {} if source_type == 'moving' else {'display': 'none'}

    @app.callback(
        Output('observer-vel-inputs', 'style'),
        Input('observer-type', 'value')
    )
    def toggle_observer_vel(observer_type):
        return {} if observer_type == 'moving' else {'display': 'none'}

    @app.callback(
        [Output('interval', 'disabled'), Output('simulation-running', 'data'), Output('time-elapsed', 'data')],
        [Input('start-btn', 'n_clicks'), Input('pause-btn', 'n_clicks'), Input('reset-btn', 'n_clicks'), Input('interval', 'n_intervals')],
        [State('simulation-running', 'data'), State('time-elapsed', 'data')]
    )
    def control_simulation(start_clicks, pause_clicks, reset_clicks, n_intervals, is_running, time_elapsed):
        ctx = callback_context
        if not ctx.triggered:
            return True, False, 0
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'reset-btn':
            return True, False, 0
        if trigger_id == 'start-btn':
            return False, True, time_elapsed
        if trigger_id == 'pause-btn':
            return True, False, time_elapsed
        if trigger_id == 'interval' and is_running:
            return False, True, round(time_elapsed + 0.1, 4)
        return True, False, time_elapsed

    @app.callback(
        [Output('simulation-graph', 'figure'), Output('frequency-display', 'children'), Output('sound-freq', 'data')],
        [Input('simulation-running', 'data'), Input('time-elapsed', 'data'), Input('freq-input', 'value'),
         Input('source-type', 'value'), Input('observer-type', 'value'),
         Input('source-x0', 'value'), Input('source-y0', 'value'),
         Input('observer-x0', 'value'), Input('observer-y0', 'value'),
         Input('source-speed', 'value'), Input('source-dir', 'value'),
         Input('observer-speed', 'value'), Input('observer-dir', 'value')]
    )
    def update_display(is_running, t, f_emit,
                       src_type, obs_type,
                       src_x0, src_y0, obs_x0, obs_y0,
                       src_speed, src_dir, obs_speed, obs_dir):
        src_x0 = src_x0 if src_x0 is not None else -200
        src_y0 = src_y0 if src_y0 is not None else 0
        obs_x0 = obs_x0 if obs_x0 is not None else 0
        obs_y0 = obs_y0 if obs_y0 is not None else 0
        f_emit = f_emit if f_emit is not None else 500
        src_speed = src_speed if src_speed is not None else (30 if src_type == 'moving' else 0)
        src_dir = src_dir if src_dir is not None else 0
        obs_speed = obs_speed if obs_speed is not None else (10 if obs_type == 'moving' else 0)
        obs_dir = obs_dir if obs_dir is not None else 0

        src_x, src_y, obs_x, obs_y = doppler_sim.compute_positions(
            t,
            (src_type, src_x0, src_y0, src_speed, src_dir),
            (obs_type, obs_x0, obs_y0, obs_speed, obs_dir)
        )

        f_perceived = doppler_sim.compute_perceived_frequency(
            f_emit, src_x, src_y, obs_x, obs_y,
            src_type, src_speed, src_dir,
            obs_type, obs_speed, obs_dir
        )

        sound_freq = f_perceived if is_running else 0
        status = "Running" if is_running else ("Paused" if t > 0 else "Stopped")
        freq_text = f"{status}  |  Emitted: {f_emit} Hz  →  Perceived: {f_perceived} Hz  |  Time: {t:.1f} s"
        if is_running:
            freq_text = f"🔊 Emitted: {f_emit} Hz  →  Perceived: {f_perceived} Hz  |  Time: {t:.1f} s"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[obs_x], y=[obs_y],
            mode='markers+text',
            marker=dict(size=16, color='#f093fb', line=dict(color='white', width=2)),
            text=['👂 Observer'],
            textposition='top center',
            textfont=dict(size=14, color='#1f2937', family='Arial Black')
        ))

        car_svg = '''
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <g transform="translate(50,50) rotate({angle}) translate(-50,-50)">
        <rect x="20" y="40" width="60" height="25" rx="5" fill="#667eea" stroke="white" stroke-width="2"/>
        <path d="M 30 40 L 35 25 L 65 25 L 70 40 Z" fill="#667eea" stroke="white" stroke-width="2"/>
        <rect x="36" y="28" width="12" height="10" rx="2" fill="#E3F2FD"/>
        <rect x="52" y="28" width="12" height="10" rx="2" fill="#E3F2FD"/>
        <circle cx="32" cy="65" r="6" fill="#2c3e50" stroke="white" stroke-width="1.5"/>
        <circle cx="68" cy="65" r="6" fill="#2c3e50" stroke="white" stroke-width="1.5"/>
        <circle cx="78" cy="45" r="2.5" fill="#FFF59D"/>
        <circle cx="78" cy="55" r="2.5" fill="#FFF59D"/>
    </g>
</svg>
        '''.format(angle=src_dir if src_type == 'moving' else 0)

        svg_encoded = urllib.parse.quote(car_svg.strip(), safe='')
        svg_data_url = f"data:image/svg+xml;charset=utf-8,{svg_encoded}"
        fig.add_layout_image(
            dict(
                source=svg_data_url,
                x=src_x,
                y=src_y,
                xref="x",
                yref="y",
                sizex=40,
                sizey=40,
                xanchor="center",
                yanchor="middle",
                layer="above"
            )
        )

        try:
            for n in range(6):
                t_emit = max(0, t - n * (1.0 / max(1, f_emit)))
                radius = 343.0 * (t - t_emit)
                if radius > 0:
                    opacity = 1 - (n / 6) * 0.7
                    fig.add_shape(
                        type="circle",
                        xref="x", yref="y",
                        x0=src_x - radius, y0=src_y - radius,
                        x1=src_x + radius, y1=src_y + radius,
                        line=dict(color=f"rgba(102, 126, 234, {opacity})", dash="dot", width=2)
                    )
        except:
            pass

        fig.update_layout(
            xaxis=dict(
                title="X Position (meters)",
                range=[-300, 300],
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)',
                zerolinecolor='rgba(0,0,0,0.2)'
            ),
            yaxis=dict(
                title="Y Position (meters)",
                range=[-150, 150],
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)',
                zerolinecolor='rgba(0,0,0,0.2)'
            ),
            showlegend=False,
            title={'text': "Real-Time Doppler Effect Visualization", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#1f2937', 'family': 'Arial Black'}},
            plot_bgcolor='rgba(249, 250, 251, 0.5)',
            paper_bgcolor='white',
            margin=dict(l=60, r=60, t=80, b=60),
            font=dict(family='Arial, sans-serif', size=12, color='#374151')
        )
        return fig, freq_text, sound_freq

    # Client-side audio callbacks
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks || n_clicks < 1) return '';
            if (typeof window._doppAudioInit === 'undefined') {
                window._doppAudioInit = true;
            }
            try {
                if (!window.audioCtx) {
                    window.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    window.gainNode = window.audioCtx.createGain();
                    window.gainNode.gain.setValueAtTime(1, window.audioCtx.currentTime);
                    window.gainNode.connect(window.audioCtx.destination);
                    window.oscillator = null;
                    window.muted = false;
                }
                if (window.audioCtx.state === 'suspended') {
                    window.audioCtx.resume();
                }
            } catch (e) {}
            return '';
        }
        """,
        Output('sound-init', 'children'),
        Input('start-btn', 'n_clicks')
    )

    app.clientside_callback(
        """
        function(n_clicks) {
            if (typeof window._doppAudioInit === 'undefined') {
                try {
                    window._doppAudioInit = true;
                    window.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    window.gainNode = window.audioCtx.createGain();
                    window.gainNode.gain.setValueAtTime(1, window.audioCtx.currentTime);
                    window.gainNode.connect(window.audioCtx.destination);
                    window.oscillator = null;
                    window.muted = false;
                    if (window.audioCtx.state === 'suspended') window.audioCtx.resume();
                } catch(e) {}
            }
            if (!n_clicks) return '';
            window.muted = (n_clicks % 2) === 1;
            try {
                if (window.gainNode && window.audioCtx) {
                    window.gainNode.gain.setValueAtTime(window.muted ? 0 : 1, window.audioCtx.currentTime);
                }
            } catch(e) {}
            return window.muted ? 'Muted' : 'Unmuted';
        }
        """,
        Output('mute-label', 'children'),
        Input('mute-btn', 'n_clicks')
    )

    app.clientside_callback(
        """
        function(freq) {
            try {
                if (typeof window._doppAudioInit === 'undefined') {
                    window._doppAudioInit = true;
                    window.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    window.gainNode = window.audioCtx.createGain();
                    window.gainNode.gain.setValueAtTime(1, window.audioCtx.currentTime);
                    window.gainNode.connect(window.audioCtx.destination);
                    window.oscillator = null;
                    window.muted = false;
                }
                if (window.audioCtx && window.audioCtx.state === 'suspended') {
                    window.audioCtx.resume();
                }
                if (freq && freq > 0 && !window.muted) {
                    if (!window.oscillator) {
                        window.oscillator = window.audioCtx.createOscillator();
                        window.oscillator.type = 'sine';
                        window.oscillator.connect(window.gainNode);
                        window.oscillator.start();
                    }
                    try {
                        window.oscillator.frequency.linearRampToValueAtTime(freq, window.audioCtx.currentTime + 0.05);
                    } catch(e) {
                        window.oscillator.frequency.setValueAtTime(freq, window.audioCtx.currentTime);
                    }
                } else {
                    if (window.oscillator) {
                        try { window.oscillator.stop(); } catch(e) {}
                        try { window.oscillator.disconnect(); } catch(e) {}
                        window.oscillator = null;
                    }
                }
            } catch (e) {}
            return '';
        }
        """,
        Output('sound-div', 'children'),
        Input('sound-freq', 'data')
    )
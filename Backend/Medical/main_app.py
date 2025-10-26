# main_app.py
import os
import plotly.io as pio
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update

from data_loader import DataLoader
from channel_manager import ChannelManager
from ai_model_manager import AIModelManager
from visualization_engine import VisualizationEngine
from aliasing_analyzer import AliasingAnalyzer
from app_state import AppStateManager


class MedicalSignalApp:
    def __init__(self):
        self.app = Dash(__name__)
        self.server = self.app.server

        # Initialize components
        self.data_loader = DataLoader()
        self.channel_manager = ChannelManager()
        self.ai_manager = AIModelManager()
        self.visualization_engine = VisualizationEngine()
        self.aliasing_analyzer = AliasingAnalyzer()
        self.state_manager = AppStateManager()

        # Setup layout and callbacks
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        # Define styles as instance variables so they're accessible in callbacks
        self.app_style = {
            'backgroundColor': '#F3F4F6',
            'color': '#111827',
            'fontFamily': 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif',
            'display': 'flex',
            'flexDirection': 'row',
            'height': '100vh',
            'overflow': 'hidden',
        }

        self.sidebar_style = {
            'width': '380px',
            'minWidth': '380px',
            'padding': '20px',
            'display': 'flex',
            'flexDirection': 'column',
            'gap': '20px',
            'overflowY': 'auto',
            'backgroundColor': '#FFFFFF',
            'borderRight': '1px solid #E5E7EB'
        }

        self.content_style = {
            'flex': 1,
            'padding': '20px',
            'display': 'flex',
            'flexDirection': 'column',
            'gap': '20px'
        }

        self.card_style = {
            'backgroundColor': '#FFFFFF',
            'borderRadius': '8px',
            'padding': '16px',
            'border': '1px solid #E5E7EB',
            'boxShadow': '0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
        }

        self.card_header_style = {
            'color': '#6B7280',
            'fontSize': '12px',
            'textTransform': 'uppercase',
            'letterSpacing': '0.05em',
            'fontWeight': '600',
            'marginBottom': '12px'
        }

        self.button_style = {
            'backgroundColor': '#3B82F6',
            'color': 'white',
            'border': 'none',
            'padding': '10px 15px',
            'borderRadius': '6px',
            'fontSize': '14px',
            'fontWeight': 'bold',
            'cursor': 'pointer',
            'width': '100%',
            'transition': 'background-color 0.2s'
        }

        self.ai_button_1d_style = {**self.button_style, 'backgroundColor': '#10B981'}
        self.ai_button_2d_style = {**self.button_style, 'backgroundColor': '#8B5CF6'}

        self.app.layout = html.Div(style=self.app_style, children=[
            # --- Controls Sidebar ---
            html.Div(style=self.sidebar_style, children=[
                html.Div([
                    html.H1("BioSignal Monitor", style={'fontSize': '24px', 'fontWeight': 'bold', 'margin': 0}),
                    html.P("ECG & EEG Real-time Analysis", style={'color': '#6B7280', 'marginTop': '4px'})
                ]),

                # --- Data Loading Card ---
                html.Div(style=self.card_style, children=[
                    html.H2("Data Source", style=self.card_header_style),
                    html.Label("Signal Type"),
                    dcc.RadioItems(id="dataset-type",
                                   options=[{"label": "ECG", "value": "ECG"}, {"label": "EEG", "value": "EEG"}],
                                   value="ECG", labelStyle={'display': 'inline-block', 'marginRight': '12px'},
                                   style={'marginTop': '8px'}),
                    html.Label("Data Directory (optional)", style={'marginTop': '12px', 'display': 'block'}),
                    dcc.Input(id="data-dir", type="text", value="", placeholder="e.g., ./data/ptbdb",
                              style={'marginTop': '8px'}),
                    html.Button("Load Data", id="load-btn", n_clicks=0,
                                style={**self.button_style, 'marginTop': '12px'}),
                    html.Div(id="load-output",
                             style={"marginTop": '12px', "fontSize": '12px', "color": '#6B7280',
                                    "whiteSpace": "pre-wrap",
                                    'minHeight': '40px'})
                ]),

                # --- Display Options Card ---
                html.Div(style=self.card_style, children=[
                    html.H2("Display Options", style=self.card_header_style),
                    html.Label("Select Patients"),
                    dcc.Dropdown(id="patients-dropdown", multi=True, placeholder="Select one or more patients"),
                    html.Label("Select Channels", style={'marginTop': '12px'}),
                    dcc.Dropdown(id="channels-dropdown", multi=True, placeholder="Auto-selects first 3 if empty"),
                    html.Button("Select All Channels", id="select-all-channels-btn", n_clicks=0,
                                style={**self.button_style, 'backgroundColor': '#6B7280', 'marginTop': '8px'}),

                    html.Label("Analysis Domain", style={'marginTop': '12px'}),
                    dcc.RadioItems(
                        id="domain-switch",
                        options=[
                            {'label': 'Time Domain', 'value': 'time'},
                            {'label': 'Frequency Domain', 'value': 'frequency'}
                        ],
                        value='time',
                        labelStyle={'display': 'inline-block', 'marginRight': '12px'}
                    ),

                    html.Label("Visualization Type", style={'marginTop': '12px'}),
                    dcc.Dropdown(id="viz-type", value="icu"),
                    html.Label("Channel Display Mode", style={'marginTop': '12px'}),
                    dcc.RadioItems(id="overlay-mode", options=[{"label": "Overlay", "value": "overlay"},
                                                               {"label": "Separate", "value": "separate"}],
                                   value="overlay", labelStyle={'display': 'inline-block', 'marginRight': '12px'}),
                    html.Div(id="channel-info", style={"marginTop": "8px", "fontSize": "12px", "color": "#6B7280"})
                ]),

                # --- Playback Control Card ---
                html.Div(style=self.card_style, children=[
                    html.H2("Playback Controls", style=self.card_header_style),
                    html.Label("Speed"),
                    dcc.Slider(id="speed", min=0.1, max=10, step=0.1, value=1,
                               marks={0.5: "0.5x", 1: "1x", 2: "2x", 5: "5x", 10: "10x"}),
                    html.Div(style={'display': 'flex', 'gap': '10px', 'marginTop': '12px'}, children=[
                        html.Div(style={'flex': 1}, children=[
                            html.Label("Update (ms)"),
                            dcc.Input(id="chunk-ms", type="number", value=200, min=20, step=10),
                        ]),
                        html.Div(style={'flex': 1}, children=[
                            html.Label("Window (s)"),
                            dcc.Input(id="display-window", type="number", value=8, min=1, step=1),
                        ])
                    ]),
                    html.Div(style={'display': 'flex', 'gap': '10px', 'marginTop': '12px'}, children=[
                        html.Button("Play", id="play-btn", n_clicks=0,
                                    style={**self.button_style, 'backgroundColor': '#10B981', 'flex': 1}),
                        html.Button("Pause", id="pause-btn", n_clicks=0,
                                    style={**self.button_style, 'backgroundColor': '#F59E0B', 'flex': 1}),
                        html.Button("Reset", id="reset-btn", n_clicks=0,
                                    style={**self.button_style, 'backgroundColor': '#EF4444', 'flex': 1}),
                    ]),
                ]),

                # --- Time Domain Analysis Card ---
                html.Div(id='time-analysis-card', style={**self.card_style}, children=[
                    html.H2("Time Domain Sampling Analysis", style=self.card_header_style),
                    html.Label("Set Sampling Period Ts (ms)", style={'marginTop': '12px'}),
                    dcc.Input(id="sampling-period-input", type="number", value=None,
                              placeholder="e.g., 20ms for 50Hz", style={'width': '100%'}),
                    html.Div(id='nyquist-info-time', style={'marginTop': '12px'}),
                ]),

                # --- Frequency Analysis Card ---
                html.Div(id='freq-analysis-card', style={**self.card_style}, children=[
                    html.H2("Frequency Analysis", style=self.card_header_style),
                    html.Div(id='nyquist-info'),
                    html.Label("Resampling Frequency (Hz) for FFT", style={'marginTop': '12px'}),
                    dcc.Input(id="resampling-freq", type="number", value=500, min=10, step=10,
                              placeholder="e.g., 500", style={'width': '100%'}),
                    html.Div(id="fft-computation-time", style={"marginTop": "12px"})
                ]),

                # --- AI Analysis Control Card ---
                html.Div(style=self.card_style, children=[
                    html.H2("AI Analysis", style=self.card_header_style),
                    html.P("Analyze the currently displayed signal window.",
                           style={'fontSize': '14px', 'color': '#6B7280', 'marginBottom': '16px'}),
                    html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'}, children=[
                        html.Button("Run 1D AI Analysis (Signal)", id="ai-analyze-btn", n_clicks=0,
                                    style=self.ai_button_1d_style),
                        html.Button("Run 2D AI Analysis (Image)", id="ai-analyze-2d-btn", n_clicks=0,
                                    style=self.ai_button_2d_style),
                    ]),
                ]),

                html.Div(style={'marginTop': 'auto'})
            ]),

            # --- Main Content Area ---
            html.Div(style=self.content_style, children=[
                # --- Main Graph ---
                html.Div(style={**self.card_style, 'flex': 1, 'minHeight': '400px', 'display': 'flex',
                                'flexDirection': 'column'},
                         children=[
                             dcc.Graph(id="main-graph", config={"displayModeBar": True}, style={'height': '100%'})
                         ]),

                # --- AI Analysis Results Section ---
                html.Div(style={**self.card_style, 'minHeight': '350px', 'maxHeight': '350px', 'display': 'flex',
                                'flexDirection': 'column'}, children=[
                    html.H2("AI Analysis Results", style=self.card_header_style),
                    html.Div(id="ai-analysis-output",
                             style={"fontSize": "14px", 'overflowY': 'auto', 'flex': 1, 'paddingRight': '10px'}),
                ]),
            ]),

            # --- Hidden Components ---
            dcc.Interval(id="interval", interval=200, n_intervals=0),
            dcc.Store(id="app-state", data=None),
        ])

    def setup_callbacks(self):
        @self.app.callback(
            Output("viz-type", "options"),
            Input("domain-switch", "value")
        )
        def update_viz_options(domain):
            if domain == 'time':
                return [
                    {"label": "Standard Monitor", "value": "icu"},
                    {"label": "Ping-Pong Overlay", "value": "pingpong"},
                    {"label": "Polar View", "value": "polar"},
                    {"label": "Cross-Recurrence Plot", "value": "crossrec"}
                ]
            else:
                return [
                    {"label": "Frequency Spectrum", "value": "icu"},
                    {"label": "Spectral Comparison", "value": "pingpong"},
                    {"label": "Spectral Polar View", "value": "polar"},
                    {"label": "Spectral Cross-Recurrence", "value": "crossrec"}
                ]

        @self.app.callback(
            Output("channels-dropdown", "value"),
            Input("select-all-channels-btn", "n_clicks"),
            State("patients-dropdown", "value"),
            prevent_initial_call=True
        )
        def select_all_channels(n_clicks, selected_patients):
            if not n_clicks or not selected_patients:
                return no_update

            patients = self.state_manager.global_data.get("patients", [])
            if not patients:
                return no_update

            try:
                pid = int(selected_patients[0])
                if pid >= len(patients):
                    return no_update
            except (ValueError, IndexError):
                return no_update

            patient = patients[pid]
            if "ecg" not in patient or patient["ecg"] is None:
                return no_update

            all_channels = [c for c in patient["ecg"].columns if c.startswith("signal_")]
            return all_channels

        @self.app.callback(
            [Output("load-output", "children"),
             Output("patients-dropdown", "options"),
             Output("channels-dropdown", "options"),
             Output("channel-info", "children")],
            [Input("load-btn", "n_clicks"),
             Input("dataset-type", "value")],
            [State("data-dir", "value")],
            prevent_initial_call=True
        )
        def load_data(nc, dataset_type, data_dir):
            if dataset_type == "EEG" and not self.data_loader.PYEDFLIB_AVAILABLE:
                return "pyedflib not installed. Install with: pip install pyedflib", [], [], ""
            if not data_dir or data_dir.strip() == "":
                auto = self.data_loader.find_dataset_directory(dataset_type, ".")
                if auto is None:
                    return f"No {dataset_type} data found automatically. Please provide directory.", [], [], ""
                data_dir = auto
            if not os.path.isdir(data_dir):
                return f"Directory not found: {data_dir}", [], [], ""

            self.state_manager.global_data["dataset_type"] = dataset_type

            if dataset_type == "EEG":
                patients = self.data_loader.load_patient_data(data_dir, dataset_type, max_samples=None,
                                                              max_patients=self.data_loader.MAX_EEG_SUBJECTS)
            else:
                patients = self.data_loader.load_patient_data(data_dir, dataset_type, max_samples=None,
                                                              max_patients=None)

            if not patients:
                return f"No {dataset_type} patients found in {data_dir}.", [], [], ""

            self.state_manager.global_data["patients"] = patients
            self.state_manager.global_data["buffers"] = {}

            for idx, p in enumerate(patients):
                try:
                    fs = float(p["header"].get("sampling_frequency", 250.0)) if p.get("header") else 250.0
                    self.state_manager.ensure_buffers_for_patient(idx, fs, display_window=8)
                except Exception as e:
                    print("Buffer init error", e)

            patient_options = [{"label": f"{p['name']} ({p['type']})", "value": idx} for idx, p in enumerate(patients)]

            channel_options = []
            channel_info = ""
            if patients:
                first_patient = patients[0]
                available_channels = [c for c in first_patient["ecg"].columns if c.startswith("signal_")]
                channel_options = [{"label": ch, "value": ch} for ch in available_channels]
                channel_info = f"{dataset_type}: {len(available_channels)} channel(s) available."

            if dataset_type == "EEG":
                msg = f"Loaded {len(patients)} EEG subjects from {data_dir}."
            else:
                msg = f"Loaded {len(patients)} ECG records from {data_dir}."

            return msg, patient_options, channel_options, channel_info

        @self.app.callback(
            Output("interval", "interval"),
            Input("chunk-ms", "value"),
            Input("speed", "value")
        )
        def adjust_interval(chunk_ms, speed):
            try:
                cm = max(20, int(float(chunk_ms)))
            except:
                cm = 200
            return cm

        @self.app.callback(
            Output("ai-analysis-output", "children"),
            [Input("ai-analyze-btn", "n_clicks"),
             Input("ai-analyze-2d-btn", "n_clicks")],
            [State("patients-dropdown", "value"),
             State("channels-dropdown", "value"),
             State("dataset-type", "value"),
             State("app-state", "data"),
             State("display-window", "value"),
             State("speed", "value")],
            prevent_initial_call=True
        )
        def run_ai_analysis(n_clicks_1d, n_clicks_2d, selected_patients, selected_channels,
                            dataset_type, app_state, display_window, speed_val):
            ctx = callback_context
            if not ctx.triggered:
                return "Click an AI Analysis button to analyze current signals"

            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            is_2d_analysis = (button_id == "ai-analyze-2d-btn")

            try:
                patients = self.state_manager.global_data.get("patients", [])
                if not patients:
                    return html.Div([
                        html.P("No patients loaded. Please load data first.",
                               style={"color": "#ff6347"})
                    ])

                selected_idxs = self.state_manager.get_selected_patient_indices(selected_patients, patients)
                signal_type = dataset_type if dataset_type else self.state_manager.global_data.get("dataset_type",
                                                                                                   "ECG")

                if is_2d_analysis:
                    # 2D analysis implementation would go here
                    return html.Div([
                        html.H4("2D AI Analysis", style={"color": "#4F46E5"}),
                        html.P("2D image-based analysis would be implemented here.",
                               style={"color": "#6B7280"})
                    ])
                else:
                    # 1D signal analysis
                    self.ai_manager.switch_signal_type(signal_type)

                    all_results = []
                    for pid in selected_idxs[:3]:
                        if pid >= len(patients):
                            continue

                        patient = patients[pid].copy()
                        patient_name = patient.get("name", f"Patient {pid}")

                        if "ecg" not in patient or patient["ecg"] is None:
                            all_results.append(html.Div([
                                html.H4(f"{patient_name}:", style={"color": "#1E40AF"}),
                                html.P("No data available for analysis", style={"color": "#EF4444"})
                            ]))
                            continue

                        pos = 0
                        if app_state and "pos" in app_state and pid < len(app_state["pos"]):
                            pos = app_state["pos"][pid]

                        if pos <= 0:
                            all_results.append(html.Div([
                                html.H4(f"{patient_name}:", style={"color": "#1E40AF"}),
                                html.P("No signal data played yet. Start playback first.",
                                       style={"color": "#EF4444"})
                            ]))
                            continue

                        signal_data = patient["ecg"].iloc[:pos].copy()

                        # Use the AI manager to analyze the data
                        analysis_result = self.ai_manager.analyze_patient_data_simple(signal_data, signal_type)

                        if analysis_result.get("success"):
                            predictions = analysis_result.get("predictions", [])

                            patient_elements = []
                            patient_elements.append(
                                html.H4(f"{patient_name} ({signal_type}):",
                                        style={"color": "#1E40AF", "marginBottom": "8px"})
                            )

                            for i, pred in enumerate(predictions[:5]):
                                confidence = pred.get("confidence", 0)
                                confidence_pct = f"{confidence * 100:.1f}%"
                                label = pred.get("label", "Unknown")

                                if confidence >= 0.8:
                                    confidence_color = "#059669"
                                    icon = "✓"
                                elif confidence >= 0.6:
                                    confidence_color = "#F59E0B"
                                    icon = "○"
                                elif confidence >= 0.4:
                                    confidence_color = "#F97316"
                                    icon = "△"
                                else:
                                    confidence_color = "#EF4444"
                                    icon = "·"

                                patient_elements.append(
                                    html.Div([
                                        html.Span(f"{i + 1}. ",
                                                  style={"fontWeight": "bold", "color": "#4B5563"}),
                                        html.Span(f"{icon} ",
                                                  style={"marginRight": "5px", "color": confidence_color}),
                                        html.Span(f"{label}",
                                                  style={"color": "#111827", "fontWeight": "bold"}),
                                        html.Span(f" ({confidence_pct})",
                                                  style={"color": confidence_color, "marginLeft": "10px"})
                                    ], style={
                                        "margin": "4px 0",
                                        "padding": "8px",
                                        "backgroundColor": "#F9FAFB",
                                        "borderRadius": "4px",
                                        "borderLeft": f"4px solid {confidence_color}"
                                    })
                                )

                            all_results.append(
                                html.Div(patient_elements,
                                         style={"marginBottom": "20px", "borderBottom": "1px solid #E5E7EB",
                                                "paddingBottom": "15px"})
                            )
                        else:
                            all_results.append(html.Div([
                                html.H4(f"{patient_name}:", style={"color": "#1E40AF"}),
                                html.P(f"Analysis failed", style={"color": "#EF4444"})
                            ]))

                    if not all_results:
                        return html.Div([
                            html.P("No analysis results available.", style={"color": "#EF4444"}),
                            html.P("Ensure patients are loaded and playback has started.",
                                   style={"color": "#6B7280", "fontSize": "12px"})
                        ])

                    return html.Div(all_results, style={"lineHeight": "1.5"})

            except Exception as e:
                print(f"[AI Analysis] Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                return html.Div([
                    html.P("Unexpected error in AI analysis.",
                           style={"color": "#EF4444"}),
                    html.P(f"Error details: {str(e)}",
                           style={"color": "#6B7280", "fontSize": "11px"}),
                    html.P("Check the console for more information.",
                           style={"color": "#6B7280", "fontSize": "12px"})
                ])

        @self.app.callback(
            [Output("main-graph", "figure"),
             Output("app-state", "data"),
             Output("fft-computation-time", "children"),
             Output("nyquist-info", "children"),
             Output("freq-analysis-card", "style"),
             Output("nyquist-info-time", "children"),
             Output("time-analysis-card", "style")],
            [Input("interval", "n_intervals"),
             Input("play-btn", "n_clicks"),
             Input("pause-btn", "n_clicks"),
             Input("reset-btn", "n_clicks"),
             Input("domain-switch", "value")],
            [State("app-state", "data"),
             State("patients-dropdown", "value"),
             State("channels-dropdown", "value"),
             State("overlay-mode", "value"),
             State("viz-type", "value"),
             State("speed", "value"),
             State("chunk-ms", "value"),
             State("display-window", "value"),
             State("resampling-freq", "value"),
             State("sampling-period-input", "value")],
            prevent_initial_call=False
        )
        def combined_update(n_intervals, n_play, n_pause, n_reset, domain, state,
                            selected, selected_channels, overlay_mode, viz_type,
                            speed, chunk_ms, display_window, resampling_freq,
                            sampling_period_ms):
            try:
                patients = self.state_manager.global_data.get("patients", [])
                if not patients:
                    empty_fig = self.visualization_engine.create_empty_figure("No patients loaded")
                    return empty_fig, {"playing": False, "pos": [], "write_idx": []}, "", "", {'display': 'none'}, "", {
                        'display': 'none'}

                # Initialize state
                state = self.state_manager.initialize_state(patients)

                # Determine trigger
                ctx = callback_context
                trigger = getattr(ctx, "triggered_id", None)
                if trigger is None and ctx.triggered:
                    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

                # Update playback state
                state = self.state_manager.update_playback_state(patients, trigger, state,
                                                                 float(chunk_ms or 200), float(speed or 1.0))

                # Parse parameters
                chunk_ms_val = max(20, float(chunk_ms or 200))
                speed_val = max(0.1, float(speed or 1.0))
                display_window_val = max(1.0, float(display_window or 8.0))

                # Ensure buffers
                for pid, p in enumerate(patients):
                    fs = float(p["header"].get("sampling_frequency", 250.0)) if p.get("header") else 250.0
                    self.state_manager.ensure_buffers_for_patient(pid, fs, display_window_val)

                # Get selected patient indices
                selected_idxs = self.state_manager.get_selected_patient_indices(selected, patients)

                # Determine overlay mode
                overlay = (overlay_mode == "overlay")

                # Build visualization
                if not selected_idxs:
                    main_fig = self.visualization_engine.create_empty_figure("No patients selected")
                    return main_fig, state, "", "", {'display': 'none'}, "", {'display': 'none'}

                pid = selected_idxs[0]
                if pid < 0 or pid >= len(patients):
                    main_fig = self.visualization_engine.create_empty_figure("Invalid patient selection")
                    return main_fig, state, "", "", {'display': 'none'}, "", {'display': 'none'}

                p = patients[pid]
                if "ecg" not in p or p["ecg"] is None:
                    main_fig = self.visualization_engine.create_empty_figure("No data for selected patient")
                    return main_fig, state, "", "", {'display': 'none'}, "", {'display': 'none'}

                # Get channels to display
                if selected_channels and isinstance(selected_channels, (list, tuple)) and len(selected_channels) > 0:
                    channels_to_display = [ch for ch in selected_channels if ch in p["ecg"].columns]
                    if not channels_to_display:
                        main_fig = self.visualization_engine.create_empty_figure("Selected channels not found")
                        return main_fig, state, "", "", {'display': 'none'}, "", {'display': 'none'}
                else:
                    all_channels = [c for c in p["ecg"].columns if c.startswith("signal_")]
                    channels_to_display = all_channels[:min(3, len(all_channels))]
                    if not channels_to_display:
                        main_fig = self.visualization_engine.create_empty_figure("No channels available")
                        return main_fig, state, "", "", {'display': 'none'}, "", {'display': 'none'}

                fs = float(p.get("header", {}).get("sampling_frequency", 250.0))
                pos = int(state["pos"][pid]) if state and "pos" in state and pid < len(state["pos"]) else len(p["ecg"])

                win = int(display_window_val * fs)
                start = max(0, pos - win)
                current_segment_df = p["ecg"].iloc[start:pos]

                if current_segment_df.shape[0] < 2:
                    main_fig = self.visualization_engine.create_empty_figure("Not enough data in window")
                    return main_fig, state, "", "", {'display': 'none'}, "", {'display': 'none'}

                # Create visualization based on domain
                if domain == 'time':
                    # Apply sampling period if specified
                    if sampling_period_ms is not None:
                        try:
                            ts_user = float(sampling_period_ms) / 1000.0
                            if ts_user > 0 and (1.0 / ts_user) < fs:
                                fs_user = 1.0 / ts_user
                                step = int(round(fs / fs_user))
                                step = max(1, step)
                                if step > 1:
                                    current_segment_df = current_segment_df.iloc[::step]
                        except (ValueError, TypeError):
                            pass

                    unit = "mV" if self.state_manager.global_data.get("dataset_type", "ECG") == "ECG" else "µV"

                    # Get previous segment for ping-pong visualization
                    prev_segment_df = None
                    if viz_type == "pingpong":
                        prev_start = max(0, start - win)
                        prev_end = start
                        prev_segment_df = p["ecg"].iloc[prev_start:prev_end] if prev_end > prev_start else None

                    main_fig = self.visualization_engine.make_time_domain_visualization(
                        viz_type, current_segment_df, channels_to_display,
                        p.get('name', 'Patient'), overlay, unit, prev_segment_df
                    )

                    # Time domain aliasing analysis
                    time_card_style = {**self.card_style}
                    nyquist_output_time = ""

                    if channels_to_display and not current_segment_df.empty:
                        first_channel_data = current_segment_df[channels_to_display[0]].values
                        time_analysis = self.aliasing_analyzer.analyze_sampling_period(
                            first_channel_data, fs, sampling_period_ms
                        )

                        if time_analysis:
                            risk_level = time_analysis.get('risk_level', 'UNKNOWN')
                            risk_color = time_analysis.get('risk_color', '#6B7280')

                            nyquist_output_time = html.Div([
                                html.P(f"Signal's Max Frequency: {time_analysis['max_signal_frequency']:.1f} Hz",
                                       style={'margin': '0px', 'fontSize': '12px'}),
                                html.P(f"Current Sampling Period: {time_analysis['current_period_ms']:.2f} ms",
                                       style={'margin': '0px', 'fontSize': '12px'}),
                                html.P(f"Required Period: {time_analysis['max_period_ms']:.2f} ms",
                                       style={'margin': '0px 0px 5px 0px', 'fontSize': '12px'}),
                                html.Div([
                                    html.Span("Status: ", style={'fontWeight': 'bold'}),
                                    html.Span(risk_level, style={'color': 'white', 'backgroundColor': risk_color,
                                                                 'padding': '2px 6px', 'borderRadius': '4px',
                                                                 'fontWeight': 'bold'})
                                ])
                            ], style={'padding': '10px', 'backgroundColor': '#F9FAFB', 'borderRadius': '6px'})

                    return main_fig, state, "", "", {'display': 'none'}, nyquist_output_time, time_card_style

                else:
                    # Frequency domain visualization
                    prev_segment_df = None
                    if viz_type == "pingpong":
                        prev_start = max(0, start - win)
                        prev_end = start
                        prev_segment_df = p["ecg"].iloc[prev_start:prev_end] if prev_end > prev_start else None

                    main_fig = self.visualization_engine.make_frequency_domain_visualization(
                        viz_type, current_segment_df, channels_to_display,
                        p.get('name', 'Patient'), fs, resampling_freq, overlay, prev_segment_df
                    )

                    # Frequency domain aliasing analysis
                    freq_card_style = {**self.card_style}
                    fft_time_output = ""
                    nyquist_output = ""

                    if channels_to_display and not current_segment_df.empty:
                        first_channel_data = current_segment_df[channels_to_display[0]].values
                        _, _, _, comp_time, f_max, aliasing_risk = self.aliasing_analyzer.calculate_fft_with_aliasing_analysis(
                            first_channel_data, fs, resampling_freq
                        )

                        if comp_time is not None:
                            time_ms = comp_time * 1000
                            color = "#3B82F6"
                            if time_ms > 50: color = "#F59E0B"
                            if time_ms > 100: color = "#EF4444"
                            fft_time_output = html.Div([
                                html.Span("FFT Compute Time: ", style={'fontWeight': 'bold'}),
                                html.Span(f"{time_ms:.1f} ms",
                                          style={'color': color, 'fontSize': '16px', 'fontWeight': 'bold'})
                            ])

                        if aliasing_risk and f_max > 0:
                            risk_level = aliasing_risk['risk_level']
                            risk_color = aliasing_risk['color']

                            nyquist_output = html.Div([
                                html.P(f"Signal's Max Frequency: {f_max:.1f} Hz",
                                       style={'margin': '0px', 'fontSize': '12px'}),
                                html.P(f"Required Nyquist Rate: {aliasing_risk['nyquist_rate']:.1f} Hz",
                                       style={'margin': '0px 0px 5px 0px', 'fontSize': '12px'}),
                                html.Div([
                                    html.Span("Status: ", style={'fontWeight': 'bold'}),
                                    html.Span(risk_level,
                                              style={'color': 'white', 'backgroundColor': risk_color,
                                                     'padding': '2px 6px',
                                                     'borderRadius': '4px', 'fontWeight': 'bold'})
                                ])
                            ], style={'padding': '10px', 'backgroundColor': '#F9FAFB', 'borderRadius': '6px'})

                    return main_fig, state, fft_time_output, nyquist_output, freq_card_style, "", {'display': 'none'}

            except Exception as e:
                print(f"[combined_update] Error: {e}")
                import traceback
                traceback.print_exc()
                empty_fig = self.visualization_engine.create_empty_figure("Error occurred")
                return empty_fig, {"playing": False, "pos": [], "write_idx": []}, "Error", "", {
                    'display': 'none'}, "Error", {
                    'display': 'none'}

    def run(self, debug=True, host="127.0.0.1", port=8052):
        self.app.run(debug=debug, host=host, port=port)


if __name__ == "__main__":
    medical_app = MedicalSignalApp()
    medical_app.run()
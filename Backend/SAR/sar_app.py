import base64
import io
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate


# ---------- Helpers ---------------------------------
def pil_from_base64(b64_string):
    return Image.open(io.BytesIO(base64.b64decode(b64_string))).convert('RGBA')


def image_to_base64_bytes(img, fmt='PNG'):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def compute_stats_and_histogram(pil_img, bins=50):
    gray = pil_img.convert('L')
    intensities = np.array(gray).flatten()
    stats = {
        'mean': round(float(np.mean(intensities)), 2), 'median': round(float(np.median(intensities)), 2),
        'stdDev': round(float(np.std(intensities)), 2), 'min': round(float(np.min(intensities)), 2),
        'max': round(float(np.max(intensities)), 2), 'pixels': int(intensities.size),
        'width': pil_img.width, 'height': pil_img.height,
    }
    hist_counts, bin_edges = np.histogram(intensities, bins=bins, range=(0, 255))
    histogram = pd.DataFrame({'intensity': 0.5 * (bin_edges[:-1] + bin_edges[1:]), 'count': hist_counts})
    return stats, histogram


def apply_threshold_to_image(pil_img, threshold_percent):
    thr_value = (threshold_percent / 100.0) * 255.0
    gray = pil_img.convert('L')
    mask = np.array(gray) < thr_value
    rgba_arr = np.array(pil_img.convert('RGBA'))
    rgba_arr[mask, :3] = 0
    return Image.fromarray(rgba_arr)


# ---------- Dash App Layout -------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'SAR Analysis Dashboard'

# --- UI STYLE CONSTANTS ---
PRIMARY_COLOR = "#4A90E2"
BACKGROUND_COLOR = "#F7F9FC"
CARD_BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#333333"
SUBTLE_TEXT_COLOR = "#666666"
BORDER_COLOR = "#EAEAEA"
FONT_FAMILY = "Inter, system-ui, sans-serif"

CARD_STYLE = {
    "backgroundColor": CARD_BACKGROUND_COLOR, "borderRadius": "12px", "padding": "24px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.05)", "border": f"1px solid {BORDER_COLOR}",
}

app.layout = html.Div(
    style={"fontFamily": FONT_FAMILY, "backgroundColor": BACKGROUND_COLOR, "padding": "40px", "minHeight": "100vh"},
    children=[
        html.Div(style={"maxWidth": "1100px", "margin": "0 auto"}, children=[
            html.Div([
                html.H1("🛰️ SAR Data Analysis Platform",
                        style={'textAlign': 'center', 'color': TEXT_COLOR, 'fontWeight': 'bold'}),
                html.P("Synthetic Aperture Radar Signal Processing & Feature Extraction",
                       style={'textAlign': 'center', 'color': SUBTLE_TEXT_COLOR, 'fontSize': '16px'}),
            ], style={'marginBottom': '20px'}),

            # Upload Area
            dcc.Upload(
                id='upload-image', children=html.Div(['📂 Drag & Drop or ', html.A('Select a SAR Image File')]),
                style={
                    'width': '100%', 'height': '100px', 'lineHeight': '100px', 'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px', 'textAlign': 'center', 'marginBottom': '20px', 'cursor': 'pointer',
                    'borderColor': BORDER_COLOR, 'backgroundColor': CARD_BACKGROUND_COLOR
                },
                accept='image/*', multiple=False
            ),

            # Tabs for content
            dcc.Tabs(id='tabs', value='overview', children=[
                dcc.Tab(label='Overview', value='overview'), dcc.Tab(label='Analysis', value='analysis'),
                dcc.Tab(label='Processing', value='processing')
            ], style={'marginBottom': '20px'}),

            # Tab content will be rendered here
            html.Div(id='tab-content'),

            # Stores for data
            dcc.Store(id='original-image-b64'), dcc.Store(id='processed-image-b64'),
            dcc.Store(id='stats-store'), dcc.Store(id='hist-store'),

            # Download components
            dcc.Download(id='download-csv'), dcc.Download(id='download-image')
        ])
    ])


@app.callback(
    [Output('original-image-b64', 'data'), Output('processed-image-b64', 'data'),
     Output('stats-store', 'data'), Output('hist-store', 'data')],
    [Input('upload-image', 'contents')],
    prevent_initial_call=True
)
def handle_upload(contents):
    if not contents: raise PreventUpdate
    try:
        _, b64 = contents.split(',', 1)
        pil_img = pil_from_base64(b64)
        stats, hist_df = compute_stats_and_histogram(pil_img)
        b64_str = image_to_base64_bytes(pil_img)
        return b64_str, b64_str, stats, hist_df.to_dict('records')
    except Exception as e:
        print(f"Upload error: {e}")
        return None, None, None, None


@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value'), Input('original-image-b64', 'data'),
     Input('stats-store', 'data'), Input('hist-store', 'data'),
     Input('processed-image-b64', 'data')]
)
def render_tab(tab, orig_b64, stats, hist_records, proc_b64):
    if not stats:
        return html.Div(style=CARD_STYLE, children=[
            html.H3('Upload SAR data to begin analysis', style={'textAlign': 'center', 'color': SUBTLE_TEXT_COLOR}),
        ])

    hist_df = pd.DataFrame(hist_records)

    if tab == 'overview':
        stat_items = {k: v for k, v in stats.items() if k in ['mean', 'median', 'stdDev', 'min', 'max', 'pixels']}
        return html.Div(style={'display': 'grid', 'gridTemplateColumns': '2fr 1fr', 'gap': '24px'}, children=[
            html.Div(style=CARD_STYLE, children=[
                html.H3('SAR Intensity Image', style={'marginTop': 0}),
                html.Img(src=f'data:image/png;base64,{proc_b64}', style={'maxWidth': '100%', 'borderRadius': '8px'})
            ]),
            html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '24px'}, children=[
                html.Div(style=CARD_STYLE, children=[
                    html.H3('Image Properties', style={'marginTop': 0}),
                    html.Div(f"Resolution: {stats['width']} x {stats['height']} px", style={'marginBottom': '5px'}),
                    html.Div(f"Total Pixels: {stats['pixels']:,}")
                ]),
                html.Div(style=CARD_STYLE, children=[
                    html.H3('Signal Statistics', style={'marginTop': 0}),
                    html.Div([
                        html.Div([html.Span(k, style={'color': SUBTLE_TEXT_COLOR}),
                                  html.Span(v, style={'fontWeight': 'bold'})],
                                 style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '5px 0'})
                        for k, v in stat_items.items()
                    ])
                ]),
            ])
        ])

    if tab == 'analysis':
        fig = go.Figure(data=[go.Bar(x=hist_df['intensity'], y=hist_df['count'], marker_color=PRIMARY_COLOR)])
        fig.update_layout(title_text='Intensity Distribution', plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(family=FONT_FAMILY, color=TEXT_COLOR), margin={'t': 40, 'b': 40})
        return html.Div(style=CARD_STYLE, children=[dcc.Graph(figure=fig)])

    if tab == 'processing':
        return html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '24px'}, children=[
            html.Div(style=CARD_STYLE, children=[
                html.H3('Threshold Filter', style={'marginTop': 0}),
                html.P("Adjust the slider to highlight pixels below a certain intensity.",
                       style={'color': SUBTLE_TEXT_COLOR}),
                dcc.Slider(id='threshold-slider', min=0, max=100, step=1, value=50, marks=None,
                           tooltip={"placement": "bottom", "always_visible": True}),
                html.Button('Apply Filter', id='apply-threshold', n_clicks=0, style={
                    'marginTop': '20px', 'width': '100%', 'backgroundColor': PRIMARY_COLOR, 'color': 'white',
                    'border': 'none', 'borderRadius': '8px', 'padding': '12px', 'fontSize': '15px', 'cursor': 'pointer'
                }),
            ]),
            html.Div(style=CARD_STYLE, children=[
                html.H3('Export Data', style={'marginTop': 0}),
                html.Button('Export Statistics (CSV)', id='export-csv', n_clicks=0,
                            style={'width': '100%', 'padding': '12px', 'marginBottom': '10px'}),
                html.Button('Export Processed Image (PNG)', id='export-image', n_clicks=0,
                            style={'width': '100%', 'padding': '12px'}),
            ])
        ])


@app.callback(
    Output('processed-image-b64', 'data', allow_duplicate=True),
    Input('apply-threshold', 'n_clicks'),
    [State('threshold-slider', 'value'), State('original-image-b64', 'data')],
    prevent_initial_call=True
)
def handle_threshold(n_clicks, threshold_value, orig_b64):
    if not orig_b64: raise PreventUpdate
    pil_img = pil_from_base64(orig_b64)
    processed_img = apply_threshold_to_image(pil_img, threshold_value)
    return image_to_base64_bytes(processed_img)


@app.callback(Output('download-csv', 'data'), Input('export-csv', 'n_clicks'), State('stats-store', 'data'),
              prevent_initial_call=True)
def export_csv(n_clicks, stats):
    if not stats: raise PreventUpdate
    df = pd.DataFrame([stats])
    return dcc.send_data_frame(df.to_csv, f"sar_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)


@app.callback(Output('download-image', 'data'), Input('export-image', 'n_clicks'), State('processed-image-b64', 'data'),
              prevent_initial_call=True)
def export_image(n_clicks, proc_b64):
    if not proc_b64: raise PreventUpdate
    return dcc.send_bytes(base64.b64decode(proc_b64), f"processed_sar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")


if __name__ == '__main__':
    app.run(debug=True, port=8053)
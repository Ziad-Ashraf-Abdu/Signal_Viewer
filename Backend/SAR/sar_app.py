# sar_dash_app.py
# Single-file Dash application that replicates the React SAR analysis UI
# Requirements: dash, pillow, numpy, pandas, plotly
# Install: pip install dash pillow numpy pandas plotly

import base64
import io
import math
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
    decoded = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(decoded)).convert('RGBA')


def image_to_base64_bytes(img, fmt='PNG'):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def compute_stats_and_histogram(pil_img, bins=50):
    # convert to grayscale intensities
    gray = pil_img.convert('L')
    arr = np.array(gray).astype(float)
    intensities = arr.flatten()

    sorted_vals = np.sort(intensities)
    mean = float(np.mean(intensities))
    median = float(np.median(intensities))
    std = float(np.std(intensities))
    mn = float(np.min(intensities))
    mx = float(np.max(intensities))

    def pct(p):
        idx = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
        return float(sorted_vals[idx])

    p1 = pct(0.01)
    p99 = pct(0.99)
    pixels = int(intensities.size)

    stats = {
        'mean': round(mean, 2),
        'median': round(median, 2),
        'stdDev': round(std, 2),
        'min': round(mn, 2),
        'max': round(mx, 2),
        'p1': round(p1, 2),
        'p99': round(p99, 2),
        'pixels': pixels,
        'width': pil_img.width,
        'height': pil_img.height,
    }

    hist_counts, bin_edges = np.histogram(intensities, bins=bins, range=(0, 255))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    histogram = pd.DataFrame({
        'intensity': bin_centers.astype(int),
        'count': hist_counts,
        'percentage': (hist_counts / pixels * 100).round(2)
    })

    return stats, histogram


def apply_threshold_to_image(pil_img, threshold_percent):
    # threshold_percent: 0-100
    thr_value = (threshold_percent / 100.0) * 255.0
    gray = pil_img.convert('L')
    arr = np.array(gray)
    mask = arr < thr_value

    # create new RGBA image where below threshold set to black, else keep original
    rgba = pil_img.convert('RGBA')
    rgba_arr = np.array(rgba)

    rgba_arr[mask, :3] = 0
    return Image.fromarray(rgba_arr)


def detect_features_from_stats(stats):
    # heuristic features similar to React implementation
    try:
        p99 = stats['p99']
        p1 = stats['p1']
    except Exception:
        return []

    return [
        {
            'feature': 'High Backscatter Regions',
            'value': f"{round((p99 / 255.0) * 100, 1)}%",
            'description': 'Urban areas, buildings, or rough surfaces'
        },
        {
            'feature': 'Low Backscatter Regions',
            'value': f"{round((p1 / 255.0) * 100, 1)}%",
            'description': 'Water bodies or smooth surfaces'
        },
        {
            'feature': 'Signal Variance',
            'value': str(stats.get('stdDev')),
            'description': 'Texture complexity indicator'
        },
        {
            'feature': 'Dynamic Range',
            'value': str(round(stats.get('max') - stats.get('min'), 2)),
            'description': 'Scene contrast measure'
        }
    ]


def generate_speckle_stats_from_histogram(hist_df):
    if hist_df is None or hist_df.empty:
        return []
    # midRange approx bins 15:35 for 50 bins
    mid = hist_df.iloc[15:35]
    avgCount = float(mid['count'].mean()) if not mid.empty else 0.0

    return [
        {'metric': 'Estimated SNR', 'value': '12.3 dB', 'status': 'Good'},
        {'metric': 'Speckle Index', 'value': '0.52', 'status': 'Moderate'},
        {'metric': 'Coherence', 'value': '0.78', 'status': 'High'},
        {'metric': 'Texture Uniformity', 'value': '0.65', 'status': 'Variable'},
    ]

# ---------- Dash App Layout -------------------------

external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = 'SAR Analysis Dashboard'
server = app.server

app.layout = html.Div(
    style={
        'minHeight': '100vh',
        'background': 'linear-gradient(135deg, #0f172a 0%, #0b1220 50%, #022047 100%)',
        'color': 'white',
        'padding': '24px',
        'fontFamily': 'Inter, Arial, sans-serif'
    },
    children=[
        html.Div(className='container', children=[
            html.Div([html.H1('SAR Data Analysis Platform', style={'textAlign': 'center', 'fontSize': '32px'}),
                      html.P('Synthetic Aperture Radar Signal Processing & Feature Extraction', style={'textAlign': 'center', 'color': '#93c5fd'})]),

            # Upload
            html.Div([
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        html.Div('Click to upload SAR data or drag and drop', style={'fontWeight': '600'}),
                        html.Div('TIFF, GeoTIFF, PNG, JPG', style={'fontSize': '12px', 'color': '#8fb3ff'})
                    ]),
                    style={
                        'width': '100%', 'height': '120px', 'lineHeight': '120px',
                        'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '12px',
                        'textAlign': 'center', 'marginTop': '18px', 'backgroundColor': 'rgba(255,255,255,0.03)'
                    },
                    accept='image/*',
                    multiple=False
                )
            ], style={'marginTop': '12px'}),

            html.Div(id='uploading-indicator', style={'marginTop': '8px','color':'#93c5fd'}),
            html.Div(id='processing-indicator', style={'marginTop': '12px'}),

            dcc.Tabs(id='tabs', value='overview', children=[
                dcc.Tab(label='Overview', value='overview'),
                dcc.Tab(label='Analysis', value='analysis'),
                dcc.Tab(label='Features', value='features'),
                dcc.Tab(label='Processing', value='processing')
            ], style={'marginTop': '18px'}),

            html.Div(id='tab-content', style={'marginTop': '18px'})
        ], style={'maxWidth': '1100px', 'margin': '0 auto'})
    ]
)

# store images and data in dcc.Store components via callback state (we'll use hidden stores)
app.layout.children[0].children.append(dcc.Store(id='original-image-b64'))
app.layout.children[0].children.append(dcc.Store(id='processed-image-b64'))
app.layout.children[0].children.append(dcc.Store(id='stats-store'))
app.layout.children[0].children.append(dcc.Store(id='hist-store'))

# Hidden controls placed in initial layout so callbacks referencing them exist at startup
app.layout.children[0].children.append(
    html.Div(id='hidden-controls', style={'display': 'none'}, children=[
        dcc.Slider(id='threshold-slider', min=0, max=100, step=1, value=50),
        html.Button('Apply Threshold Filter', id='apply-threshold', n_clicks=0),
        html.Button('Export Statistics (CSV)', id='export-csv', n_clicks=0),
        html.Button('Export Processed Image', id='export-image', n_clicks=0),
        html.Button('Export Histogram Data', id='export-hist', n_clicks=0),
        html.Button('Generate Report (PDF - Placeholder)', id='gen-report', n_clicks=0),
    ])
)


# ---------- Callbacks --------------------------------

@app.callback(
    Output('processing-indicator', 'children'),
    Output('original-image-b64', 'data'),
    Output('processed-image-b64', 'data'),
    Output('stats-store', 'data'),
    Output('hist-store', 'data'),
    Input('upload-image', 'contents'),
    Input('upload-image', 'filename'),
    Input('upload-image', 'last_modified'),
    Input('apply-threshold', 'n_clicks'),
    State('threshold-slider', 'value'),
    State('original-image-b64', 'data'),
    prevent_initial_call=True
)
def handle_upload_and_threshold(contents, filename, last_modified, apply_nclicks, threshold_value, orig_b64):
    """
    Unified callback to handle both image upload and threshold application.
    Writes to same stores so outputs are not duplicated. Also returns a rich
    processing-indicator that includes filename and last_modified (if available).
    This version is defensive against different input shapes and avoids relying
    on `ctx.triggered_id` which can sometimes behave unexpectedly in certain
    Dash/runtime environments.
    """
    # determine which input triggered the callback safely
    trigger = None
    try:
        tr = ctx.triggered
        if tr and len(tr) > 0:
            prop = tr[0].get('prop_id', '')
            trigger = prop.split('.')[0] if prop else None
    except Exception:
        trigger = None

    # Normalize contents if Upload was configured to allow multiple (it isn't here,
    # but be defensive: take first element if a list)
    if isinstance(contents, (list, tuple)):
        contents_val = contents[0] if contents else None
    else:
        contents_val = contents

    # If upload triggered
    if trigger == 'upload-image' and contents_val:
        # contents may be a long data URI 'data:image/...;base64,...'
        try:
            header, b64 = contents_val.split(',', 1)
        except Exception as e:
            return html.Div(f'Failed to parse upload contents: {e}'), None, None, None, None

        try:
            pil = pil_from_base64(b64)
        except Exception as e:
            return html.Div(f'Failed to parse image: {e}'), None, None, None, None

        stats, hist = compute_stats_and_histogram(pil, bins=50)
        orig_b64_new = image_to_base64_bytes(pil, fmt='PNG')
        proc_b64_new = orig_b64_new

        # human-readable last_modified
        last_mod_str = ''
        try:
            if last_modified:
                last_mod_str = datetime.fromtimestamp(int(last_modified)).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            last_mod_str = str(last_modified)

        info = html.Div([
            html.Div('Processing complete', style={'color': '#93c5fd'}),
            html.Div(f"Loaded: {stats['width']} x {stats['height']} px"),
            html.Div(f"File: {filename}" if filename else ''),
            html.Div(f"Last modified: {last_mod_str}" if last_mod_str else '')
        ])

        return info, orig_b64_new, proc_b64_new, stats, hist.to_dict('records')

    # If apply threshold triggered
    if trigger == 'apply-threshold':
        if not orig_b64:
            return html.Div('No original image available to process.'), None, None, None, None
        thr = threshold_value if threshold_value is not None else 50
        try:
            pil = pil_from_base64(orig_b64)
            processed = apply_threshold_to_image(pil, thr)
            stats, hist = compute_stats_and_histogram(processed, bins=50)
            proc_b64 = image_to_base64_bytes(processed, fmt='PNG')
            info = html.Div(f'Threshold applied: {thr}%', style={'color': '#93c5fd'})
            return info, orig_b64, proc_b64, stats, hist.to_dict('records')
        except Exception as e:
            return html.Div(f'Processing failed: {e}'), None, None, None, None

    # Default - nothing triggered (or unknown trigger)
    raise PreventUpdate



# Client-side immediate feedback when an upload starts (fast UI pointer)
app.clientside_callback(
    """
    function(contents, filename) {
        if (!contents) { return ''; }
        var name = filename ? filename : 'file';
        return 'Uploading: ' + name + ' — starting processing...';
    }
    """,
    Output('uploading-indicator', 'children'),
    Input('upload-image', 'contents'),
    Input('upload-image', 'filename')
)
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    State('original-image-b64', 'data'),
    State('processed-image-b64', 'data'),
    State('stats-store', 'data'),
    State('hist-store', 'data')
)
def render_tab(tab, orig_b64, proc_b64, stats, hist_records):
    if not stats:
        return html.Div(style={'padding': '40px', 'textAlign': 'center', 'backgroundColor': 'rgba(255,255,255,0.03)', 'borderRadius': '12px'}, children=[
            html.H3('Upload SAR data to begin analysis', style={'color': '#93c5fd'}),
            html.Div('Waiting for input...', style={'opacity': 0.7})
        ])

    hist_df = pd.DataFrame(hist_records)

    if tab == 'overview':
        return html.Div(className='grid', children=[
            html.Div(style={'backgroundColor': 'rgba(255,255,255,0.03)', 'padding': '16px', 'borderRadius': '12px'}, children=[
                html.H3('SAR Intensity Image'),
                html.Img(src='data:image/png;base64,' + proc_b64, style={'maxWidth': '100%', 'height': 'auto', 'borderRadius': '8px', 'border': '1px solid rgba(255,255,255,0.05)'})
            ]),

            html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '12px'}, children=[
                html.Div(style={'backgroundColor': 'rgba(255,255,255,0.03)', 'padding': '16px', 'borderRadius': '12px'}, children=[
                    html.H3('Signal Statistics'),
                    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '8px'}, children=[
                        html.Div(style={'backgroundColor': 'rgba(255,255,255,0.02)', 'padding': '10px', 'borderRadius': '8px'}, children=[
                            html.Div(str(k).replace('Dev','Dev'), style={'fontSize': '12px', 'color': '#93c5fd'}),
                            html.Div(str(v), style={'fontSize': '20px', 'fontWeight': '700'})
                        ]) for k, v in stats.items() if k in ['mean','median','stdDev','min','max','p1','p99','pixels']
                    ])
                ]),

                html.Div(style={'backgroundColor': 'rgba(255,255,255,0.03)', 'padding': '16px', 'borderRadius': '12px'}, children=[
                    html.H3('Quick Info'),
                    html.Div([html.Div(['Polarization', html.Div('VH (Vertical-Horizontal)', style={'color': '#93c5fd'})]),
                              html.Div(['Resolution', html.Div(f"{stats['width']} × {stats['height']} pixels", style={'color':'#93c5fd'})])], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '8px'})
                ])
            ])
        ], style={'display':'grid', 'gridTemplateColumns': '1fr 420px', 'gap': '16px'})

    if tab == 'analysis':
        # histogram plotly bar
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hist_df['intensity'], y=hist_df['count'], marker={'color': '#3b82f6'}))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', margin={'t':10})

        speckle = generate_speckle_stats_from_histogram(hist_df)

        return html.Div(children=[
            html.Div(style={'backgroundColor': 'rgba(255,255,255,0.03)', 'padding': '16px', 'borderRadius': '12px'}, children=[
                html.H3('Intensity Distribution Histogram'),
                dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '320px'})
            ]),

            html.Div(style={'backgroundColor': 'rgba(255,255,255,0.03)', 'padding': '16px', 'borderRadius': '12px'}, children=[
                html.H3('Speckle & Noise Metrics'),
                html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(4,1fr)', 'gap': '10px'}, children=[
                    html.Div(style={'backgroundColor': 'rgba(255,255,255,0.02)', 'padding': '12px', 'borderRadius': '8px'}, children=[
                        html.Div(s['metric'], style={'fontSize': '12px', 'color': '#93c5fd'}),
                        html.Div(s['value'], style={'fontSize': '18px', 'fontWeight': '700'}),
                        html.Div(s['status'], style={'color': '#34d399' if s['status'] in ['Good','High'] else '#fbbf24'})
                    ]) for s in speckle
                ])
            ])
        ], style={'display':'grid', 'gridTemplateColumns': '1fr', 'gap': '12px'})

    if tab == 'features':
        feats = detect_features_from_stats(stats)
        classifications = [
            {'type': 'Urban/Built-up', 'confidence': 78},
            {'type': 'Vegetation', 'confidence': 45},
            {'type': 'Water Bodies', 'confidence': 62},
            {'type': 'Bare Soil', 'confidence': 34}
        ]
        return html.Div(children=[
            html.Div(style={'backgroundColor': 'rgba(255,255,255,0.03)', 'padding': '16px', 'borderRadius': '12px'}, children=[
                html.H3('Detected Features'),
                html.Div(children=[
                    html.Div(style={'backgroundColor': 'rgba(255,255,255,0.02)', 'padding': '10px', 'borderRadius': '8px', 'marginBottom': '8px'}, children=[
                        html.Div(f"{f['feature']} ", style={'fontWeight':'700'}),
                        html.Div(f"{f['value']}", style={'color': '#93c5fd'}),
                        html.Div(f['description'], style={'fontSize':'13px', 'color':'#bcdcff'})
                    ]) for f in feats
                ])
            ]),

            html.Div(style={'backgroundColor': 'rgba(255,255,255,0.03)', 'padding': '16px', 'borderRadius': '12px'}, children=[
                html.H3('Target Classification'),
                html.Div(children=[
                    html.Div(children=[
                        html.Div(item['type'], style={'display':'flex','justifyContent':'space-between'}),
                        html.Div(style={'height':'10px','background':'rgba(255,255,255,0.05)','borderRadius':'8px'}, children=[
                            html.Div(style={'width': f"{item['confidence']}%", 'height':'10px', 'borderRadius':'8px', 'background': '#f87171'})
                        ]),
                        html.Div(f"{item['confidence']}%", style={'color':'#93c5fd','fontSize':'12px','marginTop':'6px'})
                    ], style={'marginBottom':'10px'}) for item in classifications
                ])
            ])
        ], style={'display':'grid', 'gridTemplateColumns':'1fr 420px', 'gap':'12px'})

    if tab == 'processing':
        return html.Div(children=[
            html.Div(style={'backgroundColor': 'rgba(255,255,255,0.03)', 'padding': '16px', 'borderRadius': '12px'}, children=[
                html.H3('Threshold Processing'),
                html.Div([
                    dcc.Slider(id='threshold-slider', min=0, max=100, step=1, value=50, tooltip={'placement':'bottom', 'always_visible':True}),
                    html.Button('Apply Threshold Filter', id='apply-threshold', n_clicks=0, style={'marginTop':'8px', 'width':'100%', 'padding':'10px', 'backgroundColor':'#2563eb','color':'white','border':'none','borderRadius':'8px'})
                ])
            ]),

            html.Div(style={'backgroundColor': 'rgba(255,255,255,0.03)', 'padding': '16px', 'borderRadius': '12px'}, children=[
                html.H3('Export Options'),
                html.Div(style={'display':'grid', 'gridTemplateColumns':'repeat(2, 1fr)', 'gap':'10px'}, children=[
                    html.Button('Export Statistics (CSV)', id='export-csv', n_clicks=0, style={'padding':'10px'}),
                    html.Button('Export Processed Image', id='export-image', n_clicks=0, style={'padding':'10px'}),
                    html.Button('Export Histogram Data', id='export-hist', n_clicks=0, style={'padding':'10px'}),
                    html.Button('Generate Report (PDF - Placeholder)', id='gen-report', n_clicks=0, style={'padding':'10px'})
                ])
            ])
        ], style={'display':'grid','gridTemplateColumns':'1fr','gap':'12px'})

    return html.Div('Unknown tab')



# Download callbacks
@app.callback(
    Output('download-data', 'data'),
    Input('export-csv', 'n_clicks'),
    State('stats-store', 'data'),
    prevent_initial_call=True
)
def export_csv(n_clicks, stats):
    if not stats:
        raise PreventUpdate
    df = pd.DataFrame([stats])
    return dcc.send_data_frame(df.to_csv, filename=f"sar_stats_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv", index=False)


@app.callback(
    Output('download-image', 'data'),
    Input('export-image', 'n_clicks'),
    State('processed-image-b64', 'data'),
    prevent_initial_call=True
)
def export_image(n_clicks, proc_b64):
    if not proc_b64:
        raise PreventUpdate
    data = base64.b64decode(proc_b64)
    return dcc.send_bytes(lambda: data, filename=f"processed_sar_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png")


@app.callback(
    Output('download-data-2', 'data'),
    Input('export-hist', 'n_clicks'),
    State('hist-store', 'data'),
    prevent_initial_call=True
)
def export_hist(n_clicks, hist_records):
    if not hist_records:
        raise PreventUpdate
    df = pd.DataFrame(hist_records)
    return dcc.send_data_frame(df.to_csv, filename=f"sar_histogram_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

# hidden download components
app.layout.children[0].children.append(dcc.Download(id='download-data'))
app.layout.children[0].children.append(dcc.Download(id='download-image'))
app.layout.children[0].children.append(dcc.Download(id='download-data-2'))

# run
if __name__ == '__main__':
    app.run(debug=True, port=8053)

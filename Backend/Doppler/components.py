# components.py
from dash import html, dcc
import dash_bootstrap_components as dbc

def labeled_input(label, id, value, width=80):
    return html.Div([
        html.Label(label, style={
            'display': 'inline-block', 'width': '140px', 'fontWeight': '600',
            'color': '#2c3e50', 'fontSize': '14px'
        }),
        dcc.Input(id=id, type='number', value=value, style={
            'width': f'{width}px', 'padding': '8px 12px', 'border': '2px solid #e0e0e0',
            'borderRadius': '8px', 'fontSize': '14px', 'outline': 'none'
        })
    ], style={'marginBottom': '15px'})

def header():
    return html.Div([
        html.H1("🚗 Doppler ", style={'textAlign': 'center', 'color': 'white', 'margin': '0', 'padding': '30px',
                                      'fontSize': '36px', 'fontWeight': '700', 'letterSpacing': '1px',
                                      'textShadow': '2px 2px 4px rgba(0,0,0,0.3)'}),
        html.P("Upload a WAV file to analyze aliasing and simulate Doppler effect with real audio",
               style={'textAlign': 'center', 'color': 'rgba(255,255,255,0.9)', 'margin': '0', 'paddingBottom': '20px'})
    ], style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'marginBottom': '30px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
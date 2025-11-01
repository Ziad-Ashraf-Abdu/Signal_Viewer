"""
Reusable UI components for consistent user interface across all Dash applications
"""
import dash_bootstrap_components as dbc
from dash import dcc, html

from .config import *


# ==================== BASIC UI COMPONENTS ====================

def create_upload_component(component_id, accept_types='*', multiple=False, 
                          children=None, style_overrides=None):
    """
    Create standardized file upload component - used by ALL applications
    
    Args:
        component_id: Dash component ID
        accept_types: accepted file types (e.g., 'audio/*', 'image/*', '.wav')
        multiple: whether to allow multiple files
        children: custom children content
        style_overrides: additional style properties
        
    Returns:
        dcc.Upload: configured upload component
    """
    if children is None:
        children = html.Div(['📁 Drag and Drop or ', html.A('Select File')])
    
    style = UPLOAD_STYLE.copy()
    if style_overrides:
        style.update(style_overrides)
    
    return dcc.Upload(
        id=component_id,
        children=children,
        style=style,
        multiple=multiple,
        accept=accept_types
    )


def create_card(title, content, card_id=None, style_overrides=None):
    """
    Create standardized card component - used by ALL applications
    
    Args:
        title: card title
        content: card content (can be string or Dash component)
        card_id: optional card ID
        style_overrides: additional style properties
        
    Returns:
        dbc.Card: styled card component
    """
    style = CARD_STYLE.copy()
    if style_overrides:
        style.update(style_overrides)
    
    card_content = [
        html.H3(title, style={'marginTop': 0, 'marginBottom': '15px', 'color': TEXT_COLOR})
    ]
    
    # Handle different content types
    if isinstance(content, list):
        card_content.extend(content)
    else:
        card_content.append(content)
    
    if card_id is not None:
        return dbc.Card(
            dbc.CardBody(card_content),
            style=style,
            id=card_id
        )
    else:
        # If card_id is None, do not pass the id prop at all
        return dbc.Card(
            dbc.CardBody(card_content),
            style=style
        )


def create_button(button_text, button_id, variant="primary", disabled=False, 
                 n_clicks=0, style_overrides=None, **kwargs):
    """
    Create standardized button component
    
    Args:
        button_text: button text
        button_id: Dash component ID
        variant: button style variant
        disabled: whether button is disabled
        n_clicks: initial n_clicks value
        style_overrides: additional style properties
        **kwargs: additional button properties
        
    Returns:
        html.Button or dbc.Button: styled button
    """
    base_style = get_button_style(variant, disabled)
    if style_overrides:
        base_style.update(style_overrides)
    
    # Use html.Button for full style control, dbc.Button for Bootstrap features
    if 'color' in kwargs or 'className' in kwargs:
        return dbc.Button(
            button_text,
            id=button_id,
            n_clicks=n_clicks,
            disabled=disabled,
            **kwargs
        )
    else:
        return html.Button(
            button_text,
            id=button_id,
            n_clicks=n_clicks,
            disabled=disabled,
            style=base_style,
            **kwargs
        )


def create_slider(slider_id, min_value, max_value, step, value, marks=None, 
                 disabled=False, tooltip_visible=True):
    """
    Create standardized slider component
    
    Args:
        slider_id: Dash component ID
        min_value: minimum slider value
        max_value: maximum slider value
        step: slider step size
        value: initial value
        marks: slider marks dictionary
        disabled: whether slider is disabled
        tooltip_visible: whether to show tooltip
        
    Returns:
        dcc.Slider: configured slider
    """
    return dcc.Slider(
        id=slider_id,
        min=min_value,
        max=max_value,
        step=step,
        value=value,
        marks=marks,
        disabled=disabled,
        tooltip={"placement": "bottom", "always_visible": tooltip_visible}
    )


def create_audio_player(audio_id, audio_src, style_overrides=None):
    """
    Create standardized audio player
    
    Args:
        audio_id: Dash component ID
        audio_src: audio source (URL or data URL)
        style_overrides: additional style properties
        
    Returns:
        html.Audio: configured audio player
    """
    style = {"width": "100%", "marginTop": "10px", "marginBottom": "15px"}
    if style_overrides:
        style.update(style_overrides)
    
    return html.Audio(
        id=audio_id,
        src=audio_src,
        controls=True,
        style=style
    )


# ==================== COMPOSITE UI COMPONENTS ====================

def create_analysis_card(title, stats_dict, include_histogram=False, hist_data=None):
    """
    Create analysis card with statistics and optional histogram
    
    Args:
        title: card title
        stats_dict: dictionary of statistics to display
        include_histogram: whether to include histogram
        hist_data: histogram data (if including histogram)
        
    Returns:
        dbc.Card: analysis card
    """
    content = []
    
    # Add statistics
    stats_content = []
    for key, value in stats_dict.items():
        stats_content.append(
            html.Div([
                html.Span(f"{key}: ", style={'color': SUBTLE_TEXT_COLOR}),
                html.Span(str(value), style={'fontWeight': 'bold'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '5px 0'})
        )
    
    content.append(html.Div(stats_content))
    
    # Add histogram if requested
    if include_histogram and hist_data is not None:
        from .plotting_utils import plotting_utils
        hist_fig = plotting_utils.create_histogram_plot(hist_data, f"{title} Distribution")
        content.append(dcc.Graph(figure=hist_fig))
    
    return create_card(title, content)


def create_audio_comparison_players(audio_data_dict):
    """
    Create multiple audio players for comparison (used by Human app)
    
    Args:
        audio_data_dict: dictionary of {player_name: {data: audio_data, sr: sample_rate}}
        
    Returns:
        dbc.Row: row of audio player columns
    """
    from .audio_processing import audio_processor
    
    cols = []
    color_classes = {
        'original': 'text-primary',
        'downsampled': 'text-danger', 
        'reconstructed': 'text-success',
        'aliased': 'text-warning'
    }
    
    for i, (name, audio_info) in enumerate(audio_data_dict.items()):
        if audio_info.get('data') is not None:
            audio_src = audio_processor.make_playable_wav(
                audio_info['data'], 
                audio_info.get('sr', 44100)
            )
            
            color_class = color_classes.get(name, 'text-secondary')
            
            cols.append(
                dbc.Col([
                    html.H5(name.capitalize(), className=color_class),
                    create_audio_player(f"audio-{name}", audio_src)
                ], width=12, lg=4, className="mb-3 mb-lg-0")
            )
    
    return dbc.Row(cols, className="mt-3 g-3")


def create_navigation_header(main_title, subtitle, icon="🚀", background_gradient=None):
    """
    Create standardized navigation header
    
    Args:
        main_title: main header title
        subtitle: header subtitle
        icon: header icon
        background_gradient: custom background gradient
        
    Returns:
        html.Div: header component
    """
    if background_gradient is None:
        background_gradient = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    
    return html.Div([
        html.H1(f"{icon} {main_title}", style={
            'textAlign': 'center', 
            'color': 'white', 
            'margin': '0', 
            'padding': '30px',
            'fontSize': '36px', 
            'fontWeight': '700', 
            'letterSpacing': '1px',
            'textShadow': '2px 2px 4px rgba(0,0,0,0.3)'
        }),
        html.P(subtitle, style={
            'textAlign': 'center', 
            'color': 'rgba(255,255,255,0.9)', 
            'margin': '0', 
            'paddingBottom': '20px',
            'fontSize': '16px'
        })
    ], style={
        'background': background_gradient,
        'marginBottom': '30px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
    })


def create_tab_layout(tab_data, default_tab=None):
    """
    Create standardized tab layout
    
    Args:
        tab_data: list of {label: tab_label, value: tab_value}
        default_tab: default active tab
        
    Returns:
        tuple: (dcc.Tabs, html.Div) for tabs and content area
    """
    if default_tab is None:
        default_tab = tab_data[0]['value'] if tab_data else 'tab1'
    
    tabs = dcc.Tabs(
        id='app-tabs',
        value=default_tab,
        children=[dcc.Tab(label=tab['label'], value=tab['value']) for tab in tab_data],
        style={'marginBottom': '20px'}
    )
    
    content_area = html.Div(id='tab-content')
    
    return tabs, content_area


def create_status_indicator(status, message, size="md"):
    """
    Create status indicator (loading, success, error, etc.)
    
    Args:
        status: status type ('loading', 'success', 'error', 'warning', 'info')
        message: status message
        size: indicator size ('sm', 'md', 'lg')
        
    Returns:
        html.Div: status indicator
    """
    size_styles = {
        'sm': {'padding': '8px 12px', 'fontSize': '12px'},
        'md': {'padding': '12px 16px', 'fontSize': '14px'},
        'lg': {'padding': '16px 20px', 'fontSize': '16px'}
    }
    
    status_colors = {
        'loading': {'bg': '#FFF3CD', 'border': '#FFEAA7', 'text': '#856404'},
        'success': {'bg': '#D1ECF1', 'border': '#B6E2E9', 'text': '#0C5460'},
        'error': {'bg': '#F8D7DA', 'border': '#F1AEB5', 'text': '#721C24'},
        'warning': {'bg': '#FFF3CD', 'border': '#FFEAA7', 'text': '#856404'},
        'info': {'bg': '#D1ECF1', 'border': '#B6E2E9', 'text': '#0C5460'}
    }
    
    color_info = status_colors.get(status, status_colors['info'])
    size_style = size_styles.get(size, size_styles['md'])
    
    indicator_style = {
        'backgroundColor': color_info['bg'],
        'border': f"1px solid {color_info['border']}",
        'borderRadius': '8px',
        'color': color_info['text'],
        'fontWeight': '500',
        'textAlign': 'center',
        **size_style
    }
    
    return html.Div(message, style=indicator_style)


# ==================== LAYOUT TEMPLATES ====================

def create_main_container(children, fluid=True, background_color=None, padding=True):
    """
    Create main application container
    
    Args:
        children: container children
        fluid: whether to use fluid container
        background_color: background color
        padding: whether to add padding
        
    Returns:
        dbc.Container: main container
    """
    if background_color is None:
        background_color = BACKGROUND_COLOR
    
    container_style = {
        'backgroundColor': background_color,
        'minHeight': '100vh'
    }
    
    if padding:
        container_style['padding'] = '40px 20px'
    
    return dbc.Container(
        children,
        fluid=fluid,
        style=container_style,
        className="py-4"
    )


def create_two_column_layout(left_content, right_content, left_width=8, right_width=4):
    """
    Create two-column layout
    
    Args:
        left_content: left column content
        right_content: right column content  
        left_width: left column width (1-12)
        right_width: right column width (1-12)
        
    Returns:
        dbc.Row: two-column layout
    """
    return dbc.Row([
        dbc.Col(left_content, width=12, lg=left_width, className="mb-4 mb-lg-0"),
        dbc.Col(right_content, width=12, lg=right_width)
    ], className="g-4")


def create_control_panel(controls, title="Controls", width=4):
    """
    Create standardized control panel
    
    Args:
        controls: list of control components
        title: panel title
        width: panel width (1-12)
        
    Returns:
        dbc.Col: control panel column
    """
    panel_content = [html.H4(title, style={'marginBottom': '20px', 'color': TEXT_COLOR})]
    panel_content.extend(controls)
    
    return dbc.Col(
        create_card("", panel_content),
        width=12, md=width, className="mb-4"
    )


# Global utility function
def create_data_display_store(store_id, initial_data=None):
    """
    Create hidden store for data persistence
    
    Args:
        store_id: store component ID
        initial_data: initial store data
        
    Returns:
        dcc.Store: data store component
    """
    return dcc.Store(id=store_id, data=initial_data)
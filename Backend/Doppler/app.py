# app.py
from dash import Dash, Output, Input
import dash_bootstrap_components as dbc
from layout import create_layout
from callbacks import register_callbacks

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True
app.title = "Doppler"

app.layout = create_layout()
register_callbacks(app)

# === Improved Client-Side Audio Callbacks ===

app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks || n_clicks < 1) return '';
        if (typeof window._doppAudioInit === 'undefined') {
            window._doppAudioInit = true;
            window.AudioContext = window.AudioContext || window.webkitAudioContext;
            window.audioCtx = new window.AudioContext();
            window.gainNode = window.audioCtx.createGain();
            window.gainNode.gain.value = 1;
            window.gainNode.connect(window.audioCtx.destination);
            window.oscillator = null;
            window.muted = false;
        }
        if (window.audioCtx.state === 'suspended') {
            window.audioCtx.resume();
        }
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
            window._doppAudioInit = true;
            window.AudioContext = window.AudioContext || window.webkitAudioContext;
            window.audioCtx = new window.AudioContext();
            window.gainNode = window.audioCtx.createGain();
            window.gainNode.gain.value = 1;
            window.gainNode.connect(window.audioCtx.destination);
            window.oscillator = null;
            window.muted = false;
        }
        if (!n_clicks) return '';
        window.muted = (n_clicks % 2) === 1;
        try {
            window.gainNode.gain.value = window.muted ? 0 : 1;
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
        if (typeof window._doppAudioInit === 'undefined') {
            window._doppAudioInit = true;
            window.AudioContext = window.AudioContext || window.webkitAudioContext;
            window.audioCtx = new window.AudioContext();
            window.gainNode = window.audioCtx.createGain();
            window.gainNode.gain.value = 1;
            window.gainNode.connect(window.audioCtx.destination);
            window.oscillator = null;
            window.muted = false;
        }

        if (window.audioCtx.state === 'suspended') {
            window.audioCtx.resume();
        }

        if (window.muted || !freq || freq <= 0) {
            if (window.oscillator) {
                try {
                    window.oscillator.stop();
                    window.oscillator.disconnect();
                } catch (e) {}
                window.oscillator = null;
            }
            return '';
        }

        if (!window.oscillator) {
            try {
                window.oscillator = window.audioCtx.createOscillator();
                window.oscillator.type = 'sine';
                window.oscillator.frequency.value = freq;
                window.oscillator.connect(window.gainNode);
                window.oscillator.start(0);
            } catch (e) {
                window.oscillator = null;
            }
        } else {
            try {
                const now = window.audioCtx.currentTime;
                window.oscillator.frequency.cancelScheduledValues(now);
                window.oscillator.frequency.setValueAtTime(freq, now);
            } catch (e) {}
        }
        return '';
    }
    """,
    Output('sound-div', 'children'),
    Input('sound-freq', 'data')
)

if __name__ == '__main__':
    app.run(debug=True)
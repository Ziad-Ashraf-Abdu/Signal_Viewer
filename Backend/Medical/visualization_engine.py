# visualization_engine.py
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import numpy as np
import time
from scipy.signal import resample

class VisualizationEngine:
    def __init__(self):
        pass
        
    def create_empty_figure(self, title="No data"):
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title={"text": title, "y": 0.5, "x": 0.5, "xanchor": "center", "yanchor": "middle"},
            xaxis={'visible': False},
            yaxis={'visible': False},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#6B7280'
        )
        return fig

    def xor_overlay_segments(self, prev_vals, new_vals, strict=True):
        if prev_vals is None or new_vals is None:
            return prev_vals, new_vals

        a = np.asarray(prev_vals, dtype=float).copy()
        b = np.asarray(new_vals, dtype=float).copy()

        if a.size == 0 or b.size == 0:
            return a, b

        if a.shape != b.shape:
            m = min(a.size, b.size)
            a = a[:m]
            b = b[:m]

        if strict:
            same_mask = (a == b)
        else:
            scale = max(1.0, np.nanmax(np.abs(np.concatenate([a, b]))))
            tol = 1e-9 * scale
            same_mask = np.isclose(a, b, atol=tol, rtol=0.0)

        prev_out = a.copy()
        new_out = b.copy()
        prev_out[same_mask] = np.nan
        new_out[same_mask] = np.nan

        return prev_out, new_out

    def _calculate_fft(self, signal_segment, original_fs, resampling_freq):
        start_time = time.time()

        y = signal_segment
        y = y[~np.isnan(y)]

        if len(y) < 2:
            return None, None, original_fs, 0, 0

        # Find f_max from original signal
        N_orig = len(y)
        yf_orig = np.fft.fft(y)
        xf_orig = np.fft.fftfreq(N_orig, 1 / original_fs)

        magnitudes = np.abs(yf_orig[1:N_orig // 2])
        frequencies = xf_orig[1:N_orig // 2]

        if len(magnitudes) > 0:
            noise_threshold = np.max(magnitudes) * 0.01
            significant_freqs = frequencies[magnitudes > noise_threshold]
            if len(significant_freqs) > 0:
                f_max = np.max(significant_freqs)
            else:
                f_max = 0
        else:
            f_max = 0

        # Perform resampling for plot
        resampling_freq_val = float(resampling_freq or 500.0)
        if resampling_freq_val >= original_fs:
            step = 1
        else:
            step = int(round(original_fs / resampling_freq_val))
        step = max(1, step)

        y_sampled = y[::step]
        fs_new = original_fs / step

        if len(y_sampled) < 2:
            return None, None, fs_new, 0, f_max

        # Calculate FFT on down-sampled signal
        N = len(y_sampled)
        yf = np.fft.fft(y_sampled)
        xf = np.fft.fftfreq(N, 1 / fs_new)

        xf_plot = xf[:N // 2]
        yf_plot = np.abs(yf[0:N // 2])

        computation_time = time.time() - start_time

        return xf_plot, yf_plot, fs_new, computation_time, f_max

    def make_time_domain_visualization(self, viz_type, current_segment_df, channels_to_display, 
                                     patient_name, overlay=True, unit="mV", prev_segment_df=None):
        if viz_type == "icu":
            if overlay:
                fig = go.Figure()
                for ch in channels_to_display:
                    seg = current_segment_df[["time", ch]]
                    t = (seg["time"].values - seg["time"].values[0]).astype(float)
                    y = seg[ch].values.astype(float)
                    fig.add_trace(go.Scattergl(x=t, y=y, mode="lines", name=ch, customdata=[ch] * len(t)))
                fig.update_layout(yaxis_title=f"Amplitude ({unit})")
            else:
                n = len(channels_to_display)
                fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=channels_to_display)
                for i, ch in enumerate(channels_to_display, start=1):
                    seg = current_segment_df[["time", ch]]
                    t = (seg["time"].values - seg["time"].values[0]).astype(float)
                    y = seg[ch].values.astype(float)
                    fig.add_trace(go.Scattergl(x=t, y=y, mode="lines", name=ch, customdata=[ch] * len(t)), row=i, col=1)
                    fig.update_yaxes(title_text=f"Amp ({unit})", row=i, col=1)
                fig.update_layout(showlegend=False)
            fig.update_layout(title=f"Time Domain Monitor: {patient_name}", xaxis_title="Time (s)")

        elif viz_type == "pingpong":
            if overlay:
                fig = go.Figure()
                for ch in channels_to_display:
                    y_curr = current_segment_df[ch].values
                    t = (current_segment_df["time"].values - current_segment_df["time"].values[0]).astype(float)
                    if prev_segment_df is not None and prev_segment_df.shape[0] == len(current_segment_df):
                        y_prev = prev_segment_df[ch].values
                        prev_masked, curr_masked = self.xor_overlay_segments(y_prev, y_curr)
                        fig.add_trace(go.Scattergl(x=t, y=prev_masked, mode="lines", name=f"{ch} (Prev)",
                                                   line=dict(dash='dash', color='gray'), customdata=[ch] * len(t)))
                        fig.add_trace(go.Scattergl(x=t, y=curr_masked, mode="lines", name=f"{ch} (Curr)",
                                                   customdata=[ch] * len(t)))
                    else:
                        fig.add_trace(go.Scattergl(x=t, y=y_curr, mode="lines", name=f"{ch} (Curr)",
                                                   customdata=[ch] * len(t)))
                fig.update_layout(yaxis_title=f"Amplitude ({unit})")
            else:
                n = len(channels_to_display)
                fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=channels_to_display)
                for i, ch in enumerate(channels_to_display, start=1):
                    y_curr = current_segment_df[ch].values
                    t = (current_segment_df["time"].values - current_segment_df["time"].values[0]).astype(float)
                    if prev_segment_df is not None and prev_segment_df.shape[0] == len(current_segment_df):
                        y_prev = prev_segment_df[ch].values
                        prev_masked, curr_masked = self.xor_overlay_segments(y_prev, y_curr)
                        fig.add_trace(go.Scattergl(x=t, y=prev_masked, mode="lines", name=f"Prev",
                                                   line=dict(dash='dash', color='gray'), customdata=[ch] * len(t)),
                                      row=i, col=1)
                        fig.add_trace(go.Scattergl(x=t, y=curr_masked, mode="lines", name=f"Curr", customdata=[ch] * len(t)),
                                      row=i, col=1)
                    else:
                        fig.add_trace(go.Scattergl(x=t, y=y_curr, mode="lines", name=f"Curr", customdata=[ch] * len(t)),
                                      row=i, col=1)
                    fig.update_yaxes(title_text=f"Amp ({unit})", row=i, col=1)
                fig.update_layout(showlegend=True)
            fig.update_layout(title=f"Ping-Pong Overlay: {patient_name}", xaxis_title="Time (s)")

        elif viz_type == "polar":
            time_vals = current_segment_df["time"].values
            span = time_vals[-1] - time_vals[0] if len(time_vals) > 1 and time_vals[-1] != time_vals[0] else 1.0
            theta = 360 * (time_vals - time_vals[0]) / span

            if overlay:
                fig = go.Figure()
                for ch in channels_to_display:
                    r = current_segment_df[ch].values
                    fig.add_trace(go.Scatterpolar(theta=theta, r=r, mode="lines", name=ch, customdata=[ch] * len(r)))
                fig.update_layout(polar=dict(radialaxis=dict(title=f"Amplitude ({unit})")))
            else:
                n_channels = len(channels_to_display)
                cols = min(2, n_channels) if n_channels > 1 else 1
                rows = int(np.ceil(n_channels / cols))
                specs = [[{'type': 'polar'}] * cols for _ in range(rows)]
                fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=channels_to_display,
                                    horizontal_spacing=0.15, vertical_spacing=0.15)
                for idx, ch in enumerate(channels_to_display):
                    row, col = (idx // cols) + 1, (idx % cols) + 1
                    r = current_segment_df[ch].values
                    fig.add_trace(go.Scatterpolar(theta=theta, r=r, mode="lines", name=ch, customdata=[ch] * len(r)), row=row, col=col)
                    polar_name = f'polar{idx + 1}' if idx > 0 else 'polar'
                    fig.layout[polar_name].radialaxis.title = f'Amp ({unit})'
                fig.update_layout(showlegend=False)
            fig.update_layout(title=f"Polar View: {patient_name}")

        elif viz_type == "crossrec":
            if len(channels_to_display) < 2:
                return self.create_empty_figure("Need at least 2 channels for Cross-Recurrence")

            if overlay:
                midpoint = len(channels_to_display) // 2
                set_a_ch = channels_to_display[:midpoint]
                set_b_ch = channels_to_display[midpoint:]
                if not set_a_ch or not set_b_ch:
                    return self.create_empty_figure("Need channels for both comparison sets in overlay mode")

                s1 = current_segment_df[set_a_ch].mean(axis=1).values
                s2 = current_segment_df[set_b_ch].mean(axis=1).values

                fig = go.Figure()
                hist, xedges, yedges = np.histogram2d(s1, s2, bins=80)
                fig.add_trace(go.Heatmap(z=hist.T, x=xedges, y=yedges, colorscale="Viridis",
                                         customdata=[f"Avg({','.join(set_a_ch)})"] * len(xedges)))
                fig.update_layout(
                    title=f"Cross-Recurrence: Avg({', '.join(set_a_ch)}) vs. Avg({', '.join(set_b_ch)})",
                    xaxis_title=f"Avg of Set A ({unit})",
                    yaxis_title=f"Avg of Set B ({unit})"
                )
            else:
                pairs = [(channels_to_display[i], channels_to_display[i + 1]) for i in range(0, len(channels_to_display) - 1, 2)]
                if not pairs:
                    return self.create_empty_figure("Not enough channels for pairing")
                n_pairs = len(pairs)
                fig = make_subplots(rows=n_pairs, cols=1, subplot_titles=[f"{p[0]} vs {p[1]}" for p in pairs])
                for i, (ch_a, ch_b) in enumerate(pairs, start=1):
                    s1 = current_segment_df[ch_a].values
                    s2 = current_segment_df[ch_b].values
                    hist, xedges, yedges = np.histogram2d(s1, s2, bins=80)
                    fig.add_trace(go.Heatmap(z=hist.T, x=xedges, y=yedges, colorscale="Viridis",
                                             customdata=[ch_a] * len(xedges)), row=i, col=1)
                    fig.update_xaxes(title_text=f"{ch_a} ({unit})", row=i, col=1)
                    fig.update_yaxes(title_text=f"{ch_b} ({unit})", row=i, col=1)
                fig.update_layout(title=f"Cross-Recurrence: {patient_name}")

        else:
            fig = self.create_empty_figure(f"Unknown Visualization Type")

        fig.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=60, b=40))
        return fig

    def make_frequency_domain_visualization(self, viz_type, current_segment_df, channels_to_display,
                                           patient_name, fs, resampling_freq, overlay=True, prev_segment_df=None):
        final_fs_new = fs

        if viz_type == "icu":
            if overlay:
                fig = go.Figure()
                for ch in channels_to_display:
                    y_segment = current_segment_df[ch].values
                    xf, yf, fs_new, comp_time, f_max = self._calculate_fft(y_segment, fs, resampling_freq)
                    final_fs_new = fs_new
                    if xf is not None:
                        fig.add_trace(go.Scattergl(x=xf, y=yf, mode="lines", name=ch, customdata=[ch] * len(xf)))
                fig.update_layout(yaxis_title="Magnitude")
            else:
                n = len(channels_to_display)
                fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=channels_to_display)
                for i, ch in enumerate(channels_to_display, start=1):
                    y_segment = current_segment_df[ch].values
                    xf, yf, fs_new, comp_time, f_max = self._calculate_fft(y_segment, fs, resampling_freq)
                    final_fs_new = fs_new
                    if xf is not None:
                        fig.add_trace(go.Scattergl(x=xf, y=yf, mode="lines", name=ch, customdata=[ch] * len(xf)), row=i, col=1)
                    fig.update_yaxes(title_text="Magnitude", row=i, col=1)
                fig.update_layout(showlegend=False)
            fig.update_layout(
                title=f"Frequency Spectrum: {patient_name}",
                xaxis_title=f"Frequency (Hz) - Sampled at {final_fs_new:.1f} Hz"
            )

        elif viz_type == "pingpong":
            if overlay:
                fig = go.Figure()
                for ch in channels_to_display:
                    xf_curr, yf_curr, fs_new_curr, ct_curr, fmax_curr = self._calculate_fft(
                        current_segment_df[ch].values, fs, resampling_freq)
                    final_fs_new = fs_new_curr
                    if xf_curr is not None:
                        fig.add_trace(go.Scattergl(x=xf_curr, y=yf_curr, mode="lines", name=f"{ch} (Curr)",
                                                   customdata=[ch] * len(xf_curr)))

                    if prev_segment_df is not None and prev_segment_df.shape[0] > 1:
                        xf_prev, yf_prev, _, ct_prev, fmax_prev = self._calculate_fft(prev_segment_df[ch].values, fs,
                                                                                     resampling_freq)
                        if xf_prev is not None:
                            fig.add_trace(go.Scattergl(x=xf_prev, y=yf_prev, mode="lines", name=f"{ch} (Prev)",
                                                       line=dict(dash='dash', color='gray'),
                                                       customdata=[ch] * len(xf_prev)))
                fig.update_layout(yaxis_title="Magnitude")
            else:
                n = len(channels_to_display)
                fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=channels_to_display)
                for i, ch in enumerate(channels_to_display, start=1):
                    xf_curr, yf_curr, fs_new_curr, ct_curr, fmax_curr = self._calculate_fft(
                        current_segment_df[ch].values, fs, resampling_freq)
                    final_fs_new = fs_new_curr
                    if xf_curr is not None:
                        fig.add_trace(go.Scattergl(x=xf_curr, y=yf_curr, mode="lines", name=f"Curr",
                                                   customdata=[ch] * len(xf_curr)), row=i, col=1)

                    if prev_segment_df is not None and prev_segment_df.shape[0] > 1:
                        xf_prev, yf_prev, _, ct_prev, fmax_prev = self._calculate_fft(prev_segment_df[ch].values, fs,
                                                                                     resampling_freq)
                        if xf_prev is not None:
                            fig.add_trace(go.Scattergl(x=xf_prev, y=yf_prev, mode="lines", name=f"Prev",
                                                       line=dict(dash='dash', color='gray'),
                                                       customdata=[ch] * len(xf_prev)), row=i, col=1)
                    fig.update_yaxes(title_text="Magnitude", row=i, col=1)
                fig.update_layout(showlegend=True)
            fig.update_layout(
                title=f"Spectral Comparison: {patient_name}",
                xaxis_title=f"Frequency (Hz) - Sampled at {final_fs_new:.1f} Hz"
            )

        elif viz_type == "polar":
            if overlay:
                fig = go.Figure()
                for ch in channels_to_display:
                    xf, yf, fs_new, ct, f_max = self._calculate_fft(current_segment_df[ch].values, fs, resampling_freq)
                    final_fs_new = fs_new
                    if xf is not None and len(xf) > 0:
                        theta = xf * 360 / (fs_new / 2)
                        fig.add_trace(go.Scatterpolar(theta=theta, r=yf, mode="lines", name=ch, customdata=[ch] * len(xf)))
                fig.update_layout(polar=dict(radialaxis_type="log", radialaxis_title="Magnitude"))
            else:
                n_channels = len(channels_to_display)
                cols = min(2, n_channels) if n_channels > 1 else 1
                rows = int(np.ceil(n_channels / cols))
                specs = [[{'type': 'polar'}] * cols for _ in range(rows)]
                fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=channels_to_display,
                                    horizontal_spacing=0.15, vertical_spacing=0.15)
                for idx, ch in enumerate(channels_to_display):
                    row, col = (idx // cols) + 1, (idx % cols) + 1
                    xf, yf, fs_new, ct, f_max = self._calculate_fft(current_segment_df[ch].values, fs, resampling_freq)
                    final_fs_new = fs_new
                    if xf is not None and len(xf) > 0:
                        theta = xf * 360 / (fs_new / 2)
                        fig.add_trace(go.Scatterpolar(theta=theta, r=yf, mode="lines", name=ch, customdata=[ch] * len(xf)),
                                    row=row, col=col)
                    polar_name = f'polar{idx + 1}' if idx > 0 else 'polar'
                    fig.layout[polar_name].radialaxis.type = "log"
                    fig.layout[polar_name].radialaxis.title = "Magnitude"
                fig.update_layout(showlegend=False)
            fig.update_layout(title=f"Spectral Polar View: {patient_name}")

        elif viz_type == "crossrec":
            if len(channels_to_display) < 2:
                return self.create_empty_figure("Need at least 2 channels for Cross-Recurrence")

            if overlay:
                midpoint = len(channels_to_display) // 2
                set_a_ch = channels_to_display[:midpoint]
                set_b_ch = channels_to_display[midpoint:]
                if not set_a_ch or not set_b_ch:
                    return self.create_empty_figure("Need channels for both comparison sets in overlay mode")

                xf_a_all, yfs_a, yfs_b = [], [], []
                min_len = float('inf')

                for ch in set_a_ch:
                    xf, yf, _, ct, f_max = self._calculate_fft(current_segment_df[ch].values, fs, resampling_freq)
                    if yf is not None:
                        yfs_a.append(yf)
                        xf_a_all = xf
                        min_len = min(min_len, len(yf))

                for ch in set_b_ch:
                    xf, yf, fs_new, ct, f_max = self._calculate_fft(current_segment_df[ch].values, fs, resampling_freq)
                    final_fs_new = fs_new
                    if yf is not None:
                        yfs_b.append(yf)
                        min_len = min(min_len, len(yf))

                if not yfs_a or not yfs_b or min_len == float('inf'):
                    return self.create_empty_figure("Could not compute FFT for channel sets")

                yf_a_avg = np.mean([y[:min_len] for y in yfs_a], axis=0)
                yf_b_avg = np.mean([y[:min_len] for y in yfs_b], axis=0)

                z = np.outer(yf_a_avg, yf_b_avg)
                xf_plot = xf_a_all[:min_len] if xf_a_all is not None else np.arange(min_len)

                fig = go.Figure(data=go.Heatmap(z=z, x=xf_plot, y=xf_plot, colorscale='Viridis'))
                fig.update_layout(
                    title=f"Spectral Cross-Recurrence: Avg({', '.join(set_a_ch)}) vs. Avg({', '.join(set_b_ch)})",
                    xaxis_title=f"Frequency (Hz) - Set A",
                    yaxis_title=f"Frequency (Hz) - Set B"
                )

            else:
                pairs = [(channels_to_display[i], channels_to_display[i + 1]) for i in range(0, len(channels_to_display) - 1, 2)]
                if not pairs:
                    return self.create_empty_figure("Not enough channels for pairing")
                n_pairs = len(pairs)
                fig = make_subplots(rows=n_pairs, cols=1, subplot_titles=[f"{p[0]} vs {p[1]}" for p in pairs])

                for i, (ch_a, ch_b) in enumerate(pairs, start=1):
                    xf_a, yf_a, fs_new_a, ct_a, f_max_a = self._calculate_fft(current_segment_df[ch_a].values, fs, resampling_freq)
                    xf_b, yf_b, fs_new_b, ct_b, f_max_b = self._calculate_fft(current_segment_df[ch_b].values, fs, resampling_freq)
                    final_fs_new = fs_new_a

                    if xf_a is not None and xf_b is not None:
                        z = np.outer(yf_a, yf_b)
                        fig.add_trace(go.Heatmap(z=z, x=xf_a, y=xf_b, colorscale="Viridis", customdata=[ch_a] * len(xf_a)),
                                    row=i, col=1)
                    fig.update_xaxes(title_text=f"Frequency ({ch_a})", row=i, col=1)
                    fig.update_yaxes(title_text=f"Frequency ({ch_b})", row=i, col=1)
                fig.update_layout(title="Cross-Recurrence of Frequencies")

        else:
            fig = self.create_empty_figure(f"Unknown Visualization Type")

        fig.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=60, b=40))
        return fig
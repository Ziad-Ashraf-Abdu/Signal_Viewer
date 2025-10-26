# channel_manager.py
import numpy as np
import pandas as pd

class ChannelManager:
    def __init__(self):
        pass
        
    def derive_third_ecg_channel(self, sig1, sig2, method="difference"):
        if len(sig1) != len(sig2):
            min_len = min(len(sig1), len(sig2))
            sig1, sig2 = sig1[:min_len], sig2[:min_len]

        if method == "difference":
            derived = sig2 - sig1
            print(f"[Channel Derivation] Created third ECG channel using difference method (Lead2 - Lead1)")
        elif method == "sum":
            derived = (sig1 + sig2) / 2
            print(f"[Channel Derivation] Created third ECG channel using sum method ((Lead1 + Lead2)/2)")
        elif method == "orthogonal":
            dot_product = np.dot(sig1, sig2) / (np.linalg.norm(sig1) * np.linalg.norm(sig2))
            derived = sig2 - dot_product * sig1
            print(f"[Channel Derivation] Created third ECG channel using orthogonal method")
        else:
            derived = sig2 - sig1
            print(f"[Channel Derivation] Created third ECG channel using default difference method")

        return derived

    def get_display_channels(self, patient, dataset_type, show_all_channels=False):
        if "ecg" not in patient or patient["ecg"] is None:
            return []

        available_channels = [c for c in patient["ecg"].columns if c.startswith("signal_")]

        if dataset_type == "ECG":
            if len(available_channels) == 0:
                return []
            elif len(available_channels) == 1:
                return [available_channels[0]]
            elif len(available_channels) == 2:
                return available_channels
            elif len(available_channels) >= 3:
                if show_all_channels:
                    return available_channels
                else:
                    return available_channels[:3]
        else:
            if len(available_channels) <= 3:
                return available_channels
            else:
                if show_all_channels:
                    return available_channels
                else:
                    return available_channels[:3]

        return available_channels

    def process_patient_channels(self, patient, dataset_type, show_all_channels=False):
        if "ecg" not in patient or patient["ecg"] is None:
            return {"channels": [], "derived_info": "No data available"}

        ecg_df = patient["ecg"].copy()
        available_channels = [c for c in ecg_df.columns if c.startswith("signal_")]

        result = {
            "channels": [],
            "derived_info": "",
            "original_count": len(available_channels)
        }

        if dataset_type == "ECG":
            if len(available_channels) == 0:
                result["derived_info"] = "No ECG channels available"
                return result
            elif len(available_channels) == 1:
                ch = available_channels[0]
                result["channels"] = [ch, f"{ch}_copy", f"{ch}_inverted"]
                ecg_df[f"{ch}_copy"] = ecg_df[ch]
                ecg_df[f"{ch}_inverted"] = -ecg_df[ch]
                result["derived_info"] = f"Single channel {ch} displayed with copy and inverted version"
            elif len(available_channels) == 2:
                ch1, ch2 = available_channels[0], available_channels[1]
                derived_ch = f"derived_{ch1}_{ch2}"
                derived_signal = self.derive_third_ecg_channel(ecg_df[ch1].values, ecg_df[ch2].values, method="difference")
                ecg_df[derived_ch] = derived_signal
                result["channels"] = [ch1, ch2, derived_ch]
                result["derived_info"] = f"Third channel '{derived_ch}' derived from {ch1} - {ch2}"
            else:
                if show_all_channels:
                    result["channels"] = available_channels
                    result["derived_info"] = f"Showing all {len(available_channels)} channels"
                else:
                    result["channels"] = available_channels[:3]
                    result["derived_info"] = f"Showing main 3 channels (out of {len(available_channels)} available)"
        else:
            if len(available_channels) <= 3:
                result["channels"] = available_channels
                result["derived_info"] = f"Showing all {len(available_channels)} EEG channels"
            else:
                if show_all_channels:
                    result["channels"] = available_channels
                    result["derived_info"] = f"Showing all {len(available_channels)} EEG channels"
                else:
                    result["channels"] = available_channels[:3]
                    result["derived_info"] = f"Showing main 3 EEG channels (out of {len(available_channels)} available)"

        patient["ecg"] = ecg_df
        result["processed_df"] = ecg_df

        return result
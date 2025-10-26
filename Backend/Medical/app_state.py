# app_state.py
import numpy as np

class AppStateManager:
    def __init__(self):
        self.global_data = {
            "patients": [],
            "buffers": {},
            "dataset_type": "ECG"
        }
        self.app_state = None
        
    def ensure_buffers_for_patient(self, pid, fs, display_window, rr_capacity=300):
        bufs = self.global_data["buffers"]
        if pid not in bufs:
            blen = max(1, int(round(display_window * fs)))
            bufs[pid] = {
                "signal_buffer": np.full(blen, np.nan),
                "write_idx": 0,
                "len": blen,
                "rr_buffer": np.full(rr_capacity, np.nan),
                "rr_write_idx": 0,
                "last_peak_global_index": -1,
                "direction": 1,
                "ping_position": 0.0,
                "ai_analysis": None
            }
        else:
            bufinfo = bufs[pid]
            desired_len = max(1, int(round(display_window * fs)))
            if bufinfo["len"] != desired_len:
                bufinfo["signal_buffer"] = np.full(desired_len, np.nan)
                bufinfo["len"] = desired_len
                bufinfo["write_idx"] = 0

    def update_playback_state(self, patients, trigger, state, chunk_ms_val, speed_val):
        if trigger == "play-btn":
            state["playing"] = True
        elif trigger == "pause-btn":
            state["playing"] = False
        elif trigger == "reset-btn":
            state["playing"] = False
            state["pos"] = [0] * len(patients)
            state["write_idx"] = [0] * len(patients)
            for pid in list(self.global_data.get("buffers", {}).keys()):
                buf = self.global_data["buffers"][pid]
                buf["signal_buffer"].fill(np.nan)
                buf["write_idx"] = 0
                buf["rr_buffer"].fill(np.nan)
                buf["rr_write_idx"] = 0
                buf["last_peak_global_index"] = -1
                buf["direction"] = 1
                buf["ping_position"] = 0.0

        if trigger == "interval" and state.get("playing", False):
            for pid, p in enumerate(patients):
                if not p or "ecg" not in p or p["ecg"] is None:
                    continue

                ecg = p["ecg"]
                if ecg.shape[0] == 0:
                    continue

                fs = float(p["header"].get("sampling_frequency", 250.0)) if p.get("header") else 250.0
                chunk_sec = (chunk_ms_val / 1000.0) * speed_val
                chunk_samples = max(1, int(round(chunk_sec * fs)))

                pos0 = state["pos"][pid]
                pos1 = min(len(ecg), pos0 + chunk_samples)

                if pos1 > pos0:
                    state["pos"][pid] = pos1

            try:
                if all(state["pos"][i] >= len(patients[i]["ecg"]) for i in range(len(patients))):
                    state["playing"] = False
            except Exception:
                pass

        return state

    def get_selected_patient_indices(self, selected_patients, patients):
        selected_idxs = []
        if selected_patients:
            try:
                if isinstance(selected_patients, list):
                    selected_idxs = [int(i) for i in selected_patients if 0 <= int(i) < len(patients)]
                else:
                    idx = int(selected_patients)
                    if 0 <= idx < len(patients):
                        selected_idxs = [idx]
            except:
                selected_idxs = []
        return selected_idxs if selected_idxs else [0]

    def initialize_state(self, patients):
        if self.app_state is None:
            self.app_state = {"playing": False, "pos": [0] * len(patients), "write_idx": [0] * len(patients)}
        if "pos" not in self.app_state or len(self.app_state["pos"]) != len(patients):
            self.app_state["pos"] = [0] * len(patients)
        if "write_idx" not in self.app_state or len(self.app_state["write_idx"]) != len(patients):
            self.app_state["write_idx"] = [0] * len(patients)
        if "playing" not in self.app_state:
            self.app_state["playing"] = False
        return self.app_state
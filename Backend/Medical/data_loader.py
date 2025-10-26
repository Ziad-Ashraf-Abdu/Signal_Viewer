# data_loader.py
import os
import re
import math
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from datetime import datetime

try:
    import pyedflib
    PYEDFLIB_AVAILABLE = True
except ImportError:
    PYEDFLIB_AVAILABLE = False

class DataLoader:
    def __init__(self):
        self.PYEDFLIB_AVAILABLE = PYEDFLIB_AVAILABLE
        self.MAX_EEG_SUBJECTS = 150
        self.DEFAULT_MAX_EEG_SECONDS = 60
        
    def parse_num(self, token, default=None):
        if token is None:
            return default
        token = str(token).strip()
        if token == "":
            return default
        try:
            return float(token)
        except:
            pass
        if '/' in token:
            parts = token.split('/')
            for p in parts:
                p = p.strip()
                m = re.search(r'[-+]?\d+(\.\d+)?', p)
                if m:
                    try:
                        return float(m.group(0))
                    except:
                        continue
        m = re.search(r'[-+]?\d+(\.\d+)?', token)
        if m:
            try:
                return float(m.group(0))
            except:
                return default
        return default

    def find_dataset_directory(self, dataset_type, root="."):
        if dataset_type == "ECG":
            candidates = [
                os.path.join(os.getcwd(), "data", "ptbdb"),
                os.path.join(os.getcwd(), "ptbdb"),
                os.path.join(os.getcwd(), "qtdb_data", "physionet.org", "files", "qtdb", "1.0.0"),
                os.path.join(os.getcwd(), "qtdb"),
                os.path.join(os.getcwd(), "qtdb_data"),
                os.path.join(os.getcwd(), "1.0.0"),
                os.path.join(os.getcwd(), "qtdb", "1.0.0"),
                os.path.join(os.getcwd(), "qtdb-1.0.0"),
            ]

            for d in candidates:
                if os.path.isdir(d):
                    patient_dirs = self.find_patient_directories(d)
                    if patient_dirs:
                        return d

                    for rootd, _, files in os.walk(d):
                        for f in files:
                            if f.lower().endswith('.hea'):
                                return d
        else:
            candidates = [
                os.path.join(os.getcwd(), "ASZED-153"),
                os.path.join(os.getcwd(), "ASZED_153"),
                os.path.join(os.getcwd(), "aszed-153"),
                os.path.join(os.getcwd(), "eeg_data"),
            ]

            for d in candidates:
                if os.path.isdir(d):
                    for rootd, _, files in os.walk(d):
                        for f in files:
                            if f.lower().endswith('.edf'):
                                return d

        for rootd, _, files in os.walk(root):
            for f in files:
                ext = '.hea' if dataset_type == "ECG" else '.edf'
                if f.lower().endswith(ext):
                    return rootd
        return None

    def find_patient_directories(self, data_dir):
        patient_dirs = []
        if not os.path.isdir(data_dir):
            return patient_dirs

        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                if (item.lower().startswith('patient') or
                        item.lower().startswith('subject') or
                        item.isdigit()):
                    patient_dirs.append(item_path)
        return sorted(patient_dirs)

    def read_header_file(self, path):
        if not os.path.exists(path):
            return None
        with open(path, "r", errors="ignore") as fh:
            lines = [ln.strip() for ln in fh.readlines() if ln.strip() != ""]
        if not lines:
            return None
        first = lines[0].split()
        record_name = first[0] if len(first) >= 1 else None
        num_signals = self.parse_num(first[1], default=2) if len(first) >= 2 else 2
        fs = self.parse_num(first[2], default=250.0) if len(first) >= 3 else 250.0
        num_samples = self.parse_num(first[3], default=225000) if len(first) >= 4 else 225000
        try:
            num_signals = int(num_signals)
        except:
            num_signals = int(max(1, math.floor(num_signals))) if num_signals else 2
        try:
            fs = float(fs)
        except:
            fs = 250.0
        try:
            num_samples = int(num_samples)
        except:
            num_samples = int(225000)
        signals_raw = []
        if len(lines) > 1:
            for ln in lines[1:]:
                signals_raw.append(ln.split())
        return {
            "record_name": record_name,
            "num_signals": num_signals,
            "sampling_frequency": fs,
            "num_samples": num_samples,
            "signals_raw": signals_raw
        }

    def read_dat_file(self, dat_path, header_info, max_samples=None):
        if not os.path.exists(dat_path) or header_info is None:
            return None
        try:
            raw = np.fromfile(dat_path, dtype=np.int16)
            n_signals = max(1, int(header_info.get("num_signals", 2)))
            total_samples = raw.shape[0] // n_signals
            if total_samples <= 0:
                return None
            raw = raw[: total_samples * n_signals]
            mat = raw.reshape((total_samples, n_signals))
            gains = np.ones(n_signals) * 200.0
            for i in range(min(n_signals, len(header_info.get("signals_raw", [])))):
                parts = header_info["signals_raw"][i]
                if len(parts) >= 3:
                    g = self.parse_num(parts[2], default=None)
                    if g and g > 0:
                        gains[i] = g
            cols = [f"signal_{i + 1}" for i in range(n_signals)]
            df = pd.DataFrame(mat[:, :n_signals].astype(float) / gains[:n_signals], columns=cols)
            fs = header_info.get("sampling_frequency", 250.0)
            df.insert(0, "time", np.arange(df.shape[0]) / float(fs))
            if max_samples is not None:
                df = df.iloc[: int(max_samples)].reset_index(drop=True)
            return df
        except Exception as e:
            print(f"[read_dat_file] error reading {dat_path}: {e}")
            return None

    def read_edf_file(self, edf_path, max_samples=None, attempts=8):
        if not self.PYEDFLIB_AVAILABLE:
            print("pyedflib not installed. Please: pip install pyedflib")
            return None, None

        last_exc = None
        backoff = 0.05
        for attempt in range(attempts):
            try:
                f = pyedflib.EdfReader(edf_path)
                try:
                    try:
                        n_signals = int(f.signals_in_file)
                    except Exception:
                        n_signals = int(getattr(f, "signals_in_file", 0) or 0)
                    if n_signals <= 0:
                        raise ValueError("No signals in EDF")
                    nsamps = f.getNSamples()
                    if isinstance(nsamps, (list, tuple, np.ndarray)):
                        min_samples = int(min(nsamps))
                    else:
                        min_samples = int(nsamps)
                    use_samples = min_samples if max_samples is None else min(min_samples, int(max_samples))
                    fs = None
                    try:
                        fs = int(f.getSampleFrequency(0))
                    except Exception:
                        try:
                            dur = getattr(f, "getFileDuration", lambda: None)()
                            if dur:
                                fs = max(1, int(round(use_samples / float(dur))))
                            else:
                                fs = 250
                        except Exception:
                            fs = 250
                    data = np.zeros((use_samples, n_signals), dtype=float)
                    for ch in range(n_signals):
                        sig = f.readSignal(ch)
                        if sig is None:
                            sig = np.zeros(use_samples, dtype=float)
                        if len(sig) >= use_samples:
                            sig_use = np.asarray(sig[:use_samples], dtype=float)
                        else:
                            sig_use = np.empty(use_samples, dtype=float)
                            sig_use[:len(sig)] = sig
                            sig_use[len(sig):] = np.nan
                        data[:, ch] = sig_use
                    cols = [f"signal_{i + 1}" for i in range(n_signals)]
                    df = pd.DataFrame(data, columns=cols)
                    df.insert(0, "time", np.arange(use_samples) / float(fs))
                    header = {
                        "sampling_frequency": fs,
                        "num_signals": n_signals,
                        "record_name": os.path.basename(edf_path),
                        "num_samples": use_samples
                    }
                    return df, header
                finally:
                    try:
                        f.close()
                    except Exception:
                        try:
                            f._close()
                        except Exception:
                            pass
                    try:
                        del f
                    except Exception:
                        pass
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if ("already been opened" in msg) or ("file has already been opened" in msg) or (
                        "resource temporarily unavailable" in msg) or ("i/o error" in msg):
                    time.sleep(backoff)
                    backoff = min(0.5, backoff * 1.8)
                    continue
                time.sleep(backoff)
                backoff = min(0.5, backoff * 1.8)
                continue
        print(f"[read_edf_file] Error reading {edf_path}: {last_exc}")
        return None, None

    def find_all_data_files(self, root_dir, file_extension):
        grouped_files = {}
        for root, dirs, files in os.walk(root_dir):
            matching_files = [f for f in files if f.lower().endswith(file_extension)]
            if matching_files:
                parent_key = os.path.basename(root) or root
                full_paths = [os.path.join(root, f) for f in sorted(matching_files)]
                if parent_key not in grouped_files:
                    grouped_files[parent_key] = []
                grouped_files[parent_key].extend(full_paths)
        return grouped_files

    def concatenate_ecg_files(self, file_paths, max_samples=None):
        combined_df = None
        combined_header = None
        total_samples = 0

        for file_path in file_paths:
            if file_path.endswith('.hea'):
                hea_path = file_path
                dat_path = file_path.replace('.hea', '.dat')
            elif file_path.endswith('.dat'):
                dat_path = file_path
                hea_path = file_path.replace('.dat', '.hea')
            else:
                continue

            if not os.path.exists(hea_path) or not os.path.exists(dat_path):
                print(f"[concatenate_ecg_files] Missing pair for {file_path}")
                continue

            header = self.read_header_file(hea_path)
            if header is None:
                print(f"[concatenate_ecg_files] Failed to read header: {hea_path}")
                continue

            df = self.read_dat_file(dat_path, header, max_samples=max_samples)
            if df is None:
                print(f"[concatenate_ecg_files] Failed to read data: {dat_path}")
                continue

            if combined_df is None:
                combined_df = df.copy()
                combined_header = header.copy()
                combined_header['source_files'] = [os.path.basename(file_path)]
                total_samples = len(df)
            else:
                common_signals = [col for col in df.columns if col.startswith('signal_') and col in combined_df.columns]
                if not common_signals:
                    print(f"[concatenate_ecg_files] No matching signal columns in {file_path}")
                    continue

                last_time = combined_df['time'].iloc[-1]
                fs = header.get('sampling_frequency', 250.0)
                time_increment = 1.0 / fs
                df['time'] = df['time'] + last_time + time_increment

                cols_to_concat = ['time'] + common_signals
                combined_df = pd.concat([combined_df[cols_to_concat], df[cols_to_concat]],
                                        ignore_index=True, axis=0)
                combined_header['source_files'].append(os.path.basename(file_path))
                total_samples += len(df)

        if combined_df is not None:
            combined_header['num_samples'] = total_samples
            combined_header['concatenated_files'] = len(combined_header['source_files'])
            print(f"[concatenate_ecg_files] Combined {len(combined_header['source_files'])} files, "
                  f"total samples: {total_samples}")

        return combined_df, combined_header

    def concatenate_eeg_files(self, file_paths, max_samples=None):
        if not self.PYEDFLIB_AVAILABLE:
            print("[concatenate_eeg_files] pyedflib not available")
            return None, None

        combined_df = None
        combined_header = None
        total_samples = 0

        for file_path in file_paths:
            df, header = self.read_edf_file(file_path, max_samples=max_samples)
            if df is None:
                print(f"[concatenate_eeg_files] Failed to read: {file_path}")
                continue

            if combined_df is None:
                combined_df = df.copy()
                combined_header = header.copy()
                combined_header['source_files'] = [os.path.basename(file_path)]
                total_samples = len(df)
            else:
                common_signals = [col for col in df.columns if col.startswith('signal_') and col in combined_df.columns]
                if not common_signals:
                    print(f"[concatenate_eeg_files] No matching signal columns in {file_path}")
                    continue

                last_time = combined_df['time'].iloc[-1]
                fs = header.get('sampling_frequency', 250)
                time_increment = 1.0 / fs
                df['time'] = df['time'] + last_time + time_increment

                cols_to_concat = ['time'] + common_signals
                combined_df = pd.concat([combined_df[cols_to_concat], df[cols_to_concat]],
                                        ignore_index=True, axis=0)
                combined_header['source_files'].append(os.path.basename(file_path))
                total_samples += len(df)

        if combined_df is not None:
            combined_header['num_samples'] = total_samples
            combined_header['concatenated_files'] = len(combined_header['source_files'])
            print(f"[concatenate_eeg_files] Combined {len(combined_header['source_files'])} files, "
                  f"total samples: {total_samples}")

        return combined_df, combined_header

    def apply_signal_filtering(self, signal, fs, signal_type="ECG"):
        if signal is None or len(signal) < 3:
            return signal
        if signal_type == "ECG":
            low_cutoff, high_cutoff = 0.5, 40.0
        else:
            low_cutoff, high_cutoff = 0.5, 70.0
        nyq = fs / 2.0
        low = max(low_cutoff / nyq, 1e-6)
        high = min(high_cutoff / nyq, 0.9999)
        try:
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, signal)
            return filtered
        except Exception:
            return signal

    def load_patient_data(self, data_dir, dataset_type="ECG", max_samples=None, max_patients=None,
                          max_records_per_patient=5):
        patients = []

        if dataset_type == "ECG":
            file_ext = '.dat'
            concat_func = self.concatenate_ecg_files
        else:
            if not self.PYEDFLIB_AVAILABLE:
                print("pyedflib is required to read EDF files. Please install: pip install pyedflib")
                return []
            file_ext = '.edf'
            concat_func = self.concatenate_eeg_files

        grouped_files = self.find_all_data_files(data_dir, file_ext)

        if not grouped_files:
            print(f"[load_patient_data] No {dataset_type} files found in {data_dir}")
            return []

        print(f"[load_patient_data] Found {len(grouped_files)} groups of {dataset_type} files")

        for group_idx, (group_name, file_paths) in enumerate(sorted(grouped_files.items())):
            if max_patients is not None and group_idx >= max_patients:
                break

            print(f"[load_patient_data] Processing group '{group_name}' with {len(file_paths)} files...")

            if max_records_per_patient:
                file_paths = file_paths[:max_records_per_patient]

            combined_df, combined_header = concat_func(file_paths, max_samples=max_samples)

            if combined_df is None:
                print(f"[load_patient_data] Failed to load group '{group_name}'")
                continue

            fs = combined_header.get("sampling_frequency", 250)
            signal_cols = [c for c in combined_df.columns if c.startswith("signal_")]

            for col in signal_cols:
                combined_df[col] = self.apply_signal_filtering(combined_df[col].values, fs, dataset_type)

            patient_record = {
                "name": group_name,
                "header": combined_header,
                "ecg": combined_df,
                "type": dataset_type,
                "source_directory": os.path.dirname(file_paths[0]),
                "files_concatenated": len(file_paths),
                "total_samples": combined_header.get('num_samples', len(combined_df)),
                "total_channels": len(signal_cols)
            }

            patients.append(patient_record)
            print(f"[load_patient_data] Loaded '{group_name}': {len(file_paths)} files, "
                  f"{len(combined_df)} samples, {len(signal_cols)} channels")

        print(f"[load_patient_data] Successfully loaded {len(patients)} patient groups total")
        return patients
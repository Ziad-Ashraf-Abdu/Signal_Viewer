# aliasing_analyzer.py
import numpy as np
import time
from scipy.signal import resample

class AliasingAnalyzer:
    """
    Standalone class dedicated to aliasing detection and anti-aliasing analysis
    """
    
    def __init__(self):
        self.last_analysis = {}
        
    def calculate_fft_with_aliasing_analysis(self, signal_segment, original_fs, resampling_freq):
        """
        Perform FFT analysis with comprehensive aliasing detection
        """
        start_time = time.time()

        y = signal_segment
        y = y[~np.isnan(y)]

        if len(y) < 2:
            return None, None, original_fs, 0, 0, {"risk_level": "UNKNOWN", "description": "Insufficient data"}

        # 1. Find f_max from the original, un-sampled segment
        N_orig = len(y)
        yf_orig = np.fft.fft(y)
        xf_orig = np.fft.fftfreq(N_orig, 1 / original_fs)

        # Enhanced f_max calculation for aliasing detection
        magnitudes = np.abs(yf_orig[1:N_orig // 2])
        frequencies = xf_orig[1:N_orig // 2]

        if len(magnitudes) > 0:
            # Use multiple thresholds for better aliasing detection
            noise_threshold = np.max(magnitudes) * 0.01
            significant_indices = magnitudes > noise_threshold
            
            if np.any(significant_indices):
                significant_freqs = frequencies[significant_indices]
                f_max = np.max(significant_freqs)
                
                # Additional aliasing metrics
                energy_above_nyquist = self._calculate_energy_above_nyquist(
                    yf_orig, frequencies, original_fs/2
                )
            else:
                f_max = 0
                energy_above_nyquist = 0
        else:
            f_max = 0
            energy_above_nyquist = 0

        # 2. Perform resampling for the plot with aliasing checks
        resampling_freq_val = float(resampling_freq or 500.0)
        
        # Check for potential aliasing
        aliasing_risk = self._check_aliasing_risk(f_max, resampling_freq_val, energy_above_nyquist)
        
        if resampling_freq_val >= original_fs:
            step = 1
        else:
            step = int(round(original_fs / resampling_freq_val))
        step = max(1, step)

        y_sampled = y[::step]
        fs_new = original_fs / step

        if len(y_sampled) < 2:
            return None, None, fs_new, 0, f_max, aliasing_risk

        # 3. Calculate FFT on the (potentially down-sampled) signal
        N = len(y_sampled)
        yf = np.fft.fft(y_sampled)
        xf = np.fft.fftfreq(N, 1 / fs_new)

        xf_plot = xf[:N // 2]
        yf_plot = np.abs(yf[0:N // 2])

        computation_time = time.time() - start_time

        return xf_plot, yf_plot, fs_new, computation_time, f_max, aliasing_risk

    def _calculate_energy_above_nyquist(self, yf_orig, frequencies, nyquist_freq):
        """Calculate the energy content above Nyquist frequency"""
        above_nyquist = frequencies > nyquist_freq
        if np.any(above_nyquist):
            energy = np.sum(np.abs(yf_orig[1:len(frequencies)+1][above_nyquist])**2)
            total_energy = np.sum(np.abs(yf_orig[1:len(frequencies)+1])**2)
            return energy / total_energy if total_energy > 0 else 0
        return 0

    def _check_aliasing_risk(self, f_max, sampling_freq, energy_above_nyquist=0):
        """
        Comprehensive aliasing risk assessment
        Returns: dict with risk level and details
        """
        if f_max <= 0:
            return {
                'risk_level': "UNKNOWN",
                'description': "Cannot determine signal bandwidth",
                'nyquist_rate': 0,
                'current_sampling_rate': sampling_freq,
                'max_signal_frequency': f_max
            }
            
        nyquist_rate = 2 * f_max
        safety_ratio = sampling_freq / nyquist_rate if nyquist_rate > 0 else 0
        
        if sampling_freq >= nyquist_rate * 1.2:  # 20% safety margin
            risk_level = "VERY LOW"
            description = "Excellent sampling - sufficient margin"
            color = "#10B981"
        elif sampling_freq >= nyquist_rate:
            risk_level = "LOW" 
            description = "Adequate sampling - meets Nyquist criterion"
            color = "#3B82F6"
        elif sampling_freq >= nyquist_rate * 0.8:
            risk_level = "MEDIUM"
            description = "Moderate aliasing risk - consider higher sampling rate"
            color = "#F59E0B"
        elif sampling_freq >= nyquist_rate * 0.5:
            risk_level = "HIGH"
            description = "High aliasing risk - significant signal distortion likely"
            color = "#EF4444"
        else:
            risk_level = "SEVERE"
            description = "Severe aliasing risk - signal may be unrecoverable"
            color = "#7F1D1D"
            
        # Adjust risk based on energy above Nyquist
        if energy_above_nyquist > 0.1:  # More than 10% energy above Nyquist
            if risk_level in ["LOW", "VERY LOW"]:
                risk_level = "MEDIUM"
                description += " | High-frequency content detected"
            elif risk_level == "MEDIUM":
                risk_level = "HIGH" 
                description += " | Significant high-frequency energy"
                
        return {
            'risk_level': risk_level,
            'color': color,
            'nyquist_rate': nyquist_rate,
            'current_sampling_rate': sampling_freq,
            'max_signal_frequency': f_max,
            'safety_ratio': safety_ratio,
            'energy_above_nyquist': energy_above_nyquist,
            'description': description
        }

    def analyze_sampling_period(self, signal_data, original_fs, sampling_period_ms):
        """
        Analyze time-domain sampling for aliasing issues
        """
        if sampling_period_ms is None or original_fs == 0:
            return None
            
        try:
            Ts_user = float(sampling_period_ms) / 1000.0
        except (ValueError, TypeError):
            return None

        if Ts_user <= 0:
            return None
            
        # Calculate effective sampling frequency
        effective_fs = 1.0 / Ts_user

        # Analyze signal to find maximum frequency component
        f_max = 0
        if len(signal_data) > 1:
            y = signal_data[~np.isnan(signal_data)]
            if len(y) >= 2:
                yf = np.fft.fft(y)
                xf = np.fft.fftfreq(len(y), 1 / original_fs)
                
                magnitudes = np.abs(yf[1:len(y)//2])
                frequencies = xf[1:len(y)//2]
                
                if len(magnitudes) > 0:
                    noise_threshold = np.max(magnitudes) * 0.01
                    significant_freqs = frequencies[magnitudes > noise_threshold]
                    f_max = np.max(significant_freqs) if len(significant_freqs) > 0 else 0

        # Calculate required sampling parameters
        Ts_max = 1.0 / (2 * f_max) if f_max > 0 else float('inf')
        nyquist_rate = 2 * f_max
        
        # Determine aliasing risk
        is_aliasing_risk = Ts_user > Ts_max if f_max > 0 else False
        
        # Risk assessment
        if f_max <= 0:
            risk_level = "UNKNOWN"
            color = "#6B7280"
        elif Ts_user <= Ts_max * 0.8:  # 20% safety margin
            risk_level = "VERY LOW"
            color = "#10B981"
        elif Ts_user <= Ts_max:
            risk_level = "LOW"
            color = "#3B82F6"
        elif Ts_user <= Ts_max * 1.2:
            risk_level = "MEDIUM"
            color = "#F59E0B"
        elif Ts_user <= Ts_max * 1.5:
            risk_level = "HIGH"
            color = "#EF4444"
        else:
            risk_level = "SEVERE"
            color = "#7F1D1D"

        return {
            'original_sampling_frequency': original_fs,
            'user_sampling_period_ms': sampling_period_ms,
            'effective_sampling_frequency': effective_fs,
            'max_signal_frequency': f_max,
            'nyquist_rate': nyquist_rate,
            'required_sampling_period_ms': Ts_max * 1000,
            'current_period_ms': Ts_user * 1000,
            'max_period_ms': Ts_max * 1000,
            'is_aliasing_risk': is_aliasing_risk,
            'risk_level': risk_level,
            'risk_color': color,
            'safety_margin': (Ts_max - Ts_user) / Ts_max if Ts_max > 0 else 0
        }

    def recommend_anti_aliasing_filter(self, f_max, current_fs):
        """
        Recommend anti-aliasing filter parameters
        """
        if f_max <= 0 or current_fs <= 0:
            return None
            
        nyquist = current_fs / 2
        # Conservative cutoff: 80% of Nyquist or 90% of signal max frequency
        cutoff_freq = min(f_max * 0.9, nyquist * 0.8)
        
        # Determine filter order based on required attenuation
        required_attenuation = 40
        if current_fs < 2.5 * f_max:  # Close to Nyquist
            required_attenuation = 60
            filter_order = 6
        else:
            filter_order = 4
            
        return {
            'filter_type': 'lowpass',
            'recommended_cutoff_freq': cutoff_freq,
            'filter_order': filter_order,
            'max_signal_freq': f_max,
            'nyquist_frequency': nyquist,
            'stopband_start': f_max,
            'attenuation_requirement': f'>{required_attenuation}dB above Nyquist',
            'filter_characteristics': 'Butterworth (maximally flat) recommended',
            'implementation_note': 'Use scipy.signal.butter for implementation'
        }

    def generate_aliasing_report(self, signal_data, original_fs, resampling_freq=None, sampling_period_ms=None):
        """
        Generate comprehensive aliasing analysis report
        """
        report = {
            'timestamp': time.time(),
            'original_sampling_frequency': original_fs,
            'signal_length': len(signal_data),
            'analyses': {}
        }
        
        # Frequency domain analysis
        if resampling_freq is not None:
            _, _, _, _, f_max, aliasing_risk = self.calculate_fft_with_aliasing_analysis(
                signal_data, original_fs, resampling_freq
            )
            report['analyses']['frequency_domain'] = aliasing_risk
            report['max_detected_frequency'] = f_max
            
        # Time domain analysis  
        if sampling_period_ms is not None:
            time_analysis = self.analyze_sampling_period(signal_data, original_fs, sampling_period_ms)
            report['analyses']['time_domain'] = time_analysis
            
        # Filter recommendations
        if 'max_detected_frequency' in report and report['max_detected_frequency'] > 0:
            report['filter_recommendations'] = self.recommend_anti_aliasing_filter(
                report['max_detected_frequency'], original_fs
            )
            
        # Overall risk assessment
        risks = []
        if 'frequency_domain' in report['analyses']:
            risks.append(report['analyses']['frequency_domain']['risk_level'])
        if 'time_domain' in report['analyses']:
            risks.append(report['analyses']['time_domain']['risk_level'])
            
        if risks:
            if any(r in ["SEVERE", "HIGH"] for r in risks):
                report['overall_risk'] = "HIGH"
            elif any(r == "MEDIUM" for r in risks):
                report['overall_risk'] = "MEDIUM"
            elif all(r in ["LOW", "VERY LOW"] for r in risks):
                report['overall_risk'] = "LOW"
            else:
                report['overall_risk'] = "UNKNOWN"
                
        return report

    def get_visualization_components(self, analysis_result):
        """
        Generate visualization components for the aliasing analysis
        """
        if not analysis_result:
            return None
            
        components = {}
        
        # Risk badge
        if 'risk_level' in analysis_result:
            risk_level = analysis_result['risk_level']
            color = analysis_result.get('color', '#6B7280')
            
            components['risk_badge'] = {
                'text': risk_level,
                'color': color,
                'description': analysis_result.get('description', '')
            }
            
        # Key metrics
        metrics = []
        if 'max_signal_frequency' in analysis_result:
            metrics.append({
                'label': 'Max Signal Frequency',
                'value': f"{analysis_result['max_signal_frequency']:.1f} Hz",
                'icon': '📊'
            })
            
        if 'nyquist_rate' in analysis_result:
            metrics.append({
                'label': 'Nyquist Rate',
                'value': f"{analysis_result['nyquist_rate']:.1f} Hz", 
                'icon': '🎯'
            })
            
        if 'current_sampling_rate' in analysis_result:
            metrics.append({
                'label': 'Current Rate',
                'value': f"{analysis_result['current_sampling_rate']:.1f} Hz",
                'icon': '⚡'
            })
            
        if 'safety_ratio' in analysis_result:
            ratio = analysis_result['safety_ratio']
            if ratio > 1:
                status = f"+{(ratio-1)*100:.0f}%"
                icon = '✅'
            else:
                status = f"{(ratio-1)*100:.0f}%"
                icon = '⚠️'
            metrics.append({
                'label': 'Safety Margin',
                'value': status,
                'icon': icon
            })
            
        components['metrics'] = metrics
        
        return components
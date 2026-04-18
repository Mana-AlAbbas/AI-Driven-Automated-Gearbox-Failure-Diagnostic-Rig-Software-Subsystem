"""Feature extraction for gearbox vibration windows."""

from typing import Dict, Optional, Tuple

import numpy as np
try:
    import pywt
except ImportError:
    pywt = None
from scipy import integrate as sp_integrate
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew


def _trapz(y, x) -> float:
    """Trapezoidal integration with broad NumPy and SciPy compatibility."""
    fn = getattr(np, "trapz", None)
    if fn is not None:
        return float(fn(y, x))

    fn = getattr(np, "trapezoid", None)
    if fn is not None:
        return float(fn(y, x))

    fn = getattr(sp_integrate, "trapezoid", None)
    if fn is not None:
        return float(fn(y, x))

    y = np.asarray(y)
    x = np.asarray(x)
    if y.size < 2:
        return 0.0
    return float(np.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1]) * 0.5))


class GearboxFeatureExtractor:
    def __init__(
        self,
        fs: int = 25600,
        enable_gear_specific: bool = True,
        estimate_gmf: bool = True,
        compute_sample_entropy: bool = False,
    ):
        self.fs = fs
        self.enable_gear_specific = enable_gear_specific
        self.estimate_gmf = estimate_gmf
        self.compute_sample_entropy = compute_sample_entropy

    def time_domain_features(self, x: np.ndarray) -> Dict[str, float]:
        features: Dict[str, float] = {}
        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(np.max(np.abs(x)))
        abs_mean = float(np.mean(np.abs(x)))

        features["rms"] = rms
        features["peak"] = peak
        features["crest_factor"] = peak / rms if rms > 0 else 0.0
        features["impulse_factor"] = peak / abs_mean if abs_mean > 0 else 0.0
        features["shape_factor"] = rms / abs_mean if abs_mean > 0 else 0.0

        denom = float(np.mean(np.sqrt(np.abs(x)))) ** 2
        features["clearance_factor"] = peak / denom if denom > 0 else 0.0

        features["skewness"] = float(skew(x))
        features["kurtosis"] = float(kurtosis(x, fisher=False))
        features["p2p"] = float(np.ptp(x))
        features["p2p_rms_ratio"] = features["p2p"] / rms if rms > 0 else 0.0
        features["sample_entropy"] = float(self._sample_entropy(x)) if self.compute_sample_entropy else 0.0
        return features

    def _sample_entropy(self, x, m: int = 2, r_factor: float = 0.2) -> float:
        x = np.asarray(x)
        r = r_factor * np.std(x)
        n = len(x)

        def _max_dist(x_i, x_j):
            return np.max(np.abs(x_i - x_j))

        def _count(embed_dim, tolerance):
            count = 0
            for i in range(n - embed_dim):
                for j in range(i + 1, n - embed_dim):
                    if _max_dist(x[i:i + embed_dim], x[j:j + embed_dim]) < tolerance:
                        count += 1
            return count

        if n <= m + 1:
            return 0.0
        a_count = _count(m, r)
        b_count = _count(m + 1, r)
        if b_count == 0 or a_count == 0:
            return 0.0
        return float(-np.log(a_count / b_count))

    def _welch_psd(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(x)
        nperseg = min(2048, n)
        if nperseg < 256:
            nperseg = n
        noverlap = nperseg // 2
        freq, pxx = sp_signal.welch(
            x,
            fs=self.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
        )
        return freq, pxx

    def frequency_domain_features(self, x: np.ndarray) -> Dict[str, float]:
        features: Dict[str, float] = {}
        freq, psd = self._welch_psd(x)

        if len(freq) > 1:
            freq = freq[1:]
            psd = psd[1:]

        if len(freq) == 0 or np.all(psd <= 0):
            for key in [
                "spectral_centroid",
                "spectral_spread",
                "spectral_skewness",
                "spectral_kurtosis",
                "energy_low",
                "energy_mid",
                "energy_high",
                "energy_ratio_low",
                "energy_ratio_mid",
                "energy_ratio_high",
                "peak_freq",
                "peak_magnitude_norm",
            ]:
                features[key] = 0.0
            return features

        total_energy = _trapz(psd, freq) + 1e-12
        centroid = _trapz(freq * psd, freq) / total_energy
        spread = np.sqrt(_trapz(((freq - centroid) ** 2) * psd, freq) / total_energy)

        features["spectral_centroid"] = float(centroid)
        features["spectral_spread"] = float(spread)

        skew_num = _trapz(((freq - centroid) ** 3) * psd, freq)
        features["spectral_skewness"] = float(skew_num / (total_energy * (spread ** 3) + 1e-12))

        kurt_num = _trapz(((freq - centroid) ** 4) * psd, freq)
        features["spectral_kurtosis"] = float(kurt_num / (total_energy * (spread ** 4) + 1e-12))

        nyquist = self.fs / 2
        low_mask = freq <= 0.2 * nyquist
        mid_mask = (freq > 0.2 * nyquist) & (freq <= 0.5 * nyquist)
        high_mask = freq > 0.5 * nyquist

        e_low = _trapz(psd[low_mask], freq[low_mask]) if np.any(low_mask) else 0.0
        e_mid = _trapz(psd[mid_mask], freq[mid_mask]) if np.any(mid_mask) else 0.0
        e_high = _trapz(psd[high_mask], freq[high_mask]) if np.any(high_mask) else 0.0

        features["energy_ratio_low"] = e_low / total_energy
        features["energy_ratio_mid"] = e_mid / total_energy
        features["energy_ratio_high"] = e_high / total_energy
        features["energy_low"] = e_low
        features["energy_mid"] = e_mid
        features["energy_high"] = e_high

        peak_idx = int(np.argmax(psd))
        features["peak_freq"] = float(freq[peak_idx])
        features["peak_magnitude_norm"] = float(psd[peak_idx] / total_energy)
        return features

    def cepstrum_features(self, x: np.ndarray) -> Dict[str, float]:
        features: Dict[str, float] = {}
        try:
            spectrum = np.abs(np.fft.rfft(x))
            log_spec = np.log(spectrum + 1e-10)
            cep = np.fft.irfft(log_spec)

            if len(cep) > 10:
                cep_trim = cep[5:]
                peak_idx = int(np.argmax(cep_trim)) + 5
                features["cepstrum_peak_amp"] = float(cep[peak_idx])
                features["cepstrum_peak_ratio"] = float(cep[peak_idx] / (cep[0] + 1e-10))
                features["cepstrum_rms"] = float(np.sqrt(np.mean(cep ** 2)))
                features["cepstrum_kurtosis"] = float(kurtosis(cep, fisher=False))
            else:
                features.update({
                    "cepstrum_peak_amp": 0.0,
                    "cepstrum_peak_ratio": 0.0,
                    "cepstrum_rms": 0.0,
                    "cepstrum_kurtosis": 0.0,
                })
        except Exception:
            features.update({
                "cepstrum_peak_amp": 0.0,
                "cepstrum_peak_ratio": 0.0,
                "cepstrum_rms": 0.0,
                "cepstrum_kurtosis": 0.0,
            })
        return features

    def envelope_spectrum_features(self, x: np.ndarray) -> Dict[str, float]:
        features: Dict[str, float] = {}
        try:
            analytic = sp_signal.hilbert(x)
            envelope = np.abs(analytic)
            f_env, psd_env = self._welch_psd(envelope)
            if len(f_env) > 1:
                f_env = f_env[1:]
                psd_env = psd_env[1:]

            total_energy = _trapz(psd_env, f_env) + 1e-12
            low_mask = f_env <= 200
            if np.any(low_mask):
                peak_idx = int(np.argmax(psd_env[low_mask]))
                peak_freq = f_env[low_mask][peak_idx]
                peak_energy = psd_env[low_mask][peak_idx]
                features["env_peak_freq"] = float(peak_freq)
                features["env_peak_norm_energy"] = float(peak_energy / total_energy)
            else:
                features["env_peak_freq"] = 0.0
                features["env_peak_norm_energy"] = 0.0

            if features["env_peak_freq"] > 0:
                harmonic_energy = 0.0
                for multiplier in [2, 3]:
                    mask = (f_env >= multiplier * peak_freq * 0.95) & (f_env <= multiplier * peak_freq * 1.05)
                    if np.any(mask):
                        harmonic_energy += _trapz(psd_env[mask], f_env[mask])
                features["env_harmonic_ratio"] = float(harmonic_energy / (total_energy + 1e-12))
            else:
                features["env_harmonic_ratio"] = 0.0

            features["env_entropy"] = float(
                -np.sum((psd_env / total_energy) * np.log(psd_env / total_energy + 1e-12))
            )
        except Exception:
            features.update({
                "env_peak_freq": 0.0,
                "env_peak_norm_energy": 0.0,
                "env_harmonic_ratio": 0.0,
                "env_entropy": 0.0,
            })
        return features

    def _estimate_gmf(self, psd: np.ndarray, freq: np.ndarray) -> float:
        mask = (freq >= 300) & (freq <= 2000)
        if not np.any(mask):
            return 0.0
        idx = int(np.argmax(psd[mask]))
        return float(freq[mask][idx])

    def sideband_features(self, x: np.ndarray, shaft_speed_hz: Optional[float] = None) -> Dict[str, float]:
        features: Dict[str, float] = {}
        if not self.enable_gear_specific:
            return features

        freq, psd = self._welch_psd(x)
        if len(freq) > 1:
            freq = freq[1:]
            psd = psd[1:]

        total_energy = _trapz(psd, freq) + 1e-12

        if shaft_speed_hz is not None and shaft_speed_hz > 0:
            gmf = shaft_speed_hz * 18.0
        elif self.estimate_gmf:
            gmf = self._estimate_gmf(psd, freq)
        else:
            return features

        if gmf <= 0:
            return features

        def band_energy(center, tol: float = 0.05) -> float:
            mask = (freq >= center * (1 - tol)) & (freq <= center * (1 + tol))
            return _trapz(psd[mask], freq[mask]) if np.any(mask) else 0.0

        e_gmf = band_energy(gmf)
        features["sideband_energy_ratio"] = 0.0
        if e_gmf > 0:
            side_energy = 0.0
            spacing = shaft_speed_hz if shaft_speed_hz else gmf * 0.01
            for multiplier in range(1, 4):
                side_energy += band_energy(gmf - multiplier * spacing)
                side_energy += band_energy(gmf + multiplier * spacing)
            features["sideband_energy_ratio"] = float(side_energy / (e_gmf + 1e-10))

        features["gmf_energy_norm"] = e_gmf / total_energy
        features["estimated_gmf"] = float(gmf)
        return features

    def wavelet_features(self, x: np.ndarray, wavelet: str = "db4", level: int = 5) -> Dict[str, float]:
        features: Dict[str, float] = {}
        try:
            if pywt is None:
                raise ImportError("PyWavelets is not installed.")
            max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
            use_level = min(level, max_level) if max_level > 0 else 1
            coeffs = pywt.wavedec(x, wavelet, level=use_level)
            total_energy = float(sum(np.sum(c ** 2) for c in coeffs)) + 1e-12

            for index, coeff in enumerate(coeffs):
                energy = float(np.sum(coeff ** 2))
                features[f"wavelet_norm_energy_lvl{index}"] = energy / total_energy
                features[f"wavelet_std_lvl{index}"] = float(np.std(coeff))

            entropy = 0.0
            for coeff in coeffs:
                energy = float(np.sum(coeff ** 2))
                probability = energy / total_energy
                entropy -= probability * np.log(probability + 1e-10)
            features["wavelet_entropy"] = float(entropy)
        except Exception:
            for index in range(level + 1):
                features[f"wavelet_norm_energy_lvl{index}"] = 0.0
                features[f"wavelet_std_lvl{index}"] = 0.0
            features["wavelet_entropy"] = 0.0
        return features

    def extract_all_features(self, x: np.ndarray, metadata: Optional[Dict] = None) -> Dict[str, float]:
        features: Dict[str, float] = {}
        features.update(self.time_domain_features(x))
        features.update(self.frequency_domain_features(x))
        features.update(self.cepstrum_features(x))
        features.update(self.envelope_spectrum_features(x))
        features.update(self.wavelet_features(x))

        if self.enable_gear_specific:
            speed = None
            if metadata and not metadata.get("is_time_varying", False):
                speed = metadata.get("speed")
                if not isinstance(speed, (int, float)):
                    speed = None
                elif np.isnan(speed):
                    speed = None
                else:
                    speed = float(speed)
            features.update(self.sideband_features(x, shaft_speed_hz=speed))

        return features

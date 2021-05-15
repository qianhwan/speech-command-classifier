'''
Define audio related functions

copied from https://github.com/TensorSpeech/TensorFlowASR/blob/main/tensorflow_asr/featurizers/speech_featurizers.py  # noqa
'''
import numpy as np
import librosa


def load_audio(file_name: str, sample_rate: int = 16000):
    audio, _ = librosa.load(file_name, sr=sample_rate, mono=True)
    return audio


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """ Normailize signal to [-1, 1] range """
    gain = 1.0 / (np.max(np.abs(signal)) + 1e-9)
    return signal * gain


def normalize_audio_feature(
        audio_feature: np.ndarray, per_frame=False) -> np.ndarray:
    """ Mean and variance normalization """
    axis = 1 if per_frame else None
    mean = np.mean(audio_feature, axis=axis)
    std_dev = np.sqrt(np.var(audio_feature, axis=axis) + 1e-9)
    normalized = (audio_feature - mean) / std_dev
    return normalized


def stft(
        signal: np.ndarray,
        nfft: int = 2048,
        frame_step: int = 512,
        center: bool = True):
    return np.square(
            np.abs(librosa.core.stft(
                signal, n_fft=nfft, hop_length=frame_step,
                center=center,
                window="hann")))


def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    return librosa.power_to_db(S, ref=ref, amin=amin, top_db=top_db)


def compute_log_mel_spectrogram(
        signal: np.ndarray,
        sample_rate: int = 16000,
        nfft: int = 2048,
        num_feature_bins: int = 32) -> np.ndarray:
    S = stft(signal)

    mel = librosa.filters.mel(sample_rate, nfft,
                              n_mels=num_feature_bins,
                              fmin=0.0,
                              fmax=int(sample_rate / 2))

    mel_spectrogram = np.dot(S.T, mel.T)

    return power_to_db(mel_spectrogram)


def fix_audio_length(signal: np.ndarray, n_samples: int = 16000):
    if n_samples <= len(signal):
        return signal[:n_samples]
    else:
        return np.pad(signal, (0, n_samples - len(signal)), 'constant')

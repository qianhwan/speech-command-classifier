'''
Audio augmentation
Copied from https://github.com/tugstugi/pytorch-speech-commands/blob/36f3cd9e0d2b2d50cf4b023c9af1544fdfeeb886/transforms/transforms_wav.py  # noqa
'''
import random
import numpy as np
import librosa


def should_apply_transform(prob=0.5):
    """Transforms are only randomly applied with the given probability."""
    return random.random() < prob


class ChangeAmplitude(object):
    """Changes amplitude of an audio randomly."""

    def __init__(self, amplitude_range=(0.7, 1.1)):
        self.amplitude_range = amplitude_range

    def __call__(self, signal):
        if not should_apply_transform():
            return signal

        signal = signal * random.uniform(*self.amplitude_range)
        return signal


class ChangeSpeedAndPitchAudio(object):
    """
    Change the speed of an audio.
    This transform also changes the pitch of the audio.
    """

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, signal):
        if not should_apply_transform():
            return signal

        scale = random.uniform(-self.max_scale, self.max_scale)
        speed_fac = 1.0 / (1 + scale)
        signal = np.interp(np.arange(0, len(signal), speed_fac),
                           np.arange(0, len(signal)),
                           signal).astype(np.float32)
        return signal


class StretchAudio(object):
    """Stretches an audio randomly."""

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, signal):
        if not should_apply_transform():
            return signal

        scale = random.uniform(-self.max_scale, self.max_scale)
        signal = librosa.effects.time_stretch(signal, 1+scale)
        return signal


class TimeshiftAudio(object):
    """Shifts an audio randomly."""

    def __init__(self, max_shift_seconds=0.2):
        self.max_shift_seconds = max_shift_seconds

    def __call__(self, signal, sr):
        if not should_apply_transform():
            return signal

        max_shift = (sr * self.max_shift_seconds)
        shift = random.randint(-max_shift, max_shift)
        a = -min(0, shift)
        b = max(0, shift)
        signal = np.pad(signal, (a, b), "constant")
        signal = signal[:len(signal) - a] if a else signal[b:]
        return signal

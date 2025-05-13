"""
This module abstracts the creation of a spectrogram by offering some
interface functions to create, normalize, resize, transform and plot a spectrogram.

Parameters:

- sr: sampling rate of the audio files.

- n_fft: number of time samples used to compute the discrete fourier transform.

- win_length: defines how long the window for Short Time Fourier Transform should be
in terms nr. of samples. Usually equal to n_fft.

- hop length: nr. of samples which define the distance between a window and the
following one.

- n_mels: nr. frequency bins

- fmax: maximum frequency to be considered for mel bins generation
"""

from librosa.feature import melspectrogram
from librosa import power_to_db
from librosa.display import specshow
import numpy as np
from cv2 import resize, INTER_NEAREST
import matplotlib.pyplot as plt


def get_db_mel_spectrogram(audio, n_fft, n_mels, sr):
    """
    Create a dB mel-spectrogram from an audio represented as a numpy array.
    """

    win_length = n_fft
    hop_length = win_length // 2
    fmax = 0.5 * sr

    # generate spectrogram
    mel_spec = melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        fmax=fmax,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )

    # convert to Power Spectrogram by taking modulus and squaring (already applied in melspectrogram)
    #mel_spec = np.abs(mel_spec) ** 2

    # convert power to dB
    db_mel_spec = power_to_db(mel_spec)

    return db_mel_spec


def square_spectrogram(sp, size):
    """
    Return a new spectrogram with shape (size, size), using INTER_NEAREST
    as interpolation method
    """
    return resize(sp, (size, size), interpolation=INTER_NEAREST)


def normalize_spectrogram(sp):
    """
    Given a dB-spectrogram, normalize the dB in the range [0,1]
    """
    min_val = np.min(sp)
    max_val = np.max(sp)
    if max_val == min_val:
        # Evita la divisione per zero restituendo uno spettrogramma uniforme
        return np.zeros_like(sp)
    return (sp - min_val) / (max_val - min_val)


def shift_spectrogram(sp, shift, noise_level=0.01):
    """
    Return a new spectrogram shifting the original one on the time axis by
    the required amount. Adds noise instead of zeros to simulate ambient noise.
    
    Parameters:
    - sp: Input spectrogram.
    - shift: Amount of shift on the time axis.
    - noise_level: Standard deviation of the Gaussian noise to add.
    """
    size = len(sp[1])
    new_sp = np.random.normal(0, noise_level, (size, size))  # Add Gaussian noise
    if shift > 0:
        new_sp[:, shift:] = sp[:, :-shift]
    elif shift < 0:
        new_sp[:, :shift] = sp[:, -shift:]
    else:
        new_sp = sp  # No shift
    return new_sp


def calculate_noise_level(signal, snr_db):
    """
    Calcola il livello di rumore per ottenere un SNR specifico.
    
    Parameters:
    - signal: Segnale originale.
    - snr_db: Rapporto segnale-rumore desiderato in decibel.
    
    Returns:
    - noise_level: Deviazione standard del rumore.
    """
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    return np.sqrt(noise_power)


def plot_spectrogram(sp, sr, hop_length):
    """
    Plot mel-spectrogram
    """
    specshow(sp, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel")
    plt.title("MEL scale dB-spectrogram")
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("time (s)")
    plt.ylabel("frequency (Hz)")
    plt.show()

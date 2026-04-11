import pyedflib
import numpy as np
import tqdm
import mne
from scipy.signal import welch,stft
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import euclidean

def extract_basic_features(signal):
    signal = (signal - np.mean(signal)) / np.std(signal)
    mean = np.mean(signal)
    std = np.std(signal)
    sample_entropy = np.log(np.std(np.diff(signal)))
    fuzzy_entropy = -np.log(euclidean(signal[:-1], signal[1:]) / len(signal))
    skewness = skew(signal)
    kurt = kurtosis(signal)
    return [mean, std, sample_entropy, fuzzy_entropy, skewness, kurt]

def extract_advanced_features(data, fs, window_length_sec=3):
    """
    Extract advanced features from EEG data using Short-Time Fourier Transform (STFT).

    :param data: EEG signal data.
    :param fs: Sampling frequency.
    :param window_length_sec: Window length in seconds for STFT.
    :return: Features extracted from STFT.
    """

    # Perform STFT
    f, t, Zxx = stft(data, fs, nperseg=window_length_sec*fs)

    # Extract features from STFT
    # Compute mean power for each frequency band
    power = np.mean(np.abs(Zxx)**2, axis=1)  # Mean power at each frequency

    return power

def preprocess_and_extract_features_mne_with_timestamps(file_name):
    """
    Preprocess EEG data using MNE and extract basic and advanced features.
    Prepends a timestamp to each feature array.
    """

    # Load data
    raw = mne.io.read_raw_edf(file_name, preload=True)

    # Apply bandpass filter
    raw.filter(1., 50., fir_design='firwin')

    # Select EEG channels
    raw.pick_types(meg=False, eeg=True, eog=False)

    # Define short window parameters
    window_length = 3  # Window length in seconds
    sfreq = raw.info['sfreq']  # Sampling frequency
    window_samples = int(window_length * sfreq)

    # Initialize list to store features with timestamps
    features_with_timestamps = []

    # Iterate over each window
    for start in range(0, len(raw.times), window_samples):
        end = start + window_samples
        if end > len(raw.times):
            break

        # Extract and preprocess data in this window
        window_data, times = raw[:, start:end]
        window_data = np.squeeze(window_data)

        # Get the start timestamp of this window
        timestamp = raw.times[start]

        # Extract basic and advanced features for each channel in this window
        for channel_data in window_data:
            basic_features = extract_basic_features(channel_data)
            advanced_features = extract_advanced_features(channel_data, sfreq)
            combined_features = np.concatenate([[timestamp], basic_features, advanced_features])
            features_with_timestamps.append(combined_features)

    return np.array(features_with_timestamps)

preprocess_and_extract_features_mne_with_timestamps("data/chb01/chb01_03.edf")
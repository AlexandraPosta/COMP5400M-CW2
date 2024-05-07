import math
import time

import numpy as np
from matplotlib import pyplot as patches
import pandas as pd

import pyroomacoustics as pra
from pyroomacoustics.utilities import normalize

from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder

from itertools import combinations

C = 343.0
NFFT = 256
ROOM_DIM = [5.0, 5.0]
SNR = 0.001

AUDIO_PATH = "../data/original/arctic_a0200.wav"


# CNN
def gcc_phat(x_1, x_2, FS=16000, interp=1):
    n = len(x_1) + len(x_2) - 1
    n += 1 if n % 2 else 0

    # Fourier transforms of the two signals
    X_1 = np.fft.rfft(x_1, n=n)
    X_2 = np.fft.rfft(x_2, n=n)

    # Normalize by the magnitude of FFT - because PHAT
    np.divide(X_1, np.abs(X_1), X_1, where=np.abs(X_1) != 0)
    np.divide(X_2, np.abs(X_2), X_2, where=np.abs(X_2) != 0)

    # GCC-PHAT = [X_1(f)X_2*(f)] / |X_1(f)X_2*(f)|
    # See Knapp and Carter (1976) for reference
    CC = X_1 * np.conj(X_2)
    cc = np.fft.irfft(CC, n=n * interp)

    # Maximum delay between a pair of microphones,
    # expressed in a number of samples.
    # 0.2 m is the distance between the micropones and
    # 340 m/s is assumed to be the speed of sound.
    max_len = math.ceil(0.2 / 340 * FS * interp)

    # Trim the cc vector to only include a
    # small number of samples around the origin
    cc = np.concatenate((cc[-max_len:], cc[: max_len + 1]))

    # Return the cross correlation
    return cc


def compute_gcc_matrix(observation, fs, interp=1):
    # Initialize a transformed observation, that will be populated with GCC vectors
    # of the observation
    transformed_observation = []

    # Compute GCC for every pair of microphones
    mic_1, mic_2 = [0, 1]
    x_1 = observation[:, mic_1]
    x_2 = observation[:, mic_2]

    gcc = gcc_phat(x_1, x_2, FS=fs, interp=interp)

    # Add the GCC vector to the GCC matrix
    transformed_observation.append(gcc)

    return transformed_observation


def create_observations(wav_signals, fs, samples=20, step=5, interp=1):
    # Lists of observations and labels that will be populated
    X = []

    # Loop through the signal frame and take subframes
    for i in range(0, len(wav_signals) - samples + 1, step):
        # Extract the observation from subframe
        observation = np.array(wav_signals[i : i + samples])

        # Transform observation into a GCC matrix
        X.append(compute_gcc_matrix(observation, fs, interp=interp))

    cols = [
        f"mics{mic_1+1}{mic_2+1}_{i}"
        for mic_1, mic_2 in combinations(range(2), r=2)
        for i in range(np.shape(X)[2])
    ]

    df = pd.DataFrame(data=np.reshape(X, (len(X), -1)), columns=cols)

    return df


def encode():
    x = list(range(0, 180 + 1, 10))
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit([[label] for label in x])
    return encoder


def get_CNN_prediction(X, model, encoder):
    t0_cnn = time.time()
    predictions = model.predict(X)
    y_pred = encoder.inverse_transform(predictions)

    t1_cnn = time.time()
    print(f"CNN prediction time: {t1_cnn - t0_cnn:.4f}s")

    predictions = []
    for i in y_pred:
        predictions.append(int(i))

    pred_degree = max(set(predictions), key=predictions.count)
    pred_rad = math.pi - pred_degree * math.pi / 180

    return pred_rad


def get_doa_MUSIC(room, microphones, fs):
    X = pra.transform.stft.analysis(room.mic_array.signals.T, NFFT, NFFT // 2)
    X = X.transpose([2, 1, 0])

    # Construct the new DOA object and perform localization on the frames in X
    t0_m = time.time()
    doa = pra.doa.MUSIC(microphones, fs, NFFT, c=C, num_src=1)
    doa.locate_sources(X)
    pred = doa.azimuth_recon

    if pred > np.pi:
        pred = 2 * np.pi - pred

    t1_m = time.time()
    print(f"MUSIC time taken: {t1_m - t0_m:.4f}s")

    return pred


def create_room(signal, fs, source_loc, centre_mic):
    room = pra.ShoeBox(ROOM_DIM, fs=fs, max_order=0)
    room.add_source(source_loc, signal=signal)

    microphones = np.c_[
        [centre_mic[0] - 0.1, centre_mic[1]], [centre_mic[0] + 0.1, centre_mic[1]]
    ]
    room.add_microphone_array(microphones)

    snr = 10 * math.log10(2 / SNR)
    room.simulate()

    return room, microphones

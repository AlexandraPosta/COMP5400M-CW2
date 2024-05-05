'''
	COMP5400M - CW2
    Author Name: A. Posta
    Description: Direction of Arrival (DOA) models using 
    Convolutional Neural Network (CNN) and MUSIC algorithm
'''
from itertools import combinations
import math
import numpy as np
import pandas as pd
from pyroomacoustics.utilities import normalize
import pyroomacoustics as pra
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder

class Doa():
    """ Base class to simulate the room and predict the direction of arrival of the sound source 
    
    Direction of arrival (DOA) is the direction from which the sound source is coming from.

    Attributes:
        room: pyroomacoustics room object
        microphones: array of microphone positions as [x, y] coordinates in the room
        centre_mic: centre microphone position as [x, y] coordinates in the room
        source_loc: array of source positions as [x, y] coordinates in the room
        fs: int, sampling frequency (recommended 16000 Hz)
        snr: float, signal-to-noise ratio (recommended: 0, 0.01, 0.001)
        room_dimension: array, room dimension as [length, width]
        NFFT: int, FFT size
        C: float, speed of sound
    """

    def __init__(self, room_dimensions, source_loc, centre_mic, distance_mic=0.1, snr=0):
        self.room = None
        self.microphones = None
        self.centre_mic = centre_mic
        self.source_loc = source_loc
        self.distance_mic = distance_mic
        self.fs = 16000
        self.snr = 0
        self.room_dimension = room_dimensions
        self.nfft = 256
        self.c = 343.

    def get_room(self, signal):
        """ Simulate the room with the given signal and source locations """
        self.room = pra.ShoeBox(self.room_dimension, fs=self.fs, max_order=0)
        for source in self.source_loc:
            self.room.add_source(source, signal=signal)

        self.microphones = np.c_[[self.centre_mic[0]-self.distance_mic, self.centre_mic[1]],
                                 [self.centre_mic[0]+self.distance_mic, self.centre_mic[1]]]
        self.room.add_microphone_array(self.microphones)

        if self.snr != 0:
            self.room.simulate(10*math.log10(2 / self.snr))
        else:
            self.room.simulate()


class DoaCNN(Doa):
    """ Class to predict the DOA of the sound source using a Convolutional Neural Network (CNN)

    Performs the sound signal transformation using Generalized Cross-Correlation with 
    Phase Transform (GCC-PHAT). Contructed from the base DOA class.

    Attributes:
        model: keras model object; has already been trained on 180 degrees of azimuth predictions
    """

    def __init__(self, room_dimensions, source_loc, centre_mic, distance_mic=0.1, snr=0):
        super().__init__(room_dimensions, source_loc, centre_mic, distance_mic, snr)
        self.model_name = "CNN"
        self.model = load_model('./saved_model')

    def gcc_phat(self, x_1, x_2, fs=16000, interp=1):
        """ Compute the GCC-PHAT between two signals """
        n = len(x_1) + len(x_2) - 1
        n += 1 if n % 2 else 0

        # Fourier transforms of the two signals
        _x_1 = np.fft.rfft(x_1, n=n)
        _x_2 = np.fft.rfft(x_2, n=n)

        # Normalize by the magnitude of FFT - because PHAT
        np.divide(_x_1, np.abs(_x_1), _x_1, where=np.abs(_x_1) != 0)
        np.divide(_x_2, np.abs(_x_2), _x_2, where=np.abs(_x_2) != 0)

        # GCC-PHAT = [X_1(f)X_2*(f)] / |X_1(f)X_2*(f)|
        # See Knapp and Carter (1976) for reference
        _cc = _x_1 * np.conj(_x_2)
        cc = np.fft.irfft(_cc, n=n * interp)

        # Maximum delay between a pair of microphones,
        # expressed in a number of samples.
        # 0.2 m is the distance between the micropones and
        # 340 m/s is assumed to be the speed of sound.
        max_len = math.ceil(0.2 / 340 * fs * interp)

        # Trim the cc vector to only include a
        # small number of samples around the origin
        cc = np.concatenate((cc[-max_len:], cc[:max_len+1]))

        # Return the cross correlation
        return cc


    def compute_gcc_matrix(self, observation, fs, interp=1):
        """ Compute the GCC matrix of the observation """
        # Initialize a transformed observation, that will be populated with GCC vectors
        # of the observation
        transformed_observation = []

        # Compute GCC for every pair of microphones
        mic_1, mic_2 = [0, 1]
        x_1 = observation[:, mic_1]
        x_2 = observation[:, mic_2]
        gcc = self.gcc_phat(x_1, x_2, fs=fs, interp=interp)

        # Add the GCC vector to the GCC matrix
        transformed_observation.append(gcc)
        return transformed_observation


    def create_observations(self, wav_signals, fs, samples=20, step=5, interp=1):
        """ Create observations from the wav signals"""
        # Lists of observations and labels that will be populated
        x_wav = []

        # Loop through the signal frame and take subframes
        for i in range(0, len(wav_signals) - samples + 1, step):
            # Extract the observation from subframe
            observation = np.array(wav_signals[i : i + samples])

            # Transform observation into a GCC matrix
            x_wav.append(self.compute_gcc_matrix(observation, fs, interp=interp))

        cols = [
                    f'mics{mic_1+1}{mic_2+1}_{i}' 
                        for mic_1, mic_2 in combinations(range(2), r=2)
                            for i in range(np.shape(x_wav)[2])
                ]

        df = pd.DataFrame(data=np.reshape(x_wav, (len(x_wav), -1)), columns=cols)
        return df


    def encode(self):
        """ Encode the labels (180 degrees) using one-hot encoding"""
        x = list(range(0, 180 + 1, 10))
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit([[label] for label in x])
        return encoder


    def get_prediction(self):
        """ Get the prediction of the sound source direction of arrival of the CNN """
        # Get the signals from the room simulation
        data = self.room.mic_array.signals.T
        data = np.array(normalize(data, bits=16), dtype=np.int16)
        x = self.create_observations(data, self.fs)

        # Get the predictions from the model as a list of
        # probabilities in 10 degrees increments
        # range from 0 to 180 degrees
        predictions = self.model.predict(x)
        encoder = self.encode()
        y_pred = encoder.inverse_transform(predictions)
        predictions=[]
        for i in y_pred:
            predictions.append(int(i))

        # Returns a list of 180 predicted angles probabilities in degrees; for angle
        # in 10 degrees increments 
        return predictions


class DoaMUSIC(Doa):
    """ Class to predict the doa of the sound source using the MUSIC algorithm

    Performs the sound signal transformation using Short-Time Fourier Transform (STFT)
    """
    def __init__(self, room_dimensions, source_loc, centre_mic, distance_mic=0.1, snr=0):
        super().__init__(room_dimensions, source_loc, centre_mic, distance_mic, snr)
        self.model_name = "MUSIC"

    def get_prediction(self):
        """ Get the prediction of the sound source direction of arrival of the MUSIC algorithm
        Output is float; predicted angle in degrees from 0 to 180 """
        # Perform the STFT on the signals from the room simulation
        x = pra.transform.stft.analysis(self.room.mic_array.signals.T, self.nfft, self.nfft // 2)
        x = x.transpose([2, 1, 0])

        # Construct the new DOA object and perform localisation on the frames in X
        doa = pra.doa.MUSIC(self.microphones, self.fs, self.nfft, c=self.c, num_src=1)
        doa.locate_sources(x)
        pred = doa.azimuth_recon

        # If the azimuth is larger than pi, then we need to take the complement
        if pred > np.pi:
            pred = 2 * np.pi - pred
        pred = math.degrees(pred)

        # Convert the prediction to degrees
        return pred

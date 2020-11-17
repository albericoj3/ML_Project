# From: https://www.analyticsvidhya.com/blog/2019/07/learn-build-first-speech-to-text-model-python/

import os
import librosa  # for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile  # for audio processing
import warnings

train_audio_path = 'C:/Users/jorda/PycharmProjects/MachineLearning/ML_Project/tensorflow-speech-recognition-challenge/train/audio'

labels=os.listdir(train_audio_path)

warnings.filterwarnings("ignore")

all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr=16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if (len(samples) == 8000):
            all_wave.append(samples)
            all_label.append(label)

# Convert output labels into integer encoded
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(all_label)
classes = list(le.classes_)

# Convert integer labels to a one-hot vector
from keras.utils import np_utils

y = np_utils.to_categorical(y, num_classes=len(labels))

# Reshape 2D array to 3D
all_wave = np.array(all_wave).reshape(-1, 8000, 1)

# Train on 80% validate on 20%
from sklearn.model_selection import train_test_split

x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y), stratify=y, test_size=0.2,
                                            random_state=777, shuffle=True)

import librosa
import librosa.display
import moviepy.editor as mp
import os
import numpy as np
import soundfile as sf
import IPython.display as ipd
from IPython.display import Audio
import matplotlib.pyplot as plt


def plot_time_series(data):
    plt.figure(figsize=(10,7))

    librosa.display.waveplot(data, sr=22050)

    plt.xlabel('time', fontsize=14)
    plt.ylabel('amplitude', fontsize=14)

    plt.show()


def adding_white_noise(data, sr=22050, noise_rate=0.005):
    # noise 방식으로 일반적으로 쓰는 잡음 끼게 하는 겁니다.
    wn = np.random.randn(len(data))
    data_wn = data + noise_rate * wn
    plot_time_series(data_wn)
    sf.write('white_noise.wav', data_wn, sr)  # 저장
    print('White Noise 저장 성공')
    ipd.display(Audio('white_noise.wav'))

    return data


def shifting_sound(data, sr=22050, roll_rate=0.1):
    # 그냥 [1, 2, 3, 4] 를 [4, 1, 2, 3]으로 만들어주는겁니다.
    data_roll = np.roll(data, int(len(data) * roll_rate))
    plot_time_series(data_roll)
    sf.write('rolling_sound.wav', data_roll, sr)
    print('rolling_sound 저장 성공')
    ipd.display(Audio('rolling_sound.wav'))

    return data


def stretch_sound(data, sr=22050, rate=0.8):
    # stretch 해주는 것 테이프 늘어진 것처럼 들린다.
    stretch_data = librosa.effects.time_stretch(data, rate)
    plot_time_series(stretch_data)
    sf.write('stretch_data.wav', stretch_data, sr)
    print('stretch_data 저장 성공')
    ipd.display(Audio('stretch_data.wav'))

    return data


def reverse_sound(data, sr=22050):
    # 거꾸로 재생
    data_len = len(data)
    data = np.array([data[len(data) - 1 - i] for i in range(len(data))])
    plot_time_series(data)
    sf.write('reverse_data.wav', data, sr)
    ipd.display(Audio('reverse_data.wav'))

    return data


def minus_sound(data, sr=22050):
    # 위상을 뒤집는 것으로서 원래 소리와 똑같이 들린다.
    temp_numpy = (-1) * data
    plot_time_series(temp_numpy)
    sf.write('minus_data.wav', temp_numpy, sr)
    ipd.display(Audio('minus_data.wav'))

    return data
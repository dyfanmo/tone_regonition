import os
import pandas as pd
import librosa.display
import warnings
import numpy as np
import IPython.display as ipd
from IPython.display import display
from matplotlib import pyplot as plt

from . import utils


warnings.filterwarnings("ignore")

if not os.path.isdir(utils.FIGURE_PATH):
    os.mkdir(utils.FIGURE_PATH)


def display_tone_dist(chinese_words, df, file_path='tone_distribution.png'):
    """
    Plot a comparison of the tone distribution of all the Chinese Words and
    the tone distribution of the sampled data
    """
    chinese_df = pd.DataFrame(chinese_words, columns=['word'])
    chinese_df['tone'] = chinese_df['word'].apply(lambda word: utils.text_to_tone(word))
    tone_count0 = chinese_df['tone'].value_counts()
    tone_per0 = tone_count0 / tone_count0.sum()
    tone_count1 = df['tone'].value_counts()
    tone_per1 = tone_count1 / tone_count1.sum()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    axs[0].set_ylabel('Per %')
    fig.text(0.5, 0.04, 'Tones', ha='center')

    for i, data in enumerate([tone_per0, tone_per1]):
        axs[i].bar(data.index, data)
        t = axs[i].set_xticks(data.index)

    axs[0].set_title('Real-world Tone Distribution')
    axs[1].set_title('Sampled Data Tone Distribution')
    plt.savefig(utils.FIGURE_PATH + file_path)


def play_tones(df):
    """ Play an example of each tone pronunciation """
    for i in range(1, 5):
        tone = df[df['tone'] == i]
        audio = tone.sample(1)
        display(ipd.Audio(utils.AUDIO_PATH +  audio.id.iloc[0]))


def display_duration(df):
    """ A histogram of the audio data duration """
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    hist = ax.hist(df['duration'], bins=80)
    ax.set_title('Duration')
    plt.savefig(utils.FIGURE_PATH + 'old_duration.png')


def display_wave_plots(df):
    """ Plot the wave plots of random audio files """
    n_rows = 1
    n_cols = 5
    plt.figure(figsize=(n_cols * 3.3, n_rows * 2.8))
    for row in range(n_rows):
        for col in range(n_cols):
            for i in range(n_cols * n_rows):
                audio = df.sample(1)
                file_path = utils.AUDIO_PATH + audio.id.iloc[0]
                wav, rate = librosa.load(file_path, sr=None)
                index = n_cols * row + col
                plt.subplot(n_rows, n_cols, index + 1)
            librosa.display.waveplot(wav, rate)  # plot wave
            plt.title(f'Tone: {audio.tone.iloc[0]}')
            plt.tight_layout()
    plt.savefig(utils.FIGURE_PATH + '/wave_plots.png')


def display_spectrogram(df):
    """ Plot spectrograms of random audio files """
    n_rows = 1
    n_cols = 5
    plt.figure(figsize=(n_cols * 3.3, n_rows * 2.8))
    for row in range(n_rows):
        for col in range(n_cols):
            for i in range(n_cols * n_rows):
                audio = df.sample(1)
                file_path = utils.AUDIO_PATH + audio.id.iloc[0]
                specgram = utils.get_melspectrogram_db(file_path, duration=3)
                img = utils.specgram_to_image(specgram)
                index = n_cols * row + col
                plt.subplot(n_rows, n_cols, index + 1)
            librosa.display.specshow(img, cmap='viridis')
            plt.title(f'Tone:{audio.tone.iloc[0]}')
    plt.savefig(utils.FIGURE_PATH + 'mel_specgram.png')


def compare_duration(audio_data):
    """ Display the comparison between the duration before and after silence """
    fig, ax = plt.subplots(1, 2, figsize=(17, 6), sharex=True, sharey=True)
    duration_list = [audio_data.new_duration, audio_data.duration]
    for i in range(2):
        ax[i].hist(duration_list[i], bins=50)
        ax[i].set_xlabel('Duration (min)')
        if i == 1:
            ax[i].set_title('Original Duration Distribution')
        else:
            ax[i].set_title('New Duration Distribution')
    plt.savefig(utils.FIGURE_PATH + 'compare_duration.png')


def display_outliers(df):
    """ Residual plots of PCA whilst labelling the anomalies """
    inliers = df[df['anomaly'] == 1]
    outliers = df[df['anomaly'] == -1]

    fig = plt.figure(figsize=(13, 9))
    ax = fig.gca(projection='3d')
    for data in [inliers, outliers]:
        ax.scatter(data.PC1, data.PC2, data.PC3, alpha=0.2)
    ax.legend(['Inliers', 'Outliers'])
    ax.set_title('Anomalies')

    plt.savefig(utils.FIGURE_PATH + 'pca_outliers.png')


def play_audio_quality(df):
    """ Play audio files classified as the lowest and highest quality """
    high_quality = df[df['sound_quality'] == 0].sample(1).squeeze()
    low_quality = df[df['sound_quality'] == 2].sample(1).squeeze()
    for ser_i in [high_quality, low_quality]:
        file_path = utils.get_audio_path(ser_i)
        print(f'Tone: {ser_i.tone}\tSound Quality: {ser_i.sound_quality}\tAudio Type: {ser_i.audio_type}')
        display(ipd.Audio(file_path))


def play_audio_length(df):
    """ Play audio file before and after silence """
    df = df.sample(1).squeeze()
    print(f'Word: {df.word}, Tone: {df.tone}')
    for file_path in [utils.CLEAN_PATH, utils.AUDIO_PATH]:
        display(ipd.Audio(file_path + df.id))


def play_anomalies(df, num=1):
    """ Play audio files classified as inliers or outliers """
    for i in range(num):
        outlier = df[df['anomaly'] == -1].sample(1).squeeze()
        inlier = df[df['anomaly'] == 1].sample(1).squeeze()
        for ser_i in [inlier, outlier]:
            file_path = utils.get_audio_path(ser_i)
            print(f'Tone: {ser_i.tone}\tAnomaly: {ser_i.anomaly}\tAudio Type: {ser_i.audio_type}')
            display(ipd.Audio(file_path))


def play_aug(df):
    """ Play each version of the augmented data """
    row = df.sample(1)
    file_path = utils.CLEAN_PATH + row.id.iloc[0]
    wav, rate = librosa.load(file_path)

    wn = np.random.randn(len(wav))
    wav_wn = wav + 0.005 * wn
    wav_dp = utils.change_pitch(wav, rate, deep=True)
    wav_hi = utils.change_pitch(wav, rate, deep=False)
    for wav_i in [wav, wav_wn, wav_dp, wav_hi]:
        display(ipd.Audio(wav_i, rate=rate))


def display_aug(df):
    """ Plot each version of the augmented data """
    row = df.sample(1)
    file_path = utils.CLEAN_PATH + row.id.iloc[0]
    wav, rate = librosa.load(file_path)

    wn = np.random.randn(len(wav))
    wav_wn = wav + 0.005 * wn
    wav_dp = utils.change_pitch(wav, rate, deep=True)
    wav_hi = utils.change_pitch(wav, rate, deep=False)
    wav_all = [('Original', wav), ('With Noise', wav_wn), ('Deep', wav_dp), ('High', wav_hi)]
    fig, axs = plt.subplots(4, 1, figsize=(17, 10), sharey=True, sharex=True)
    for i, wav_i in enumerate(wav_all):
        axs[i].set_title(wav_i[0])
        axs[i].plot(wav_i[1])
    plt.savefig(utils.FIGURE_PATH + 'wave_plots_augmented.png')


def display_pca_types(df):
    """ Residual plots of PCA whilst labelling the audio types """
    cl = df[df['audio_type'] == 'CL']
    wn = df[df['audio_type'] == 'WN']
    dp = df[df['audio_type'] == 'DP']
    hf = df[df['audio_type'] == 'HF']

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for data in [cl, wn, dp, hf]:
        ax.scatter(data.PC1, data.PC2, data.PC3, alpha=0.2)
    ax.legend(['Original', 'With Noise', 'Deep Pitch', 'High Pitch'])
    ax.set_title('Audio Types')

    plt.savefig(utils.FIGURE_PATH + 'pca_audio_types.png')


def display_pca_tones(df):
    """ Residual plots of PCA whilst labelling the tones """
    tone1 = df[df['tone'] == 1]
    tone2 = df[df['tone'] == 2]
    tone3 = df[df['tone'] == 3]
    tone4 = df[df['tone'] == 4]

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for data in [tone1, tone2, tone3, tone4]:
        ax.scatter(data.PC1, data.PC2, data.PC3, alpha=0.2)
    ax.legend([1, 2, 3, 4])
    ax.set_title('Tones')
    plt.savefig(utils.FIGURE_PATH + 'pca_tones.png')


def display_model_loss(model):
    """ Plot the train and validation loss of each epoch """
    name = model.__class__.__name__
    tl = np.load(f'{utils.DATA_PATH}/scores/tl-{name[:4].upper()}.npy')
    vl = np.load(f'{utils.DATA_PATH}/scores/vl-{name[:4].upper()}.npy')

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharey=True)
    ax.plot(tl)
    ax.plot(vl)
    ax.set_xlabel('Epochs')
    ax.legend(['Train Loss', 'Valid Loss'])
    ax.set_title(name)

    plt.savefig(f'{utils.FIGURE_PATH}/loss-{name[:3]}.png')


def compare_model_loss(*models):
    """ Plot and compare the model's train and validation loss for each epoch """
    names = []
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    axs[0].set_title('Train Loss')
    axs[1].set_title('Valid Loss')
    axs[1].set_xlabel('Epochs')
    for model in models:
        name = model.__class__.__name__
        names.append(name)
        tl = np.load(f'{utils.DATA_PATH}/scores/tl-{name[:4].upper()}.npy')
        vl = np.load(f'{utils.DATA_PATH}/scores/vl-{name[:4].upper()}.npy')
        axs[0].plot(tl)
        axs[1].plot(vl)
    axs[0].legend(names)
    axs[1].legend(names)
    plt.savefig(utils.FIGURE_PATH + 'compare_loss.png')


def play_long_audio(df):
    """ Play audio files longer than 2 seconds after silence """
    long_audio = df[df.new_duration > 2]
    for i in range(2):
        sample = long_audio.sample(1)
        display(ipd.Audio(utils.CLEAN_PATH + sample.id.iloc[0]))

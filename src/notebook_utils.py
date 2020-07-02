import sys
import librosa.display
import warnings
import os.path as path
import IPython.display as ipd
from IPython.display import display
import matplotlib.pyplot as plt

ROOT_DIR = path.abspath(path.join(__file__ ,"../.."))
sys.path.insert(1, f"{ROOT_DIR}")
from src.utils import *

warnings.filterwarnings("ignore")


def display_tone_distributions(chinese_words, df):
    chinese_df = pd.DataFrame(chinese_words, columns=['word'])
    chinese_df['tone'] = chinese_df['word'].apply(lambda word: text_to_tone(word))
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


def play_tones(df):
    for i in range(1, 5):
        tone = df[df['tone'] == i]
        audio = tone.sample(1)
        display(ipd.Audio(AUDIO_PATH + '/' + audio.id.iloc[0]))


def display_duration(df):
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    hist = ax.hist(df['duration'], bins=80)
    ax.set_title('Duration')


def display_waveplots(df):
    n_rows = 1
    n_cols = 5
    plt.figure(figsize=(n_cols * 3.3, n_rows * 2.8))
    for row in range(n_rows):
        for col in range(n_cols):
            for i in range(n_cols * n_rows):
                audio = df.sample(1)
                path = f'{AUDIO_PATH}/{audio.id.iloc[0]}'
                wav, sr = librosa.load(path, sr=None)
                index = n_cols * row + col
                plt.subplot(n_rows, n_cols, index + 1)
            librosa.display.waveplot(wav, sr)  # plot wavefile
            plt.title(f'Tone: {audio.tone.iloc[0]}')
            plt.tight_layout()


def display_specgrams(df):
    n_rows = 1
    n_cols = 5
    plt.figure(figsize=(n_cols * 3.3, n_rows * 2.8))
    for row in range(n_rows):
        for col in range(n_cols):
            for i in range(n_cols * n_rows):
                audio = df.sample(1)
                path = f'{AUDIO_PATH}/{audio.id.iloc[0]}'
                spegram = get_melspectrogram_db(path, duration=3)
                img = specgram_to_image(spegram)
                index = n_cols * row + col
                plt.subplot(n_rows, n_cols, index + 1)
            librosa.display.specshow(img, cmap='viridis')
            plt.title(f'Tone:{audio.tone.iloc[0]}')


def display_duration_comparison(audio_data):
    fig, ax = plt.subplots(1, 2, figsize=(17, 6), sharex=True, sharey=True)
    duration_list = [audio_data.new_duration, audio_data.duration]
    for i in range(2):
        ax[i].hist(duration_list[i], bins=50)
        ax[i].set_xlabel('Duataion (min)')
        if i == 1:
            ax[i].set_title('Orginal Duration Distribution')
        else:
            ax[i].set_title('New Duration Distribution')


def display_outliers(df):
    inliers = df[df['anomaly'] == 1]
    outliers = df[df['anomaly'] == -1]

    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca(projection='3d')
    for data in [inliers, outliers]:
        ax.scatter(data.PC1, data.PC2, data.PC3, alpha=0.4)
    ax.legend(['Inliers', 'Outliers'])
    ax.set_title('Anomalies')


def play_audio_quality(df):
    high_quality = df[df['sound_quality'] == 0].sample(1).squeeze()
    low_quality = df[df['sound_quality'] == 2].sample(1).squeeze()
    for ser_i in [high_quality, low_quality]:
        file_path = get_audio_path(ser_i)
        print(f'Tone: {ser_i.tone}\tSound Quality: {ser_i.sound_quality}\tAudio Type: {ser_i.audio_type}')
        display(ipd.Audio(file_path))


def play_audio_length(df):
    df = df.sample(1).squeeze()
    print(f'Word: {df.word}, Tone: {df.tone}')
    for path in [CLEAN_PATH, AUDIO_PATH]:
        display(ipd.Audio(f'{path}/{df.id}'))


def play_anomalies(df, num=1):
    for i in range(num):
        outlier = df[df['anomaly'] == -1].sample(1).squeeze()
        inlier = df[df['anomaly'] == 1].sample(1).squeeze()
        for ser_i in [inlier, outlier]:
            file_path = get_audio_path(ser_i)
            print(f'Tone: {ser_i.tone}\tAnomaly: {ser_i.anomaly}\tAudio Type: {ser_i.audio_type}')
            display(ipd.Audio(file_path))


def play_aug(df):
    row = df.sample(1)
    path = f'{CLEAN_PATH}/{row.id.iloc[0]}'
    wav, sr = librosa.load(path)

    wn = np.random.randn(len(wav))
    wav_wn = wav + 0.005 * wn
    wav_dp = change_pitch(wav, sr, deep=True)
    wav_hi = change_pitch(wav, sr, deep=False)
    wav_rl = np.roll(wav, sr)
    for wav_i in [wav, wav_wn, wav_dp, wav_hi, wav_rl]:
        display(ipd.Audio(wav_i, rate=sr))


def display_aug(df):
    row = df.sample(1)
    path = f'{CLEAN_PATH}/{row.id.iloc[0]}'
    wav, sr = librosa.load(path)

    wn = np.random.randn(len(wav))
    wav_wn = wav + 0.005 * wn
    wav_dp = change_pitch(wav, sr, deep=True)
    wav_hi = change_pitch(wav, sr, deep=False)
    wav_rl = np.roll(wav, sr)
    wav_all = [('Orginal', wav), ('With Noise', wav_wn), ('Deep', wav_dp), ('High', wav_hi), ('Audio Roll', wav_rl)]
    fig, axs = plt.subplots(5,1 , figsize=(17, 10), sharey=True, sharex=True)
    for i, wav_i in enumerate(wav_all):
        axs[i].set_title(wav_i[0])
        axs[i].plot(wav_i[1])


def display_pca_types(df):
    cl = df[df['audio_type'] == 'CL']
    wn = df[df['audio_type'] == 'WN']
    dp = df[df['audio_type'] == 'DP']
    hf = df[df['audio_type'] == 'HF']
    rl = df[df['audio_type'] == 'RL']

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for data in [cl, wn, dp, hf, rl]:
        ax.scatter(data.PC1, data.PC2, data.PC3, alpha=0.4)
    ax.legend(['Original', 'With Noise', 'Deep Pitch', 'High Pitch', 'Rolled Audio'])
    ax.set_title('Audio Types')


def display_pca_tones(df):
    tone1 = df[df['tone'] == 1]
    tone2 = df[df['tone'] == 2]
    tone3 = df[df['tone'] == 3]
    tone4 = df[df['tone'] == 4]

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for data in [tone1, tone2, tone3, tone4]:
        ax.scatter(data.PC1, data.PC2, data.PC3, alpha=0.4)
    ax.legend([1, 2, 3, 4])
    ax.set_title('Tones')


def display_model_loss(model):
    name = model.__class__.__name__
    tl = np.load(f'{DATA_PATH}/scores/tl-{name[:4].upper()}.npy')
    vl = np.load(f'{DATA_PATH}/scores/vl-{name[:4].upper()}.npy')

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharey=True)
    ax.plot(tl)
    ax.plot(vl)
    ax.set_xlabel('Epochs')
    ax.legend(['Train Loss', 'Valid Loss'])
    ax.set_title(name)


def compare_model_loss(*models):
    names = []
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    axs[0].set_title('Train Loss')
    axs[1].set_title('Valid Loss')
    axs[1].set_xlabel('Epochs')
    for model in models:
        name = model.__class__.__name__
        names.append(name)
        tl = np.load(f'{DATA_PATH}/scores/tl-{name[:4].upper()}.npy')
        vl = np.load(f'{DATA_PATH}/scores/vl-{name[:4].upper()}.npy')
        axs[0].plot(tl)
        axs[1].plot(vl)
    axs[0].legend(names)
    axs[1].legend(names)


def play_long_audio(df):
    long_audio = df[df.new_duration > 2]
    for i in range(2):
        sample = long_audio.sample(1)
        display(ipd.Audio(f'{CLEAN_PATH}/{sample.id.iloc[0]}'))

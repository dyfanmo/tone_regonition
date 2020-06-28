import pinyin
import librosa
import pickle
from torch.utils.data import Dataset
import speech_recognition as sr
import numpy as np
import pandas as pd
from tqdm import tqdm
import os.path as path

ROOT_DIR = path.abspath(path.join(__file__ ,"../.."))
DATA_PATH = f"{ROOT_DIR}/data"
AUDIO_PATH = f"{DATA_PATH}/raw/Audio"
CLEAN_PATH = f"{DATA_PATH}/processed/Audio/Clean"
AUGMENTED_PATH = f'{DATA_PATH}/processed/Audio/Augmented'
INTERIM_PATH = f'{DATA_PATH}/interim'
SCORES_PATH = f'{DATA_PATH}/scores'
MODELS_PATH = f"{ROOT_DIR}/models"

# insert API-KEY from https://api.forvo.com/
API_KEY = '2e3cb6041dea7f1b91e9d75e5b1c1415'


def speech_to_text(filename):
    r = sr.Recognizer()
    transcripts = []
    with sr.AudioFile(filename) as source:
        audio_text = r.listen(source)
        response = r.recognize_google(audio_text, language="zh-TW", show_all=True)
        if response:

            for alternative in response['alternative']:
                transcript = alternative['transcript']
                transcripts.append(transcript)

    if transcripts:
        return transcripts
    else:
        return []


def text_to_tone(text):
    tone = ''
    try:
        pin = pinyin.get(text, format='numerical')
        if '1' in pin:
            tone = 1
        elif '2' in pin:
            tone = 2
        elif '3' in pin:
            tone = 3
        elif '4' in pin:
            tone = 4
    except:
        pass

    if tone:
        return tone
    else:
        return np.nan


def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def get_melspectrogram_db(file_path, duration=2, n_fft=2048, hop_length=256, n_mels=128, fmin=20, fmax=8300):
    wav, sr = librosa.load(file_path)
    input_length = sr * duration
    wav = librosa.util.fix_length(wav, round(input_length))

    spec = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
                                          hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    return spec_db


def specgram_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj


def get_audio_path(ser):
    try:
        if ser.audio_type == 'CL':
            file_path = f"{ROOT_DIR}/data/processed/Audio/Clean/{ser.id}"
        else:
            file_path = f"{ROOT_DIR}/data/processed/Audio/Augmented/{ser.id}"
    except:
        file_path = f"{ROOT_DIR}/data/raw/Audio/{ser.id}"
    return file_path


class ToneData(Dataset):
    def __init__(self, df, out_col, duration):
        self.df = df
        self.data = []
        self.labels = []
        self.c2i = {}
        self.i2c = {}
        self.tones = sorted(df[out_col].unique())
        self.duration = duration
        for i, tone in enumerate(self.tones):
            self.c2i[tone] = i
            self.i2c[i] = tone
        for index in tqdm(df.index):
            ser_i = df.loc[index]
            file_path = get_audio_path(ser_i)
            specgram = get_melspectrogram_db(file_path, duration=self.duration)
            self.data.append(specgram_to_image(specgram)[np.newaxis, ...])
            self.labels.append(self.c2i[ser_i['labels']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def change_pitch(wav, sr, deep=True):
    bins_per_octave = 12
    pitch_change = np.random.uniform(0.5, 1.5)
    if deep:
        pitch_change = pitch_change * -1
    wav = librosa.effects.pitch_shift(wav, sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)
    return wav


def get_tone_distrubtion(chinese_words):
    chinese_df = pd.DataFrame(chinese_words, columns=['word'])
    chinese_df['tone'] = chinese_df['word'].apply(lambda word: text_to_tone(word))
    tone_count = chinese_df['tone'].value_counts()
    tone_per = tone_count / tone_count.sum()
    return tone_per

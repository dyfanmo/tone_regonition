import re
import torch
import pinyin
import librosa
import pickle
from torch.utils.data import Dataset
import speech_recognition as sr
import numpy as np
import pandas as pd
from tqdm import tqdm
import os.path as path
from IPython import get_ipython


def convert_index_to_strings(indexes):
    indexes_str = []
    for index in indexes:
        try:
            indexes_str.append(str(index))
        except TypeError:
            pass
    return indexes_str


def speech_to_text(filename):
    """ Transcribe audio files into text """
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
    """ Retrieve the tone of a character """

    pin = pinyin.get(text, format='numerical')
    temp = re.findall(r'\d+', pin)
    tone = list(map(int, temp))

    if len(tone) == 1:
        tone.append(6)
    return tuple(tone)


def match_target_amplitude(aChunk, target_dBFS):
    """ Normalize given audio chunk """
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def get_melspectrogram_db(file_path, duration=3, n_fft=2048, hop_length=512, n_mels=128):
    """ Build Mel Spectrogram """
    wav, rate = librosa.load(file_path)
    input_length = rate * duration
    wav = librosa.util.fix_length(wav, round(input_length))

    spec = librosa.feature.melspectrogram(wav, sr=rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    return spec_db


def specgram_to_image(spec, eps=1e-6):
    """ Covert Mel Spectrogram into an image """
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


def save_object(obj, filename):
    """ Save an object"""
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """ Load an object """
    with open(filename, 'rb') as data:
        obj = pickle.load(data)
    return obj


def get_audio_path(ser):
    """ Get the audio path of a file """
    try:
        if ser.audio_type == 'CL':
            file_path = f"{ROOT_DIR}/data/processed/Audio/Clean/{ser.id}"
        else:
            file_path = f"{ROOT_DIR}/data/processed/Audio/Augmented/{ser.id}"
    except FileNotFoundError:
        file_path = f"{ROOT_DIR}/data/raw/Audio/{ser.id}"
    return file_path


class MultiTaskDataset(Dataset):
    """ Building spectrogram and add its label into a numpy array """
    def __init__(self, df, duration):
        self.df = df
        self.data = []
        self.labels = []
        self.duration = duration

        self.duration = duration

        for index in tqdm(df.index):
            row = df.loc[index]
            file_path = get_audio_path(row)
            specgram = get_melspectrogram_db(file_path, duration=self.duration)
            self.data.append(specgram_to_image(specgram)[np.newaxis, ...])
            self.labels.append(torch.tensor(tuple(np.array(row.tones) - 1)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def change_pitch(wav, rate, deep=True):
    """ Manipulate the pitch within an audio file """
    bins_per_octave = 12
    pitch_change = np.random.uniform(0.5, 1.5)
    if deep:
        pitch_change = pitch_change * -1
    wav = librosa.effects.pitch_shift(wav, rate, n_steps=pitch_change, bins_per_octave=bins_per_octave)
    return wav


def get_tone_dist(chinese_words):
    """ Get the distribution of tones from all the Chinese characters """
    chinese_df = pd.DataFrame(chinese_words, columns=['word'])
    chinese_df['tone'] = chinese_df['word'].apply(lambda word: text_to_tone(word))
    tone_count = chinese_df['tone'].value_counts()
    tone_per = tone_count / tone_count.sum()
    return tone_per


def isnotebook():
    """ Return boolean if code is implemented within a notebook """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


ROOT_DIR = path.abspath(path.join(__file__, "../.."))
DATA_PATH = ROOT_DIR + "/data/"
MODELS_PATH = ROOT_DIR + "/models/"
FIGURE_PATH = ROOT_DIR + '/figures/'
AUDIO_PATH = DATA_PATH + "raw/Audio/"
PROCESSED_PATH = DATA_PATH + "processed/"
PICKLE_PATH = PROCESSED_PATH + 'Pickle/'
CLEAN_PATH = PROCESSED_PATH + "Audio/Clean/"
AUGMENTED_PATH = PROCESSED_PATH + "Audio/Augmented/"
INTERIM_PATH = DATA_PATH + "interim/"
SCORES_PATH = DATA_PATH + "scores/"
import os
import io
import wave
import uuid
import requests
import pandas as pd
import hanzidentifier

from tqdm import tqdm
from pydub import AudioSegment

from ..utils import DATA_PATH, PROCESSED_PATH,  AUDIO_PATH, PICKLE_PATH, load_object, text_to_tone


def filter_words(words):
    """ Filter all the simplified Chinese words less than 2 characters """
    split_words = words.split('[')
    chinese_words = []
    for word in split_words:
        split_words2 = word.split('\n')
        for word2 in split_words2:
            if len(word2) <= 2:
                if hanzidentifier.is_simplified(word2):
                    if word2 not in chinese_words:
                        chinese_words.append(word2)

    return chinese_words


def make_chinese_words_list():
    """ Make a list of unique Chinese words from the text file """
    file_path = DATA_PATH + "external/HSK2013.txt"
    with open(file_path, "r") as word_list:
        words = word_list.read()

    chinese_words = filter_words(words)

    return chinese_words


def extract_or_load_words():
    """ Extract all the chinese words from dictionary or load them """
    try:
        chinese_words = load_object(PROCESSED_PATH + "chinese_characters.txt")
    except FileNotFoundError:
        chinese_words = make_chinese_words_list()
    return chinese_words


def request_pronunciations(words, api_key, limit, num_samples=500):
    """ Request the URLs for each audio file """
    audio_urls = []
    sampled_words = pd.Series(words).sample(num_samples).to_list()
    for word in tqdm(sampled_words):
        try:
            api_url = f'https://apifree.forvo.com/action/word-pronunciations/format/json/word/{word}/ \
                        language/zh/order/rate-desc/limit/{limit}/key/{api_key}/'
            r = requests.get(api_url)
            data = r.json()
            data_items = data['items']

            for i in range(len(data_items)):
                audio_urls.append((data_items[i]['pathmp3'], word))
        except TypeError:
            pass

    return audio_urls


def save_pronunciations(audio_urls):
    """ Save each audio file """
    data_list = []

    os.makedirs(DATA_PATH + 'raw', exist_ok=True)
    os.makedirs(AUDIO_PATH, exist_ok=True)

    for i in tqdm(range(0, len(audio_urls))):
        try:
            unique_filename = str(uuid.uuid4())
            r = requests.get(audio_urls[i][0])
            audio_data = r.content
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            audio_id = f'audio_{unique_filename[:7]}.wav'
            audio.export(AUDIO_PATH + audio_id, format='wav')
            data_list.append((audio_id, audio_urls[i][1]))
        except IndexError:
            pass

    return data_list


def build_dataframe(data_list):
    """ Build a data frame of the audio data """
    df = pd.DataFrame(data_list, columns=['id', 'word'])
    df['tones'] = df['word'].apply(lambda word: text_to_tone(word))
    df['nframes'] = df['id'].apply(lambda f: wave.open(AUDIO_PATH + f).getnframes())
    df['duration'] = df['id'].apply(lambda f: wave.open(AUDIO_PATH + f).getnframes() /
                                    wave.open(AUDIO_PATH + f).getframerate())

    try:
        old_df = pd.read_pickle(PICKLE_PATH + 'audio_df.pkl')
        df = pd.concat([old_df, df], ignore_index=True, sort=True)
        df = df.drop_duplicates('id')
    except FileNotFoundError:
        pass
    return df


API_KEY = ''

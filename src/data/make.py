from tqdm import tqdm
import os
import io
import pickle
import pandas as pd
import wave
import uuid
import time
import requests
import hanzidentifier
from pydub import AudioSegment

from .. import utils


def make_chinese_words_list():
    """ Make a list of unique  Chinese characters from the Chinese dictionary """
    file_path = utils.DATA_PATH + "external/cedict_1_0_ts_utf-8_mdbg.txt"
    with open(file_path, "r", encoding='utf-8') as word_list:
        words = word_list.read()

    chinese_words = []
    for word in tqdm(words):
        if hanzidentifier.is_simplified(word):
            if word not in chinese_words:
                chinese_words.append(word)

    return chinese_words


def extract_or_load_words():
    """ Extract all the chinese words from dictionary or load them """
    try:
        chinese_words = utils.load_object(utils.PROCESSED_PATH + "chinese_characters.txt")
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


def request_timer():
    """ Print a timer of how long the request quota will restart """
    time_now = time.strftime("%H")
    reset_time = '23'
    wait_time = int(reset_time) - int(time_now)
    if wait_time <= 1:
        time_now = time.strftime("%H:%M:%S")
        reset_time = '23:00:00'
        factors = (60, 1, 1 / 60)
        time_now = sum(i * j for i, j in zip(map(int, time_now.split(':')), factors))
        reset_time = sum(i * j for i, j in zip(map(int, reset_time.split(':')), factors))
        wait_time = int(reset_time) - int(time_now)
        if wait_time > 0:
            print(f'Requests will reset in {int(round(reset_time - time_now))} minutes!')
        else:
            print(f'Requests will reset in 24 hours!')
    else:
        print(f'Requests will reset in {wait_time} hours!')


def save_pronunciations(audio_urls):
    """ Save each audio file """
    data_list = []

    os.makedirs(utils.DATA_PATH + 'raw', exist_ok=True)
    os.makedirs(utils.AUDIO_PATH, exist_ok=True)

    for i in tqdm(range(0, len(audio_urls))):
        unique_filename = str(uuid.uuid4())
        r = requests.get(audio_urls[i][0])
        audio_data = r.content
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        audio_id = f'audio_{unique_filename[:11]}.wav'
        audio.export(utils.AUDIO_PATH + audio_id, format='wav')
        data_list.append((audio_id, audio_urls[i][1]))

    return data_list


def build_dataframe(data_list):
    """ Build a data frame of the audio data """
    df = pd.DataFrame(data_list, columns=['id', 'word'])
    df['tone'] = df['word'].apply(lambda word: utils.text_to_tone(word))
    df['nframes'] = df['id'].apply(lambda f: wave.open(utils.AUDIO_PATH + f).getnframes())
    df['duration'] = df['id'].apply(lambda f: wave.open(utils.AUDIO_PATH + f).getnframes() /
                                    wave.open(utils.AUDIO_PATH + f).getframerate())
    df['labels'] = df['tone'].apply(lambda tone: tone - 1)

    try:
        old_df = pd.read_pickle(utils.PICKLE_PATH + 'audio_df.pkl')
        df = pd.concat([old_df, df], ignore_index=True, sort=True)
        df = df.drop_duplicates('id')
    except FileNotFoundError:
        pass
    return df


API_KEY = ''

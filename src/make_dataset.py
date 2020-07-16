import os
import io
import wave
import uuid
import time
import requests
import hanzidentifier
from pydub import AudioSegment

from utils import *
from configs.data import *


def extract_words(print_total=True):
    try:
        with open(f"{ROOT_DIR}/data/processed/chinese_characters.txt", "rb") as fp:
            words = pickle.load(fp)
    except:
        file_path = f"{ROOT_DIR}/data/external/cedict_1_0_ts_utf-8_mdbg.txt"
        with open(file_path, "r") as word_list:
            words = word_list.read()

        chi_words = []
        for word in tqdm(words):
            if hanzidentifier.is_simplified(word):
                if word not in chi_words:  # ensure words are not repeated
                    chi_words.append(word)

        with open(f"{ROOT_DIR}/data/processed/chinese_characters.txt", "wb") as f:
            pickle.dump(words, f)
    if print_total:
        print(f"# Total words: {len(words)}.")
    return words


def request_pronunciations(words, api_key, limit):
    audio_urls = []  # list of all the audio samples api request urls
    words = pd.Series(words).sample(500).to_list()
    for word in tqdm(words):
        try:
            api_url = f'https://apifree.forvo.com/action/word-pronunciations/format/json/word/{word}/ \
                        language/zh/order/rate-desc/limit/{limit}/key/{api_key}/'
            r = requests.get(api_url)
            data = r.json()
            data_items = data['items']

            for i in range(len(data_items)):
                audio_urls.append((data_items[i]['pathmp3'], word))
        except:
            pass
    return audio_urls


def request_timer():
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
    data_list = []

    if not os.path.isdir(DATA_PATH + '/raw'):
        os.mkdir(DATA_PATH + '/raw')

    if not os.path.isdir(AUDIO_PATH):
        os.mkdir(AUDIO_PATH)

    for i in tqdm(range(0, len(audio_urls))):
        try:
            unique_filename = str(uuid.uuid4())
            r = requests.get(audio_urls[i][0])
            audio_data = r.content
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            audio_id = f'audio_{unique_filename[:11]}.wav'  # name of each audio file
            audio.export(f'{AUDIO_PATH}/{audio_id}', format='wav')
            data_list.append((audio_id, audio_urls[i][1]))
        except:
            pass
    return data_list


def build_dataframe(data_list):
    df = pd.DataFrame(data_list, columns=['id', 'word'])
    df['tone'] = df['word'].apply(lambda word: text_to_tone(word))
    df['nframes'] = df['id'].apply(lambda f: wave.open(f'{AUDIO_PATH}/{f}').getnframes())
    df['duration'] = df['id'].apply(lambda f: wave.open(f'{AUDIO_PATH}/{f}').getnframes() /
                                    wave.open(f'{AUDIO_PATH}/{f}').getframerate())
    df['labels'] = df['tone'].apply(lambda tone: tone - 1)

    try:
        old_df = pd.read_pickle(f'{ROOT_DIR}/data/processed/Pickle/audio_df.pkl')
        df = pd.concat([old_df, df], ignore_index=True, sort=True)
        df = df.drop_duplicates('id')
        print('Appending Previous Data Set!')
    except:
        pass
    return df


if __name__ == "__main__":

    print("Extracting words!")
    chinese_words = extract_words()
    chinese_df = pd.DataFrame(chinese_words, columns=['word'])
    chinese_df['tone'] = chinese_df['word'].apply(lambda word: text_to_tone(word))

    config = get_config()
    API_KEY = config.api_key

    if API_KEY:
        print('Requesting data!')
        mp3_urls = request_pronunciations(chinese_words, API_KEY, config.limit)

        print('Request limit is reached!')
        request_timer()

        if mp3_urls:
            print('Saving data!')
            audio_list = save_pronunciations(mp3_urls)

            audio_df = build_dataframe(audio_list)
            print(f"# Total Samples: {len(audio_df)}.")
            audio_df.to_pickle(f'{ROOT_DIR}/data/processed/Pickle/audio_df.pkl')
            print('Process Done!')
        else:
            print('Try again later!')
    else:
        print('API-KEY not found!')
        print('Insert API-KEY from https://api.forvo.com/ to request data!')

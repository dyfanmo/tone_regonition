import pandas as pd
from src.data.make import API_KEY, save_pronunciations, request_pronunciations, extract_or_load_words, build_dataframe, request_timer
from src.utils import PROCESSED_PATH, PICKLE_PATH, text_to_tone, save_object

if __name__ == '__main__':
    print("Extracting words!")
    chinese_words = extract_or_load_words()
    save_object(chinese_words, PROCESSED_PATH + 'chinese_characters.txt')
    chinese_df = pd.DataFrame(chinese_words, columns=['word'])
    chinese_df['tone'] = chinese_df['word'].apply(lambda word: text_to_tone(word))

    if API_KEY:
        print('Requesting data!')
        mp3_urls = request_pronunciations(chinese_words, API_KEY, 2)

        print('Request limit is reached!')
        request_timer()

        if mp3_urls:
            print('Saving data!')
            audio_list = save_pronunciations(mp3_urls)

            audio_df = build_dataframe(audio_list)
            print(f"# Total Samples: {len(audio_df)}.")
            file_path = PICKLE_PATH + 'audio_df.pkl'
            audio_df.to_pickle(file_path)
            print('Process Done!')
        else:
            print('Try again later!')
    else:
        print('API-KEY not found!')
        print('Insert API-KEY from https://api.forvo.com/ to request data!')

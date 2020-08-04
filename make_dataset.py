import pandas as pd
from src.data import make as mkds
from src import utils

print("Extracting words!")
chinese_words = mkds.extract_or_load_words()
utils.save_object(chinese_words, utils.PROCESSED_PATH + 'chinese_characters.txt')
chinese_df = pd.DataFrame(chinese_words, columns=['word'])
chinese_df['tone'] = chinese_df['word'].apply(lambda word: utils.text_to_tone(word))

if mkds.API_KEY:
    print('Requesting data!')
    mp3_urls = mkds.request_pronunciations(chinese_words, mkds.API_KEY, 2)

    print('Request limit is reached!')
    mkds.request_timer()

    if mp3_urls:
        print('Saving data!')
        audio_list = mkds.save_pronunciations(mp3_urls)

        audio_df = mkds.build_dataframe(audio_list)
        print(f"# Total Samples: {len(audio_df)}.")
        file_path = utils.PICKLE_PATH + 'audio_df.pkl'
        audio_df.to_pickle(file_path)
        print('Process Done!')
    else:
        print('Try again later!')
else:
    print('API-KEY not found!')
    print('Insert API-KEY from https://api.forvo.com/ to request data!')

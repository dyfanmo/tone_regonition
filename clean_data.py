import pandas as pd
from src.data import clean as cln
from src.data import make as mkds
from src import utils

audio_data = pd.read_pickle('data/processed/Pickle/audio_df.pkl')
audio_data = audio_data.dropna()

print('Removing Silence!')
df_count = len(audio_data)
audio_data = cln.remove_silence_save(audio_data)
audio_data = audio_data[audio_data.new_duration < 2]
print(f"# Samples Removed: {df_count - len(audio_data)} ")

print('Data Augmentation!')
audio_data_aug = cln.audio_augmentation(audio_data)
print(f"# Samples Added: {len(audio_data_aug) - len(audio_data)} ")

print('Removing Outliers!')
df_count = len(audio_data_aug)
audio_pca = cln.pca_audio(audio_data_aug)
audio_pca = cln.detect_outliers(audio_pca)
audio_inlier = audio_pca[audio_pca['anomaly'] == 1]
print(f"# Samples Removed: {df_count - len(audio_inlier)} ")


print('Speech Recognition Assessment!')
df_count = len(audio_inlier)
audio_cln = cln.speech_recognition_assessment(audio_inlier)
audio_best = audio_cln[audio_cln['sound_quality'] <= 1]
print(f"# Samples Removed: {df_count - len(audio_best)} ")

print('Sampling Data!')
chinese_words = mkds.extract_or_load_words()
tone_per = utils.get_tone_dist(chinese_words)
audio_pr = cln.sample_tone_per(audio_best, tone_per)

audio_pr.to_pickle("data/processed/Pickle/audio_pr.pkl")
print(f"# Data Set Size: {len(audio_pr)}")
print('Process Done!')
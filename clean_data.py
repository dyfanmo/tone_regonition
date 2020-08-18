import pandas as pd
from src.data.clean import remove_silence_save, audio_augmentation, pca_audio, detect_outliers, \
    speech_recognition_assessment

audio_data = pd.read_pickle('data/processed/Pickle/audio_df.pkl')
audio_data = audio_data.dropna()

print('Removing Silence!')
df_count = len(audio_data)
audio_data = remove_silence_save(audio_data)
audio_data = audio_data[audio_data.new_duration < 2]
print(f"# Samples Removed: {df_count - len(audio_data)} ")

print('Data Augmentation!')
audio_data_aug = audio_augmentation(audio_data)
print(f"# Samples Added: {len(audio_data_aug) - len(audio_data)} ")

print('Removing Outliers!')
df_count = len(audio_data_aug)
audio_pca = pca_audio(audio_data_aug)
audio_pca = detect_outliers(audio_pca)
audio_inlier = audio_pca[audio_pca['anomaly'] == 1]
print(f"# Samples Removed: {df_count - len(audio_inlier)} ")

print('Speech Recognition Assessment!')
df_count = len(audio_inlier)
audio_cln = speech_recognition_assessment(audio_inlier)
audio_best = audio_cln[audio_cln['sound_quality'] <= 1]
print(f"# Samples Removed: {df_count - len(audio_best)} ")

audio_best.reset_index(drop=True, inplace=True)
audio_best.to_pickle("data/processed/Pickle/audio_pr.pkl")
print(f"# Data Set Size: {len(audio_best)}")
print('Process Done!')
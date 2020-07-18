from src.data.clean_data import *


if __name__ == "__main__":

    audio_data = pd.read_pickle(f"{ROOT_DIR}/data/processed/Pickle/audio_df.pkl")
    audio_data = audio_data.dropna()

    print("Removing Silence!")
    df_count = len(audio_data)
    audio_data = remove_silence(audio_data)
    audio_data = audio_data[audio_data.new_duration < 2]
    print(f"# Samples Removed: {df_count - len(audio_data)} ")

    print("Data Augmentation!")
    audio_data_aug = audio_augmentation(audio_data)

    print("Removing Outliers!")
    df_count = len(audio_data_aug)
    audio_pca = pca_audio(audio_data_aug)
    audio_pca = detect_outliers(audio_pca)
    audio_inlier = audio_pca[audio_pca["anomaly"] == 1]
    audio_inlier.reset_index(inplace=True, drop=True)
    print(f"# Samples Removed: {df_count - len(audio_inlier)} ")

    print("Speech Recognition Assessment!")
    df_count = len(audio_inlier)
    audio_cln = speech_recognition_assessment(audio_inlier)
    audio_best = audio_cln[audio_cln["sound_quality"] <= 1]
    print(f"# Samples Removed: {df_count - len(audio_best)} ")

    print("Sampling Data!")
    chinese_words = extract_words(print_total=False)
    tone_per = get_tone_distrubtion(chinese_words)
    audio_pr = sample_tone_per(audio_best, tone_per)

    audio_pr.to_pickle(f"{ROOT_DIR}/data/processed/Pickle/audio_pr.pkl")
    print(f"# Data Set Size: {len(audio_pr)}")
    print("Process Done!")

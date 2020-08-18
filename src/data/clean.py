import os
import wave
from tqdm import tqdm
import librosa
import pandas as pd
import numpy as np
import hanzidentifier
import soundfile as sf
from os import listdir
from pydub import AudioSegment

from sklearn.decomposition import PCA
from pydub.silence import split_on_silence
from sklearn.ensemble import IsolationForest
from ..utils import match_target_amplitude, change_pitch, get_audio_path, speech_to_text, load_object, save_object, \
    PICKLE_PATH, CLEAN_PATH, PROCESSED_PATH, AUDIO_PATH, AUGMENTED_PATH, text_to_tone, get_melspectrogram_db

pd.options.mode.chained_assignment = None


def get_dir_files(dir_path):
    """ Return file names or make a dir """
    try:
        dir_files = listdir(dir_path)
    except FileNotFoundError:
        os.mkdir(dir_path)
        dir_files = []
    return dir_files


def save_silenced_audio(chunks, file_path):
    """ Save the silenced version of the audio files """
    for chunk in chunks:
        silence_chunk = AudioSegment.silent(duration=200)
        audio_chunk = silence_chunk + chunk + silence_chunk
        normalized_chunk = match_target_amplitude(audio_chunk, -20.0)

        normalized_chunk.export(file_path, bitrate="192k", format="wav")


def add_silenced_df(df):
    """ Add silenced audio information into the data frame"""
    df['new_nframes'] = df['id'].apply(lambda f: wave.open(CLEAN_PATH + f).getnframes())
    df['new_duration'] = df['id'].apply(lambda f: wave.open(CLEAN_PATH + f).getnframes() /
                                                  wave.open(CLEAN_PATH + f).getframerate())
    return df


def remove_silence_save(df):
    """ Remove silence and save audio """
    os.makedirs(PROCESSED_PATH + 'Audio', exist_ok=True)
    cln_files = get_dir_files(CLEAN_PATH)

    for index in tqdm(df.index):
        id_i = df['id'].loc[index]
        audio = AudioSegment.from_wav(os.path.join(AUDIO_PATH, id_i))
        file_path = os.path.join(CLEAN_PATH, id_i)

        if id_i not in cln_files:
            chunks = split_on_silence(audio, min_silence_len=300, silence_thresh=-30, keep_silence=300)

            if chunks:
                save_silenced_audio(chunks, file_path)
            else:
                df.drop(index=index, inplace=True)

    df = add_silenced_df(df)
    return df


def make_augmented_names(row):
    """ Make names for the augmented files"""
    id_name = row.id[6:]
    id_wn = f'audioWN_{id_name}'
    id_dp = f'audioDP_{id_name}'
    id_hi = f'audioHF_{id_name}'

    return id_wn, id_dp, id_hi


def manipulate_audio_data(wav, sample_rate):
    """ Make changes to the audio files """
    wn = np.random.randn(len(wav))
    wav_wn = wav + 0.005 * wn
    wav_dp = change_pitch(wav, sample_rate, deep=True)
    wav_hi = change_pitch(wav, sample_rate, deep=False)

    return wav_wn, wav_dp, wav_hi


def make_augmented_files(row, id_wn, id_dp, id_hi):
    """ Make the variations of the different augmented audio files """
    file_path = CLEAN_PATH + row.id
    wav, sample_rate = librosa.load(file_path)
    wav_wn, wav_dp, wav_hi = manipulate_audio_data(wav, sample_rate)
    wav_list = [(id_wn, wav_wn), (id_dp, wav_dp), (id_hi, wav_hi)]

    return wav_list, sample_rate


def save_aug(df, row, wav_list, sample_rate):
    """ Save augmented audio files """
    for wav_i in wav_list:
        id_i = wav_i[0]
        row['id'] = id_i
        row['audio_type'] = id_i[5:7]
        df = pd.concat([df, pd.DataFrame(row).T])
        sf.write(AUGMENTED_PATH + id_i, wav_i[1], sample_rate)
    return df


def append_previous_aug(df, row, *id_list):
    """ Append previous augmented data """
    for id_i in id_list:
        row['id'] = id_i
        row['audio_type'] = id_i[5:7]
        df = pd.concat([df, pd.DataFrame(row).T])
    return df


def save_aug_dataframe(df, df_aug):
    """ Save the augmented file's information to a data frame """
    df['audio_type'] = 'CL'
    df = pd.concat([df, df_aug])
    df.reset_index(drop=True, inplace=True)
    return df


def audio_augmentation(df):
    """
    Amusement the data by manipulating frequencies or injected noise
    Save the augmented data
     """
    aug_files = get_dir_files(AUGMENTED_PATH)
    df_aug = pd.DataFrame()

    for index in tqdm(df.index):
        row = df.loc[index]
        id_wn, id_dp, id_hi = make_augmented_names(row)

        if id_hi not in aug_files:
            wav_list, sr = make_augmented_files(row, id_wn, id_dp, id_hi)
            df_aug = save_aug(df_aug, row, wav_list, sr)

        else:
            df_aug = append_previous_aug(df_aug, row, id_wn, id_dp, id_hi)

    df = save_aug_dataframe(df, df_aug)
    return df


def get_transcripts(ser_i):
    """ Get the speech recognition transcripts from the audio files"""
    file_path = get_audio_path(ser_i)
    transcripts = speech_to_text(file_path)
    return transcripts


def get_cleaned_audio():
    """ Get information information about the previously cleaned audio """
    try:
        old_df = pd.read_pickle(PICKLE_PATH + 'audio_cln.pkl')
        cleaned_audio = load_object(PICKLE_PATH + 'cleaned_audio.pkl')
    except FileNotFoundError:
        old_df = {}
        cleaned_audio = []

    return old_df, cleaned_audio


def append_old_data_to_dataframe(df, df_cln, old_df, index, id_i):
    """ Add information from the old cleaned data set to the new one """
    df.drop(index=index, inplace=True)
    row = old_df.loc[old_df['id'] == id_i]
    df_cln = pd.concat([df_cln, row])
    return df, df_cln


def setup_clean_dataframe(df):
    """ Setup the columns and an empty data frame for Speech Recognition Analysis  """
    df['transcripts'] = np.nan
    df['sound_quality'] = np.nan
    df_cln = pd.DataFrame()

    return df, df_cln


def clean_transcripts(transcripts):
    """ Remove transcripts with more than one character or another language """
    cln_transcripts = []
    for text in transcripts:
        if pd.notnull(text):
            if len(text) <= 2:
                if hanzidentifier.has_chinese(text):
                    cln_transcripts.append(text)
    return cln_transcripts


def grade_audio_quality(df, index, transcripts):
    """ Grade the quality of the audio files based on the transcripts """

    word = df['word'].loc[index]
    tones = df['tones'].loc[index]

    pred_tones = np.array([text_to_tone(x) for x in transcripts])
    if word in transcripts:
        df['sound_quality'].loc[index] = 0
    elif tones in pred_tones:
        df['sound_quality'].loc[index] = 1
    else:
        df['sound_quality'].loc[index] = 2

    return df


def asses_audio_quality(df, index, transcripts):
    """
    Asses audio files based on the transcripts
    Remove audio files not suitable for training
    """
    if transcripts:
        cln_transcripts = clean_transcripts(transcripts)

        if cln_transcripts:
            df['transcripts'].loc[index] = cln_transcripts
            df = grade_audio_quality(df, index, cln_transcripts)
        else:
            df.drop(index=index, inplace=True)
    else:
        df.drop(index=index, inplace=True)

    return df


def speech_recognition_assessment(df):
    """
    Use speech recognition to listen to audio files to return a list of predicted words.
    If all predicted words are longer than 1 character or another language, the data is removed.
    If all predicted words or its tone is different from the audio data, the data is removed.
    """
    df, df_cln = setup_clean_dataframe(df)
    old_df, cleaned_audio = get_cleaned_audio()

    for index in tqdm(df.index):
        single_audio_file = df.loc[index]
        id_i = single_audio_file.id

        if id_i in cleaned_audio:
            df, df_cln = append_old_data_to_dataframe(df, df_cln, old_df, index, id_i)
        else:
            transcripts = get_transcripts(single_audio_file)
            df = asses_audio_quality(df, index, transcripts)
            cleaned_audio.append(id_i)

    df_after_audio_assessment = df
    if cleaned_audio:
        df_after_audio_assessment = pd.concat([df, df_cln])

    df_after_audio_assessment.to_pickle(PICKLE_PATH + 'audio_cln.pkl')
    save_object(cleaned_audio, PICKLE_PATH + 'cleaned_audio.pkl')

    return df_after_audio_assessment


def get_audio_features(df):
    """ Normalise and retrieve features from the spectrogram """
    fft_all = []
    for index in tqdm(df.index):
        ser_i = df.loc[index]
        file_path = get_audio_path(ser_i)
        specgram = get_melspectrogram_db(file_path, duration=2)
        fft_all.append(specgram)

    fft_all = np.array(fft_all)
    fft_all = (fft_all - np.mean(fft_all)) / np.std(fft_all)
    fft_all = fft_all.reshape(fft_all.shape[0], fft_all.shape[1] * fft_all.shape[2])

    return fft_all


def pca_audio(df):
    """ Perform PCA on the audio and return data into data frame """
    fft_all = get_audio_features(df)
    pca = PCA(n_components=3)
    fft_all = pca.fit_transform(fft_all)
    df['PC1'] = fft_all[:, 0]
    df['PC2'] = fft_all[:, 1]
    df['PC3'] = fft_all[:, 2]

    return df


def detect_outliers(df):
    """ Classify the outliers in the data set """
    fft_all = df[['PC1', 'PC2', 'PC3']].to_numpy()
    clf = IsolationForest(n_estimators=200, max_samples='auto', contamination=float(.01),
                          max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
    clf.fit(fft_all)

    anomaly = clf.predict(fft_all)
    df['anomaly'] = anomaly
    return df


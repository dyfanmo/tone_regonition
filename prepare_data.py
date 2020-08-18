import pandas as pd
from src.utils import PICKLE_PATH
from src.data.prepare import split_data, build_training_data, save_training_data

file_path = PICKLE_PATH + 'audio_pr.pkl'
audio_pr = pd.read_pickle(file_path)

print('Train Test Split!')
train, valid, test = split_data(audio_pr, 0.05, 0.10)

print("Preparing Data!")
train_loader, valid_loader, test_loader = build_training_data(train, valid, test, 32, 32, 32)
save_training_data(train_loader, valid_loader, test_loader)

print('Process Done!')
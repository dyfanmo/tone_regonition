import sys
import pandas as pd
import os.path as path
from torch.utils.data import DataLoader


ROOT_DIR = path.abspath(path.join(__file__ ,"../../.."))
sys.path.insert(2, ROOT_DIR)
from src.configs.data import *
import src.utils as utils

INTERIM_PATH = f'{ROOT_DIR}/data/interim/'


def split_data(df, test_per, val_per):
    TEST_PCT = test_per
    VAL_PCT = val_per
    test_size = int(len(df) * TEST_PCT)
    val_size = int(len(df) * VAL_PCT)
    test = df.iloc[:test_size]
    valid = df.iloc[test_size:test_size + val_size]
    train = df.iloc[val_size + test_size:]
    print(f"# Train Size: {len(train)}")
    print(f"# Valid Size: {len(valid)}")
    print(f"# Test Size: {len(test)}")
    return train, valid, test


def prepare_data(train, valid, test, duration=2, train_batch_size=16, val_batch_size=16, test_batch_size=16):
    train_data = utils.ToneData(train, 'labels', duration)
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)

    valid_data = utils.ToneData(valid, 'labels', duration)
    valid_loader = DataLoader(valid_data, batch_size=val_batch_size, shuffle=True)

    test_data = utils.ToneData(test, 'labels', duration)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    config = get_config()
    audio_pr = pd.read_pickle(f"{ROOT_DIR}/data/processed/audio_pr.pkl")

    print('Train Test Split!')
    train, valid, test = split_data(audio_pr, config.test_per, config.val_per)

    print("Preparing Data!")
    train_loader, valid_loader, test_loader = prepare_data(train, valid, test, 2, config.train_batch_size,
                                                    config.val_batch_size, config.test_batch_size)

    utils.save_object(train_loader, f'{INTERIM_PATH}/train_loader.pkl')
    utils.save_object(valid_loader, f'{INTERIM_PATH}/valid_loader.pkl')
    utils.save_object(test_loader, f'{INTERIM_PATH}/test_loader.pkl')

    print('Process Done!')

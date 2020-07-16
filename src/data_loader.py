import os
from torch.utils.data import DataLoader

from configs.data import *
from utils import *


def split_data(df, test_per, val_per):
    df = df.sample(frac=1)
    test_size = int(len(df) * test_per)
    val_size = int(len(df) * val_per)
    test_set = df.iloc[:test_size]
    valid_set = df.iloc[test_size:test_size + val_size]
    train_set = df.iloc[val_size + test_size:]
    print(f"# Train Size: {len(train_set)}")
    print(f"# Valid Size: {len(valid_set)}")
    print(f"# Test Size: {len(test_set)}")
    return train_set, valid_set, test_set


def prepare_data(train_set, valid_set, test_set, train_batch_size, val_batch_size, test_batch_size, duration=2):
    train_data = ToneData(train_set, 'labels', duration)
    train_pr = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)

    valid_data = ToneData(valid_set, 'labels', duration)
    valid_pr = DataLoader(valid_data, batch_size=val_batch_size, shuffle=True)

    test_data = ToneData(test_set, 'labels', duration)
    test_pr = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

    if not os.path.isdir(INTERIM_PATH):
        os.mkdir(INTERIM_PATH)

    save_object(train_pr, f'{INTERIM_PATH}/train_loader.pkl')
    save_object(valid_pr, f'{INTERIM_PATH}/valid_loader.pkl')
    save_object(test_pr, f'{INTERIM_PATH}/test_loader.pkl')

    return train_pr, valid_pr, test_pr


if __name__ == "__main__":
    config = get_config()
    audio_pr = pd.read_pickle(f"{ROOT_DIR}/data/processed/Pickle/audio_pr.pkl")

    print('Train Test Split!')
    train, valid, test = split_data(audio_pr, config.test_per, config.val_per)

    print("Preparing Data!")
    train_loader, valid_loader, test_loader = prepare_data(train, valid, test, config.train_batch_size,
                                                           config.val_batch_size, config.test_batch_size)

    print('Process Done!')

import os
from torch.utils.data import DataLoader

from ..utils import MultiTaskDataset, save_object, INTERIM_PATH


def split_data(df, test_per, val_per):
    """ Split the data into a train, valid and test set"""
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


def build_training_data(train_set, valid_set, test_set, train_batch_size, val_batch_size, test_batch_size, duration=2):
    """ Covert the audio samples into training data"""
    train_data = MultiTaskDataset(train_set, duration)
    train_pr = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=2)

    valid_data = MultiTaskDataset(valid_set, duration)
    valid_pr = DataLoader(valid_data, batch_size=val_batch_size, shuffle=True, num_workers=2)

    test_data = MultiTaskDataset(test_set, duration)
    test_pr = DataLoader(test_data, batch_size=test_batch_size, shuffle=True, num_workers=2)

    return train_pr, valid_pr, test_pr


def save_training_data(train, valid, test):
    """ Save training data """
    os.makedirs(INTERIM_PATH, exist_ok=True)

    save_object(train, INTERIM_PATH + 'train_loader.pkl')
    save_object(valid, INTERIM_PATH + 'valid_loader.pkl')
    save_object(test, INTERIM_PATH + 'test_loader.pkl')
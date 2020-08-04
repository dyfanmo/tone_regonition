import os
import torch
import time
import getpass
import random
import numpy as np
import multiprocessing
from torch import nn

from src import utils
from src.configs import get_config
from src.model.train import train_model
from src.model.models import SimpleCNN, ComplexCNN, DeepCNN

config = get_config()
print(config)

time_now = time.strftime("%Y-%m-%d_%H:%M:%S")
print(time_now)

print('Current directory:', os.getcwd())
print('Current user:', getpass.getuser())

print('PyTorch Version:', torch.__version__)
print('# CPUs:', multiprocessing.cpu_count())
print('# GPUs:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('Current cuda device:', torch.cuda.current_device())
    print('Device name:', torch.cuda.get_device_name(0))

print('Seed:', config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)

train_loader = utils.load_object(utils.INTERIM_PATH + "train_loader.pkl")
valid_loader = utils.load_object(utils.INTERIM_PATH + "valid_loader.pkl")

loss_func = nn.CrossEntropyLoss()
simpleCNN = SimpleCNN()
complexCNN = ComplexCNN()
deepCNN = DeepCNN()

if config.model == 'ALL':
    for model in [simpleCNN, complexCNN, deepCNN]:
        train_model(config, model, loss_func, train_loader, valid_loader)
elif config.model == 'SIM':
    train_model(config, simpleCNN, loss_func, train_loader, valid_loader)
elif config.model == 'COM':
    train_model(config, complexCNN, loss_func, train_loader, valid_loader)
elif config.model == 'DEP':
    train_model(config, deepCNN, loss_func, train_loader, valid_loader)


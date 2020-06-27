import os
import sys
import tqdm
import torch
import time
import getpass
import random
import numpy as np
import os.path as path
import multiprocessing
from torch import optim
from torch import nn

ROOT_DIR = path.abspath(path.join(__file__ ,"../../.."))
sys.path.insert(3, ROOT_DIR)
from src.utils import *
from src.configs.train import *
from src.models.models import SimpleCNN, ComplexCNN, DeepCNN

MODELS_PATH = f"{ROOT_DIR}/models"
INTERIM_PATH = f"{ROOT_DIR}/data/interim"
PROCESSED_PATH = f"{ROOT_DIR}/data/processed"


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_model(config, model, loss_fn, train_loader, valid_loader):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model.to(device)
    if torch.cuda.is_available():
        print(f'Loaded model on {torch.cuda.device_count()} GPUs!')

    model_name = model.__class__.__name__
    model_name = str(model_name.strip())
    print(f'#===== {model_name} =====#')

    print('#===== Parameters =====#')
    for name, p in model.named_parameters():
        print(name, '\t', list(p.size()))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('# Total Parameters:', count_parameters(model))

    print('#==== Weight Initialization ====#')

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')

    model.apply(weights_init)

    # ------ Optimizer ------#

    params = model.parameters()
    if config.optim.lower() == 'adam':
        optimizer = optim.Adam(params, lr=config.lr, )
    elif config.optim.lower() == 'amsgrad':
        optimizer = optim.Adam(params, lr=config.lr)
    elif config.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(params, lr=config.lr)

    print('Training starts!')
    train_losses = []
    valid_losses = []
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)
    best_acc = 0.0
    val_loss_min = np.Inf

    for epoch in tqdm(range(1, config.epochs + 1)):
        model.train()

        # ------ Learning Rate Decay ------#
        def setlr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return optimizer

        def lr_decay(optimizer, epoch):
            if epoch % 20 == 0:
                new_lr = config.lr / (10 ** (epoch // 20))
                optimizer = setlr(optimizer, new_lr)
                print(f'Changed learning rate to {new_lr}')
            return optimizer

        if config.lr_decay:
            optimizer = lr_decay(optimizer, epoch)

        batch_losses = []
        for i, data in enumerate(train_loader):
            x, y = data
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        train_losses.append(batch_losses)
        print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')

        model.eval()
        trace_y = []
        trace_yhat = []
        batch_losses = []
        for i, data in enumerate(valid_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())
            batch_losses.append(loss.item())
        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        valid_acc = np.mean(trace_yhat.argmax(axis=1) == trace_y)
        valid_loss = np.mean(valid_losses[-1])
        print(f'Epoch - {epoch} Valid-Loss :{valid_loss} Valid-Accuracy : {valid_acc}')

        # ------ Checkpoint ------#
        filename = f"{MODELS_PATH}/{model_name}.pt"
        if valid_loss < val_loss_min:
            torch.save(model.state_dict(), filename)
            val_loss_min = valid_loss

        if valid_acc > best_acc:
            best_acc = valid_acc

        # ------ Early Stopping ------#
        if config.early_stop:
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    tl = np.mean(train_losses, axis=1)
    vl = np.mean(valid_losses, axis=1)
    np.save(f'{PROCESSED_PATH}/scores/tl-{model_name[:3]}.npy', tl)
    np.save(f'{PROCESSED_PATH}/scores/vl-{model_name[:3]}.npy', vl)
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Min val Loss: {:4f}'.format(val_loss_min))
    return model


if __name__ == "__main__":
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

    train_loader = load_object(f"{INTERIM_PATH}/train_loader.pkl")
    valid_loader = load_object(f"{INTERIM_PATH}/valid_loader.pkl")

    loss_fn = nn.CrossEntropyLoss()
    simpleCNN = SimpleCNN()
    complexCNN = ComplexCNN()
    deepCNN = DeepCNN()

    if config.model == 'ALL':
        for model in [simpleCNN, complexCNN, deepCNN]:
            train_model(config, model, loss_fn, train_loader, valid_loader)
    elif config.model == 'SIM':
        train_model(config, simpleCNN, loss_fn, train_loader, valid_loader)
    elif config.model == 'COM':
        train_model(config, complexCNN, loss_fn, train_loader, valid_loader)
    elif config.model == 'DEP':
        train_model(config, deepCNN, loss_fn, train_loader, valid_loader)






import os
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from tqdm import tqdm

from .. import utils


if utils.isnotebook():
    from tqdm.notebook import tqdm


class EarlyStopping:
    """ Perform early stopping during training """
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


def count_parameters(model):
    """ Count the parameters in a model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weights_init(m):
    """ Initialise weights """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')


def choose_optimiser(model, optim_name, lr):
    """ Choose an optimiser """
    params = model.parameters()
    if optim_name.lower() == 'adam':
        optimiser = optim.Adam(params, lr=lr)
    elif optim_name.lower() == 'amsgrad':
        optimiser = optim.Adam(params, lr=lr)
    elif optim_name.lower() == 'adagrad':
        optimiser = optim.Adagrad(params, lr=lr)
    return optimiser


def save_checkpoint(model, val_loss_min, valid_loss, valid_acc, best_acc):
    """ Save model each time the validation loss decreases """
    os.makedirs(utils.MODELS_PATH, exist_ok=True)
    model_name = model.__class__.__name__
    filename = utils.MODELS_PATH + model_name + ".pt"
    if valid_loss < val_loss_min:
        torch.save(model.state_dict(), filename)
        val_loss_min = valid_loss

    if valid_acc > best_acc:
        best_acc = valid_acc

    return val_loss_min, best_acc


def print_training_info(model, print_info):
    if print_info:
        if torch.cuda.is_available():
            print(f'Loaded model on {torch.cuda.device_count()} GPUs!')

    model_name = model.__class__.__name__
    model_name = str(model_name.strip())
    if print_info:
        print(f'#===== {model_name} =====#')

        print('#===== Parameters =====#')
        for name, p in model.named_parameters():
            print(name, '\t', list(p.size()))

    if print_info:
        print('# Total Parameters:', count_parameters(model))

        print('#==== Weight Initialization ====#')
    model.apply(weights_init)

    if print_info:
        print('Training starts!')


def train_model(configs, model, loss_func, train_pr, valid_pr, print_info=True, print_epoch=1):
    """ Train and save model """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model.to(device)

    model_name = model.__class__.__name__
    model_name = str(model_name.strip())

    print_training_info(model, print_info)

    # ------ Optimizer ------#
    optimizer = choose_optimiser(model, configs.optim, configs.lr)

    train_losses = []
    valid_losses = []
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True)
    best_acc = 0.0
    val_loss_min = np.Inf

    for epoch in tqdm(range(1, configs.epochs + 1)):
        if epoch % print_epoch == 0 or epoch == 1:
            print(f'Epoch: {epoch}/{configs.epochs}')
        model.train()

        batch_losses = []
        for i, data in enumerate(train_pr):
            x, y = data
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_func(y_hat, y)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        train_losses.append(batch_losses)

        model.eval()
        trace_y = []
        trace_yhat = []
        batch_losses = []
        for i, data in enumerate(valid_pr):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_func(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())
            batch_losses.append(loss.item())
        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        valid_acc = np.mean(trace_yhat.argmax(axis=1) == trace_y)
        valid_loss = np.mean(valid_losses[-1])

        if epoch % print_epoch == 0 or epoch == 1:
            print(f'Train-Loss : {np.mean(train_losses[-1])}\tValid-Loss :{valid_loss}\tValid-Accuracy : {valid_acc}')

        # ------ Checkpoint ------#
        val_loss_min, best_acc = save_checkpoint(model, val_loss_min, valid_loss, valid_acc, best_acc)
        # ------ Early Stopping ------#
        if configs.early_stop:
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    tl = np.mean(train_losses, axis=1)
    vl = np.mean(valid_losses, axis=1)

    os.makedirs(utils.SCORES_PATH, exist_ok=True)

    np.save(f'{utils.SCORES_PATH}tl-{model_name[:4].upper()}.npy', tl)
    np.save(f'{utils.SCORES_PATH}vl-{model_name[:4].upper()}.npy', vl)
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Min val Loss: {:4f}'.format(val_loss_min))







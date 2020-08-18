import os
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from tqdm import tqdm

from ..utils import isnotebook, MODELS_PATH, SCORES_PATH


if isnotebook():
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


def save_checkpoint(model, val_loss_min, valid_loss, valid_acc1, valid_acc2, best_acc):
    """ Save model each time the validation loss decreases """
    os.makedirs(MODELS_PATH, exist_ok=True)

    model_name = model.__class__.__name__
    filename = MODELS_PATH + model_name + ".pt"
    valid_acc = (valid_acc1 + valid_acc2) / 2
    if valid_loss < val_loss_min:
        torch.save(model.state_dict(), filename)
        val_loss_min = valid_loss

    if valid_acc > best_acc:
        best_acc = valid_acc

    return val_loss_min, best_acc


def print_training_info(model, print_info):
    """ Print the training information """
    if print_info:
        if torch.cuda.is_available():
            print(f'Loaded model on {torch.cuda.device_count()} GPUs!')

        model_name = model.__class__.__name__
        model_name = str(model_name.strip())

        print(f'#===== {model_name} =====#')

        print('#===== Parameters =====#')
        for name, p in model.named_parameters():
            print(name, '\t', list(p.size()))

        print('# Total Parameters:', count_parameters(model))

        print('#==== Weight Initialization ====#')


def choose_device():
    """ Set the device type to train on """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    return device


def setup_train_variables():
    """ Set up the variables needed to train a model"""
    train_losses = []
    valid_losses = []
    best_acc = 0.0
    val_loss_min = np.Inf

    return train_losses, valid_losses, best_acc, val_loss_min


def compute_multi_loss(loss_func, y1, y2, yhat1, yhat2):
    """ Compute the average loss of two y values """
    l1 = loss_func(yhat1, y1)
    l2 = loss_func(yhat2, y2)
    loss = (l1 + l2) / 2
    return loss


def compute_train_set(model, optimizer, loss_func, device, train_set, train_losses):
    """ Compute and return the losses of the train set """
    model.train()
    batch_losses = []
    for i, data in enumerate(train_set):
        x, y = data
        x = x.to(device, dtype=torch.float32)
        y_a = y[:, 0].to(device, dtype=torch.long)
        y_b = y[:, 1].to(device, dtype=torch.long)
        optimizer.zero_grad()
        y_hat1, y_hat2 = model(x)
        loss = compute_multi_loss(loss_func, y_a, y_b, y_hat1, y_hat2)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_losses.append(batch_losses)

    return model, train_losses


def append_trace_y_values(y1, y2, yhat1, yhat2, trace_y1, trace_y2, trace_yhat1, trace_yhat2):
    """ Append trace y values """
    trace_y1.append(y1.cpu().detach().numpy())
    trace_yhat1.append(yhat1.cpu().detach().numpy())
    trace_y2.append(y2.cpu().detach().numpy())
    trace_yhat2.append(yhat2.cpu().detach().numpy())

    return trace_y1, trace_y2, trace_yhat1, trace_yhat2


def setup_trace_y_variables():
    """ Set up all trace y variables """
    trace_y1 = []
    trace_yhat1 = []
    trace_y2 = []
    trace_yhat2 = []
    return trace_y1, trace_y2, trace_yhat1, trace_yhat2


def concat_trace_y_values(trace_y1, trace_y2, trace_yhat1, trace_yhat2):
    """ Concatenate the trace y variables """
    trace_y1 = np.concatenate(trace_y1)
    trace_y2 = np.concatenate(trace_y2)
    trace_yhat1 = np.concatenate(trace_yhat1)
    trace_yhat2 = np.concatenate(trace_yhat2)
    return trace_y1, trace_y2, trace_yhat1, trace_yhat2


def compute_val_set(model, loss_func, device, valid_set, valid_losses):
    """ Compute and return the scores of the validation set """
    model.eval()
    trace_y1, trace_y2, trace_yhat1, trace_yhat2 = setup_trace_y_variables()
    batch_losses = []
    for i, data in enumerate(valid_set):
        x, y = data
        x = x.to(device, dtype=torch.float32)
        y1 = y[:, 0].to(device, dtype=torch.long)
        y2 = y[:, 1].to(device, dtype=torch.long)
        yhat1, yhat2 = model(x)
        loss = compute_multi_loss(loss_func, y1, y2, yhat1, yhat2)
        trace_y1, trace_y2, trace_yhat1, trace_yhat2 = append_trace_y_values(y1, y2, yhat1, yhat2, trace_y1, trace_y2,
                                                                             trace_yhat1, trace_yhat2)
        batch_losses.append(loss.item())
    valid_losses.append(batch_losses)
    trace_y1, trace_y2, trace_yhat1, trace_yhat2 = concat_trace_y_values(trace_y1, trace_y2, trace_yhat1, trace_yhat2)
    valid_acc1 = np.mean(trace_yhat1.argmax(axis=1) == trace_y1)
    valid_acc2 = np.mean(trace_yhat2.argmax(axis=1) == trace_y2)
    valid_loss = np.mean(valid_losses[-1])

    return model, valid_loss, valid_acc1, valid_acc2, valid_losses


def print_iteration(epoch, print_epoch, configs, train_losses, valid_loss=np.zeros(0), tone_a_valid_acc=np.zeros(0),
                    tone_b_valid_accuracy=np.zeros(0)):
    """ Print epoch information after a certain number of iterations """
    if epoch % print_epoch == 0 or epoch == 1:
        if valid_loss.size == 0:
            print(f'Epoch: {epoch}/{configs.epochs}')
        else:
            print(f'Train-Loss: {np.mean(train_losses[-1])}\tValid-Loss: {valid_loss}')
            print(f'ToneA Valid-Accuracy: {tone_a_valid_acc}\tToneB Valid Accuracy: {tone_b_valid_accuracy}')


def perform_earlystopping(model, configs, early_stopping, valid_loss):
    """ Return boolean whether or not to perform early stopping """
    if configs.early_stop:
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            return True
    return False


def save_model_scores(train_losses, valid_losses, model_name):
    """ Save the loss scores of the model """
    tl = np.mean(train_losses, axis=1)
    vl = np.mean(valid_losses, axis=1)

    os.makedirs(SCORES_PATH, exist_ok=True)

    np.save(f'{SCORES_PATH}tl-{model_name[:4].upper()}.npy', tl)
    np.save(f'{SCORES_PATH}vl-{model_name[:4].upper()}.npy', vl)


def train_model(configs, model, loss_func, train_pr, valid_pr, print_info=True, print_epoch=1):
    """ Train and save model """
    device = choose_device()
    model.to(device)

    model_name = model.__class__.__name__
    print_training_info(model, print_info)
    model.apply(weights_init)
    if print_info:
        print('Training starts!')

    optimizer = choose_optimiser(model, configs.optim, configs.lr)
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True)

    train_losses, valid_losses, best_acc, val_loss_min = setup_train_variables()

    for epoch in tqdm(range(1, configs.epochs + 1)):
        print_iteration(epoch, print_epoch, configs, train_losses)

        model, train_losses = compute_train_set(model, optimizer, loss_func, device, train_pr, train_losses)
        model, valid_loss, valid_acc1, valid_acc2, valid_losses = compute_val_set(model, loss_func, device,
                                                                                              valid_pr, valid_losses)

        print_iteration(epoch, print_epoch, configs, train_losses, valid_loss, valid_acc1,
                    valid_acc2)

        val_loss_min, best_acc = save_checkpoint(model, val_loss_min, valid_loss, valid_acc1, valid_acc2, best_acc)

        if perform_earlystopping(model, configs, early_stopping, valid_loss):
            break

    save_model_scores(train_losses, valid_losses, model_name)

    print('Best val Acc: {:4f}'.format(best_acc))
    print('Min val Loss: {:4f}'.format(val_loss_min))
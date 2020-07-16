import tqdm
import os
import torch
import time
import getpass
import random
import multiprocessing
from torch import optim
from torch import nn

from utils import *
from configs.train import *
from models import *

if __name__ != "__main__":
    from tqdm.notebook import tqdm


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


def train_model(configs, model, loss_func, train_pr, valid_pr, print_info=True, print_epoch=1):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model.to(device)

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

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if print_info:
        print('# Total Parameters:', count_parameters(model))

        print('#==== Weight Initialization ====#')

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')

    model.apply(weights_init)

    # ------ Optimizer ------#

    params = model.parameters()
    if configs.optim.lower() == 'adam':
        optimizer = optim.Adam(params, lr=configs.lr)
    elif configs.optim.lower() == 'amsgrad':
        optimizer = optim.Adam(params, lr=configs.lr)
    elif configs.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(params, lr=configs.lr)

    if print_info:
        print('Training starts!')

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
            print(
                f'Train-Loss : {np.mean(train_losses[-1])}\tValid-Loss :{valid_loss}\tValid-Accuracy : {valid_acc}')

        # ------ Checkpoint ------#
        if not os.path.isdir(MODELS_PATH):
            os.mkdir(MODELS_PATH)
        filename = f"{MODELS_PATH}/{model_name}.pt"
        if valid_loss < val_loss_min:
            torch.save(model.state_dict(), filename)
            val_loss_min = valid_loss

        if valid_acc > best_acc:
            best_acc = valid_acc

        # ------ Early Stopping ------#
        if configs.early_stop:
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    tl = np.mean(train_losses, axis=1)
    vl = np.mean(valid_losses, axis=1)

    if not os.path.isdir(SCORES_PATH):
        os.mkdir(SCORES_PATH)

    np.save(f'{SCORES_PATH}/tl-{model_name[:4].upper()}.npy', tl)
    np.save(f'{SCORES_PATH}/vl-{model_name[:4].upper()}.npy', vl)
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Min val Loss: {:4f}'.format(val_loss_min))


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






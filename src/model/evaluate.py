import torch
from tqdm import tqdm
import numpy as np

from .. import utils

device = torch.device('cpu')

if utils.isnotebook():
    from tqdm.notebook import tqdm


def test_models(test_pr, *models):
    """ Evaluate the models in the test set """
    results = {}
    for model in models:
        model_name = model.__class__.__name__
        model_name = str(model_name.strip())
        model.load_state_dict(torch.load(f'{utils.MODELS_PATH}/{model_name}.pt', map_location='cpu'))
        with torch.no_grad():
            trace_y = []
            trace_yhat = []
            for data in tqdm(test_pr):
                x, y = data
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.long)
                y_hat = model(x)  # returns a list,
                trace_y.append(y.cpu().detach().numpy())
                trace_yhat.append(y_hat.cpu().detach().numpy())
            trace_y = np.concatenate(trace_y)
            trace_yhat = np.concatenate(trace_yhat)
            accuracy = np.mean(trace_yhat.argmax(axis=1) == trace_y)
            acc_score = {model_name: accuracy}
            results.update(acc_score)
    return results




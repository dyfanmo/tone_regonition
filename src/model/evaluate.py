import torch
from tqdm import tqdm
import numpy as np

from .train import append_trace_y_values, concat_trace_y_values, setup_trace_y_variables
from ..utils import isnotebook, MODELS_PATH

device = torch.device('cpu')

if isnotebook():
    from tqdm.notebook import tqdm


def load_trained_mddel(model):
    model_name = model.__class__.__name__
    model_name = str(model_name.strip())
    model.load_state_dict(torch.load(f'{MODELS_PATH}/{model_name}.pt', map_location='cpu'))
    return model


def test_models(test_pr, *models):
    """ Evaluate the models in the test set """
    results = []
    for model in models:
        model_name = model.__class__.__name__
        model = load_trained_mddel(model)
        with torch.no_grad():
            trace_y1, trace_y2, trace_yhat1, trace_yhat2 = setup_trace_y_variables()
            for data in tqdm(test_pr):
                x, y = data
                x = x.to(device, dtype=torch.float32)
                y1 = y[:, 0].to(device, dtype=torch.long)
                y2 = y[:, 1].to(device, dtype=torch.long)
                yhat1, yhat2 = model(x)
                trace_y1, trace_y2, trace_yhat1, trace_yhat2 = append_trace_y_values(y1, y2, yhat1, yhat2, trace_y1,
                                                                                     trace_y2,
                                                                                     trace_yhat1, trace_yhat2)
            trace_y1, trace_y2, trace_yhat1, trace_yhat2 = concat_trace_y_values(trace_y1, trace_y2, trace_yhat1,
                                                                                 trace_yhat2)

            valid_acc1 = np.mean(trace_yhat1.argmax(axis=1) == trace_y1)
            valid_acc2 = np.mean(trace_yhat2.argmax(axis=1) == trace_y2)
            avg_acc = (valid_acc1 + valid_acc2)/2
            acc_score = {model_name: {"ToneA Acc": valid_acc1, 'ToneB Acc': valid_acc2, 'Avg Acc': avg_acc}}
            results.append(acc_score)
    return results




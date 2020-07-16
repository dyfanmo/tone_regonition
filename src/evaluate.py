import tqdm
import torch

from utils import *
from configs.train import *
from models import *

device = torch.device('cpu')

if __name__ != "__main__":
    from tqdm.notebook import tqdm


def test_model(model, test_pr):
    model_name = model.__class__.__name__
    model_name = str(model_name.strip())
    model.load_state_dict(torch.load(f'{MODELS_PATH}/{model_name}.pt', map_location='cpu'))
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
    return acc_score


if __name__ == "__main__":
    config = get_config()
    test_loader = load_object(f"{INTERIM_PATH}/test_loader.pkl")

    simpleCNN = SimpleCNN()
    complexCNN = ComplexCNN()
    deepCNN = DeepCNN()

    results = {}
    print('Evaluation start!')
    if config.model == 'ALL':
        for model in [simpleCNN, complexCNN, deepCNN]:
            result = test_model(model, test_loader)
            results.update(result)
    elif config.model == 'SIM':
        result = test_model(simpleCNN, test_loader)
        results.update(result)
    elif config.model == 'COM':
        result = test_model(complexCNN, test_loader)
        results.update(result)
    elif config.model == 'DEP':
        result = test_model(deepCNN, test_loader)
        results.update(result)

    print('Test Set Evaluation Result!')
    print('Accuracy Score')
    for name, score in results.items():
        print(f"{name}: " + "{0:.0%}".format(score))

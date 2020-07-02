import sys
import tqdm
import os.path as path

ROOT_DIR = path.abspath(path.join(__file__ ,"../../.."))

sys.path.insert(1, ROOT_DIR)
from src.utils import *
from src.configs.train import *
from src.models.models import *


device = torch.device('cpu')


def test_model(model, test_loader):
    name = model.__class__.__name__
    name = str(name.strip())
    model.load_state_dict(torch.load(f'{MODELS_PATH}/{name}.pt',  map_location='cpu'))
    with torch.no_grad():
        trace_y = []
        trace_yhat = []
        for data in tqdm(test_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)  # returns a list,
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        accuracy = np.mean(trace_yhat.argmax(axis=1) == trace_y)
        results = {name: accuracy}
    return results


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

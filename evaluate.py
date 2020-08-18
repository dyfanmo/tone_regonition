from src import utils
from src.model.evaluate import test_models
from src.configs import get_config
from src.model.models import SimpleCNN, ComplexCNN, DeepCNN

config = get_config()
test_loader = utils.load_object(utils.INTERIM_PATH + "test_loader.pkl")

simpleCNN = SimpleCNN()
complexCNN = ComplexCNN()
deepCNN = DeepCNN()

print('Evaluation start!')
if config.model == 'ALL':
    results = test_models(test_loader, simpleCNN, complexCNN, deepCNN)
elif config.model == 'SIM':
    results = test_models(test_loader, simpleCNN)
elif config.model == 'COM':
    results = test_models(test_loader, complexCNN)
elif config.model == 'DEP':
    results = test_models(test_loader, deepCNN)

print('Test Set Evaluation Result!')
print('Accuracy Score')
for result in results:
    print(result)
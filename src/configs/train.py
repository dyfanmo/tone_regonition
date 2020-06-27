import argparse
import pprint
import yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


def read_config(path):
    return Config.load(path)


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', type=str, default='ALL',
                        choices=['SIM', 'COM', 'DEP', 'ALL'],
                        help='SIM: SimpleCNN / COM: ComplexCNN / DEP: DeepCNN / ALL: All models')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='num_epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='early stopping')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='learning rate')
    parser.add_argument('--optim', type=str, default='adam',
                        choices=['adam', 'amsgrad', 'adagrad'],
                        help='optimizer')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')
    parser.add_argument('--early_stop', type=str2bool, default=True,
                        help='implementing early stopping')
    parser.add_argument('--lr_decay', type=str2bool, default=False,
                        help='learning rate decay')

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()

    # Save
    config.save('config.txt')

    # Load
    loaded_config = read_config('config.txt')

    assert config.__dict__ == loaded_config.__dict__

    import os

    os.remove('config.txt')

import argparse

from training.trainer import Trainer, ModelArch
from training_config import NeuralNetConfig
from customer.customers import CustomerObjectMap

parser = argparse.ArgumentParser(description='Command line interface to train various '
                                             'viNet models')
parser.add_argument('--train_config', '-tc',
                    default=r"E:\viNet_RnD\Training\Config\v3\vinet.settings.henok.ini",
                    help='Training configuration file (.ini)')

parser.add_argument('--version', '-v', default="v3",
                    help="vinet version to train (v2, v3)")
parser.add_argument('--customer', '-c', default='GWA', help="The customer or site for this network")
parser.add_argument('--epochs', '-e', default=1, help="Training epochs")


if __name__ == "__main__":
    """
    NOTE - this training script has not been validated for site deployment. 
    """
    args = parser.parse_args()
    settings_file = args.train_config
    assert args.customer in CustomerObjectMap.keys()
    customer = CustomerObjectMap[args.customer]

    trainer = None
    config = None
    models = ['B0', 'B2', 'B3', 'InceptionV3' 'Xception']
    if args.version == 'v3':
        config = NeuralNetConfig(settings_file)
        for model in models:
            config.set_net(model)
            trainer = Trainer(ModelArch.EfficientNet, customer)
            trainer.run(config)
    elif args.version == 'v2':
        trainer = Trainer(ModelArch.Inception, customer)

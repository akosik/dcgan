"""
DCGan MNIST Trainer

Architecture guidelines for stable Deep Convolutional GANs from paper:
- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided
convolutions (generator).
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.
"""
import sys
import yaml
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from models.generator import Generator
from models.discriminator import Discriminator
from train import train
from utils.weight_init import normal_initialization as norm_init_closure

logging.basicConfig(filename='log/train.log', level=logging.INFO)
root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

p = argparse.ArgumentParser(description="DCGan")
p.add_argument('-c', '--config', required=True,
      help='Config file path')
p.add_argument('--cuda', action='store_true',
      help='Use CUDA (false)')
p.add_argument('--num-workers', type=int, default=0,
               help='Num data loaders (0, load on main thread)')
p.add_argument('--logging-interval', type=int, default=10,
               help='Num batches between logging outputs (10)')
p.add_argument('--image-gen-interval', type=int, default=1000,
               help='Num batches between saving generator output sample (1000)')

args = p.parse_args()
with open(args.config,'r') as conf:
    config = yaml.load(conf)

torch.manual_seed(config['seed'])

if torch.cuda.is_available(): torch.cuda.manual_seed_all(config['seed'])

def main():
    device = torch.device("cuda" if args.cuda else "cpu" )

    train_loader = DataLoader(
        datasets.MNIST("data/mnist",
                       train=True,
                       download=True,
                       transform=transforms.Compose(
                           [transforms.Resize(config['data']['dim']),
                            transforms.ToTensor(),
                            transforms.Normalize([config['data']['normalize_to_mean']],
                                                 [config['data']['normalize_to_std']])])),
        batch_size=config['hparams']['bsz'],
        shuffle=True)

    models = {'gen': Generator(config).to(device),
              'discrim': Discriminator(config).to(device)}

    # init weights
    norm_init = norm_init_closure(config)
    for mk in models.keys():
        models[mk].apply(norm_init)

    optimizers = {'gen': optim.Adam(models['gen'].parameters(),
                                    lr=config['hparams']['lr'],
                                    betas=(config['hparams']['beta_1'],
                                           config['hparams']['beta_2'])),
                  'discrim': optim.Adam(models['discrim'].parameters(),
                                        lr=config['hparams']['lr'],
                                        betas=(config['hparams']['beta_1'],
                                               config['hparams']['beta_2']))}

    criterion = nn.BCELoss()

    for epoch in range(config['hparams']['n_epochs']):
        train(args, config, device, models, criterion, optimizers, train_loader, epoch)

if __name__ == "__main__":
    main()

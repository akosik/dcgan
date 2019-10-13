import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        # channel nums from paper
        self.conv = nn.Sequential(DiscriminatorBlock(config, config['data']['nchan'], 16, use_batchnorm=False),
                                   DiscriminatorBlock(config, 16, 32),
                                   DiscriminatorBlock(config, 32, 64),
                                   DiscriminatorBlock(config, 64, 128))

        downsampled_input_dim = config['data']['dim'] // 2 ** 4
        self.fc = nn.Sequential(nn.Linear(downsampled_input_dim ** 2 * 128, 1),
                                nn.Sigmoid())

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        return self.fc(conv_out)



class DiscriminatorBlock(nn.Module):
    def __init__(self, config, in_filt, out_filt, use_batchnorm=True):
        super(DiscriminatorBlock, self).__init__()

        seq_list = [nn.Conv2d(in_filt, out_filt, 3, 2, 1),
                    nn.LeakyReLU(config['hparams']['relu_leak_slope'],
                                 inplace=True),
                    nn.Dropout2d(0.25)]

        if use_batchnorm:
            seq_list.append(nn.BatchNorm2d(out_filt, 0.8))

        self.model = nn.Sequential(*seq_list)

    def forward(self, x):
        return self.model(x)

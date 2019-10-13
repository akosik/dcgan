import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        self.downsampled_x_dim = config['data']['dim'] // 8 # downsampled input x size
        self.z_dim = config['gen']['zdim']
        self.feat_map_dim = config['gen']['feat_map_dim']
        # project hidden rep to flat downsampled image
        self.fc = nn.Linear(self.z_dim, self.z_dim * self.downsampled_x_dim ** 2)
        # short stack of deconvs for smaller imgs than paper
        # follow general transpose convolution: decrease channels, increase img size
        self.conv = nn.Sequential(nn.BatchNorm2d(self.z_dim), # N x zdim x *x.size()
                                  nn.ConvTranspose2d(self.z_dim, self.feat_map_dim * 2, 4, 2, 1),
                                  nn.BatchNorm2d(self.feat_map_dim * 2),
                                  nn.LeakyReLU(config['hparams']['relu_leak_slope'],
                                               inplace=True),
                                  nn.ConvTranspose2d(self.feat_map_dim * 2, self.feat_map_dim, 4, 2, 1),
                                  nn.BatchNorm2d(self.feat_map_dim),
                                  nn.LeakyReLU(config['hparams']['relu_leak_slope'],
                                               inplace=True),
                                  nn.ConvTranspose2d(self.feat_map_dim, config['data']['nchan'], 4, 2, 1),
                                  nn.Tanh())

    def forward(self, z):
        fc_out = self.fc(z)
        fc_out = fc_out.view(fc_out.size(0), self.z_dim, self.downsampled_x_dim, self.downsampled_x_dim)
        x_fake = self.conv(fc_out)
        return x_fake

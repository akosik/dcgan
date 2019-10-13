import logging
import os
import numpy as np
import time

import torch
from torchvision.utils import save_image
from torch.autograd import Variable

def train(args, config, device, models, criterion, optimizers, train_loader, epoch):
    start = time.time()
    for i, (xs,_) in enumerate(train_loader):
        # targets
        valid_x = Variable(torch.FloatTensor(xs.size(0), 1, device=device).fill_(1.0),
                           requires_grad=False)
        fake_x = Variable(torch.FloatTensor(xs.size(0), 1, device=device).fill_(0.0),
                          requires_grad=False)

        # convert xs to float if not already
        real_xs = Variable(xs.type(dtype=torch.float))
        z = Variable(torch.FloatTensor(np.random.normal(0.0, 1.0,
                                                        (xs.size(0),
                                                         config['gen']['zdim']))))

        # g step
        gen_xs = models['gen'](z)

        g_loss = criterion(models['discrim'](gen_xs), valid_x)
        optimizers['gen'].zero_grad()
        g_loss.backward()
        optimizers['gen'].step()


        # d step
        d_loss = criterion(models['discrim'](real_xs), valid_x)

        gen_xs_as_input = gen_xs.detach()
        d_loss += criterion(models['discrim'](gen_xs_as_input), fake_x)
        d_loss /= 2

        optimizers['discrim'].zero_grad()
        d_loss.backward()
        optimizers['discrim'].step()

        # print progress
        if (epoch * len(train_loader) + i) % args.logging_interval == 0:
            logging.info('Epoch {0} | Iter {1} | Gen Loss {2:.6f} | '
                      'Discrim Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, g_loss / (i + 1),
                          d_loss / (i + 1), 1000 * (time.time() - start) / (i + 1)))
        if (epoch * len(train_loader) + i) % args.image_gen_interval == 0:
            if not os.path.isdir("gen_images"):
                os.mkdir("gen_images")
            save_image(gen_xs.data[:49],
                       "gen_images/{}.png".format(epoch * len(train_loader) + i), nrow=5, normalize=True)

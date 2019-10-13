import torch.nn as nn

def normal_initialization(config):
    def init(module):
        class_name = module.__class__.__name__
        if "Conv" in class_name:
            nn.init.normal_(module.weight.data,
                            config['weight_init']['conv']['mean'],
                            config['weight_init']['conv']['std'])
        elif "BatchNorm2d" in class_name:
            nn.init.normal_(module.weight.data,
                            config['weight_init']['batch_norm']['weight']['mean'],
                            config['weight_init']['batch_norm']['weight']['std'],)
            nn.init.constant_(module.bias.data,
                              config['weight_init']['batch_norm']['bias']['const'])
    return init

import torch.nn as nn
from helper import GroupNorm, Swish, ResidualBlock, UpSampleBlock, DownSampleBlock, NonLocalBlock

class Decoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        channels = [512, 256, 256, 128, 128]
        att_resolution = [16]
        num_res_blocks = 3
        resolution = 16
        layers = [nn.Conv2d(args.latent_dim, channels[0], 3, 1, 1),
                  ResidualBlock(channels[0], channels[0]),
                  NonLocalBlock(channels[0]),
                  ResidualBlock(channels[0], channels[0])]
        in_channels = channels[0]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in att_resolution:
                    layers.append(NonLocalBlock(in_channels))
                if i != 0:
                    layers.append(UpSampleBlock(in_channels))
                    resolution *= 2
        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
                
            
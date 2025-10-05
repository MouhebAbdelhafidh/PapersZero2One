import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from codebook import CodeBook

class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(args).to(device=args.device)
        self.decoder = Decoder(args).to(device=args.device)
        self.codebook = CodeBook(args).to(device=args.device)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self.after_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)

    def forward(self, imgs):
        encoded_imgs = self.encoder(imgs)
        quant_conv_imgs = self.quant_conv(encoded_imgs)
        codebook_mapping, codebook_indices, codebook_loss = self.codebook(quant_conv_imgs)
        after_quant_conv_imgs = self.after_quant_conv(codebook_mapping)
        decoded_imgs = self.decoder(after_quant_conv_imgs)

        return decoded_imgs, codebook_indices, codebook_loss
    
    # Encoding for Transformer
    def encode(self, imgs):
        encoded_imgs = self.encoder(imgs)
        quant_conv_imgs = self.quant_conv(encoded_imgs)
        codebook_mapping, codebook_indices, codebook_loss = self.codebook(quant_conv_imgs)

        return encoded_imgs, codebook_indices, codebook_loss
    
    # Decoding for Transformer
    def decode(self, z):
        after_quant_conv_imgs = self.after_quant_conv(z)
        decoded_imgs = self.decoder(after_quant_conv_imgs)

        return decoded_imgs
    
    # Weighting vector between the VQ Loss and the GAN loss
    def compute_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_gr = torch.autograd(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_gr = torch.autograd(gan_loss, last_layer_weight, retain_graph=True)[0]

        l = torch.norm(perceptual_loss_gr) / (torch.norm(gan_loss_gr) + 1e-4)
        l = torch.clamp(l, 0, 1e4).detach()
        return 0.8 * l

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
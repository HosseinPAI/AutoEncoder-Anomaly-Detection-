# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:43:04 2020

@author: MrHossein
"""
import torch.nn as nn


# ---------------------------------------------------------------------------------------------
#                                The Encoder Module
# ---------------------------------------------------------------------------------------------
class encoder_module(nn.Module):
    def __init__(self, batch_norm=False):
        super(encoder_module, self).__init__()

        self.batch_norm = batch_norm
        layers = [
            nn.BatchNorm2d(3),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),  # 64*64*32
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 32*32*32
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # 32*32*32
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),  # 16*16*64
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 16*16*64
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),  # 8*8*128
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),  # 8*8*64
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),  # 8*8*32
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=100, kernel_size=8, stride=1, padding=0),  # 1*1*100
            nn.LeakyReLU(),
        ]

        if self.batch_norm:
            layers.insert(2, nn.BatchNorm2d(32))
            layers.insert(5, nn.BatchNorm2d(32))
            layers.insert(8, nn.BatchNorm2d(32))
            layers.insert(11, nn.BatchNorm2d(64))
            layers.insert(14, nn.BatchNorm2d(64))
            layers.insert(17, nn.BatchNorm2d(128))
            layers.insert(20, nn.BatchNorm2d(64))
            layers.insert(23, nn.BatchNorm2d(32))
            layers.insert(26, nn.BatchNorm2d(100))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


# ---------------------------------------------------------------------------------------------
#                                The Encoder Module
# ---------------------------------------------------------------------------------------------
class decoder_module(nn.Module):
    def __init__(self, batch_norm=False):
        super(decoder_module, self).__init__()

        self.batch_norm = batch_norm

        layers = [
            nn.ConvTranspose2d(in_channels=100, out_channels=32, kernel_size=8, stride=1, padding=0),  # 8*8*32
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # 8*8*64
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 8*8*128
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),  # 16*16*64
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 16*16*64
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),  # 32*32*32
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # 32*32*32
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 64*64*32
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),  # 128*128*3
        ]

        if self.batch_norm:
            layers.insert(1, nn.BatchNorm2d(32))
            layers.insert(4, nn.BatchNorm2d(64))
            layers.insert(7, nn.BatchNorm2d(128))
            layers.insert(10, nn.BatchNorm2d(64))
            layers.insert(13, nn.BatchNorm2d(64))
            layers.insert(16, nn.BatchNorm2d(32))
            layers.insert(19, nn.BatchNorm2d(32))
            layers.insert(22, nn.BatchNorm2d(32))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


# ---------------------------------------------------------------------------------------------
#                                The Auto Encoder Module
# ---------------------------------------------------------------------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, batch_normalization=False):
        super(AutoEncoder, self).__init__()

        self.encoder_model = encoder_module(batch_norm=batch_normalization)
        self.decoder_model = decoder_module(batch_norm=batch_normalization)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder Part
        encoder_output = self.encoder_model(x)
        decoder_output = self.decoder_model(encoder_output)

        return self.sigmoid(decoder_output)

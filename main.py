import torch
from torch import nn
from model import *


LATENT_SIZE = 120
ENCODER_INPUT_RESOLUTION = 128


def main() :


    # 확률적 z 
    if (is_stochastic_encoder) :
        mu = 0
        sigma = 1
        z = torch.randn(LATENT_SIZE) * sigma + mu 
    else :
        z = torch.randn(LATENT_SIZE)



    # 모델 생성
    generator = Generator()
    encoder = Encoder()
    discriminator = Discriminator()

    # 학습
    

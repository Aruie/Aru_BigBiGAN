
import torch
from torch import nn
from torch.nn import functional as F

from resnet50 import ResNet50


# Z to X
class Generator(nn.Module) :
    def __init__(self) :
        super(Generator, self).__init__()

        


        

    def forward(self, z):
        None


# X to Z
class Encoder(nn.Module) :
    def __init__(self) :
        super(Encoder, self).__init__()

        self.base = ResNet50(include_top = False)
        self.dense1 = nn.Linear(1024, )


    def forward(self, x) :
        None


class Discriminator(nn.Module) :
    def __init__(self) :
        super(Discriminator, self).__init__()


        self.H = PartHJ(120)
        #self.F = 


    def forward(self, x) :
        None



class PartHJ(nn.Module) :
    def __init__(self, H_INPUT, const_k = 8):
        super(PartHJ, self).__init__()
        
        self.layer1 = nn.Linear(H_INPUT, 2048)
        self.layer2 = nn.Linear(2048, 2048)
        self.layer3 = nn.Linear(2048, 2048)
        self.layer4 = nn.Linear(2048, 2048)
        self.layer5 = nn.Linear(2048, 2048)
        self.layer6 = nn.Linear(2048, 2048)
        self.layer7 = nn.Linear(2048, 2048)
        self.layer8 = nn.Linear(2048, 2048)


class SelfAttention(nn.Module) :
    def __init__(self, input_channels, k=8) :
        super(SelfAttention, self).__init__()
        channels = input_channels
        # f, g, h 정의
        self.part_f = nn.Conv2d(channels, channels//k, kernel_size=1)
        self.part_g = nn.Conv2d(channels, channels//k, kernel_size=1)
        self.part_h = nn.Conv2d(channels, channels//k, kernel_size=1)
        self.part_v = nn.Conv2d(channels//k, channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x) :
        batch, c, width, height = x.size()
        location_num = height * width

        f = self.part_f(x).view(batch, -1, location_num).permute(0,2,1)
        g = self.part_g(x).view(batch, -1, location_num)
        beta = self.softmax(torch.bmm(f,g))

        h = self.part_h(x).view(batch, -1, location_num)
        h2 = torch.bmm(h, beta).view(batch, -1, width, height)
        o = self.part_v(h2)

        y = o * self.gamma + x
        return y





        
if __name__ == '__main__' :
    model = SelfAttention(16)

    x = torch.randn((1, 16, 8, 8))
    y = model(x)
    print(model)
    print(x.shape, y.shape)
    #print( sum(p.numel() for p in model.parameters()) )
    #for layer in model.parameters() :
    #    print(layer)

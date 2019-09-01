
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
    def __init__(self, H_INPUT):
        super(PartHJ, self)__init__()
        
        self.layer1 = nn.Linear(H_INPUT, 2048)
        self.layer2 = nn.Linear(2048, 2048)
        self.layer3 = nn.Linear(2048, 2048)
        self.layer4 = nn.Linear(2048, 2048)
        self.layer5 = nn.Linear(2048, 2048)
        self.layer6 = nn.Linear(2048, 2048)
        self.layer7 = nn.Linear(2048, 2048)
        self.layer8 = nn.Linear(2048, 2048)






        
if __name__ == '__main__' :
    model = ResNet50()

    x = torch.randn((1, 3, 224, 224))

    y = model(x)

    print(model)

    print(x.shape, y)
    #print( sum(p.numel() for p in model.parameters()) )
    #for layer in model.parameters() :
    #    print(layer)

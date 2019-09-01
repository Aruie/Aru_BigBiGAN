
import torch
from torch import nn
from torch.nn import functional as F


class ResNet50(nn.Module) :
    def __init__(self, include_top = True) :
        super(ResNet50, self).__init__()
        self.include_top = include_top

        self.conv = nn.Conv2d(3, 64, (7,7), 2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.pad = nn.ZeroPad2d(1)
        self.maxpool = nn.MaxPool2d(3, stride = 2)

        self.residual_block = nn.ModuleList() 

        step_channels = [64, 128, 256, 512]

        self.residual_block.append(ResidualBlock(step_channels[0], True, True))
        self.residual_block.append(ResidualBlock(step_channels[0], False))
        self.residual_block.append(ResidualBlock(step_channels[0], False))

        self.residual_block.append(ResidualBlock(step_channels[1], True))
        self.residual_block.append(ResidualBlock(step_channels[1], False))
        self.residual_block.append(ResidualBlock(step_channels[1], False))
        self.residual_block.append(ResidualBlock(step_channels[1], False))

        self.residual_block.append(ResidualBlock(step_channels[2], True))
        self.residual_block.append(ResidualBlock(step_channels[2], False))
        self.residual_block.append(ResidualBlock(step_channels[2], False))
        self.residual_block.append(ResidualBlock(step_channels[2], False))
        self.residual_block.append(ResidualBlock(step_channels[2], False))
        self.residual_block.append(ResidualBlock(step_channels[2], False))
        
        self.residual_block.append(ResidualBlock(step_channels[3], True))
        self.residual_block.append(ResidualBlock(step_channels[3], False))
        self.residual_block.append(ResidualBlock(step_channels[3], False))

        if (self.include_top) :
            self.dense = nn.Linear(2048, 1000)

    def forward(self, x) :
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pad(x)
        x = self.maxpool(x)
        
        for i, layer in enumerate(self.residual_block) :
            x = layer(x)

        # Global Average Pooling
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

        if self.include_top :
            x = self.dense(x)
            x = F.softmax(x)

        return x

class ResidualBlock(nn.Module) :
    def __init__(self, channels, is_start, is_intro = False) :
        super(ResidualBlock, self).__init__()
        self.is_start = is_start
        
        if self.is_start :
            if is_intro == True :
                self.conv1 = nn.Conv2d(channels, channels, 1, stride = 1)
                self.convsc = nn.Conv2d(channels, channels*4, 1)
            else : 
                self.conv1 = nn.Conv2d(channels*2, channels, 1, stride = 2)
                self.convsc = nn.Conv2d(channels*2, channels*4, 1, stride = 2)
        else :
            self.conv1 = nn.Conv2d(channels*4, channels, 1)

        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * 4, 1)
        self.bn3 = nn.BatchNorm2d(channels * 4)

    def forward(self, x) :
        if self.is_start == False :
            sc = x
        else :
            sc = self.convsc(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x = torch.add(x, sc)
        x = F.relu(x)
        
        return x

        
if __name__ == '__main__' :
    model = ResNet50(False)

    x = torch.randn((1, 3, 224, 224))

    y = model(x)

    print(y.shape)

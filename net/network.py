import torch
import torch.nn as nn
import sys
sys.path.append('..')
from net.parts import single_block,single_block1,fusionblock

class EnhanceNet(nn.Module):
    def __init__(self,channel=64):
        super().__init__()
        self.inc = single_block(16,32)
        self.layer1 = nn.Sequential(
                nn.MaxPool2d(2),
                single_block1(32,64)
        )
        self.layer2 = nn.Sequential(
                nn.MaxPool2d(2),
                single_block1(64,128)
        )
        self.layer3 = nn.Sequential(
                nn.MaxPool2d(2),
                single_block1(128,256)
        )
        self.up1 = nn.ConvTranspose2d(256,128,2,2)
        self.layer4 = nn.Sequential(single_block(256,128))
        self.up2 = nn.ConvTranspose2d(128,64,2,2)
        self.layer5 = nn.Sequential(single_block(128,64))
        self.up3 = nn.ConvTranspose2d(64,32,2,2)
        self.layer6 = nn.Sequential(single_block(64,32))
        self.output = nn.Sequential(
                nn.Conv2d(32,12,1),
                nn.ReLU()
        )
    def forward(self,x):
        I = torch.cat((0.8*x,x,1.2*x,1.5*x),dim=1)
        inc = self.inc(I)
        layer1 = self.layer1(inc)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)

        up1 = self.up1(layer3)
        layer4 = torch.cat((up1,layer2),dim=1)
        layer4 = self.layer4(layer4)
        up2 = self.up2(layer4)
        layer5 = torch.cat((up2,layer1),dim=1)
        layer5 = self.layer5(layer5)
        up3 = self.up3(layer5)
        layer6 = torch.cat((up3,inc),dim=1)
        layer6 = self.layer6(layer6)
        output = self.output(layer6)
        print(output.shape)
        output = torch.nn.functional.pixel_shuffle(output,2)
        return output


if __name__ == "__main__":
    net = EnhanceNet()
    rand = torch.rand(1,4,1024,1024)
    output = net(rand)
    print(output.shape)

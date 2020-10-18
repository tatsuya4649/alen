# this file is a part of network
import torch
import torch.nn as nn
from utils.downsample import downsample

class seblock(nn.Module):
    #channel attention block(CAB)
    def __init__(self,in_channel=32,out_channel=32):
        super().__init__()
        self.fc = nn.Sequential(
                nn.Linear(in_channel,out_channel),
                nn.ReLU(),
                nn.Linear(out_channel,out_channel),
                nn.Sigmoid()
        )
    def forward(self,x):
        _,c,h,w = x.shape
        avg = nn.AvgPool2d([h,w],stride=0)
        y = avg(x)
        y = y.view(1,1,1,-1)
        y = self.fc(y)
        y = y.view(1,-1,1,1)
        output = x * y
        return output

class single_block(nn.Module):
    # CAB + Conv
    def __init__(self,in_channel=32,out_channel=32):
        super().__init__()
        self.conv1 = nn.Sequential(
                seblock(in_channel,in_channel),
                nn.Conv2d(in_channel,out_channel,3,padding=1),
                nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
                nn.Conv2d(out_channel,out_channel,3,padding=1),
                nn.LeakyReLU(0.2)
        )
    def forward(self,x):
        conv1_output = self.conv1(x)
        conv2_output = self.conv2(conv1_output)
        return conv2_output

class nonlocalblock(nn.Module):
    # non-local operation block
    def __init__(self,channel=32,avg_kernel=32):
        super().__init__()
        self.channel = channel
        self.theta = nn.Conv2d(channel,self.channel,1)
        self.phi = nn.Conv2d(channel,self.channel,1)
        self.g = nn.Conv2d(channel,self.channel,1)
        self.conv = nn.Conv2d(self.channel,channel,1)
        self.avg = nn.AvgPool2d([avg_kernel,avg_kernel],stride=avg_kernel)
    def forward(self,x):
        _,_,H,W = x.shape
        u = self.avg(x)
        b,c,h,w = u.shape
        theta_output = self.theta(u).view(b,self.channel,-1).permute(0,2,1)
        phi_output = self.phi(u)
        phi_output = downsample(phi_output)
        g_output = self.g(u)
        g_output = downsample(g_output).permute(0,2,1)

        theta_output = torch.matmul(theta_output,phi_output)
        theta_output = torch.nn.functional.softmax(theta_output,dim=-1)
        
        y = torch.matmul(theta_output,g_output)
        y = y.permute(0,2,1).contiguous()
        y = y.view(b,self.channel,h,w)
        y = self.conv(y)
        y = torch.nn.functional.interpolate(y,size=[H,W])
        return y


class single_block1(nn.Module):
    def __init__(self,in_channel=32,out_channel=32):
        super().__init__()
        self.nonlocalblock = nonlocalblock(in_channel)
        self.seblock = seblock(2*in_channel,2*in_channel)
        self.fusion = nn.Sequential(
                nn.Conv2d(2*in_channel,out_channel,3,padding=1),
                nn.LeakyReLU(0.2)
        )
        self.conv1 = nn.Sequential(
                nn.Conv2d(out_channel,out_channel,3,padding=1),
                nn.LeakyReLU(0.2)
        )
    def forward(self,x):
        nonlocal_output = self.nonlocalblock(x)
        output_cat = torch.cat((nonlocal_output,x),dim=1)
        seblock_output = self.seblock(output_cat)
        fusion_output = self.fusion(seblock_output)
        conv1_output = self.conv1(fusion_output)
        return conv1_output

class fusionblock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.fc = nn.Linear(in_channel,out_channel)
        self.fusion = nn.Sequential(
                nn.Conv2d(out_channel*2,out_channel,1),
                nn.LeakyReLU(0.2)
        )
    def forward(self,x,y):
        _,_,h,w = y.shape
        x = self.fc(x)
        x = x.view(1,-1,1,1)
        x = x.repeat(1,1,h,w)
        x = torch.cat((x,y),dim=1)
        x = self.fusion(x)
        return x

import torch
import torch.nn as nn

def downsample(x):
    b,c,h,w = x.shape
    avg = nn.AvgPool2d([4,4],stride=4)
    avg_output = avg(x).view(b,c,-1)
    return avg_output

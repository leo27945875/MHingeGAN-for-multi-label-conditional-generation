import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm 


def GetActivation(name):
    if name == "ReLU":
        return nn.ReLU()
    elif name == "LeakyReLU":
        return nn.LeakyReLU(0.1)
    elif name == "SiLU":
        return nn.SiLU()
    else:
        raise ValueError("Invalid activation function !")


class Identity(nn.Module):
    def forward(self, x):
        return x


class GlobalPool2d(nn.Module):
    def __init__(self, poolType):
        super().__init__()
        self.poolType = poolType

    def forward(self, x):
        if self.poolType == "sum":
            return torch.sum(x, dim=(2, 3), keepdim=True)
        elif self.poolType == "avg":
            return torch.mean(x, dim=(2, 3), keepdim=True)
        else:
            raise ValueError(f"Invalid pooling type {self.poolType} !")


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, numClass, numFeature, momentum=0.1):
        super().__init__()
        self.weight = nn.Linear(numClass, numFeature)
        self.bias   = nn.Linear(numClass, numFeature)
        self.bn     = nn.BatchNorm2d(numFeature, momentum=momentum, affine=False)
        self.Initialize()

    def forward(self, input, c):
        # c = c / torch.sqrt(torch.sum(c, dim=-1, keepdim=True))
        output = self.bn(input)
        weight = self.weight(c).unsqueeze(-1).unsqueeze(-1)
        bias   = self.bias  (c).unsqueeze(-1).unsqueeze(-1)
        return weight * output + bias
 
    def Initialize(self):
        nn.init.orthogonal_(self.weight.weight.data)
        nn.init.zeros_(self.bias.weight.data)


class UpSampleBlock(nn.Module):
    def __init__(self, numClass, inChannel, outChannel, kernelSize, stride, padding, activation, isSN=True, isRes=False):
        super().__init__()
        if isSN:
            self.conv = spectral_norm(nn.Conv2d(inChannel, outChannel, kernelSize, stride, padding))
        else:
            self.conv = nn.Conv2d(inChannel, outChannel, kernelSize, stride, padding)
        
        if isRes:
            self.skip = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                spectral_norm(nn.Conv2d(inChannel, outChannel, 1, 1, 0)) if isSN else nn.Conv2d(inChannel, outChannel, 1, 1, 0)
            )
        else:
            self.skip = None
        
        self.norm = ConditionalBatchNorm2d(numClass, inChannel)
        self.act = GetActivation(activation)
        self.upSample = nn.UpsamplingNearest2d(scale_factor=2) 

    def forward(self, x, c):
        h = self.norm(x, c)
        h = self.act(h)
        h = self.upSample(h)
        h = self.conv(h)
        return h + self.skip(x) if self.skip else h


class DownSampleBlock(nn.Module):
    def __init__(self, inChannel, outChannel, kernelSize, stride, padding, activation, isSN=True, isRes=False):
        super().__init__()
        if isSN:
            self.conv = spectral_norm(nn.Conv2d(inChannel, outChannel, kernelSize, stride, padding))
        else:
            self.conv = nn.Conv2d(inChannel, outChannel, kernelSize, stride, padding)
        
        if isRes:
            self.skip = nn.Sequential(
                nn.AvgPool2d(2),
                spectral_norm(nn.Conv2d(inChannel, outChannel, 1, 1, 0)) if isSN else nn.Conv2d(inChannel, outChannel, 1, 1, 0)
            )
        else:
            self.skip = None

        self.act = GetActivation(activation)
        self.downSample = nn.AvgPool2d(2)

    def forward(self, x):
        h = self.conv(x)
        h = self.act(h)
        h = self.downSample(h)
        return h + self.skip(x) if self.skip else h


class BatchStd(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        b, h, w = x.size(0), x.size(2), x.size(3)
        std = torch.std(x, dim=0).mean().repeat(b, 1, h, w)
        return torch.cat([x, std], dim=1)


class SelfAttention(nn.Module):
    def __init__(self, inChannel, isSN=True):
        super().__init__()
        self.inChannel = inChannel

        self.qConv = spectral_norm(nn.Conv2d(inChannel, inChannel // 8, 1)) if isSN else nn.Conv2d(inChannel, inChannel // 8, 1)
        self.kConv = spectral_norm(nn.Conv2d(inChannel, inChannel // 8, 1)) if isSN else nn.Conv2d(inChannel, inChannel // 8, 1)
        self.vConv = spectral_norm(nn.Conv2d(inChannel, inChannel // 8, 1)) if isSN else nn.Conv2d(inChannel, inChannel // 8, 1)
        self.oConv = spectral_norm(nn.Conv2d(inChannel // 8, inChannel, 1)) if isSN else nn.Conv2d(inChannel // 8, inChannel, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        b, c, h, w = x.shape

        q = self.qConv(x).view(b, c // 8, h * w).permute(0, 2, 1)
        k = self.kConv(x).view(b, c // 8, h * w)
        v = self.vConv(x).view(b, c // 8, h * w)
        
        atten = F.softmax(q @ k, dim=-1)

        out = (v @ atten.permute(0, 2, 1)).view(b, c // 8, h, w)
        out = self.oConv(out) * self.gamma + x
        return out, atten


class Generator(nn.Module):
    def __init__(self, noiseDim, baseChannel, numClass, activation="LeakyReLU", isAttention=True, isSN=True, isRes=False):
        super().__init__()
        
        self.linear = spectral_norm(nn.Linear(noiseDim, baseChannel * 4 * 4)) if isSN else nn.Linear(noiseDim, baseChannel * 4 * 4)  # -> 4*4
        self.conv1 = UpSampleBlock(numClass, baseChannel // 1, baseChannel // 2, 3, 1, 1, activation, isSN, isRes)                   # -> 8*8
        self.conv2 = UpSampleBlock(numClass, baseChannel // 2, baseChannel // 4, 3, 1, 1, activation, isSN, isRes)                   # -> 16*16
        self.conv3 = UpSampleBlock(numClass, baseChannel // 4, baseChannel // 8, 3, 1, 1, activation, isSN, isRes)                   # -> 32*32
        self.conv4 = UpSampleBlock(numClass, baseChannel // 8, 3               , 3, 1, 1, activation, isSN, isRes)                   # -> 64*64

        self.atten = SelfAttention(baseChannel // 8) if isAttention else None
    
    def forward(self, x, c):
        h = self.linear(x).view(x.size(0), -1, 4, 4)
        h = self.conv1(h, c)
        h = self.conv2(h, c)
        h = self.conv3(h, c)

        if self.atten is not None:
            h, atten = self.atten(h)
        else:
            h, atten = h, None

        img = torch.tanh(self.conv4(h, c))
        return img, atten

        
class Discriminator(nn.Module):
    def __init__(self, baseChannel, numClass, activation="LeakyReLU", isAttention=True, isSN=True, isBatchStd=False, isAux=False, isRes=False, poolType="sum"):
        super().__init__()

        self.conv0 = DownSampleBlock(3               , baseChannel // 8, 3, 1, 1, activation, isSN, isRes)  # -> 32*32
        self.conv1 = DownSampleBlock(baseChannel // 8, baseChannel // 4, 3, 1, 1, activation, isSN, isRes)  # -> 16*16
        self.conv2 = DownSampleBlock(baseChannel // 4, baseChannel // 2, 3, 1, 1, activation, isSN, isRes)  # -> 8*8
        self.conv3 = DownSampleBlock(baseChannel // 2, baseChannel // 1, 3, 1, 1, activation, isSN, isRes)  # -> 4*4

        self.atten = SelfAttention(baseChannel // 8) if isAttention else None

        if isBatchStd:
            self.pool = nn.Sequential(
                BatchStd(),
                DownSampleBlock(baseChannel + 1, baseChannel, 3, 1, 1, activation, isSN),
                GlobalPool2d(poolType),
                nn.Flatten()
            )
        else:
            self.pool = nn.Sequential(
                GlobalPool2d(poolType),
                nn.Flatten()
            )

        self.linearCon = spectral_norm(nn.Linear(numClass, baseChannel)) if isSN else nn.Linear(numClass, baseChannel)
        self.linearOut = spectral_norm(nn.Linear(baseChannel, 1))        if isSN else nn.Linear(baseChannel, 1)
        if isAux:
            self.linearAux = spectral_norm(nn.Linear(baseChannel, numClass)) if isSN else nn.Linear(baseChannel, numClass)
        else:
            self.linearAux = None
    
    def forward(self, x, c):
        h = self.conv0(x)

        if self.atten is not None:
            h, atten = self.atten(h)
        else:
            h, atten = h, None

        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.pool(h)

        proj = self.Project(self.linearCon(c), h)
        out  = self.linearOut(h) + proj

        if self.linearAux is not None:
            cls = self.linearAux(h)
        else:
            cls = None
        
        return out, cls, atten
    
    def Project(self, x, y):
        return torch.sum(x * y, dim=-1, keepdim=True)


class GeneratorLoss(nn.Module):
    def __init__(self, lossType):
        super().__init__()
        self.lossType = lossType

    def forward(self, fakeOut, fakeLogit=None, label=None, auxWeight=0.1):
        if fakeLogit is None or label is None:
            return -fakeOut.mean()
        else:
            return -fakeOut.mean() + auxWeight * self.GetClassifyLoss(fakeLogit, label)
    
    def GetClassifyLoss(self, logit, label):
        if self.lossType == "hinge":
            c = label * 2 - 1
            return F.relu(1 - logit * c).sum(dim=-1).mean()
        elif self.lossType == "bce":
            sig = torch.sigmoid(logit)
            return -(label * torch.log(sig) + (1 - label) * torch.log(1 - sig)).mean()


class DiscriminatorLoss(nn.Module):
    def __init__(self, lossType):
        super().__init__()
        self.lossType = lossType

    def forward(self, realOut, fakeOut, realLogit=None, fakeLogit=None, label=None):
        disLoss = (F.relu(1 - realOut) + F.relu(1 + fakeOut)).mean()
        drift   = (realOut ** 2).mean()
        if (realLogit is None and fakeLogit is None) or label is None:
            return disLoss + 0.001 * drift
        else:
            if self.lossType == "hinge":
                return disLoss + self.GetClassifyLoss(realLogit, label) + 0.001 * drift
            elif self.lossType == "bce":
                return disLoss + self.GetClassifyLoss(realLogit, label) + self.GetClassifyLoss(fakeLogit, label) + 0.001 * drift
    
    def GetClassifyLoss(self, logit, label):
        if self.lossType == "hinge":
            c = label * 2 - 1
            return F.relu(1 - logit * c).sum(dim=-1).mean()
        elif self.lossType == "bce":
            sig = torch.sigmoid(logit)
            return -(label * torch.log(sig) + (1 - label) * torch.log(1 - sig)).sum(dim=-1).mean()


if __name__ == '__main__':

    c = torch.ones(4, 24)
    x = torch.ones(4, 32)
    g = Generator(32, 256, 24)
    d = Discriminator(256, 24)
    
    img, gAtten = g(x, c)
    out, dAtten = d(img, c)

    print(img.size(), gAtten[0].size(), gAtten[1].size())
    print(out.size(), dAtten[0].size(), dAtten[1].size())
    
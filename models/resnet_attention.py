import torch.nn as nn
import torchvision.models as models

class Attention(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(c,c,1),nn.Sigmoid())

    def forward(self,x):
        return x*self.a(x)

class ResNetAttention(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.attn = Attention(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512,2)

    def forward(self,x):
        x = self.features(x)
        x = self.attn(x)
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)

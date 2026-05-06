import torch.nn as nn
from torchvision.models import vit_b_16

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vit_b_16(weights=None)
        self.model.heads = nn.Linear(self.model.heads.head.in_features,2)

    def forward(self,x):
        return self.model(x)

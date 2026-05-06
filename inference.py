import torch
from PIL import Image
from data.transforms import get_tf
from models.resnet_attention import ResNetAttention

model=ResNetAttention()
model.load_state_dict(torch.load("outputs/models/resnet_attention.pth"))
model.eval()

img=Image.open("sample.jpg").convert("RGB")
img=get_tf(256)(img).unsqueeze(0)

print(model(img).argmax(1).item())
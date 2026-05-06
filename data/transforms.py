from torchvision import transforms

def get_tf(size):
    return transforms.Compose([
        transforms.Resize((size,size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

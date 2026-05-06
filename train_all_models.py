import torch,yaml
from torch.utils.data import DataLoader,random_split
from data.dataset import FoodDataset
from data.transforms import get_tf
from models.cnn import CNN
from models.resnet import ResNet
from models.resnet_attention import ResNetAttention
from models.transformer import TransformerModel
from utils.metrics import get_metrics
from utils.logger import save_results

config=yaml.safe_load(open("config.yaml"))

dataset=FoodDataset("data/images",get_tf(config["image_size"]))

n=len(dataset)
train,val,test=random_split(dataset,[int(n*0.7),int(n*0.15),n-int(n*0.85)])

train_loader=DataLoader(train,batch_size=config["batch_size"],shuffle=True)
test_loader=DataLoader(test,batch_size=config["batch_size"])

models={
    "cnn":CNN(),
    "resnet":ResNet(),
    "resnet_attention":ResNetAttention(),
    "transformer":TransformerModel()
}

for name,model in models.items():
    opt=torch.optim.Adam(model.parameters(),lr=config["lr"])
    loss_fn=torch.nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        model.train()
        for x,y in train_loader:
            pred=model(x)
            loss=loss_fn(pred,y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    y_true=[];y_pred=[]
    model.eval()
    with torch.no_grad():
        for x,y in test_loader:
            p=model(x).argmax(1)
            y_true+=y.tolist()
            y_pred+=p.tolist()

    metrics=get_metrics(y_true,y_pred)
    save_results(name,metrics)
    torch.save(model.state_dict(),f"outputs/models/{name}.pth")
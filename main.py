import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from models.cnn import CNN
from models.resnet import ResNet
from models.resnet_attention import ResNetAttention
from models.transformer import TransformerModel

os.makedirs("outputs", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(
    root="./dataset",
    train=True,
    download=True,
    transform=transform
)

small_dataset, _ = random_split(dataset, [2000, len(dataset) - 2000])

train_size = int(0.7 * len(small_dataset))
val_size = int(0.15 * len(small_dataset))
test_size = len(small_dataset) - train_size - val_size

train_data, val_data, test_data = random_split(
    small_dataset,
    [train_size, val_size, test_size]
)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

models = {
    "CNN": CNN(),
    "ResNet": ResNet(),
    "ResNetAttention": ResNetAttention(),
    "Transformer": TransformerModel()
}

results = []

for name, model in models.items():

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(2):

        model.train()

        for x, y in train_loader:

            y = y % 2

            pred = model(x)

            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():

        for x, y in test_loader:

            y = y % 2

            pred = model(x).argmax(1)

            y_true.extend(y.numpy())
            y_pred.extend(pred.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,5))
    plt.imshow(cm)
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()
    plt.savefig(f"outputs/{name}_confusion.png")
    plt.close()

    torch.save(model.state_dict(), f"outputs/{name}.pth")

results_df = pd.DataFrame(results)

results_df.to_csv("outputs/results.csv", index=False)

plt.figure(figsize=(8,5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.savefig("outputs/model_comparison.png")
plt.close()

print(results_df)
print("Completed Successfully")
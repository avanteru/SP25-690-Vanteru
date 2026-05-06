import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def confusion(y,yhat,name):
    cm=confusion_matrix(y,yhat)
    sns.heatmap(cm,annot=True)
    plt.savefig(f"outputs/plots/{name}_cm.png")
    plt.clf()

def bar(results):
    names=results["model"]
    acc=results["accuracy"]
    plt.bar(names,acc)
    plt.savefig("outputs/plots/model_compare.png")
    plt.clf()

def failures(images,labels,preds):
    fig,ax=plt.subplots(2,2)
    k=0
    for i in range(len(images)):
        if labels[i]!=preds[i] and k<4:
            ax[k//2][k%2].imshow(images[i].permute(1,2,0))
            ax[k//2][k%2].set_title(f"T:{labels[i]} P:{preds[i]}")
            k+=1
    plt.savefig("outputs/plots/failures.png")
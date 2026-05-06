from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def get_metrics(y,yhat):
    return {
        "accuracy":accuracy_score(y,yhat),
        "precision":precision_score(y,yhat),
        "recall":recall_score(y,yhat),
        "f1":f1_score(y,yhat)
    }
import pandas as pd

def save_results(name,metrics):
    df=pd.DataFrame([{"model":name,**metrics}])
    try:
        old=pd.read_csv("outputs/results.csv")
        df=pd.concat([old,df])
    except:
        pass
    df.to_csv("outputs/results.csv",index=False)

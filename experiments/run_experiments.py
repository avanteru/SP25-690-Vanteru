import pandas as pd
from utils.plots import bar

df=pd.read_csv("outputs/results.csv")
bar(df)

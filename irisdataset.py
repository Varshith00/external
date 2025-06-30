import pandas as pd
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
data.frame.to_csv("iris.csv", index=False)
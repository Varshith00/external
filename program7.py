# train_and_save.py
from sklearn.linear_model import LogisticRegression
import joblib

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

model = LogisticRegression().fit(X, y)
joblib.dump(model, 'model.joblib')
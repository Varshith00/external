from sklearn.linear_model import LogisticRegression
X= [[1],[2],[3],[4]]
y=[0,1,0,1]
model = LogisticRegression()
model.fit()
print(model.predict([[1]]))
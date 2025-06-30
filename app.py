import streamlit as st
from sklearn.linear_model import LogisticRegression

# Minimal training data
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
model = LogisticRegression().fit(X, y)

# UI and prediction
val = st.number_input("Enter a value", value=1.5)
if st.button("Predict"):
    pred = model.predict([[val]])
    st.write(f"Prediction: {int(pred[0])}")
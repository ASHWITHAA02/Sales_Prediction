import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sales_prediction_model.pkl")

model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    advertising = float(request.form["advertising"])
    price = float(request.form["price"])
    discount = float(request.form["discount"])
    season = float(request.form["season"])

    features = np.array([[advertising, price, discount, season]])

    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Sales: {prediction:.0f} Units"
    )

if __name__ == "__main__":
    app.run(debug=True)
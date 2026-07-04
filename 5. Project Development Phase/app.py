from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("floods.save")
scaler = joblib.load("transform.save")


# Home Page
@app.route("/")
def home():
    return render_template("home.html")


# Prediction Page
@app.route("/Predict")
def predict_page():
    return render_template("index.html")


# Predict Flood
@app.route("/predict", methods=["POST"])
def predict():

    temp = float(request.form["temp"])
    humidity = float(request.form["humidity"])
    cloud = float(request.form["cloud"])
    annual = float(request.form["annual"])
    janfeb = float(request.form["janfeb"])
    marmay = float(request.form["marmay"])
    junsep = float(request.form["junsep"])
    octdec = float(request.form["octdec"])
    avgjune = float(request.form["avgjune"])
    sub = float(request.form["sub"])

    data = pd.DataFrame([[temp, humidity, cloud, annual,
                          janfeb, marmay, junsep,
                          octdec, avgjune, sub]],
                        columns=[
                            "Temp",
                            "Humidity",
                            "Cloud Cover",
                            "ANNUAL",
                            "Jan-Feb",
                            "Mar-May",
                            "Jun-Sep",
                            "Oct-Dec",
                            "avgjune",
                            "sub"
                        ])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        return render_template("chance.html")
    else:
        return render_template("no_chance.html")


if __name__ == "__main__":
    app.run(debug=True)
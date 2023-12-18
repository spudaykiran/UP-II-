
from flask import Flask, render_template, jsonify, request
import joblib
# from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("Model.pkl", "rb"))

heartModel = pickle.load(open("HeartModel.pkl", "rb"))
kidneyModel = pickle.load(open("KidneyModel.pkl", "rb"))
liverModel = pickle.load(open("LiverModel.pkl", "rb"))
diabetesModel = pickle.load(open("DiabetesModel.pkl", "rb"))

@app.route("/")
def home_page():
    return render_template('index1.html')

@app.route("/Diabetes_form.html")
def diabetes_page():
    return render_template('Diabetes_form.html')

@app.route("/Heart_form.html")
def heart_page():
    return render_template('Heart_form.html')

@app.route("/HIV_form.html")
def hiv_page():
    return render_template('HIV_form.html')

@app.route("/Kidney_form.html")
def kidney_page():
    return render_template("Kidney_form.html")

@app.route("/Liver_form.html")
def liver_page():
    return render_template('Liver_form.html')

@app.route("/Lung_form.html")
def lung_page():
    return render_template('Lung_form.html')


@app.route("/predict", methods=['POST'])
def predict():
    # model = joblib.load('Random_Forest_Mode_J.joblib')
    int_data = [int(x) for x in request.form.values()]
    # print(int_data)
    data = [np.array(int_data)]
    pred = model.predict(data)
    # print("prediction is : ", pred[0])
    if pred[0] == 1:
        return render_template("index.html", prediction_text = "The result is {} - DIABETES POSITIVE".format(pred))
    elif pred[0] == 0:
         return render_template("index.html", prediction_text = "The result is {} - DIABETES NEGITIVE".format(pred))


@app.route("/heartPredict", methods=['POST'])
def heart_predict():
    heart_data = [int(x) for x in request.form.values()]
    data = [np.array(heart_data)]
    pred = heartModel.predict(data)
    if pred[0] == 1:
        output = "You are Diagnosed with Heart Disease"
    elif pred[0] == 0:
        output = "You are Not Diagnosed with Heart Disease"

@app.route("/diabetesPredict", methods=['POST'])
def diabetes_predict():
    heart_data = [int(x) for x in request.form.values()]
    data = [np.array(heart_data)]
    pred = diabetesModel.predict(data)
    if pred[0] == 1:
        output = "You are Diagnosed with Diabetes Disease"
    elif pred[0] == 0:
        output = "You are Not Diagnosed with Diabetes Disease"
    print(output)

@app.route("/liverPredict", methods=['POST'])
def liver_predict():
    heart_data = [int(x) for x in request.form.values()]
    data = [np.array(heart_data)]
    pred = liverModel.predict(data)
    if pred[0] == 1:
        output = "You are Diagnosed with Liver Disease"
    elif pred[0] == 0:
        output = "You are Not Diagnosed with Liver Disease"

@app.route("/lungPredict", methods=['POST'])
def lung_predict():
    heart_data = [int(x) for x in request.form.values()]
    data = [np.array(heart_data)]
    pred = lungModel.predict(data)
    if pred[0] == 1:
        output = "You are Diagnosed with Lung Disease"
    elif pred[0] == 0:
        output = "You are Not Diagnosed with Lung Disease"

@app.route("/kidneyPredict", methods=['POST'])
def kidney_predict():
    heart_data = [int(x) for x in request.form.values()]
    data = [np.array(heart_data)]
    pred = kidneyModel.predict(data)
    if pred[0] == 1:
        output = "You are Diagnosed with Kidney Disease"
    elif pred[0] == 0:
        output = "You are Not Diagnosed with Kidney Disease"

@app.route("/hivPredict", methods=['POST'])
def hiv_predict():
    heart_data = [int(x) for x in request.form.values()]
    data = [np.array(heart_data)]
    pred = hivModel.predict(data)
    if pred[0] == 1:
        output = "You are Diagnosed with HIV Disease"
    elif pred[0] == 0:
        output = "You are Not Diagnosed with HIV Disease"

if __name__ == "_main_":
    app.run(debug=True)
    # print(predict())
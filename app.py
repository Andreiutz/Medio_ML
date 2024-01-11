from flask import Flask, request, render_template
import requests
import pickle
import pandas as pd
import joblib
import numpy as np
import sys

from utils import DateTransformer, BasicTransformations, CityTransformer, BasicExperiments, MultiOutputClassifier
 
app = Flask(__name__)

patient_model = pickle.load(open('Models/Patient.pkl', 'rb'))

with open('api_key.txt', 'r') as f:
    key = f.read()
    print(key)
    

def get_weather(city, interval):
    url = 'https://api.weatherbit.io/v2.0/forecast/daily'
    request = f"{url}?key={key}&city={city}&days={interval}"
    response = requests.get(request).json()['data']

    df = pd.DataFrame(response, columns=['max_temp','min_temp',  'precip', 'pres', 'wind_spd', 'datetime', 'sunset_ts', 'sunrise_ts'])
    df['Insolat'] = ((df['sunset_ts'] - df['sunrise_ts']) / 3600 ).round(2)

    df.drop(columns=['sunset_ts', 'sunrise_ts'], inplace=True)
    df.rename(columns={'min_temp': 'Min', 'max_temp': 'Max', 'precip': 'Prec', 'pres': 'Press', 'wind_spd': 'Wind', 'datetime': 'Date'}, inplace=True)
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/patient', methods=['GET', 'POST'])
def patient():
    city, sex, age, ht, af, cihd, vi, copd, comorbidities = None, None, None, None, None, None, None, None, []
    table = []
    data = request.get_json()
    if request.method == 'POST':
        sex = data["Sex"]
        age = data["Age"]
        ht = data["HT"]
        af = data["AF"]
        cihd = data["CIHD"]
        vi = data["VI"]
        copd = data["COPD"]
        max = data["Max"]
        min = data["Min"]
        prec = data["Prec"]
        press = data["Press"]
        wind = data["Wind"]
        insolat = data["Insolat"]
        date = data["Date"]
        features = {
            "Date": [date],
            "Sex": [sex],
            "Age": [age],
            "HT": [ht],
            "AF": [af],
            "CIHD": [cihd],
            "COPD": [copd],
            "VI": [vi],
            "Max": [max],
            "Min": [min],
            "Prec": [prec],
            "Press": [press],
            "Wind": [wind],
            "Insolat": [insolat]
        }
        features = pd.DataFrame(features)
        risks = patient_model.predict_proba(features)
        hf = [risks[0][0][1]]
        rf = [risks[1][0][1]]
        ci = [risks[2][0][1]]
        predictions = pd.DataFrame({'Heart Failure': hf, 'Respiratory Failure': rf, 'Cerebrovascular Infarction': ci})
        return predictions.to_dict()

if __name__ == '__main__':
    app.run(port=5000, debug=True)
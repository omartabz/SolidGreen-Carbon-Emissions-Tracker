"""
Solid Green Predictor Flask Server

This file contains a Flask server that contains one endpoint which can be used to
make a prediction of the CO2 Emissions(g/km) of a vehicle given the:
    * Fuel Consumption Comb (mpg)
    * Fuel Consumption Comb (L/100 km)
    * Fuel Consumption Hwy (L/100 km)
    * Fuel Consumption City (L/100 km)
    * Fuel Type
    * Transmission
    * Cylinders
    * Engine Size(L)
    * Vehicle Class
    * Model
    * Make
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import pickle

from flask import Flask, request, jsonify
from flask_cors import CORS


def preprocess(df):
    label_encoders = {}
    for col in ["Make", "Fuel Type", "Vehicle Class", "Transmission", "Model"]:
        df[col] = df[col].astype(str)  # Ensure all values are strings
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    df.fillna(0, inplace=True)
    return df, label_encoders


df = pd.read_excel('CO2 Emissions_Canada.xlsx')
df, label_encoders = preprocess(df)

features = ["Fuel Consumption Comb (mpg)", "Fuel Consumption Comb (L/100 km)",
            "Fuel Consumption Hwy (L/100 km)", "Fuel Consumption City (L/100 km)", "Fuel Type",
            "Transmission", "Cylinders", "Engine Size(L)", "Vehicle Class", "Model", "Make"]

X = df[features]
y = df["CO2 Emissions(g/km)"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        vehicle_info = request.get_json()
        print(vehicle_info)
        input_data = []
        input_data.append(vehicle_info["Fuel Consumption Comb (mpg)"])
        input_data.append(vehicle_info["Fuel Consumption Comb (L/100 km)"])
        input_data.append(vehicle_info["Fuel Consumption Hwy (L/100 km)"])
        input_data.append(vehicle_info["Fuel Consumption City (L/100 km)"])
        input_data.append(vehicle_info["Fuel Type"])
        input_data.append(vehicle_info["Transmission"])
        input_data.append(vehicle_info["Cylinders"])
        input_data.append(vehicle_info["Engine Size(L)"])
        input_data.append(vehicle_info["Vehicle Class"])
        input_data.append(vehicle_info["Model"])
        input_data.append(vehicle_info["Make"])

        file_path = "linear_regress.pkl"
        loaded_model = pickle.load(open(file_path, "rb"))
        
        # Convert input data into a DataFrame
        input_df = pd.DataFrame([input_data], columns=features)
        #
        for col in ["Make", "Model", "Vehicle Class", "Transmission", "Fuel Type"]:
            if col in label_encoders:
                # Ensure consistency with training data
                input_df[col] = input_df[col].astype(str)
                input_df[col] = label_encoders[col].transform(input_df[col])

        input_scaled = scaler.transform(input_df[features])

        # Predict using the model
        result = loaded_model.predict(input_scaled)
        co2_emission_prediction = result[0]
        return jsonify({"CO2 Emissions(g/km)": co2_emission_prediction}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(port=4000, debug=True)

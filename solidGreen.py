from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculatio
import re
import  numpy as np
import pandas as pd
import hashlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, jsonify
# from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__)
CORS(app)

# Preprocessing and loading dataset
def preprocess(df):
    label_encoder = LabelEncoder()
    df['Make'] = label_encoder.fit_transform(df['Make'])
    df['Fuel Type'] = label_encoder.fit_transform(df['Fuel Type'])
    df['Vehicle Class'] = label_encoder.fit_transform(df['Vehicle Class'])
    df['Transmission'] = label_encoder.fit_transform(df['Transmission'])
    df['Model'] = df['Model'].astype(str)
    df['Model'] = label_encoder.fit_transform(df['Model'])
    df.fillna(0, inplace=True)
    return df

df = pd.read_excel('/content/CO2 Emissions_Canada.xlsx')
df = preprocess(df)

features = ["CO2 Emissions(g/km)","Fuel Consumption Comb (mpg)","Fuel Consumption Comb (L/100 km)",
            "Fuel Consumption Hwy (L/100 km)","Fuel Consumption City (L/100 km)","Fuel Type",
            "Transmission","Cylinders","Engine Size(L)","Vehicle Class","Model","Make"]

X = df[features]
y = df["CO2 Emissions(g/km)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
joblib.dump(lr_model, 'best_model.joblib')

model = joblib.load('best_model.joblib')

@app.route('/api/predict', methods=['GET'])
def predict():
    try:
        input_data = request.json
        input_df = pd.DataFrame([input_data])
        input_df = preprocess(input_df)
        input_scaled = scaler.transform(input_df[features])
        predictions = model.predict(input_scaled)
        return jsonify({'prediction': predictions[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

# if __name__ == "__main__":
#     app.run(port=4000)


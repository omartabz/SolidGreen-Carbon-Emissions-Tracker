import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

import pickle

from flask import Flask, request, jsonify

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


df = pd.read_excel('/home/raphaeltabengwa/Solid green/CO2 Emissions_Canada.xlsx')
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

app = Flask(__name__)



@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # TODO: Contains the vehicle data to use for the prediction
        # vehicle_info = request.get_json()
        file_path = "linear_reg.pkl"
        loaded_model = pickle.load(open(file_path, "rb"))
        # TODO: move away from the score function to the predict function to actually predict
        # the emissions for the given vehicle - this is the result you want to return! 
        # NOTE: You can use collab to figure out the predict protocol.
        #result = loaded_model.predict(data ...)
        # TODO: remove the following line when the predict line is in
        result = loaded_model.score(X_test_scaled, y_test)
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(port=4000)
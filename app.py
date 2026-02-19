from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)  # Allows HTML frontend to talk to Flask

# Load your saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Get values from frontend
        distance     = float(data['distance'])
        traffic      = int(data['traffic'])       # 0=Low, 1=Medium, 2=High
        prep_time    = float(data['prep_time'])
        experience   = float(data['experience'])
        weather      = data['weather']            # "Clear","Foggy","Rainy","Snowy","Windy"
        time_of_day  = data['time_of_day']        # "Morning","Afternoon","Evening","Night"
        vehicle      = data['vehicle']            # "Bike","Car","Scooter"

        # Build the feature DataFrame (must match training columns exactly)
        sample = pd.DataFrame({
            "Distance_km":            [distance],
            "Traffic_Level":          [traffic],
            "Preparation_Time_min":   [prep_time],
            "Courier_Experience_yrs": [experience],

            # Weather one-hot
            "Weather_Clear":  [1 if weather == "Clear"  else 0],
            "Weather_Foggy":  [1 if weather == "Foggy"  else 0],
            "Weather_Rainy":  [1 if weather == "Rainy"  else 0],
            "Weather_Snowy":  [1 if weather == "Snowy"  else 0],
            "Weather_Windy":  [1 if weather == "Windy"  else 0],

            # Time of Day one-hot
            "Time_of_Day_Afternoon": [1 if time_of_day == "Afternoon" else 0],
            "Time_of_Day_Evening":   [1 if time_of_day == "Evening"   else 0],
            "Time_of_Day_Morning":   [1 if time_of_day == "Morning"   else 0],
            "Time_of_Day_Night":     [1 if time_of_day == "Night"     else 0],

            # Vehicle Type one-hot
            "Vehicle_Type_Bike":    [1 if vehicle == "Bike"   else 0],
            "Vehicle_Type_Car":     [1 if vehicle == "Car"    else 0],
            "Vehicle_Type_Scooter": [1 if vehicle == "Scooter" else 0],
        })

        prediction = model.predict(sample)
        delivery_time = round(float(prediction[0]), 1)

        return jsonify({'delivery_time': delivery_time, 'status': 'success'})

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400


if __name__ == '__main__':
    print("✅ Flask server running at http://127.0.0.1:5000")
    app.run(debug=True)

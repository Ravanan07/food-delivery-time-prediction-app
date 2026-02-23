# 🍕 Food Delivery Time Prediction App

A Machine Learning web application that predicts food delivery time
based on distance, weather, traffic, and other real-world factors.

---

## 🚀 About The Project

This project uses a Linear Regression machine learning model trained
on real food delivery data to predict how long a delivery will take.
The model is served via a Flask REST API and connected to a modern
HTML/CSS/JS frontend.

---

## ✨ Features

- ML Powered - Linear Regression model trained on real data
- REST API - Flask backend with /predict endpoint
- Modern UI - Dark themed responsive HTML/CSS frontend
- Multiple Features - Considers 8+ real-world factors
- Real-time Prediction - Instant result on form submit
- One-Hot Encoding - Handles categorical variables properly

---

## 🛠️ Tech Stack

**Machine Learning**
- Python 3.x
- Scikit-learn (Linear Regression)
- Pandas (Data preprocessing)
- Pickle (Model saving/loading)

**Backend**
- Flask (REST API)
- Flask-CORS (Cross Origin support)

**Frontend**
- HTML5
- CSS3
- JavaScript

---

## 📁 Project Structure

food-delivery-time-prediction-app/
├── app.py          ← Flask backend API
├── model.pkl       ← Trained ML model
├── Food.py         ← Model training code
├── index.html      ← Frontend UI
├── index.css       ← Styling
└── README.md       ← Project documentation

---

## ⚙️ How to Run

**Step 1 - Clone the Repository**
git clone https://github.com/Ravanan07/food-delivery-time-prediction-app.git
cd food-delivery-time-prediction-app

**Step 2 - Install Dependencies**
pip install flask flask-cors pandas scikit-learn

**Step 3 - Run Flask Server**
py app.py

**Step 4 - Open Frontend**
Open index.html in your browser and start predicting!

---

Linear Regression was selected for simplicity and interpretability.

---



## 👨‍💻 Author

Ramkumar
GitHub: https://github.com/Ravanan07

---

If you found this project helpful, please give it a star! ⭐

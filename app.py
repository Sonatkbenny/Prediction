from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Load datasets
loan_data = pd.read_csv('loan_prediction_dataset.csv')
temp_data = pd.read_csv('temperature_classification_dataset.csv')
sugar_data = pd.read_csv('blood_sugar_prediction_dataset.csv')

def check_accuracy(score):
    return 0.9 <= score <= 1.0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/polynomial', methods=['GET', 'POST'])
def polynomial():
    if request.method == 'POST':
        try:
            # Prepare loan prediction data
            X = loan_data[['Income', 'Credit_Score']].values
            y = loan_data['Loan_Amount'].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            poly = PolynomialFeatures(degree=2)
            X_poly_train = poly.fit_transform(X_train)
            X_poly_test = poly.transform(X_test)
            
            model = LinearRegression()
            model.fit(X_poly_train, y_train)
            
            # Make prediction
            income = float(request.form['income'])
            credit_score = float(request.form['credit_score'])
            
            input_data = poly.transform([[income, credit_score]])
            prediction = model.predict(input_data)[0]
            
            return render_template('polynomial.html', 
                                prediction=f"${prediction:,.2f}")
        except Exception as e:
            return render_template('polynomial.html', error=str(e))
            
    return render_template('polynomial.html')

@app.route('/logistic', methods=['GET', 'POST'])
def logistic():
    if request.method == 'POST':
        try:
            # Prepare temperature classification data
            X = temp_data[['Humidity', 'Pressure']].values
            y = temp_data['High_Temperature'].values
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # Initialize and fit the model with class weights
            model = LogisticRegression(class_weight='balanced', random_state=42)
            model.fit(X_train, y_train)
            
            # Make prediction
            humidity = float(request.form['humidity'])
            pressure = float(request.form['pressure'])
            
            # Scale the input data
            input_data = scaler.transform([[humidity, pressure]])
            prediction = model.predict(input_data)[0]
            
            # Get probability of high temperature
            prob_high = model.predict_proba(input_data)[0][1]
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, model.predict(X_test))
            
            # Convert prediction to human-readable format
            prediction_text = "High Temperature" if prediction == 1 else "Low Temperature"
            
            return render_template('logistic.html',
                                prediction=prediction_text,
                                probability=f"{prob_high:.2%}",
                                accuracy=f"{accuracy:.2f}",
                                accuracy_check=check_accuracy(accuracy))
        except Exception as e:
            return render_template('logistic.html', error=str(e))
            
    return render_template('logistic.html')

@app.route('/knn', methods=['GET', 'POST'])
def knn():
    if request.method == 'POST':
        try:
            # Prepare blood sugar data
            X = sugar_data[['Age', 'BMI']].values
            y = sugar_data['Blood_Sugar_Level'].values
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # Use KNeighborsRegressor for continuous value prediction
            model = KNeighborsRegressor(n_neighbors=5)
            model.fit(X_train, y_train)
            
            # Make prediction
            age = float(request.form['age'])
            bmi = float(request.form['bmi'])
            
            # Scale the input data
            input_data = scaler.transform([[age, bmi]])
            prediction = model.predict(input_data)[0]
            
            # Determine blood sugar category based on standard ranges
            if prediction < 140:
                category = "Normal"
            elif prediction < 200:
                category = "Pre-diabetes"
            else:
                category = "Diabetes"
            
            return render_template('knn.html',
                                prediction=category,
                                blood_sugar=f"{prediction:.1f}")
        except Exception as e:
            return render_template('knn.html', error=str(e))
            
    return render_template('knn.html')

if __name__ == '__main__':
    app.run(debug=True)

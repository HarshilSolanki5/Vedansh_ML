from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
best_xgb = joblib.load('best_xgb.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input values from the form
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])
        heart_rate = int(request.form['heart_rate'])
        body_temp = float(request.form['body_temp'])
        bmi = float(request.form['bmi'])

        # Make a prediction
        X_array = np.array([[height, weight, duration, heart_rate, body_temp, bmi]])
        prediction = best_xgb.predict(X_array)

        # Render the result template with the prediction
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
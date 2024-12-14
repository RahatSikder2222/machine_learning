from flask import Flask, request, jsonify, render_template
import pickle
import gzip
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the compressed pickle file
with gzip.open('model.pkl.gz', 'rb') as f:
    model = pickle.load(f)





# Define the homepage route
@app.route('/')
def home():
    return render_template('index.html')  # HTML file for user interface

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    form_data = request.form
    
    # Extract input features
    features = [
        float(form_data['CreditScore']),
        float(form_data['Geography']),
        float(form_data['Gender']),
        int(form_data['Age']),
        int(form_data['Tenure']),
        float(form_data['Balance']),
        int(form_data['NumOfProducts']),
        int(form_data['HasCrCard']),
        int(form_data['IsActiveMember']),
        float(form_data['EstimatedSalary'])
    ]

    # Preprocess categorical features (manual encoding for simplicity)
    geography_dict = {'France': 0, 'Germany': 1, 'Spain': 2}
    gender_dict = {'Male': 0, 'Female': 1}
    features[1] = geography_dict[features[1]]
    features[2] = gender_dict[features[2]]

    # Convert features to numpy array and reshape
    final_features = np.array(features).reshape(1, -1)

    # Predict using the loaded model
    prediction = model.predict(final_features)
    output = 'Exited' if prediction[0] == 1 else 'Not Exited'

    # Return the prediction result
    return render_template('index.html', prediction_text=f'The customer is predicted to: {output}')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

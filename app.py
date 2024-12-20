from flask import Flask, request, render_template
import pickle
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model (replace 'cancer.pkl' with your actual file path)
with open('cancer.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        features = [
            float(request.form['concavity_worst']),
            float(request.form['compactness_worst']),
            float(request.form['symmetry_worst']),
            float(request.form['concavity_mean']),
            float(request.form['texture_worst']),
            float(request.form['concave_points_worst']),
            float(request.form['perimeter_mean']),
            float(request.form['smoothness_worst']),
            float(request.form['concave_points_mean']),
            float(request.form['compactness_mean'])
        ]
    except ValueError:
        return "Please enter valid numerical values for all features."
    
    # Convert the input features to a numpy array and reshape for prediction
    features_array = np.array(features).reshape(1, -1)
    
    # Make prediction using the loaded model
    prediction = model.predict(features_array)
    
    # Check the prediction output (1 = Malignant, 0 = Benign)
    diagnosis = "Malignant" if prediction[0] == 1 else "Benign"
    
    return render_template('index.html', diagnosis=diagnosis)

# Run the app
if __name__ == "__main__":
    # Retrieve the PORT environment variable or use the default port 5000 for local testing
    port = int(os.environ.get("PORT", 10000))
    # Run the app on all available network interfaces
    app.run(host="0.0.0.0", port=port)

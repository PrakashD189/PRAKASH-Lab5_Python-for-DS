from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the label encoders if you saved them
with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    car_name = request.form['Car_Name']
    year = int(request.form['Year'])
    present_price = float(request.form['Present_Price'])
    kms_driven = int(request.form['Kms_Driven'])
    fuel_type = request.form['Fuel_Type']
    seller_type = request.form['Seller_Type']
    transmission = request.form['Transmission']
    owner = int(request.form['Owner'])

    # Prepare the input data for prediction
    input_data = np.array([[car_name, year, present_price, kms_driven, fuel_type, seller_type, transmission, owner]])

    # Encode categorical variables
    for column in ['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission']:
        if column in label_encoders:
            input_data[0][0] = label_encoders['Car_Name'].transform([car_name])[0]
            input_data[0][4] = label_encoders['Fuel_Type'].transform([fuel_type])[0]
            input_data[0][5] = label_encoders['Seller_Type'].transform([seller_type])[0]
            input_data[0][6] = label_encoders['Transmission'].transform([transmission])[0]

    # Make prediction
    prediction = model.predict(input_data)

    return f'The predicted selling price is: {prediction[0]:.2f}'

if __name__ == '__main__':
    app.run(debug=True)
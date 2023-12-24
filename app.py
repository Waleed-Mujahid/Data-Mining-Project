from flask import Flask, jsonify, request
import pandas as pd
import pickle

# Initialize_ the Flask application
heart_attack_model_weights = open('./model_checkpoints/heart_attack_rf.pkl', 'rb')
heart_attack_model = pickle.load(heart_attack_model_weights)
heart_attack_scaler_file = open('./model_checkpoints/heart_attack_scaler.pkl', 'rb')
heart_attack_scaler = pickle.load(heart_attack_scaler_file)

# Initialize_ the Flask application
app = Flask(__name__)

def make_row_heart_attack_prediction(request_data):
    data = pd.DataFrame(request_data["items"])
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    data[columns_to_scale] = heart_attack_scaler.transform(data[columns_to_scale])
    rows = data.tail(request_data["items"].__len__())

    return rows

@app.route('/api/predict/heart_attack', methods=['POST'])
def predict_heart_attack():
    # Handle POST request
    request_data = request.json
    rows = make_row_heart_attack_prediction(request_data)
    predictions = heart_attack_model.predict(rows)
    response_data = {"predictions": [int(x) for x in predictions]}

    return jsonify(response_data)

@app.route('/api/predict/heart_disease', methods=['POST'])
def predict_heart_disease():
    # Handle POST request
    request_data = request.json
    predictions = predict_heart_attack(request_data)
    response_data = {"predictions": predictions}

    return jsonify(response_data)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)

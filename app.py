from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

CLUSTER_NAMES = {
    0: "Low Risk / Standard Home",
    1: "High Value / Premium Home",
    2: "High Risk / High Claim Profile"
}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([[
            data['home_age_years'],
            data['home_size_sqft'],
            data['estimated_home_value'],
            data['claim_history_count'],
            data['safety_score'],
            data['annual_premium_usd']
        ]])
        scaled_features = scaler.transform(features)
        cluster_id = int(model.predict(scaled_features)[0])
        return jsonify({
            'cluster_id': cluster_id,
            'risk_profile': CLUSTER_NAMES.get(cluster_id, "Unknown Profile")
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

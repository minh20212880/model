!pip install flask-ngrok
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Khởi tạo Flask app
app = Flask(__name__)

# Đường dẫn đến file mô hình
MODEL_PATH = "lgbm"  # Thay bằng đường dẫn chính xác đến mô hình trên Drive

# Load mô hình
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Định nghĩa API để dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded properly!'})

    try:
        # Lấy dữ liệu JSON từ request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'})

        # Chuyển dữ liệu thành DataFrame
        input_data = pd.DataFrame([data])

        # Thực hiện dự đoán
        prediction = model.predict(input_data)

        # Trả kết quả về dạng JSON
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

# Chạy Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

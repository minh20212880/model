from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Khởi tạo Flask app
app = Flask(__name__)

# Đường dẫn đến file mô hình
MODEL_PATH = "lgbm"  # Thay bằng đường dẫn chính xác đến mô hình trên GitHub hoặc hệ thống

# Load mô hình
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Hàm mã hóa dữ liệu categorical
def preprocess_input(input_data):
    # Ví dụ: mã hóa cột 'Gender' thành số (1: Male, 0: Female)
    if 'Gender' in input_data.columns:
        input_data['Gender'] = input_data['Gender'].map({'Male': 1, 'Female': 0})
    
    # Kiểm tra và mã hóa các cột categorical khác nếu cần
    # Thêm mã hóa tùy thuộc vào đặc trưng của dữ liệu bạn sử dụng

    return input_data

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

        # Tiền xử lý dữ liệu (mã hóa các cột categorical)
        input_data = preprocess_input(input_data)

        # Chuyển đổi tất cả cột trong DataFrame thành kiểu dữ liệu serializable
        input_data = input_data.astype(float)

        # Thực hiện dự đoán
        prediction = model.predict(input_data)

        # Trả kết quả dự đoán
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

# Chạy Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

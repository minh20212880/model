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

# Hàm chuyển đổi dữ liệu thành kiểu JSON serializable
def convert_to_serializable(data):
    if isinstance(data, np.ndarray):  # Nếu là NumPy array
        return data.astype(float).tolist()
    elif isinstance(data, pd.DataFrame):  # Nếu là Pandas DataFrame
        return data.astype(float).to_dict(orient="records")
    elif isinstance(data, (np.float64, np.int64)):  # Nếu là kiểu số NumPy
        return float(data) if isinstance(data, np.float64) else int(data)
    else:
        return data  # Nếu đã là kiểu serializable

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

        # Chuyển đổi tất cả cột trong DataFrame thành kiểu dữ liệu serializable
        input_data = input_data.astype(float)

        # Thực hiện dự đoán
        prediction = model.predict(input_data)

        # Chuyển kết quả dự đoán sang kiểu JSON serializable
        response = {'prediction': convert_to_serializable(prediction[0])}

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

# Chạy Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

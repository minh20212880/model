from flask import Flask, request, Response
import joblib
import pandas as pd
import numpy as np
import json  # Sử dụng json từ thư viện tiêu chuẩn

# Khởi tạo Flask app
app = Flask(__name__)

MODEL_PATH = "lgbm"  # Thay bằng đường dẫn chính xác đến mô hình của bạn

# Load mô hình
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Hàm chuyển đổi kiểu dữ liệu NumPy thành kiểu dữ liệu cơ bản của Python
def convert_numpy_to_python(data):
    if isinstance(data, list):
        return [convert_numpy_to_python(item) for item in data]
    elif isinstance(data, (np.int64, np.float64)):
        return data.item()
    return data

# Định nghĩa API để dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return Response(
            json.dumps({'error': 'Model not loaded properly!'}),
            status=500,
            mimetype='application/json'
        )

    try:
        # Lấy dữ liệu JSON từ request
        data = request.get_json()
        if not data:
            return Response(
                json.dumps({'error': 'No input data provided'}),
                status=400,
                mimetype='application/json'
            )

        # Chuyển dữ liệu thành DataFrame
        input_data = pd.DataFrame([data])

        # Thực hiện dự đoán
        prediction = model.predict(input_data)

        # Chuyển đổi kiểu dữ liệu NumPy thành Python cơ bản
        prediction_python = convert_numpy_to_python(prediction.tolist())

        # Trả kết quả về JSON
        response = {
            'prediction': prediction_python
        }
        return Response(
            json.dumps(response),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        # Trả lỗi nếu có
        error_response = {
            'error': str(e)
        }
        return Response(
            json.dumps(error_response),
            status=500,
            mimetype='application/json'
        )

# Chạy Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

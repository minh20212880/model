
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Khởi tạo Flask app
app = Flask(__name__)

# Đường dẫn đến file mô hình
MODEL_PATH = "lgbm"  # Thay bằng đường dẫn thực tế đến file mô hình của bạn

# Load mô hình
try:
    model = joblib.load(MODEL_PATH)
    print("Model đã được load thành công!")
except Exception as e:
    print(f"Lỗi khi load mô hình: {e}")
    model = None

# Định nghĩa API để dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Mô hình chưa được load đúng cách!'})

    try:
        # Nhận dữ liệu JSON từ request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Không có dữ liệu đầu vào!'})

        # Chuyển đổi dữ liệu JSON thành DataFrame
        input_data = pd.DataFrame([data])

        # Dự đoán bằng pipeline (bao gồm tiền xử lý)
        prediction = model.predict(input_data)

        # Chuyển đổi kết quả thành dạng JSON-serializable
        prediction = prediction.tolist()

        # Trả kết quả về
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

# Chạy Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

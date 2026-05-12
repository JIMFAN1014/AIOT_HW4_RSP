import argparse
import os
import sys
import time
from collections import Counter, deque

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import preprocess_flatten_image, preprocess_mobilenet_image


def load_smart_model(model_path):
    ext = os.path.splitext(model_path)[1].lower()

    if ext == '.pkl':
        print('[ML] Traditional model (.pkl) detected')
        import joblib
        model = joblib.load(model_path)
        return model, 'ml'

    if ext in ['.h5', '.keras']:
        print('[TF] TensorFlow/Keras model detected')
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        return model, 'tf'

    if ext in ['.pt', '.pth']:
        print('[Torch] PyTorch model detected')
        import torch
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        return model, 'torch'

    raise ValueError(f'❌ 不支援的模型格式: {ext}')


def preprocess_and_predict(model, model_type, roi):
    if model_type == 'ml':
        features = preprocess_flatten_image(roi).reshape(1, -1)
        prediction = model.predict(features)[0]
        return int(prediction), 1.0

    if model_type == 'tf':
        features = np.expand_dims(preprocess_mobilenet_image(roi), axis=0)
        prediction_probs = model.predict(features, verbose=0)[0]
        prediction = int(np.argmax(prediction_probs))
        confidence = float(np.max(prediction_probs))
        return prediction, confidence

    if model_type == 'torch':
        import torch
        rgb = preprocess_mobilenet_image(roi)
        features = np.expand_dims(rgb.transpose((2, 0, 1)), axis=0)
        tensor_features = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(tensor_features)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        return int(predicted.item()), float(confidence.item())

    raise ValueError(f'❌ 未知的模型類型: {model_type}')


def main():
    parser = argparse.ArgumentParser(description='Rock Paper Scissors Smart Inference')
    parser.add_argument('-m', '--model', type=str, default='rps_svm_model.pkl', help='Path to the model file')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ 錯誤：找不到模型檔案 '{args.model}'")
        return

    print(f'⏳ 載入模型 {args.model} 中...')
    try:
        model, model_type = load_smart_model(args.model)
        print('✅ 模型載入成功！')
    except Exception as error:
        print(f'❌ 載入模型失敗：{error}')
        return

    labels = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('❌ 無法開啟攝影機')
        return

    prev_time = 0
    prediction_interval = 3
    frame_count = 0
    result_text = 'Waiting'
    confidence_text = 'N/A'
    confidence_threshold = 0.75 if model_type == 'tf' else 0.0
    prediction_history = deque(maxlen=5)

    print("🎥 啟動智慧型攝影機... (按下 'q' 離開)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print('❌ 無法從攝影機讀取畫面')
            break

        h, w, _ = frame.shape
        box_size = 300
        start_x = max(0, w // 2 - box_size // 2)
        start_y = max(0, h // 2 - box_size // 2)
        end_x = min(w, start_x + box_size)
        end_y = min(h, start_y + box_size)
        roi = frame[start_y:end_y, start_x:end_x]

        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
        cv2.putText(frame, 'Put Hand Here', (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        try:
            if frame_count % prediction_interval == 0:
                prediction, confidence = preprocess_and_predict(model, model_type, roi)
                confidence_text = f'{confidence:.2f}'

                if confidence >= confidence_threshold:
                    prediction_history.append(prediction)
                    stable_prediction = Counter(prediction_history).most_common(1)[0][0]
                    result_text = labels.get(stable_prediction, 'Unknown')
                elif model_type == 'tf':
                    result_text = 'Uncertain'
        except Exception as error:
            result_text = 'Error'
            print(f'⚠️ 預測時發生錯誤: {error}')
            break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        frame_count += 1

        cv2.putText(frame, f'Prediction: {result_text}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f'Confidence: {confidence_text}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Type: {model_type.upper()}', (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow('Smart Camera Inference', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

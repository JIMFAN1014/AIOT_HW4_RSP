import cv2
import numpy as np
import argparse
import time
import os

def load_smart_model(model_path):
    """根據副檔名動態載入對應的模型與函式庫"""
    ext = os.path.splitext(model_path)[1].lower()
    
    if ext == '.pkl':
        print("🔍 偵測到傳統機器學習模型 (.pkl)")
        import joblib
        model = joblib.load(model_path)
        return model, "ml"
        
    elif ext in ['.h5', '.keras']:
        print("🔍 偵測到 TensorFlow/Keras 深度學習模型")
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        return model, "tf"
        
    elif ext in ['.pt', '.pth']:
        print("🔍 偵測到 PyTorch 深度學習模型")
        import torch
        model = torch.load(model_path, map_location='cpu') # 確保在 RPi 的 CPU 上能跑
        model.eval()
        return model, "torch"
        
    else:
        raise ValueError(f"❌ 不支援的模型格式: {ext}")

def preprocess_and_predict(model, model_type, roi):
    """根據模型類型，進行對應的影像前處理並回傳預測結果 (0: Rock, 1: Paper, 2: Scissors)"""
    if model_type == "ml":
        # 機器學習 (SVM, Random Forest)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        features = resized.flatten() / 255.0
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        return int(prediction)
        
    elif model_type == "tf":
        # TensorFlow / Keras (CNN)
        # 通常 CNN 會吃 RGB 圖片，形狀 (1, 64, 64, 3) 
        resized = cv2.resize(roi, (64, 64))
        # OpenCV 預設是 BGR，轉為 RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        features = rgb / 255.0
        features = np.expand_dims(features, axis=0) # 加上 batch 維度: (1, 64, 64, 3)
        prediction_probs = model.predict(features, verbose=0)
        return int(np.argmax(prediction_probs))
        
    elif model_type == "torch":
        # PyTorch (CNN)
        import torch
        resized = cv2.resize(roi, (64, 64))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # PyTorch 預設是 (N, C, H, W)
        features = rgb.transpose((2, 0, 1)) / 255.0
        features = np.expand_dims(features, axis=0)
        tensor_features = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(tensor_features)
            _, predicted = torch.max(outputs, 1)
        return int(predicted.item())

def main():
    parser = argparse.ArgumentParser(description="Rock Paper Scissors Smart Inference")
    parser.add_argument('-m', '--model', type=str, default='rps_svm_model.pkl',
                        help='Path to the model file')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ 錯誤：找不到模型檔案 '{args.model}'")
        return

    print(f"⏳ 載入模型 {args.model} 中...")
    try:
        model, model_type = load_smart_model(args.model)
        print("✅ 模型載入成功！")
    except Exception as e:
        print(f"❌ 載入模型失敗：{e}")
        return

    labels = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
    cap = cv2.VideoCapture(0)
    prev_time = 0

    print("🎥 啟動智慧型攝影機... (按下 'q' 離開)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 無法從攝影機讀取畫面")
            break

        h, w, _ = frame.shape
        box_size = 300
        start_x = max(0, w // 2 - box_size // 2)
        start_y = max(0, h // 2 - box_size // 2)
        end_x = min(w, start_x + box_size)
        end_y = min(h, start_y + box_size)
        
        roi = frame[start_y:end_y, start_x:end_x]
        
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
        cv2.putText(frame, "Put Hand Here", (start_x, start_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        try:
            # 根據模型類型進行預測
            prediction = preprocess_and_predict(model, model_type, roi)
            result_text = labels.get(prediction, "Unknown")
        except Exception as e:
            result_text = "Error"
            print(f"⚠️ 預測時發生錯誤: {e}")

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"Prediction: {result_text}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Type: {model_type.upper()}", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Smart Camera Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
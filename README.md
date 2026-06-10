# RSP_demo

這是一個以 **Raspberry Pi 4** 為目標平台的「剪刀、石頭、布（Rock / Paper / Scissors）」手勢辨識專案。

目前專案已完成：
- 傳統機器學習模型訓練與測試
- 多個替代模型實驗
- 深度學習模型訓練與測試
- 即時攝影機推論程式 `carema.py`
- 可在 Windows 本機與 Raspberry Pi 上進行模型驗證與 Demo

本 README 主要整理：
1. 專案目的
2. 專案結構
3. 模型與目前成果
4. 環境需求
5. 如何安裝與執行
6. 如何在 Raspberry Pi / VNC 上進行 Demo
7. 後續報告整理建議

---

# 1. 專案目標

本專案用來完成課程作業，核心要求如下：

- 在 Raspberry Pi 4 上成功執行模型測試與即時攝影機推論
- 比較多種模型架構的效果
- 至少呈現：
  - Accuracy
  - Precision
  - Recall
  - F1-score
- 最後選出較佳模型進行即時手勢辨識 Demo

目前實驗方向包含：
- Baseline SVM
- HOG + SVM
- PCA + Linear SVM
- PCA + Random Forest
- Random Forest
- Small CNN
- MobileNetV2
- EfficientNetB0

---

# 2. 專案結構

```text
RSP_demo/
├─ dataset/
│  ├─ train/
│  │  ├─ rock/
│  │  ├─ paper/
│  │  └─ scissors/
│  └─ test/
│     ├─ rock/
│     ├─ paper/
│     └─ scissors/
│
├─ demo/
│  ├─ carema.py
│  ├─ test.py
│  ├─ requirements.txt
│  ├─ rps_svm_model.pkl
│  ├─ rps_hog_svm_model.pkl
│  ├─ rps_pca_svm_model.pkl
│  ├─ rps_pca_rf_model.pkl
│  ├─ rps_rf_model.pkl
│  ├─ rps_small_cnn.keras
│  ├─ rps_mobilenetv2.keras
│  └─ rps_efficientnetb0.keras
│
├─ train/
│  ├─ requirements.txt
│  ├─ train_svm.py
│  ├─ train_hog_svm.py
│  ├─ train_pca_svm.py
│  ├─ train_pca_rf.py
│  ├─ train_rf.py
│  ├─ train_small_cnn.py
│  ├─ train_mobilenetv2.py
│  └─ train_efficientnetb0.py
│
├─ model_utils.py
├─ README.md
├─ README_old.md
└─ 聊天紀錄.md
```

---

# 3. 各檔案用途說明

## 3.1 dataset/

資料集分成：
- `train/`：訓練資料
- `test/`：測試資料

每個資料夾內包含三類：
- `rock`
- `paper`
- `scissors`

目前程式都依照這個資料夾結構讀資料。

---

## 3.2 model_utils.py

這是整個專案的共用工具模組，內容包含：

- 類別標籤常數
- 資料集路徑處理
- 影像載入
- 傳統模型前處理
- CNN / MobileNetV2 前處理
- HOG 特徵擷取
- 共用評估輸出

目前重要常數：
- `IMAGE_SIZE = (64, 64)`：傳統模型使用
- `CNN_IMAGE_SIZE = (64, 64)`：Small CNN 使用
- `MOBILENET_IMAGE_SIZE = (96, 96)`：MobileNetV2 使用
- `EFFICIENTNET_IMAGE_SIZE = (224, 224)`：EfficientNetB0 使用

---

## 3.3 train/

### `train/train_svm.py`
訓練 baseline SVM。

流程：
- 灰階
- resize 到 `64x64`
- flatten
- SVM(RBF)

---

### `train/train_hog_svm.py`
訓練 HOG + SVM。

流程：
- 灰階
- resize 到 `64x64`
- HOG 特徵
- SVM

---

### `train/train_pca_svm.py`
訓練 PCA + Linear SVM。

流程：
- 灰階
- resize 到 `64x64`
- flatten
- StandardScaler
- PCA
- LinearSVC

---

### `train/train_pca_rf.py`
訓練 PCA + Random Forest。

流程：
- 灰階
- resize 到 `64x64`
- flatten
- StandardScaler
- PCA
- RandomForestClassifier

---

### `train/train_rf.py`
訓練直接使用 flatten 特徵的 Random Forest。

---

### `train/train_small_cnn.py`
訓練小型自建 CNN。

輸入：
- RGB
- `64x64`

架構概念：
- Conv(32) x2 + MaxPool
- Conv(64) x2 + MaxPool
- Conv(128) + MaxPool
- GlobalAveragePooling
- Dense + Dropout
- Softmax(3類)

---

### `train/train_mobilenetv2.py`
訓練 MobileNetV2。

輸入：
- RGB
- `96x96`

使用 Keras / TensorFlow 的 MobileNetV2 做 transfer learning。

---

### `train/train_efficientnetb0.py`
訓練 EfficientNetB0。

輸入：
- RGB
- `224x224`

使用 Keras / TensorFlow 的 EfficientNetB0 做 transfer learning。

---

## 3.4 demo/

### `demo/test.py`
統一測試入口。

功能：
- 可載入 `.pkl`
- 可載入 `.keras` / `.h5`
- 自動依模型類型選正確前處理
- 輸出 Accuracy、Precision、Recall、F1-score

---

### `demo/carema.py`
即時攝影機推論程式。

功能：
- 開啟攝影機
- 顯示 ROI 區域
- 載入指定模型做即時預測
- 顯示：
  - Prediction
  - FPS
  - Model Type

目前支援：
- `.pkl`：傳統模型
- `.keras` / `.h5`：TensorFlow/Keras 模型
- `.pt` / `.pth`：PyTorch 模型（目前專案未主用）

---

# 4. 目前模型成果

目前已完成的主要實驗結果如下：

| 模型 | 類型 | Accuracy |
|---|---|---:|
| Baseline SVM | 傳統 ML | 68.28% |
| HOG + SVM | 傳統 ML | 63.44% |
| PCA + Linear SVM | 傳統 ML | 61.29% |
| PCA + Random Forest | 傳統 ML | 52.15% |
| Random Forest | 傳統 ML | 56.45% |
| Small CNN | 深度學習 | 83.33% |
| MobileNetV2 | 深度學習 | 93.82% |
| EfficientNetB0 | 深度學習 | 87.37% |

目前最值得作為報告主比較的兩個模型：
- **Small CNN**（83.33%）
- **EfficientNetB0**（87.37%，實機 Demo 主力）

但其他模型仍保留，作為完整實驗歷程與比較依據。

---

# 5. 環境需求

## 5.1 Windows 本機

目前專案在 Windows 本機是使用 `uv` 建立虛擬環境測試。

建議版本：
- Python 3.13
- uv

---

## 5.2 Raspberry Pi

建議環境：
- Raspberry Pi 4
- Python 3
- uv
- 可使用 VNC 連進 GUI 桌面
- 有接攝影機

---

# 6. 安裝方式

本專案統一以 **uv** 作為套件與虛擬環境管理工具。

## 6.1 Windows 本機安裝

在專案根目錄執行：

```bash
uv venv .venv
uv pip install -r train/requirements.txt
uv pip install -r demo/requirements.txt
```

如果只想做推論測試，可至少安裝：

```bash
uv pip install -r demo/requirements.txt
```

---

## 6.2 Raspberry Pi 安裝

先進入專案資料夾後，建議和 Windows 一樣統一使用 `uv`：

```bash
uv venv .venv
uv pip install -r train/requirements.txt
uv pip install -r demo/requirements.txt
```

如果樹莓派只做 Demo、不重新訓練模型，也可只安裝：

```bash
uv venv .venv
uv pip install -r demo/requirements.txt
```

---

# 7. 如何訓練模型

以下指令都在專案根目錄執行。

## 7.1 Baseline SVM

```bash
uv run python train/train_svm.py
```

---

## 7.2 HOG + SVM

```bash
uv run python train/train_hog_svm.py
```

---

## 7.3 PCA + Linear SVM

```bash
uv run python train/train_pca_svm.py
```

---

## 7.4 PCA + Random Forest

```bash
uv run python train/train_pca_rf.py
```

---

## 7.5 Random Forest

```bash
uv run python train/train_rf.py
```

---

## 7.6 Small CNN

```bash
uv run python train/train_small_cnn.py
```

---

## 7.7 MobileNetV2

```bash
uv run python train/train_mobilenetv2.py
```

---

## 7.8 EfficientNetB0

```bash
uv run python train/train_efficientnetb0.py
```

---

# 8. 如何測試模型

統一測試腳本：

```bash
uv run python demo/test.py -m <模型檔>
```

## 範例

### Baseline SVM
```bash
uv run python demo/test.py -m rps_svm_model.pkl
```

### Small CNN
```bash
uv run python demo/test.py -m rps_small_cnn.keras
```

### MobileNetV2
```bash
uv run python demo/test.py -m rps_mobilenetv2.keras
```

### EfficientNetB0
```bash
uv run python demo/test.py -m rps_efficientnetb0.keras
```

---

# 9. 如何做即時攝影機 Demo

## 9.1 Small CNN

```bash
uv run python demo/carema.py -m demo/rps_small_cnn.keras
```

## 9.2 MobileNetV2

```bash
uv run python demo/carema.py -m demo/rps_mobilenetv2.keras
```

## 9.3 EfficientNetB0

```bash
uv run python demo/carema.py -m demo/rps_efficientnetb0.keras
```

## 9.4 Baseline SVM

```bash
uv run python demo/carema.py -m demo/rps_svm_model.pkl
```

---

# 10. Raspberry Pi / VNC 使用說明

本專案後續預計在 Raspberry Pi 4 上使用，並透過 **VNC** 觀看 GUI。

這代表：
- `carema.py` 的 OpenCV 視窗可保留
- 可直接在 VNC 桌面中看到攝影機畫面
- 不需要另外改成純 SSH 無 GUI 模式

建議實測流程：
1. 先測 `Small CNN`
2. 再測 `MobileNetV2`
3. 若需要更強的 transfer learning 基準，可再測 `EfficientNetB0`
4. 比較：
   - 啟動速度
   - FPS
   - 預測穩定度
   - Rock / Paper / Scissors / Error 的表現

補充：
- `EfficientNetB0` 可能比 `MobileNetV2` 在 Raspberry Pi 4 上更慢，但若對實際光線變化更穩，仍值得作為比較模型。
- 若最終目標是錄製 Demo 影片，仍應綜合考慮辨識穩定度與 FPS，而不是只看離線準確率。

最後選一個最適合錄影片的模型。

---

# 11. 報告撰寫建議

目前專案保留了所有做過的模型，因此報告可以分成兩層：

## 11.1 完整實驗歷程
可提到：
- baseline SVM
- HOG + SVM
- PCA + Linear SVM
- PCA + Random Forest
- Random Forest
- Small CNN
- MobileNetV2

## 11.2 最終主比較重點
建議主講兩個模型：
- **Small CNN**
- **MobileNetV2**

原因：
- 兩者都明確超過 baseline
- 一個代表自建 CNN
- 一個代表成熟的輕量化 transfer learning 模型
- 敘事最完整，也最好寫出比較

---

# 12. 作業繳交項目

## Part 1 — 實機執行截圖

截圖存放於 `part_1/` 資料夾：

- `part_1/test.py.jpg`：於 Raspberry Pi 4 執行 `python3 test.py` 的結果截圖
- `part_1/carema.py.jpg`：於 Raspberry Pi 4 執行 `python3 carema.py` 的攝影機推論截圖

---

## Part 2 — Demo 展示影片

**YouTube 連結**：https://youtu.be/hNdXaP3jzTQ

影片包含 10 次手勢辨識（Rock × 3、Paper × 3、Scissors × 3、Error × 1），使用模型為 **EfficientNetB0**，實測 FPS 約 22.5。

---

## Part 3 — 報告

完整報告請見：[報告.md](報告.md)

---

# 13. 備註

- 舊版 README 已備份為：`README_old.md`
- 專案中的對話與開發歷程記錄在：`聊天紀錄.md`
- 完整報告內容請見：`報告.md`

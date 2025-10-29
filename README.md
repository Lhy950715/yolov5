# 使用YOLOv5 官方 COCO128 資料集訓練流程（在 Google Colab 執行）

參考網站:https://blog.csdn.net/Mr_Wanderer/article/details/119521955

##  第 1 步：啟用 GPU
1. 開啟 [Google Colab](https://colab.research.google.com/)
2. 點選上方選單：**執行階段 → 變更執行階段類型**
3. 將「硬體加速器」選為 **GPU**，然後按「儲存」。

<img width="711" height="650" alt="step 1" src="https://github.com/user-attachments/assets/d9c2cd0f-ef66-42cc-a7d6-fe0337d1ebb3" />

4. 確認 GPU 可用：
```python
!nvidia-smi
```

<img width="971" height="528" alt="step 1 1" src="https://github.com/user-attachments/assets/4bc3f650-7112-4905-97d4-c9efda1e7906" />

## 第 2 步：安裝 YOLOv5 並載入環境

```python
# 下載 YOLOv5 原始碼
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5

# 安裝依賴套件
!pip install -r requirements.txt
```
<img width="1766" height="578" alt="step 2" src="https://github.com/user-attachments/assets/761f056a-5c91-4f58-8fb6-2fafd40f3efb" />

<img width="1680" height="712" alt="step 2 1" src="https://github.com/user-attachments/assets/4df56376-ea20-4698-85cc-ab618ad975f1" />

## 第 3 步：檢查資料集

### COCO128 為 YOLOv5 官方內建的小型資料集，會自動下載。
因為參考網站的口罩資料集無法存取，所以直接用 YOLOv5 官方內建資料集

可執行以下指令確認：

```python
!cat data/coco128.yaml
```
會看到路徑指向：
```bath
train: ../datasets/coco128/images/train2017
val: ../datasets/coco128/images/train2017
```
<img width="1335" height="698" alt="step 3" src="https://github.com/user-attachments/assets/c5cba4e5-f956-422a-a076-081864c12237" />

## 第 4 步：開始訓練

執行下面這行指令，使用官方資料訓練 YOLOv5s（小型模型）：
```python
!python train.py --data data/coco128.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --epochs 3 --batch-size 16
```
說明：

- `--data`: 指定資料集設定檔（這裡是官方 coco128）

- `--cfg`: 模型架構（yolov5s = small）

- `--weights`: 預訓練權重（會自動下載）

- `--epochs`: 訓練 3 個回合（可改大一點）

- `--batch-size`: 每批訓練的圖片數

<img width="1769" height="708" alt="step 4" src="https://github.com/user-attachments/assets/cb639232-f594-4869-88c7-8421806965d9" />

<img width="932" height="620" alt="step 4 1" src="https://github.com/user-attachments/assets/21c53850-619b-4034-8cff-cfa196a39339" />

可以看到完成後，結果會儲存在： **runs/train/exp/**

## 第 5 步：查看訓練結果

執行以下程式可以查看訓練結果：

```python
!ls runs/train/exp
```

<img width="815" height="348" alt="step 5" src="https://github.com/user-attachments/assets/fa09c8e6-b70c-4e95-bf6a-db9118aa3734" />

主要內容如下：

### 1. 圖表與曲線
- `results.png` → 訓練過程總覽圖（loss、precision、recall 等）
- `F1_curve.png` → F1 分數曲線
- `P_curve.png` → Precision 曲線
- `R_curve.png` → Recall 曲線
- `PR_curve.png` → Precision-Recall 曲線
- `confusion_matrix.png` → 混淆矩陣，檢視模型對各類別預測正確/錯誤情況
- `labels.jpg` → 真實標籤分佈
- `labels_correlogram.jpg` → 標籤相關性圖（觀察不同類別標籤間的關聯）

### 2. 訓練樣本示意
- `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg` → 訓練過程中取樣 batch 的圖像，方便檢查標註和預處理是否正確
- `val_batch0_labels.jpg`, `val_batch1_labels.jpg`, `val_batch2_labels.jpg` → 驗證資料的標註示意圖
- `val_batch0_pred.jpg`, `val_batch1_pred.jpg`, `val_batch2_pred.jpg` → 驗證資料的模型預測結果圖

### 3. 模型設定與超參數
- `hyp.yaml` → 超參數設定檔（learning rate、augment 等）
- `opt.yaml` → 訓練選項配置（batch-size、epoch 等）

### 4. 訓練紀錄與結果
- `results.csv` → 訓練過程數據紀錄（每 epoch 的 loss、precision、recall 等）
- `events.out.tfevents.*` → TensorBoard 訓練日誌，可用於可視化訓練過程

### 5. 模型權重
- `weights/` → 存放訓練得到的模型權重：
  - `best.pt` → 最佳模型（驗證集表現最好）
  - `last.pt` → 最後一輪模型權重

## 第 6 步：使用訓練好的模型做推論（偵測）

```python
!python detect.py --weights runs/train/exp/weights/best.pt --source data/images
```
偵測結果會存在：**runs/detect/exp/**

<img width="1771" height="304" alt="step 6" src="https://github.com/user-attachments/assets/84218fe6-bbc0-4259-aa31-d0fbe6e69f96" />

##  第 7 步：在 Colab 中顯示偵測結果
```python
from IPython.display import Image
Image(filename='runs/detect/exp/zidane.jpg')
```
<img width="557" height="108" alt="step 7" src="https://github.com/user-attachments/assets/cd5c658b-2430-4e26-803b-b24505a53672" />

## 輸出結果:

![output](https://github.com/user-attachments/assets/7975d1d8-8e90-447b-ac92-dfe6bed61e9d)


## 心得與收穫

這次在 Google Colab 上使用 YOLOv5 進行 COCO128 資料集訓練，整體流程非常順暢。由於使用的是官方資料集，因此不需自行準備標註或影像資料，適合**初學者**快速上手深度學習影像辨識的訓練流程。透過 train.py 指令可以輕鬆完成模型訓練與評估，而結果圖 results.png 也能清楚地觀察到損失值（loss）與準確率的變化。

在推論階段輸出的偵測照片中，可以清楚看出模型對物件的辨識框與標籤呈現得非常準確，這代表模型與資料之間的相關度相當高，能夠有效識別出影像中的關鍵物件。從這次實作中，我更深入理解了 YOLOv5 的訓練架構，也體驗到在 Colab 環境中進行 GPU 加速訓練的便利性。未來若要應用於自訂資料集，只要替換資料集與 data.yaml 即可延伸使用，是一個非常靈活且實用的影像辨識工具。

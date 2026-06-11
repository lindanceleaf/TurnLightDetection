# CCTV Turn Signal YOLO Training

這個資料夾已經放好一套最小可用流程，先以白天資料 `yolo_dataset_day` 訓練四類偵測模型：

| id | class |
|---:|---|
| 0 | left_on |
| 1 | left_off |
| 2 | right_on |
| 3 | right_off |

## 1. 建議建立 Python 3.11 環境

目前這台電腦預設是 Python 3.13。若 `torch` 安裝失敗，建議另外開 Python 3.11 環境：

```powershell
conda create -n turn_signal python=3.11 -y
conda activate turn_signal
```

安裝 PyTorch CUDA 版時，RTX 3080 要確認裝到 CUDA 版，不要裝成 CPU 版。

建議直接跑：

```powershell
.\install_cuda_pytorch.ps1
```

確認 GPU：

```powershell
python check_cuda.py
```

如果看到 `True` 和 `NVIDIA GeForce RTX 3080` 就可以開始訓練。

## 2. 檢查資料集

```powershell
python audit_dataset.py
```

要特別看：

- `left_off` 和 `right_off` 數量是否太少
- train/val 是否有同一個 `clip_xxxx` 重疊
- label 是否有座標超出 `0..1`

目前原始白天資料的 train/val 有 clip 重疊。建議先建立一份依 clip 重新切分的資料集：

```powershell
python prepare_clip_split.py --out yolo_dataset_day_clip_split
```

然後用新資料訓練：

```powershell
python train_day.py --data yolo_dataset_day_clip_split/data.yaml --model yolov8s.pt --imgsz 960 --batch 8 --epochs 100 --device 0 --name yolov8s_img960_clip_split
```

## 3. 先訓練白天模型

快速確認流程可以先跑：

```powershell
.\train_fast.ps1
```

RTX 3080 建議先跑這個 baseline：

```powershell
.\train_balanced_3080.ps1
```

如果顯存還有空，可以試：

```powershell
python train_day.py --model yolov8s.pt --imgsz 1280 --batch 4 --epochs 100 --device 0 --name yolov8s_img1280
```

或：

```powershell
python train_day.py --model yolov8m.pt --imgsz 960 --batch 4 --epochs 100 --device 0 --name yolov8m_img960
```

輸出模型會在：

```text
runs/turn_signal_day/<run_name>/weights/best.pt
```

## 4. 驗證模型

```powershell
python validate_model.py --model runs/turn_signal_day/yolov8s_img960/weights/best.pt
```

重點不要只看 mAP，要看 confusion matrix。這題最重要的是：

- `left_on` vs `left_off`
- `right_on` vs `right_off`
- `left_*` vs `right_*`

## 5. 對圖片或影片推論

單張圖片或資料夾：

```powershell
python predict_video.py --model runs/turn_signal_day/yolov8s_img960/weights/best.pt --source path/to/image_or_folder
```

影片建議開 tracking，因為方向燈會閃，單張 frame 可能剛好是暗的：

```powershell
python predict_video.py --model runs/turn_signal_day/yolov8s_img960/weights/best.pt --source path/to/video.mp4 --track
```

輸出會在：

```text
runs/predict_turn_signal/result
```

如果手邊沒有原始影片，可以先把 val frame 合成 demo mp4：

```powershell
python make_demo_video.py --clip clip_0001 --out demo_clip_0001.mp4 --max-frames 300 --fps 20
python predict_video.py --model runs/detect/runs/turn_signal_day/yolov8n_img640_fast/weights/best.pt --source demo_clip_0001.mp4 --track --imgsz 640 --name fast_demo_clip_0001
```

## 6. 這題的實務判斷策略

訓練時先讓 YOLO 學四類：

```text
left_on / left_off / right_on / right_off
```

實際跑 CCTV 影片時，不建議只看單一 frame。建議用 `--track` 追同一台車，然後對最近 30 到 60 frames 的結果做投票或平均：

```text
同一台車連續多幀被判成 left_on -> 左轉有打燈
同一台車連續多幀被判成 left_off -> 左轉沒打燈
```

這樣會比單張圖片穩很多。

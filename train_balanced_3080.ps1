$ErrorActionPreference = "Stop"

python train_day.py `
  --data yolo_dataset_day_clip_split/data.yaml `
  --model yolov8s.pt `
  --imgsz 960 `
  --batch 16 `
  --epochs 50 `
  --patience 12 `
  --device 0 `
  --name yolov8s_img960_b16_balanced

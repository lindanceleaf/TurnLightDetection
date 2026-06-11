$ErrorActionPreference = "Stop"

python train_day.py `
  --data yolo_dataset_day_clip_split/data.yaml `
  --model yolov8n.pt `
  --imgsz 640 `
  --batch 32 `
  --epochs 20 `
  --patience 8 `
  --device 0 `
  --name yolov8n_img640_fast

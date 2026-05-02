# ◈ YOLOForge — No-Code YOLO Training Platform

> Train any YOLO model without writing a single line of code.  
> Powered by **LangGraph** (Python backend) + **React/Vite** (JSX frontend).

---

## 📁 Project Structure

```
YOLOForge/
├── frontend/          ← React + Vite (JSX)
│   ├── src/
│   │   ├── App.jsx    ← Full platform UI (1273 lines)
│   │   └── main.jsx   ← React entry point
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
└── backend/           ← Python + LangGraph + FastAPI
    ├── tools/
    │   └── yolo_tools.py   ← 9 @tool functions
    ├── graph/
    │   └── pipeline.py     ← LangGraph StateGraph (7 nodes)
    ├── api/
    │   └── server.py       ← FastAPI (12 routes)
    ├── utils/
    │   └── schemas.py      ← Pydantic models
    ├── main.py             ← Uvicorn entrypoint
    └── requirements.txt
```

---

## 🚀 Quick Start

### 1. Start the Backend

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
# → API running at http://localhost:8000
# → Docs at http://localhost:8000/docs
```

### 2. Start the Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run dev server
npm run dev
# → App running at http://localhost:3000
```

### 3. Open in Browser

Go to **http://localhost:3000**

---

## 🧠 Features

### Frontend (React JSX)
- **5 YOLO Tasks** — Detect, Segment, Classify, Pose, OBB
- **48+ Models** — YOLOv5 → YOLO26, RT-DETR, YOLO-World, YOLO-NAS
- **Live Augmentation Preview** — 5 CV scenes rendered on canvas, real-time updates
  - Street · Aerial · Medical · Wildlife · Factory
  - HSV color, rotation, flip, mosaic, erasing — all visualized instantly
- **Dataset Format Guide** — YOLO, COCO JSON, Pascal VOC, Roboflow
- **Full Hyperparameter Control** — all YOLO training params with sliders
- **Fine-tune or Train from Scratch** — per model
- **Platform** — Google Colab or Kaggle notebook output

### Backend (LangGraph Python)
- **7-node StateGraph Pipeline**:
  ```
  validate → clean → split → eda → yaml → notebook → readme
  ```
- **9 LangGraph `@tool` functions**:
  - `validate_config` — model/task compat, HP range checks
  - `clean_dataset` — remove corrupt, duplicates, fix labels
  - `split_dataset` — train/val/test YOLO folder structure
  - `generate_data_yaml` — auto-detect or write data.yaml
  - `analyze_dataset` — class distribution, imbalance ratio
  - `generate_notebook_cells` — 26-cell training notebook
  - `assemble_ipynb` — valid .ipynb JSON
  - `build_yaml_string` — lightweight yaml preview
  - `generate_readme` — full README with config table
- **12 FastAPI Routes**:
  - `POST /generate` — full pipeline
  - `POST /upload-dataset` — ZIP upload + extract
  - `POST /download/notebook` — stream .ipynb
  - `POST /tools/validate` — config check only
  - `POST /tools/clean/{session_id}`
  - `POST /tools/split/{session_id}`
  - `POST /tools/eda/{session_id}`
  - `GET  /pipeline/graph` — topology info
  - `GET  /tools` — list all tools
  - `GET  /health`

---

## 📓 Generated Notebook

The generated `.ipynb` notebook contains **26 cells** covering:

1. Install dependencies (`ultralytics`, etc.)
2. Mount Google Drive / Kaggle workspace
3. Upload & extract dataset ZIP
4. **Dataset cleaning** (corrupt, duplicates, label fix)
5. **Train / Val / Test split**
6. **data.yaml generation** (auto class detection)
7. **EDA** — class distribution, bbox stats, plots
8. Load YOLO model (fine-tune `.pt` or scratch `.yaml`)
9. **Full training** with all hyperparameters
10. Training curves visualization
11. **Validation** — mAP50, mAP50-95, precision, recall
12. Confusion matrix display
13. **Model export** — ONNX, TorchScript, TFLite, CoreML
14. Save to Google Drive / Kaggle output
15. **Inference preview** — sample predictions grid
16. Quick inference snippet

---

## 🛠 Build for Production

```bash
# Frontend
cd frontend
npm run build        # Output in frontend/dist/

# Backend (with gunicorn for production)
cd backend
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 📋 Supported Models

| Family | Models | Tasks |
|--------|--------|-------|
| YOLOv5 | n/s/m/l/x | detect |
| YOLOv8 | n/s/m/l/x | detect, segment, classify, pose, obb |
| YOLOv9 | t/s/m/c/e | detect, segment |
| YOLOv10 | n/s/m/b/l/x | detect, segment |
| YOLO11 | n/s/m/l/x | detect, segment, classify, pose, obb |
| YOLO26 | n/s/m | detect |
| RT-DETR | L/X | detect |
| YOLO-World | S/M/L | detect |
| YOLO-NAS | S/M/L | detect |

---

*YOLOForge — Built with LangGraph + FastAPI + React*

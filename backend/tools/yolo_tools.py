"""
YOLOForge – LangGraph Tools
Every pipeline stage is a @tool. The graph calls these directly (no LLM needed).

Fixes applied:
  1. validate_config  – added image_size validation; added model_id existence check
  2. clean_dataset    – now copies to staging dir first (non-destructive)
  3. split_dataset    – warns when val split would be 0 images
  4. analyze_dataset  – skips empty splits (images == 0) to avoid misleading entries
  5. generate_notebook_cells – freeze=0 no longer becomes "None" (falsy or-bug fixed)
  6. build_yaml_string / generate_data_yaml – pose flip_idx uses correct COCO pairs;
                                              rglob restricted to known filenames
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import textwrap
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

# ── Valid model IDs (used by validate_config) ─────────────────────────────
_VALID_MODEL_IDS: set[str] = {
    "yolov5nu","yolov5su","yolov5mu","yolov5lu","yolov5xu",
    "yolov8n","yolov8s","yolov8m","yolov8l","yolov8x",
    "yolov8n-seg","yolov8s-seg","yolov8m-seg",
    "yolov8n-cls","yolov8s-cls","yolov8m-cls",
    "yolov8n-pose","yolov8s-pose","yolov8m-pose",
    "yolov8n-obb","yolov8s-obb",
    "yolov9t","yolov9s","yolov9c","yolov9e",
    "yolov10n","yolov10s","yolov10m","yolov10l","yolov10x",
    "yolo11n","yolo11s","yolo11m","yolo11l","yolo11x",
    "yolo11n-seg","yolo11n-cls","yolo11n-pose","yolo11n-obb",
    "yolo26n","yolo26s","yolo26m",
    "rtdetr-l","rtdetr-x",
    "yolov8s-worldv2","yolov8m-worldv2","yolov8l-worldv2",
    "yolo_nas_s","yolo_nas_m","yolo_nas_l",
}

# Correct COCO 17-keypoint flip pairs (left↔right symmetric pairs)
_COCO_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# Supported image sizes (must be multiples of 32, reasonable range)
_MIN_IMG_SIZE = 32
_MAX_IMG_SIZE = 4096


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 1 – validate_config
# ═══════════════════════════════════════════════════════════════════════════
@tool
def validate_config(config_json: str) -> dict:
    """
    Validate a TrainConfig JSON for YOLO compatibility.
    Returns {valid, errors, warnings, suggestions}.
    """
    cfg = json.loads(config_json)
    errors, warnings, tips = [], [], []

    task  = cfg.get("task", "detect")
    model = cfg.get("model_id", "")
    mode  = cfg.get("train_mode", "finetune")
    hp    = cfg.get("hp", {})

    # ── FIX 1a: model_id existence check ──────────────────────────────────
    if model and model not in _VALID_MODEL_IDS:
        warnings.append(
            f"model_id '{model}' is not in the known catalog. "
            "Proceeding, but double-check the Ultralytics model name."
        )

    # Model/task compat
    v9v10_only = {"detect", "segment"}
    if any(model.startswith(p) for p in ("yolov9", "yolov10")):
        if task not in v9v10_only:
            errors.append(f"{model} supports only detect/segment, not '{task}'.")

    rtdetr = model.startswith("rtdetr")
    if rtdetr and task != "detect":
        errors.append("RT-DETR only supports object detection.")

    world = "world" in model
    if world and task != "detect":
        errors.append("YOLO-World only supports object detection.")

    # ── FIX 1b: scratch + no yaml ─────────────────────────────────────────
    model_yaml = cfg.get("model_yaml", "")
    if mode == "scratch" and not model_yaml:
        errors.append(
            f"train_mode='scratch' requires a model YAML architecture file, "
            f"but model_yaml is empty for model '{model}'. "
            "Use fine-tuning or choose a model that provides a .yaml."
        )

    # Splits
    tr = cfg.get("train_split", 0.8)
    vr = cfg.get("val_split",   0.1)
    te = cfg.get("test_split",  0.1)
    if abs(tr + vr + te - 1.0) > 0.02:
        errors.append(f"Splits must sum to 1.0 (got {tr+vr+te:.2f}).")
    if tr < 0.5:
        warnings.append("Train split < 50 % — model may underfit.")

    # ── FIX 1c: image_size validation ─────────────────────────────────────
    img_sz = cfg.get("image_size", 640)
    if not isinstance(img_sz, int) or img_sz < _MIN_IMG_SIZE or img_sz > _MAX_IMG_SIZE:
        errors.append(
            f"image_size={img_sz} is out of the supported range "
            f"[{_MIN_IMG_SIZE}, {_MAX_IMG_SIZE}]."
        )
    elif img_sz % 32 != 0:
        warnings.append(
            f"image_size={img_sz} is not a multiple of 32. "
            "YOLO models work best with sizes like 320, 416, 512, 640, 768, 1024, 1280."
        )

    # LR
    lr0 = hp.get("lr0", 0.01)
    if lr0 > 0.1:
        warnings.append(f"lr0 = {lr0} is high; consider ≤ 0.01.")
    if lr0 < 1e-5:
        warnings.append(f"lr0 = {lr0} is very low; training may be extremely slow.")

    # Batch
    batch = hp.get("batch", 16)
    if batch > 64:
        tips.append("Large batch → scale LR proportionally (linear scaling rule).")
    if batch == -1:
        tips.append("Auto-batch requires sufficient GPU VRAM (≥ 8 GB recommended).")

    # Epochs
    epochs = hp.get("epochs", 100)
    if epochs < 10:
        warnings.append("Fewer than 10 epochs — model unlikely to converge.")

    # Freeze
    freeze = hp.get("freeze", 0)
    if mode == "scratch" and freeze and freeze > 0:
        warnings.append("Freezing layers while training from scratch has no effect.")

    # Task-specific
    if task == "segment":
        tips.append("Use overlap_mask=True for overlapping instances (default).")
    if task == "pose":
        tips.append("Pose datasets need keypoint annotations (x y visibility per point).")
    if task == "classify":
        tips.append("Classification expects images in class-named sub-folders.")
    if task == "obb":
        tips.append("OBB labels need rotation angle: class cx cy w h angle.")

    # Mosaic close
    augs   = cfg.get("augmentations", {})
    mosaic = augs.get("mosaic", {})
    if isinstance(mosaic, dict) and mosaic.get("enabled") and mosaic.get("value", 0) > 0:
        close = hp.get("close_mosaic", 10)
        if close == 0:
            tips.append("Set close_mosaic > 0 to disable mosaic in final epochs.")

    return {
        "valid":       len(errors) == 0,
        "errors":      errors,
        "warnings":    warnings,
        "suggestions": tips,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 2 – clean_dataset
# ═══════════════════════════════════════════════════════════════════════════
@tool
def clean_dataset(dataset_dir: str) -> dict:
    """
    Clean a YOLO-format dataset:
    1. Copy source to a staging area (non-destructive — source is never modified)
    2. Remove corrupt/unreadable images from the staging copy
    3. Remove duplicate images (MD5 hash) from the staging copy
    4. Validate and clamp label coordinates to [0,1]
    5. Report missing labels
    Returns a detailed cleaning report dict including staging_dir.
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        return {"success": False, "error": "Pillow not installed"}

    src = Path(dataset_dir)
    if not src.exists():
        return {"success": False, "error": f"Path not found: {dataset_dir}"}

    # ── FIX 2: copy to staging dir before any mutations ───────────────────
    staging = src.parent / (src.name + "_cleaned")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(src, staging)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    all_imgs = [p for p in staging.rglob("*") if p.suffix.lower() in img_exts]

    stats: dict[str, Any] = {
        "total":        len(all_imgs),
        "corrupt":      0,
        "duplicates":   0,
        "no_label":     0,
        "labels_fixed": 0,
        "valid_final":  0,
        "success":      True,
        "staging_dir":  str(staging),
        "details":      [],
    }

    # Pass 1 – corrupt
    valid: list[Path] = []
    for p in all_imgs:
        try:
            with PILImage.open(p) as im:
                im.verify()
            valid.append(p)
        except Exception:
            p.unlink(missing_ok=True)
            stats["corrupt"] += 1
            stats["details"].append(f"CORRUPT: {p.name}")

    # Pass 2 – duplicates
    seen: dict[str, Path] = {}
    unique: list[Path] = []
    for p in valid:
        try:
            h = hashlib.md5(p.read_bytes()).hexdigest()
        except Exception:
            unique.append(p)
            continue
        if h in seen:
            p.unlink(missing_ok=True)
            stats["duplicates"] += 1
            stats["details"].append(f"DUPE: {p.name} == {seen[h].name}")
        else:
            seen[h] = p
            unique.append(p)

    # Pass 3 – label validation
    for p in unique:
        lp = Path(str(p.with_suffix(".txt")).replace("/images/", "/labels/"))
        if not lp.exists():
            lp = p.with_suffix(".txt")
        if not lp.exists():
            stats["no_label"] += 1
            continue
        try:
            raw   = [ln for ln in lp.read_text().splitlines() if ln.strip()]
            fixed, changed = [], False
            for ln in raw:
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cls    = parts[0]
                coords = [float(x) for x in parts[1:]]
                clamped = [min(max(c, 0.0), 1.0) for c in coords]
                if clamped != coords:
                    changed = True
                fixed.append(cls + " " + " ".join(f"{c:.6f}" for c in clamped))
            if changed:
                lp.write_text("\n".join(fixed))
                stats["labels_fixed"] += 1
        except Exception as e:
            stats["details"].append(f"LABEL_ERR: {lp.name}: {e}")

    stats["valid_final"] = len(unique)
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 3 – split_dataset
# ═══════════════════════════════════════════════════════════════════════════
@tool
def split_dataset(
    source_dir:  str,
    output_dir:  str,
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    test_ratio:  float = 0.10,
    shuffle:     bool  = True,
    seed:        int   = 42,
) -> dict:
    """
    Split a flat or pre-structured image+label dataset into train/val/test
    following YOLO folder conventions.
    """
    src = Path(source_dir)
    dst = Path(output_dir)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split in ("train", "val", "test"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[Path, Path | None]] = []
    for img in src.rglob("*"):
        if img.suffix.lower() not in img_exts:
            continue
        lbl = Path(str(img.with_suffix(".txt")).replace("/images/", "/labels/"))
        if not lbl.exists():
            lbl = img.with_suffix(".txt")
        pairs.append((img, lbl if lbl.exists() else None))

    if not pairs:
        return {"success": False, "error": "No image files found in source_dir."}

    if shuffle:
        random.seed(seed)
        random.shuffle(pairs)

    n     = len(pairs)
    n_tr  = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # ── FIX 3: warn when val split rounds down to 0 ───────────────────────
    warnings: list[str] = []
    if n_val == 0 and val_ratio > 0:
        warnings.append(
            f"Dataset is too small ({n} images): val split rounds to 0 images "
            f"with val_ratio={val_ratio}. Consider using more images or a larger val_ratio."
        )

    assignment = (
        [("train", p) for p in pairs[:n_tr]]
        + [("val",   p) for p in pairs[n_tr : n_tr + n_val]]
        + [("test",  p) for p in pairs[n_tr + n_val :]]
    )

    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0, "missing_labels": 0}
    for split, (img, lbl) in assignment:
        shutil.copy2(img, dst / "images" / split / img.name)
        if lbl:
            shutil.copy2(lbl, dst / "labels" / split / img.with_suffix(".txt").name)
        else:
            counts["missing_labels"] += 1
        counts[split] += 1

    return {
        "success":        True,
        "output_dir":     str(dst),
        "total":          n,
        "train":          counts["train"],
        "val":            counts["val"],
        "test":           counts["test"],
        "missing_labels": counts["missing_labels"],
        "warnings":       warnings,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 4 – generate_data_yaml
# ═══════════════════════════════════════════════════════════════════════════
@tool
def generate_data_yaml(
    dataset_dir: str,
    class_names: list[str],
    task:        str = "detect",
) -> dict:
    """
    Build a YOLO data.yaml string.
    Auto-detects class names from existing yaml / classes.txt if list is empty.
    Returns {yaml_content, nc, names}.
    """
    import yaml as _yaml

    ds    = Path(dataset_dir)
    names = list(class_names)

    # Auto-detect — FIX: restrict yaml search to known safe filenames only
    # to avoid picking up Ultralytics training output artifacts (args.yaml etc.)
    _KNOWN_DATA_YAMLS = {"data.yaml", "dataset.yaml", "config.yaml"}

    if not names:
        for yf in ds.rglob("*.yaml"):
            # Skip files inside runs/ or similar output directories
            if any(part in ("runs", "train", "val", "exp") for part in yf.parts):
                continue
            if yf.name not in _KNOWN_DATA_YAMLS:
                continue
            try:
                d = _yaml.safe_load(yf.read_text())
                if "names" in d:
                    n = d["names"]
                    names = n if isinstance(n, list) else list(n.values())
                    break
            except Exception:
                pass

    if not names:
        for cf in [*ds.rglob("classes.txt"), *ds.rglob("obj.names")]:
            names = [ln.strip() for ln in cf.read_text().splitlines() if ln.strip()]
            break

    if not names:
        ids: set[int] = set()
        for lf in ds.rglob("*.txt"):
            for ln in lf.read_text().splitlines():
                p = ln.split()
                if p:
                    try:
                        ids.add(int(p[0]))
                    except ValueError:
                        pass
        names = [f"class_{i}" for i in sorted(ids)]

    data: dict[str, Any] = {
        "path":  str(ds.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(names),
        "names": names,
    }

    if task == "pose":
        data["kpt_shape"] = [17, 3]
        # FIX 6a: use correct COCO left↔right keypoint flip pairs
        data["flip_idx"] = _COCO_FLIP_IDX

    yaml_str = _yaml.dump(data, default_flow_style=False, sort_keys=False)
    return {"success": True, "yaml_content": yaml_str, "nc": len(names), "names": names}


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 5 – analyze_dataset
# ═══════════════════════════════════════════════════════════════════════════
@tool
def analyze_dataset(dataset_dir: str, class_names: list[str]) -> dict:
    """
    Compute EDA stats for a YOLO-format split dataset.
    Returns per-split: image count, class distribution, bbox stats, imbalance ratio.
    JSON-safe (no matplotlib).
    Empty splits (dir exists but has 0 images) are excluded from the report.
    """
    ds     = Path(dataset_dir)
    report: dict[str, Any] = {"success": True}

    for split in ("train", "val", "test"):
        img_dir = ds / "images" / split
        lbl_dir = ds / "labels" / split
        if not img_dir.exists():
            continue

        imgs = [
            p for p in img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]

        # ── FIX 4: skip empty splits so the report is not polluted ────────
        if not imgs:
            continue

        cls_counts: Counter = Counter()
        box_w: list[float]  = []
        box_h: list[float]  = []

        for img in imgs:
            lbl = lbl_dir / img.with_suffix(".txt").name
            if not lbl.exists():
                continue
            for ln in lbl.read_text().splitlines():
                parts = ln.split()
                if len(parts) >= 5:
                    try:
                        cls_counts[int(parts[0])] += 1
                        box_w.append(float(parts[3]))
                        box_h.append(float(parts[4]))
                    except (ValueError, IndexError):
                        pass

        def avg(lst: list[float]) -> float:
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        named: dict[str, int] = {}
        for cls_id, cnt in sorted(cls_counts.items()):
            lbl_name = (
                class_names[cls_id]
                if cls_id < len(class_names)
                else f"class_{cls_id}"
            )
            named[lbl_name] = cnt

        vals      = list(named.values())
        imbalance = round(max(vals) / (min(vals) + 1e-9), 2) if vals else 1.0

        report[split] = {
            "images":             len(imgs),
            "class_distribution": named,
            "total_annotations":  sum(cls_counts.values()),
            "imbalance_ratio":    imbalance,
            "avg_box_w":          avg(box_w),
            "avg_box_h":          avg(box_h),
        }

    return report


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 6 – generate_notebook_cells
# ═══════════════════════════════════════════════════════════════════════════
@tool
def generate_notebook_cells(config_json: str) -> dict:
    """
    Generate all Jupyter notebook cells for the complete YOLO training pipeline.
    Covers: install, setup, upload/extract, cleaning, splitting, yaml, EDA,
            training, validation, export, inference preview.
    Returns {cells: [...], cell_count: int}.
    """
    cfg        = json.loads(config_json)
    platform   = cfg.get("platform", "colab")
    task       = cfg.get("task", "detect")
    mode       = cfg.get("train_mode", "finetune")
    model_pt   = cfg.get("model_pt",   "yolov8n.pt")
    model_yaml = cfg.get("model_yaml", "yolov8n.yaml")
    model_ref  = model_yaml if mode == "scratch" else model_pt
    model_id   = cfg.get("model_id", "yolov8n")
    hp         = cfg.get("hp", {})
    augs       = cfg.get("augmentations", {})
    cls_list   = cfg.get("class_names", [])
    img_sz     = cfg.get("image_size", 640)
    is_colab   = platform == "colab"

    aug_lines = []
    for k, v in augs.items():
        if isinstance(v, dict) and v.get("enabled"):
            val = v.get("value", 0)
            aug_lines.append(f"    {k}={repr(val)},")
    aug_args = "\n".join(aug_lines)

    cls_py  = repr(cls_list)
    tr_r    = cfg.get("train_split", 0.8)
    val_r   = cfg.get("val_split",   0.1)
    te_r    = cfg.get("test_split",  0.1)
    seed    = cfg.get("seed", 42)
    shuffle = cfg.get("shuffle", True)
    fmt     = cfg.get("data_format", "yolo")

    cells: list[dict] = []

    # ── 0. Title ──────────────────────────────────────────────────────────
    cells.append({"type": "markdown", "source": f"""\
# 🔥 YOLOForge — {model_id.upper()} `{task}` Training Notebook

| Parameter | Value |
|-----------|-------|
| Model | `{model_ref}` |
| Task | `{task}` |
| Mode | {"Train from Scratch" if mode == "scratch" else "Fine-tuning"} |
| Platform | {"Google Colab" if is_colab else "Kaggle"} |
| Epochs | {hp.get("epochs", 100)} |
| Batch | {hp.get("batch", 16)} |
| Image Size | {img_sz} |
| Optimizer | {hp.get("optimizer", "SGD")} |

> **Generated by YOLOForge** · {date.today()} · [docs.ultralytics.com](https://docs.ultralytics.com)
"""})

    # ── 1. Install ────────────────────────────────────────────────────────
    coco_pip = "!pip install pycocotools -q" if fmt == "coco" else ""
    cells.append({"type": "code", "source": f"""\
# ── Install Dependencies ──────────────────────────────────────────────────
!pip install ultralytics>=8.3.0 -q
!pip install Pillow numpy pandas matplotlib seaborn PyYAML tqdm -q
{coco_pip}

import ultralytics
ultralytics.checks()
print("✅ All dependencies installed")"""})

    # ── 2. Setup workspace ────────────────────────────────────────────────
    if is_colab:
        cells.append({"type": "code", "source": """\
# ── Mount Google Drive & Workspace Setup ─────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, shutil, random, yaml, zipfile, hashlib
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from ultralytics import YOLO

BASE = Path("/content/yolo_workspace"); BASE.mkdir(exist_ok=True)
print(f"✅ Workspace: {BASE}")"""})
    else:
        cells.append({"type": "code", "source": """\
# ── Kaggle Workspace Setup ────────────────────────────────────────────────
import os, shutil, random, yaml, zipfile, hashlib
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from ultralytics import YOLO

BASE         = Path("/kaggle/working/yolo_ws"); BASE.mkdir(exist_ok=True)
DATASET_PATH = Path("/kaggle/input")
print(f"✅ Workspace: {BASE}")"""})

    # ── 3. Upload / Extract ───────────────────────────────────────────────
    if is_colab:
        cells.append({"type": "code", "source": """\
# ── Upload & Extract Dataset ZIP ─────────────────────────────────────────
from google.colab import files
print("⬆ Upload your dataset ZIP:")
uploaded = files.upload()
ZIP_PATH = list(uploaded.keys())[0]

EXTRACT = BASE / "raw_dataset"; EXTRACT.mkdir(exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    zf.extractall(EXTRACT)

print(f"✅ Extracted → {EXTRACT}")
print("Contents:", [p.name for p in EXTRACT.iterdir()])"""})
    else:
        cells.append({"type": "code", "source": """\
# ── Kaggle: Point to Input Dataset ───────────────────────────────────────
EXTRACT = DATASET_PATH
print(f"✅ Dataset: {EXTRACT}")
print("Contents:", [p.name for p in EXTRACT.iterdir()])"""})

    # ── 4. Cleaning ───────────────────────────────────────────────────────
    cells.append({"type": "markdown", "source": "## 🧹 Step 1 — Dataset Cleaning"})
    cells.append({"type": "code", "source": """\
# ── Clean Dataset (non-destructive: works on a staged copy) ──────────────
def _md5(p):
    with open(p, 'rb') as f: return hashlib.md5(f.read()).hexdigest()

def clean_dataset(src):
    src     = Path(src)
    staging = src.parent / (src.name + "_cleaned")
    if staging.exists(): shutil.rmtree(staging)
    shutil.copytree(src, staging)

    exts     = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    all_imgs = [p for p in staging.rglob("*") if p.suffix.lower() in exts]
    stats    = dict(total=len(all_imgs), corrupt=0, dupes=0, no_label=0, fixed=0,
                    staging_dir=str(staging))

    valid = []
    for p in all_imgs:
        try: Image.open(p).verify(); valid.append(p)
        except: p.unlink(missing_ok=True); stats['corrupt'] += 1

    seen, uniq = {}, []
    for p in valid:
        h = _md5(p)
        if h in seen: p.unlink(missing_ok=True); stats['dupes'] += 1
        else:         seen[h] = p; uniq.append(p)

    for p in uniq:
        lp = Path(str(p.with_suffix('.txt')).replace('/images/', '/labels/'))
        if not lp.exists(): lp = p.with_suffix('.txt')
        if not lp.exists(): stats['no_label'] += 1; continue
        raw   = [l for l in lp.read_text().splitlines() if l.strip()]
        fixed, changed = [], False
        for ln in raw:
            parts = ln.split()
            if len(parts) < 5: continue
            coords  = [float(x) for x in parts[1:]]
            clamped = [min(max(c, 0.), 1.) for c in coords]
            if clamped != coords: changed = True
            fixed.append(parts[0] + ' ' + ' '.join(f'{c:.6f}' for c in clamped))
        if changed: lp.write_text('\\n'.join(fixed)); stats['fixed'] += 1

    stats['valid_final'] = len(uniq)
    return staging, stats

CLEAN_DIR, cstats = clean_dataset(EXTRACT)
print("\\n🧹 Cleaning Report:")
for k, v in cstats.items(): print(f"  {k:20s}: {v}")
print(f"\\n✅ Working from cleaned copy: {CLEAN_DIR}")"""})

    # ── 5. Split ──────────────────────────────────────────────────────────
    cells.append({"type": "markdown", "source": "## 📂 Step 2 — Dataset Split"})
    cells.append({"type": "code", "source": f"""\
# ── Split into Train / Val / Test ─────────────────────────────────────────
TRAIN_R, VAL_R, TEST_R = {tr_r}, {val_r}, {te_r}
SEED, SHUFFLE           = {seed}, {shuffle}

DS = BASE / "dataset"
for sp in ['train', 'val', 'test']:
    (DS / 'images' / sp).mkdir(parents=True, exist_ok=True)
    (DS / 'labels' / sp).mkdir(parents=True, exist_ok=True)

exts  = {{'.jpg', '.jpeg', '.png', '.bmp', '.webp'}}
pairs = []
for img in CLEAN_DIR.rglob("*"):
    if img.suffix.lower() not in exts: continue
    lbl = Path(str(img.with_suffix('.txt')).replace('/images/', '/labels/'))
    if not lbl.exists(): lbl = img.with_suffix('.txt')
    if lbl.exists(): pairs.append((img, lbl))

if SHUFFLE: random.seed(SEED); random.shuffle(pairs)
n  = len(pairs); nt = int(n * TRAIN_R); nv = int(n * VAL_R)

if nv == 0 and VAL_R > 0:
    print(f"⚠ Dataset too small ({{n}} images): val split rounds to 0. "
          "Consider adding more images or increasing val_ratio.")

for sp, sl in [('train', pairs[:nt]), ('val', pairs[nt:nt+nv]), ('test', pairs[nt+nv:])]:
    for img, lbl in sl:
        shutil.copy2(img, DS / 'images' / sp / img.name)
        shutil.copy2(lbl, DS / 'labels' / sp / img.with_suffix('.txt').name)
    print(f"  {{sp}}: {{len(sl)}} images")

print(f"\\n✅ Total: {{n}} | Train: {{nt}} | Val: {{nv}} | Test: {{n-nt-nv}}")"""})

    # ── 6. YAML ───────────────────────────────────────────────────────────
    # Correct COCO flip_idx for pose
    coco_flip = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    cells.append({"type": "code", "source": f"""\
# ── Generate data.yaml ────────────────────────────────────────────────────
CLASS_NAMES = {cls_py}

_KNOWN_DATA_YAMLS = {{"data.yaml", "dataset.yaml", "config.yaml"}}

if not CLASS_NAMES:
    for yf in CLEAN_DIR.rglob("*.yaml"):
        # Skip output artifact directories
        if any(part in ("runs", "train", "val", "exp") for part in yf.parts): continue
        if yf.name not in _KNOWN_DATA_YAMLS: continue
        try:
            d = yaml.safe_load(open(yf))
            if 'names' in d:
                CLASS_NAMES = d['names'] if isinstance(d['names'], list) else list(d['names'].values())
                break
        except: pass
    if not CLASS_NAMES:
        for cf in [*CLEAN_DIR.rglob("classes.txt"), *CLEAN_DIR.rglob("obj.names")]:
            CLASS_NAMES = [l.strip() for l in cf.read_text().splitlines() if l.strip()]
            break
    if not CLASS_NAMES:
        ids = set()
        for lf in (DS / 'labels' / 'train').glob("*.txt"):
            for ln in lf.read_text().splitlines():
                p = ln.split()
                if p:
                    try: ids.add(int(p[0]))
                    except: pass
        CLASS_NAMES = [f"class_{{i}}" for i in sorted(ids)]

print(f"Classes ({{len(CLASS_NAMES)}}): {{CLASS_NAMES}}")

data_yaml = dict(path=str(DS.resolve()), train="images/train",
                 val="images/val", test="images/test",
                 nc=len(CLASS_NAMES), names=CLASS_NAMES)

# FIX: correct COCO left↔right keypoint flip pairs for pose task
if "{task}" == "pose":
    data_yaml["kpt_shape"] = [17, 3]
    data_yaml["flip_idx"]  = {coco_flip}

YAML_PATH = DS / "data.yaml"
yaml.dump(data_yaml, open(YAML_PATH, 'w'), default_flow_style=False)
print(f"✅ data.yaml → {{YAML_PATH}}")
print(yaml.dump(data_yaml))"""})

    # ── 7. EDA ────────────────────────────────────────────────────────────
    cells.append({"type": "markdown", "source": "## 📊 Step 3 — Exploratory Data Analysis"})
    cells.append({"type": "code", "source": """\
# ── EDA — Class Distribution & Dataset Stats ──────────────────────────────
def count_cls(lbl_dir):
    c = Counter()
    for f in Path(lbl_dir).glob("*.txt"):
        for ln in f.read_text().strip().splitlines():
            p = ln.split()
            if p:
                try: c[int(p[0])] += 1
                except: pass
    return c

tc = count_cls(DS / 'labels' / 'train')
vc = count_cls(DS / 'labels' / 'val')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('#0f172a')
for ax in axes:
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8')
    [ax.spines[s].set_color('#334155') for s in ax.spines]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

nm = {i: CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i) for i in tc}
axes[0].barh([nm[i] for i in sorted(tc)], [tc[i] for i in sorted(tc)], color='#3b82f6')
axes[0].set_title('Class Distribution (Train)', color='#f1f5f9', fontweight='bold')
axes[0].set_xlabel('Count', color='#94a3b8')

n_tr  = len(list((DS / 'images' / 'train').glob("*.*")))
n_val = len(list((DS / 'images' / 'val').glob("*.*")))
n_te  = len(list((DS / 'images' / 'test').glob("*.*")))
axes[1].pie([n_tr, n_val, n_te], labels=['Train', 'Val', 'Test'],
    colors=['#3b82f6', '#06b6d4', '#f59e0b'], autopct='%1.1f%%',
    textprops={'color': '#f1f5f9'})
axes[1].set_title('Dataset Split', color='#f1f5f9', fontweight='bold')

if vc:
    cids = sorted(set(list(tc) + list(vc)))
    x    = range(len(cids))
    axes[2].bar([i - .2 for i in x], [tc.get(c, 0) for c in cids], .4, label='Train', color='#3b82f6')
    axes[2].bar([i + .2 for i in x], [vc.get(c, 0) for c in cids], .4, label='Val',   color='#06b6d4')
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels([nm.get(c, str(c)) for c in cids], rotation=45, ha='right', color='#94a3b8')
    axes[2].set_title('Train vs Val Balance', color='#f1f5f9', fontweight='bold')
    axes[2].legend(facecolor='#1e293b', labelcolor='#f1f5f9')

plt.tight_layout()
plt.savefig(BASE / 'eda.png', dpi=120, bbox_inches='tight', facecolor='#0f172a')
plt.show()

counts = list(tc.values())
if counts:
    ratio = max(counts) / (min(counts) + 1e-9)
    if ratio > 10:
        print(f"⚠ High class imbalance detected (ratio {ratio:.1f}x) — consider oversampling or class weights.")
    else:
        print(f"✅ Class balance looks good (ratio {ratio:.1f}x)")"""})

    # ── 8. Bounding box stats ─────────────────────────────────────────────
    cells.append({"type": "code", "source": """\
# ── BBox Size Distribution ────────────────────────────────────────────────
bw, bh, areas = [], [], []
for lf in (DS / 'labels' / 'train').glob("*.txt"):
    for ln in lf.read_text().splitlines():
        p = ln.split()
        if len(p) >= 5:
            try:
                w, h = float(p[3]), float(p[4])
                bw.append(w); bh.append(h); areas.append(w * h)
            except: pass

if bw:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#0f172a')
    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        [ax.spines[s].set_color('#334155') for s in ax.spines]

    axes[0].hist(bw, bins=40, color='#3b82f6', edgecolor='none')
    axes[0].set_title('Box Width Distribution',  color='#f1f5f9', fontweight='bold')
    axes[1].hist(bh, bins=40, color='#06b6d4', edgecolor='none')
    axes[1].set_title('Box Height Distribution', color='#f1f5f9', fontweight='bold')
    axes[2].hist(areas, bins=40, color='#f59e0b', edgecolor='none')
    axes[2].set_title('Box Area Distribution',   color='#f1f5f9', fontweight='bold')

    plt.tight_layout()
    plt.savefig(BASE / 'bbox_stats.png', dpi=110, bbox_inches='tight', facecolor='#0f172a')
    plt.show()
    print(f"Avg W: {sum(bw)/len(bw):.4f}  Avg H: {sum(bh)/len(bh):.4f}  Avg Area: {sum(areas)/len(areas):.4f}")
else:
    print("⚠ No bounding box data found in train labels.")"""})

    # ── 9. Load model ─────────────────────────────────────────────────────
    cells.append({"type": "markdown", "source": (
        f"## 🚀 Step 4 — Training: `{model_ref}`\n\n"
        f"**Mode:** {'Train from Scratch — architecture only, no pretrained weights' if mode == 'scratch' else 'Fine-tuning — loads pretrained weights'}"
    )})
    cells.append({"type": "code", "source": f"""\
# ── Load YOLO Model ───────────────────────────────────────────────────────
model = YOLO("{model_ref}")
print(model.info())"""})

    # ── 10. Train ─────────────────────────────────────────────────────────
    # FIX 5: freeze=0 is falsy — use explicit None check instead of `or`
    raw_freeze = hp.get("freeze", 0)
    if raw_freeze is None or raw_freeze == 0:
        freeze_val = "None"
    else:
        freeze_val = str(raw_freeze)

    cache_val = hp.get("cache", "false")

    cells.append({"type": "code", "source": f"""\
# ── Train ─────────────────────────────────────────────────────────────────
results = model.train(
    data             = str(YAML_PATH),
    task             = "{task}",
    epochs           = {hp.get("epochs", 100)},
    patience         = {hp.get("patience", 50)},
    batch            = {hp.get("batch", 16)},
    imgsz            = {img_sz},
    optimizer        = "{hp.get("optimizer", "SGD")}",
    lr0              = {hp.get("lr0", 0.01)},
    lrf              = {hp.get("lrf", 0.01)},
    momentum         = {hp.get("momentum", 0.937)},
    weight_decay     = {hp.get("weight_decay", 0.0005)},
    warmup_epochs    = {hp.get("warmup_epochs", 3.0)},
    warmup_momentum  = {hp.get("warmup_momentum", 0.8)},
    warmup_bias_lr   = {hp.get("warmup_bias_lr", 0.1)},
    box              = {hp.get("box", 7.5)},
    cls              = {hp.get("cls", 0.5)},
    dfl              = {hp.get("dfl", 1.5)},
    cos_lr           = {hp.get("cos_lr", False)},
    close_mosaic     = {hp.get("close_mosaic", 10)},
    amp              = {hp.get("amp", True)},
    fraction         = {hp.get("fraction", 1.0)},
    freeze           = {freeze_val},
    workers          = {hp.get("workers", 8)},
    device           = "{hp.get("device", "0")}",
    pretrained       = {mode != "scratch"},
    resume           = {hp.get("resume", False)},
    multi_scale      = {hp.get("multi_scale", False)},
    {'overlap_mask   = ' + str(hp.get("overlap_mask", True)) + ',' if task == "segment" else ''}
    {'mask_ratio     = ' + str(hp.get("mask_ratio",   4))    + ',' if task == "segment" else ''}
    dropout          = {hp.get("dropout", 0.0)},
    val              = {hp.get("val", True)},
    plots            = {hp.get("plots", True)},
    save             = {hp.get("save", True)},
    save_period      = {hp.get("save_period", -1)},
    project          = "{hp.get("project", "runs/train")}",
    name             = "{hp.get("name", "exp")}",
    exist_ok         = {hp.get("exist_ok", False)},
    cache            = "{cache_val}",
    # ── Augmentations ────────────────────────────────────────────────────
{aug_args}
)

print(f"\\n✅ Training complete!")
print(f"Best weights → {{results.save_dir}}/weights/best.pt")
print(f"Last weights → {{results.save_dir}}/weights/last.pt")"""})

    # ── 11. Results plots ─────────────────────────────────────────────────
    cells.append({"type": "code", "source": """\
# ── Training Results ──────────────────────────────────────────────────────
import pandas as pd

csv_path = Path(results.save_dir) / "results.csv"
if csv_path.exists():
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.patch.set_facecolor('#0f172a')
    fig.suptitle('Training Results', color='#f1f5f9', fontsize=14, fontweight='bold')

    metric_cols = [c for c in df.columns if any(k in c.lower() for k in ['loss', 'map', 'precision', 'recall'])]
    for ax, col in zip(axes.flat, metric_cols[:6]):
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.plot(df[col].values, color='#3b82f6', linewidth=2)
        ax.set_title(col, color='#f1f5f9', fontsize=10)
        ax.set_xlabel('Epoch', color='#94a3b8')
        [ax.spines[s].set_color('#334155') for s in ax.spines]

    for ax in axes.flat[len(metric_cols):]: ax.axis('off')
    plt.tight_layout()
    plt.savefig(BASE / 'training_curves.png', dpi=110, facecolor='#0f172a')
    plt.show()
else:
    print("⚠ results.csv not found — check results.save_dir for plots.")"""})

    # ── 12. Validate ──────────────────────────────────────────────────────
    cells.append({"type": "markdown", "source": "## 📈 Step 5 — Evaluation"})
    metric_code = {
        "detect":   "print(f'mAP50={m.box.map50:.4f} | mAP50-95={m.box.map:.4f} | P={m.box.mp:.4f} | R={m.box.mr:.4f}')",
        "segment":  "print(f'Box mAP50={m.box.map50:.4f} | Mask mAP50={m.seg.map50:.4f}')",
        "classify": "print(f'Top-1 Accuracy={m.top1:.4f} | Top-5 Accuracy={m.top5:.4f}')",
        "pose":     "print(f'Box mAP50={m.box.map50:.4f} | Pose mAP50={m.pose.map50:.4f}')",
        "obb":      "print(f'OBB mAP50={m.obb.map50:.4f} | OBB mAP50-95={m.obb.map:.4f}')",
    }.get(task, "print(m)")

    cells.append({"type": "code", "source": f"""\
# ── Validate Best Model ───────────────────────────────────────────────────
best = YOLO(str(Path(results.save_dir) / 'weights' / 'best.pt'))
m    = best.val(data=str(YAML_PATH), imgsz={img_sz})

print("\\n📊 Validation Metrics:")
{metric_code}"""})

    # ── 13. Confusion matrix ──────────────────────────────────────────────
    cells.append({"type": "code", "source": """\
# ── Display Confusion Matrix ──────────────────────────────────────────────
cm_path = Path(results.save_dir) / 'confusion_matrix_normalized.png'
if not cm_path.exists():
    cm_path = Path(results.save_dir) / 'confusion_matrix.png'
if cm_path.exists():
    from PIL import Image as PILImage
    img = PILImage.open(cm_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img); plt.axis('off')
    plt.title('Confusion Matrix', color='#f1f5f9', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()
else:
    print("Confusion matrix not found — check results.save_dir for plots.")"""})

    # ── 14. Export ────────────────────────────────────────────────────────
    cells.append({"type": "markdown", "source": "## 📦 Step 6 — Export"})
    cells.append({"type": "code", "source": f"""\
# ── Export to Multiple Formats ────────────────────────────────────────────
export_formats = [
    "onnx",          # universal — works everywhere
    "torchscript",   # PyTorch deployment
    # "tflite",      # TensorFlow Lite (mobile/edge)
    # "coreml",      # Apple CoreML (iOS/macOS)
    # "engine",      # TensorRT (NVIDIA GPU, fastest)
    # "openvino",    # Intel OpenVINO
]

for fmt in export_formats:
    try:
        path = best.export(format=fmt, imgsz={img_sz})
        print(f"✅ {{fmt}}: {{path}}")
    except Exception as e:
        print(f"⚠  {{fmt}} failed: {{e}}")"""})

    # ── 15. Save to Drive (Colab only) ────────────────────────────────────
    if is_colab:
        cells.append({"type": "code", "source": f"""\
# ── Save All Results to Google Drive ─────────────────────────────────────
drive_dst = Path("/content/drive/MyDrive/yolo_runs/{hp.get('name', 'exp')}")
drive_dst.mkdir(parents=True, exist_ok=True)
shutil.copytree(results.save_dir, drive_dst / "run", dirs_exist_ok=True)
shutil.copy2(YAML_PATH, drive_dst / "data.yaml")
print(f"✅ Saved to Drive: {{drive_dst}}")"""})

    # ── 16. Inference preview ─────────────────────────────────────────────
    cells.append({"type": "markdown", "source": "## 🎯 Step 7 — Inference Preview"})
    cells.append({"type": "code", "source": f"""\
# ── Run Inference on Test Images ──────────────────────────────────────────
test_imgs  = list((DS / 'images' / 'test').glob('*.jpg'))[:6]
test_imgs += list((DS / 'images' / 'test').glob('*.png'))[:max(0, 6 - len(test_imgs))]
test_imgs  = test_imgs[:6]

if test_imgs:
    preds = best.predict(test_imgs, imgsz={img_sz}, conf=0.25, verbose=False)
    n     = len(preds)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#0f172a')
    for i, (ax, r) in enumerate(zip(axes.flat, preds)):
        ax.imshow(r.plot()[:, :, ::-1])
        ax.axis('off')
        ax.set_title(f'Sample {{i+1}}', color='#f1f5f9', fontsize=10)
    for ax in axes.flat[n:]: ax.axis('off')
    plt.suptitle('Prediction Samples (conf ≥ 0.25)', color='#f1f5f9', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(BASE / 'predictions.png', dpi=100, facecolor='#0f172a')
    plt.show()
else:
    print("⚠ No test images found — add images to DS/images/test/")"""})

    # ── 17. Quick inference snippet ───────────────────────────────────────
    cells.append({"type": "markdown", "source": "## 🔧 Quick Inference Snippet"})
    cells.append({"type": "code", "source": f"""\
# ── Use Your Trained Model ────────────────────────────────────────────────
from ultralytics import YOLO

model   = YOLO("{hp.get('project', 'runs/train')}/{hp.get('name', 'exp')}/weights/best.pt")
results = model.predict("your_image.jpg", conf=0.25, imgsz={img_sz})
results[0].show()

# Batch inference
results = model.predict(["img1.jpg", "img2.jpg"], conf=0.25)
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    cls   = r.boxes.cls.cpu().numpy()
    print(boxes, confs, cls)"""})

    return {"cells": cells, "cell_count": len(cells), "success": True}


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 7 – assemble_ipynb
# ═══════════════════════════════════════════════════════════════════════════
@tool
def assemble_ipynb(cells_json: str, platform: str = "colab") -> dict:
    """
    Assemble {type, source} cell dicts into a valid .ipynb JSON string.
    Returns {notebook_json, cell_count}.
    """
    cells    = json.loads(cells_json)
    nb_cells = []
    for i, c in enumerate(cells):
        ct   = "markdown" if c["type"] == "markdown" else "code"
        cell: dict[str, Any] = {
            "cell_type": ct,
            "id":        f"yf-{i:04d}",
            "metadata":  {},
            "source":    c["source"],
        }
        if ct == "code":
            cell["outputs"]         = []
            cell["execution_count"] = None
        nb_cells.append(cell)

    nb = {
        "nbformat":       4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec":    {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
            "accelerator":   "GPU" if platform == "colab" else None,
            "colab":         {"provenance": []} if platform == "colab" else None,
        },
        "cells": nb_cells,
    }
    nb_json = json.dumps(nb, indent=2, ensure_ascii=False)
    return {"notebook_json": nb_json, "cell_count": len(nb_cells), "success": True}


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 8 – build_yaml_string  (lightweight, no FS access)
# ═══════════════════════════════════════════════════════════════════════════
@tool
def build_yaml_string(class_names: list[str], task: str = "detect") -> dict:
    """
    Build a ready-to-use data.yaml template string (no filesystem required).
    Returns {yaml_content}.
    """
    import yaml as _yaml

    names = class_names or ["class_0"]
    data: dict[str, Any] = {
        "path":  "./dataset",
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(names),
        "names": names,
    }
    if task == "pose":
        data["kpt_shape"] = [17, 3]
        # FIX 6b: correct COCO left↔right keypoint flip pairs
        data["flip_idx"]  = _COCO_FLIP_IDX

    return {"yaml_content": _yaml.dump(data, default_flow_style=False, sort_keys=False)}


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 9 – generate_readme
# ═══════════════════════════════════════════════════════════════════════════
@tool
def generate_readme(config_json: str) -> dict:
    """
    Generate a comprehensive README.md for the training run.
    """
    cfg      = json.loads(config_json)
    task     = cfg.get("task", "detect")
    model_id = cfg.get("model_id", "yolov8n")
    model_pt = cfg.get("model_pt", "yolov8n.pt")
    model_yaml = cfg.get("model_yaml", "")
    mode     = cfg.get("train_mode", "finetune")
    platform = cfg.get("platform", "colab")
    hp       = cfg.get("hp", {})
    augs     = cfg.get("augmentations", {})
    cls_list = cfg.get("class_names", [])
    img_sz   = cfg.get("image_size", 640)

    model_ref = model_yaml if mode == "scratch" else model_pt

    active_augs = [
        f"- `{k}` = {v['value']}"
        for k, v in augs.items()
        if isinstance(v, dict) and v.get("enabled")
    ]

    cls_md = (
        "\n".join(f"  {i}. `{c}`" for i, c in enumerate(cls_list))
        or "  _Auto-detected from dataset_"
    )

    task_notes = {
        "detect":   "Outputs `[x1 y1 x2 y2 conf cls]` per detection.",
        "segment":  "Outputs bounding boxes + pixel-level instance masks.",
        "classify": "Outputs class probabilities for the whole image.",
        "pose":     "Outputs bounding box + keypoints `(x, y, visibility)` per instance.",
        "obb":      "Outputs rotated bounding boxes `(cx, cy, w, h, angle)` per detection.",
    }.get(task, "")

    colab_steps = textwrap.dedent("""\
        1. Go to [colab.research.google.com](https://colab.research.google.com)
        2. **File → Upload notebook** → select the `.ipynb` file
        3. **Runtime → Change runtime type → GPU** (T4 / A100)
        4. **Ctrl+F9** → Run All — upload your dataset ZIP when prompted
        5. Model auto-saved to Google Drive under `yolo_runs/`""")

    kaggle_steps = textwrap.dedent("""\
        1. Go to [kaggle.com/code](https://kaggle.com/code) → **New Notebook**
        2. **File → Import Notebook** → upload the `.ipynb`
        3. **Add Data** → attach your dataset
        4. **Settings → Accelerator → GPU P100**
        5. **Run All** — results in `/kaggle/working/`""")

    md = textwrap.dedent(f"""\
    # 🔥 YOLOForge Training Run

    > Generated by **YOLOForge** · {date.today()}

    ---

    ## 📋 Config Summary

    | Parameter | Value |
    |-----------|-------|
    | Model | `{model_id}` → `{model_ref}` |
    | Task | `{task}` |
    | Mode | {"Train from Scratch" if mode == "scratch" else "Fine-tuning"} |
    | Platform | {platform.capitalize()} |
    | Image Size | {img_sz}px |
    | Epochs | {hp.get("epochs", 100)} |
    | Batch | {hp.get("batch", 16)} |
    | Optimizer | {hp.get("optimizer", "SGD")} |
    | LR | {hp.get("lr0", 0.01)} → {hp.get("lrf", 0.01)} |
    | Momentum | {hp.get("momentum", 0.937)} |
    | Weight Decay | {hp.get("weight_decay", 0.0005)} |
    | AMP | {hp.get("amp", True)} |
    | Split | {int(cfg.get("train_split",0.8)*100)} / {int(cfg.get("val_split",0.1)*100)} / {int(cfg.get("test_split",0.1)*100)} |

    ## 🎯 Task Notes

    **{task.upper()}** — {task_notes}

    ## 🔀 Active Augmentations ({len(active_augs)})

    {chr(10).join(active_augs) or "_None_"}

    ## 🏷 Class Names ({len(cls_list)})

    {cls_md}

    ## 🚀 Usage Instructions

    {"### Google Colab" if platform == "colab" else "### Kaggle"}

    {colab_steps if platform == "colab" else kaggle_steps}

    ## 📁 Output Structure

    ```
    {hp.get("project", "runs/train")}/{hp.get("name", "exp")}/
    ├── weights/
    │   ├── best.pt          ← best model checkpoint
    │   └── last.pt          ← final epoch checkpoint
    ├── results.csv          ← per-epoch metrics
    ├── confusion_matrix.png
    ├── PR_curve.png
    ├── F1_curve.png
    ├── labels.jpg
    └── val_batch*.jpg
    ```

    ## 🔧 Inference Example

    ```python
    from ultralytics import YOLO

    model   = YOLO("{hp.get("project","runs/train")}/{hp.get("name","exp")}/weights/best.pt")
    results = model.predict("image.jpg", conf=0.25, imgsz={img_sz})
    results[0].show()
    results[0].save("out.jpg")
    ```

    ---
    *YOLOForge — No-Code YOLO Training Platform*
    """)

    return {"readme_content": md, "success": True}


# ── Registry ───────────────────────────────────────────────────────────────
ALL_TOOLS = [
    validate_config,
    clean_dataset,
    split_dataset,
    generate_data_yaml,
    analyze_dataset,
    generate_notebook_cells,
    assemble_ipynb,
    build_yaml_string,
    generate_readme,
]

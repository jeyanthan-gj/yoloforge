"""
YOLOForge – Pydantic schemas & TypedDict state
"""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ─── Augmentation ─────────────────────────────────────────────
class AugParam(BaseModel):
    enabled: bool = True
    value: float | str = 0.0


class AugConfig(BaseModel):
    hsv_h:       AugParam = AugParam(enabled=True,  value=0.015)
    hsv_s:       AugParam = AugParam(enabled=True,  value=0.7)
    hsv_v:       AugParam = AugParam(enabled=True,  value=0.4)
    degrees:     AugParam = AugParam(enabled=False, value=0.0)
    translate:   AugParam = AugParam(enabled=True,  value=0.1)
    scale:       AugParam = AugParam(enabled=True,  value=0.5)
    shear:       AugParam = AugParam(enabled=False, value=0.0)
    perspective: AugParam = AugParam(enabled=False, value=0.0)
    flipud:      AugParam = AugParam(enabled=False, value=0.0)
    fliplr:      AugParam = AugParam(enabled=True,  value=0.5)
    mosaic:      AugParam = AugParam(enabled=True,  value=1.0)
    mixup:       AugParam = AugParam(enabled=False, value=0.0)
    copy_paste:  AugParam = AugParam(enabled=False, value=0.0)
    erasing:     AugParam = AugParam(enabled=False, value=0.4)
    crop_fraction: AugParam = AugParam(enabled=False, value=1.0)


# ─── Hyperparameters ──────────────────────────────────────────
class HyperParams(BaseModel):
    epochs:           int   = 100
    patience:         int   = 50
    batch:            int   = 16
    optimizer:        str   = "SGD"
    device:           str   = "0"
    workers:          int   = 8
    fraction:         float = 1.0
    lr0:              float = 0.01
    lrf:              float = 0.01
    cos_lr:           bool  = False
    warmup_epochs:    float = 3.0
    warmup_momentum:  float = 0.8
    warmup_bias_lr:   float = 0.1
    momentum:         float = 0.937
    weight_decay:     float = 0.0005
    dropout:          float = 0.0
    box:              float = 7.5
    cls:              float = 0.5
    dfl:              float = 1.5
    close_mosaic:     int   = 10
    amp:              bool  = True
    multi_scale:      bool  = False
    freeze:           int   = 0
    overlap_mask:     bool  = True
    mask_ratio:       int   = 4
    resume:           bool  = False
    val:              bool  = True
    plots:            bool  = True
    save:             bool  = True
    save_period:      int   = -1
    project:          str   = "runs/train"
    name:             str   = "exp"
    exist_ok:         bool  = False
    cache:            str   = "false"


# ─── Full Training Config ─────────────────────────────────────
class TrainConfig(BaseModel):
    # Task & Model
    task:          str   = "detect"    # detect|segment|classify|pose|obb
    model_id:      str   = "yolov8n"
    model_pt:      str   = "yolov8n.pt"
    model_yaml:    str   = "yolov8n.yaml"
    train_mode:    str   = "finetune"  # finetune|scratch
    platform:      str   = "colab"    # colab|kaggle

    # Dataset
    data_format:   str   = "yolo"     # yolo|coco|voc|roboflow
    image_size:    int   = 640
    train_split:   float = 0.80
    val_split:     float = 0.10
    test_split:    float = 0.10
    shuffle:       bool  = True
    seed:          int   = 42
    class_names:   list[str] = Field(default_factory=list)

    # Augmentations & HPs
    augmentations: AugConfig    = Field(default_factory=AugConfig)
    hp:            HyperParams  = Field(default_factory=HyperParams)


# ─── LangGraph State ──────────────────────────────────────────
class PipelineState(TypedDict):
    config:           dict
    # tool results
    validation:       Optional[dict]
    cleaning_report:  Optional[dict]
    split_report:     Optional[dict]
    yaml_content:     Optional[str]
    eda_report:       Optional[dict]
    notebook_cells:   Optional[list]
    notebook_json:    Optional[str]
    data_yaml:        Optional[str]
    readme_md:        Optional[str]
    # control
    messages:         list[str]
    errors:           list[str]
    completed:        list[str]
    current:          str


# ─── API Request / Response ───────────────────────────────────
class GenerateReq(BaseModel):
    config: TrainConfig

class GenerateResp(BaseModel):
    success:          bool
    notebook_json:    Optional[str]  = None
    data_yaml:        Optional[str]  = None
    readme_md:        Optional[str]  = None
    cleaning_report:  Optional[dict] = None
    split_report:     Optional[dict] = None
    eda_report:       Optional[dict] = None
    validation:       Optional[dict] = None
    messages:         list[str]      = Field(default_factory=list)
    errors:           list[str]      = Field(default_factory=list)

class ValidateReq(BaseModel):
    config: TrainConfig

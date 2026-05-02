"""
YOLOForge – FastAPI Server

Fixes applied:
  1. Added DELETE /sessions/{session_id} endpoint for explicit session cleanup.
  2. Added background cleanup task: sessions older than SESSION_TTL_HOURS are
     removed automatically on each /generate call (lazy expiry, zero extra deps).
  3. /tools/split now surfaces split warnings in its response.
"""
from __future__ import annotations

import io
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from graph.pipeline import run_pipeline
from tools.yolo_tools import (
    analyze_dataset,
    build_yaml_string,
    clean_dataset,
    generate_notebook_cells,
    generate_readme,
    split_dataset,
    validate_config,
    assemble_ipynb,
)
from utils.schemas import GenerateReq, GenerateResp, TrainConfig, ValidateReq

# ─── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "YOLOForge API",
    description = "No-Code YOLO Training Pipeline — LangGraph Backend",
    version     = "2.1.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR      = Path(tempfile.gettempdir()) / "yoloforge_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Sessions older than this will be cleaned up lazily on each /generate call.
SESSION_TTL_HOURS = 24


# ─── Session helpers ───────────────────────────────────────────────────────
def _session_dir(session_id: str) -> Path:
    return UPLOAD_DIR / session_id


def _cleanup_expired_sessions() -> None:
    """Remove session directories older than SESSION_TTL_HOURS."""
    cutoff = time.time() - SESSION_TTL_HOURS * 3600
    for child in UPLOAD_DIR.iterdir():
        if child.is_dir():
            try:
                if child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# ROOT & HEALTH
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/")
async def root():
    return {
        "message": "Welcome to YOLOForge API",
        "status": "online",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "YOLOForge", "version": "2.1.0"}


# ═══════════════════════════════════════════════════════════════════════════
# UPLOAD DATASET
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a dataset ZIP archive.
    Returns session_id + structure preview.
    """
    suf = Path(file.filename or "ds.zip").suffix
    if suf not in {".zip", ".gz", ".tgz"}:
        raise HTTPException(400, f"Unsupported type: {suf}")

    session_id  = Path(tempfile.mktemp(prefix="yf_", dir="")).name
    session_dir = _session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    raw      = await file.read()
    zip_path = session_dir / (file.filename or "dataset.zip")
    zip_path.write_bytes(raw)

    extract_dir = session_dir / "raw"
    extract_dir.mkdir(exist_ok=True)

    try:
        if zip_path.suffix == ".zip":
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
                members = zf.namelist()
        else:
            import tarfile
            with tarfile.open(zip_path) as tf:
                tf.extractall(extract_dir)
                members = tf.getnames()
    except Exception as e:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(400, f"Extraction failed: {e}")

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs  = [p for p in extract_dir.rglob("*") if p.suffix.lower() in img_exts]
    txts  = list(extract_dir.rglob("*.txt"))
    yamls = list(extract_dir.rglob("*.yaml")) + list(extract_dir.rglob("*.yml"))
    jsons = list(extract_dir.rglob("*.json"))
    xmls  = list(extract_dir.rglob("*.xml"))

    fmt = "yolo"
    if jsons and any("annotation" in p.name.lower() for p in jsons):
        fmt = "coco"
    elif xmls:
        fmt = "voc"
    elif yamls and imgs:
        fmt = "roboflow"

    return {
        "session_id":      session_id,
        "extract_dir":     str(extract_dir),
        "file_size_mb":    round(len(raw) / 1024 / 1024, 2),
        "images_found":    len(imgs),
        "labels_found":    len(txts),
        "yamls_found":     len(yamls),
        "detected_format": fmt,
        "total_files":     len(members),
        "sample_paths":    [
            str(p.relative_to(extract_dir))
            for p in list(extract_dir.rglob("*"))[:12]
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# FULL PIPELINE  →  GENERATE
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/generate", response_model=GenerateResp)
async def generate(
    req:        GenerateReq,
    background: BackgroundTasks,
    session_id: Optional[str] = Query(None),
):
    """
    Run the complete LangGraph pipeline:
    validate → clean → split → eda → yaml → notebook → readme
    """
    # Lazily expire old sessions in background (no extra dependencies).
    background.add_task(_cleanup_expired_sessions)

    cfg_dict = req.config.model_dump()

    dataset_path: Optional[str] = None
    if session_id:
        raw_dir = _session_dir(session_id) / "raw"
        if raw_dir.exists():
            dataset_path = str(raw_dir)

    try:
        state = run_pipeline(config=cfg_dict, dataset_path=dataset_path)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")

    return GenerateResp(
        success         = len(state.get("errors", [])) == 0,
        notebook_json   = state.get("notebook_json"),
        data_yaml       = state.get("data_yaml"),
        readme_md       = state.get("readme_md"),
        cleaning_report = state.get("cleaning_report"),
        split_report    = state.get("split_report"),
        eda_report      = state.get("eda_report"),
        validation      = state.get("validation"),
        messages        = state.get("messages", []),
        errors          = state.get("errors",   []),
    )


# ═══════════════════════════════════════════════════════════════════════════
# DOWNLOAD  →  .ipynb stream
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/download/notebook")
async def download_notebook(
    req:        GenerateReq,
    background: BackgroundTasks,
    session_id: Optional[str] = Query(None),
):
    """Generate and stream the .ipynb file."""
    background.add_task(_cleanup_expired_sessions)

    cfg_dict = req.config.model_dump()

    dataset_path = None
    if session_id:
        raw_dir = _session_dir(session_id) / "raw"
        if raw_dir.exists():
            dataset_path = str(raw_dir)

    state    = run_pipeline(config=cfg_dict, dataset_path=dataset_path)
    nb_json  = state.get("notebook_json") or "{}"
    filename = (
        f"yoloforge_{cfg_dict.get('model_id','yolov8n')}"
        f"_{cfg_dict.get('platform','colab')}.ipynb"
    )

    return StreamingResponse(
        io.BytesIO(nb_json.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ═══════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════
@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Explicitly delete an upload session and all its files.
    Call this after /generate has completed to free disk space immediately.
    """
    sd = _session_dir(session_id)
    if not sd.exists():
        raise HTTPException(404, f"Session '{session_id}' not found.")
    shutil.rmtree(sd, ignore_errors=True)
    return {"deleted": session_id, "success": True}


@app.get("/sessions")
async def list_sessions():
    """List active upload sessions with their sizes and ages."""
    now = time.time()
    sessions = []
    for child in UPLOAD_DIR.iterdir():
        if child.is_dir():
            try:
                mtime   = child.stat().st_mtime
                age_h   = round((now - mtime) / 3600, 1)
                size_mb = round(
                    sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
                    / 1024 / 1024,
                    2,
                )
                sessions.append({
                    "session_id": child.name,
                    "age_hours":  age_h,
                    "size_mb":    size_mb,
                    "expired":    age_h > SESSION_TTL_HOURS,
                })
            except OSError:
                pass
    return {"sessions": sessions, "ttl_hours": SESSION_TTL_HOURS}


# ═══════════════════════════════════════════════════════════════════════════
# INDIVIDUAL TOOL ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/tools/validate")
async def api_validate(req: ValidateReq):
    """Validate config without running full pipeline."""
    return validate_config.invoke({"config_json": req.config.model_dump_json()})


@app.post("/tools/cells-preview")
async def api_cells_preview(req: GenerateReq):
    """Generate notebook cells for preview only (no filesystem ops)."""
    cfg = req.config.model_dump()
    return generate_notebook_cells.invoke({"config_json": json.dumps(cfg)})


@app.post("/tools/yaml-preview")
async def api_yaml_preview(class_names: list[str], task: str = "detect"):
    """Preview data.yaml content."""
    return build_yaml_string.invoke({"class_names": class_names, "task": task})


@app.post("/tools/clean/{session_id}")
async def api_clean(session_id: str):
    """
    Run cleaning on an uploaded dataset session.
    Returns staging_dir in the result — downstream calls should use that path.
    """
    raw_dir = _session_dir(session_id) / "raw"
    if not raw_dir.exists():
        raise HTTPException(404, "Session not found")
    return clean_dataset.invoke({"dataset_dir": str(raw_dir)})


@app.post("/tools/split/{session_id}")
async def api_split(
    session_id:  str,
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    test_ratio:  float = 0.10,
    shuffle:     bool  = True,
    seed:        int   = 42,
):
    """Split an uploaded dataset. Includes warnings for small datasets."""
    raw_dir = _session_dir(session_id) / "raw"
    out_dir = _session_dir(session_id) / "split"
    if not raw_dir.exists():
        raise HTTPException(404, "Session not found")
    result = split_dataset.invoke({
        "source_dir":  str(raw_dir),
        "output_dir":  str(out_dir),
        "train_ratio": train_ratio,
        "val_ratio":   val_ratio,
        "test_ratio":  test_ratio,
        "shuffle":     shuffle,
        "seed":        seed,
    })
    return result


@app.post("/tools/eda/{session_id}")
async def api_eda(session_id: str, class_names: list[str] = []):
    """Run EDA on a split dataset session. Empty splits are excluded from report."""
    split_dir = _session_dir(session_id) / "split"
    if not split_dir.exists():
        raise HTTPException(404, "Split not found — run /tools/split first")
    return analyze_dataset.invoke(
        {"dataset_dir": str(split_dir), "class_names": class_names}
    )


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH META
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/pipeline/graph")
async def graph_info():
    """Return pipeline graph topology."""
    return {
        "nodes": ["validate", "clean", "split", "eda", "yaml", "notebook", "readme", "abort"],
        "edges": [
            {"from": "START",    "to": "validate"},
            {"from": "validate", "to": "clean",    "cond": "no errors"},
            {"from": "validate", "to": "abort",    "cond": "has errors"},
            {"from": "clean",    "to": "split"},
            {"from": "split",    "to": "eda"},
            {"from": "eda",      "to": "yaml"},
            {"from": "yaml",     "to": "notebook"},
            {"from": "notebook", "to": "readme"},
            {"from": "readme",   "to": "END"},
        ],
        "tools": [
            t.name
            for t in __import__(
                "tools.yolo_tools", fromlist=["ALL_TOOLS"]
            ).ALL_TOOLS
        ],
    }


@app.get("/tools")
async def list_tools():
    from tools.yolo_tools import ALL_TOOLS
    return [{"name": t.name, "description": t.description[:120]} for t in ALL_TOOLS]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

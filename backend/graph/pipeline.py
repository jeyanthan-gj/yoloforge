"""
YOLOForge – LangGraph StateGraph Pipeline

Fixes applied:
  1. dataset_path added to PipelineState TypedDict (was missing, caused type errors
     and a false sense of safety around the injected field).
  2. node_split now RETURNS dataset_path in its patch dict instead of mutating
     state in-place (LangGraph ignores in-place mutations; only returned dicts
     are merged into state).  node_eda therefore receives the correct split output
     directory rather than the original pre-split upload path.
  3. node_clean returns the staging_dir from clean_dataset so downstream nodes
     use the cleaned copy, not the original source.
  4. split_report warnings are surfaced into pipeline messages.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from tools.yolo_tools import (
    analyze_dataset,
    assemble_ipynb,
    build_yaml_string,
    clean_dataset,
    generate_notebook_cells,
    generate_readme,
    split_dataset,
    validate_config,
)


# ─── Helper ────────────────────────────────────────────────────────────────
def _call(tool_fn, **kwargs) -> dict:
    """Invoke a LangChain @tool synchronously."""
    return tool_fn.invoke(kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# STATE  (FIX 1: dataset_path declared in TypedDict)
# ═══════════════════════════════════════════════════════════════════════════
class PipelineState(TypedDict):
    config:           dict
    # optional uploaded-dataset path — injected by run_pipeline, updated by nodes
    dataset_path:     Optional[str]          # ← FIX 1: was missing from TypedDict
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


# ═══════════════════════════════════════════════════════════════════════════
# NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def node_validate(state: PipelineState) -> dict:
    cfg_json = json.dumps(state["config"])
    result   = _call(validate_config, config_json=cfg_json)

    msgs = list(state.get("messages", []))
    errs = list(state.get("errors",   []))

    msgs.append("⚙ Config validation complete")
    for w in result.get("warnings",    []): msgs.append(f"  ⚠ {w}")
    for s in result.get("suggestions", []): msgs.append(f"  💡 {s}")
    for e in result.get("errors",      []): errs.append(f"CONFIG: {e}")

    return {
        "validation": result,
        "messages":   msgs,
        "errors":     errs,
        "current":    "validate",
        "completed":  state.get("completed", []) + ["validate"],
    }


def node_clean(state: PipelineState) -> dict:
    msgs = list(state.get("messages", []))
    errs = list(state.get("errors",   []))
    ds   = state.get("dataset_path")

    if not ds or not Path(ds).exists():
        msgs.append("ℹ No dataset uploaded — cleaning embedded in notebook")
        return {
            "cleaning_report": {"skipped": True},
            "messages":  msgs,
            "errors":    errs,
            "current":   "clean",
            "completed": state.get("completed", []) + ["clean"],
        }

    msgs.append(f"🧹 Cleaning dataset at {ds}")
    result = _call(clean_dataset, dataset_dir=ds)

    # FIX 2a: propagate the staging_dir so downstream nodes use the cleaned copy
    new_dataset_path = state.get("dataset_path")
    if result.get("success") and result.get("staging_dir"):
        new_dataset_path = result["staging_dir"]
        msgs.append(
            f"  ✅ {result['valid_final']} valid images "
            f"({result['corrupt']} corrupt, {result['duplicates']} dupes, "
            f"{result['labels_fixed']} labels fixed)"
        )
        msgs.append(f"  📁 Working from cleaned copy: {new_dataset_path}")
    else:
        errs.append(f"CLEAN: {result.get('error', 'unknown')}")

    return {
        "cleaning_report": result,
        "dataset_path":    new_dataset_path,   # ← propagate staging dir
        "messages":        msgs,
        "errors":          errs,
        "current":         "clean",
        "completed":       state.get("completed", []) + ["clean"],
    }


def node_split(state: PipelineState) -> dict:
    msgs = list(state.get("messages", []))
    errs = list(state.get("errors",   []))
    cfg  = state["config"]
    ds   = state.get("dataset_path")

    if not ds or not Path(ds).exists():
        msgs.append("ℹ No dataset — splitting embedded in notebook")
        return {
            "split_report":  {"skipped": True},
            "messages":      msgs,
            "errors":        errs,
            "current":       "split",
            "completed":     state.get("completed", []) + ["split"],
        }

    out_dir = str(Path(ds).parent / "split")
    msgs.append(f"📂 Splitting → {out_dir}")

    result = _call(
        split_dataset,
        source_dir  = ds,
        output_dir  = out_dir,
        train_ratio = cfg.get("train_split", 0.80),
        val_ratio   = cfg.get("val_split",   0.10),
        test_ratio  = cfg.get("test_split",  0.10),
        shuffle     = cfg.get("shuffle", True),
        seed        = cfg.get("seed", 42),
    )

    # FIX 3: surface split warnings into pipeline messages
    for w in result.get("warnings", []):
        msgs.append(f"  ⚠ {w}")

    # FIX 2b: RETURN updated dataset_path in the patch dict instead of
    # mutating state in-place (LangGraph ignores in-place mutations).
    new_dataset_path = state.get("dataset_path")
    if result.get("success"):
        new_dataset_path = result["output_dir"]
        msgs.append(
            f"  ✅ Train {result['train']} | Val {result['val']} | Test {result['test']}"
        )
    else:
        errs.append(f"SPLIT: {result.get('error')}")

    return {
        "split_report":  result,
        "dataset_path":  new_dataset_path,   # ← FIX: returned in patch, not mutated
        "messages":      msgs,
        "errors":        errs,
        "current":       "split",
        "completed":     state.get("completed", []) + ["split"],
    }


def node_eda(state: PipelineState) -> dict:
    msgs = list(state.get("messages", []))
    errs = list(state.get("errors",   []))
    cfg  = state["config"]
    ds   = state.get("dataset_path")          # now correctly points to split output
    spl  = state.get("split_report", {})

    if ds and spl and not spl.get("skipped") and Path(ds).exists():
        msgs.append("📊 Running EDA")
        result = _call(
            analyze_dataset,
            dataset_dir = ds,
            class_names = cfg.get("class_names", []),
        )
        if result.get("success") and result.get("train"):
            tr = result["train"]
            msgs.append(
                f"  Train: {tr['images']} imgs, "
                f"{tr['total_annotations']} ann, "
                f"imbalance {tr['imbalance_ratio']}x"
            )
            if tr["imbalance_ratio"] > 10:
                msgs.append("  ⚠ High class imbalance — consider oversampling")
        # Warn about any empty splits that were excluded from the report
        for split in ("val", "test"):
            if split not in result and spl.get(split, 0) == 0:
                msgs.append(
                    f"  ℹ {split} split has 0 images — excluded from EDA report"
                )
    else:
        msgs.append("ℹ No dataset — EDA embedded in notebook")
        result = {"skipped": True}

    return {
        "eda_report": result,
        "messages":   msgs,
        "errors":     errs,
        "current":    "eda",
        "completed":  state.get("completed", []) + ["eda"],
    }


def node_yaml(state: PipelineState) -> dict:
    msgs = list(state.get("messages", []))
    errs = list(state.get("errors",   []))
    cfg  = state["config"]

    msgs.append("📄 Building data.yaml")
    result = _call(
        build_yaml_string,
        class_names = cfg.get("class_names", []),
        task        = cfg.get("task", "detect"),
    )

    return {
        "data_yaml": result.get("yaml_content", ""),
        "messages":  msgs,
        "errors":    errs,
        "current":   "yaml",
        "completed": state.get("completed", []) + ["yaml"],
    }


def node_notebook(state: PipelineState) -> dict:
    msgs = list(state.get("messages", []))
    errs = list(state.get("errors",   []))
    cfg  = state["config"]

    msgs.append(f"📓 Building {cfg.get('platform','colab')} notebook")

    cells_result = _call(generate_notebook_cells, config_json=json.dumps(cfg))
    if not cells_result.get("success"):
        errs.append("NOTEBOOK: cell generation failed")
        return {
            "messages":  msgs,
            "errors":    errs,
            "current":   "notebook",
            "completed": state.get("completed", []) + ["notebook"],
        }

    cells    = cells_result["cells"]
    cell_cnt = cells_result["cell_count"]
    msgs.append(f"  ✅ {cell_cnt} cells generated")

    asm = _call(
        assemble_ipynb,
        cells_json = json.dumps(cells),
        platform   = cfg.get("platform", "colab"),
    )
    nb_json = asm.get("notebook_json", "")
    msgs.append(f"  ✅ Notebook assembled ({asm.get('cell_count')} cells)")

    return {
        "notebook_cells": cells,
        "notebook_json":  nb_json,
        "messages":       msgs,
        "errors":         errs,
        "current":        "notebook",
        "completed":      state.get("completed", []) + ["notebook"],
    }


def node_readme(state: PipelineState) -> dict:
    msgs = list(state.get("messages", []))
    errs = list(state.get("errors",   []))
    cfg  = state["config"]

    msgs.append("📋 Generating README.md")
    result = _call(generate_readme, config_json=json.dumps(cfg))
    msgs.append("  ✅ README complete")

    return {
        "readme_md": result.get("readme_content", ""),
        "messages":  msgs,
        "errors":    errs,
        "current":   "readme",
        "completed": state.get("completed", []) + ["readme"],
    }


def node_abort(state: PipelineState) -> dict:
    msgs = list(state.get("messages", []))
    msgs.append("❌ Pipeline aborted due to config errors:")
    for e in state.get("errors", []):
        msgs.append(f"  • {e}")
    return {"messages": msgs, "current": "abort"}


# ─── Conditional routing ────────────────────────────────────────────────────
def route_after_validate(state: PipelineState) -> Literal["clean", "abort"]:
    return "abort" if state.get("errors") else "clean"


# ═══════════════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ═══════════════════════════════════════════════════════════════════════════
def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("validate", node_validate)
    g.add_node("abort",    node_abort)
    g.add_node("clean",    node_clean)
    g.add_node("split",    node_split)
    g.add_node("eda",      node_eda)
    g.add_node("yaml",     node_yaml)
    g.add_node("notebook", node_notebook)
    g.add_node("readme",   node_readme)

    g.add_edge(START, "validate")
    g.add_conditional_edges(
        "validate",
        route_after_validate,
        {"clean": "clean", "abort": "abort"},
    )
    g.add_edge("abort",    END)
    g.add_edge("clean",    "split")
    g.add_edge("split",    "eda")
    g.add_edge("eda",      "yaml")
    g.add_edge("yaml",     "notebook")
    g.add_edge("notebook", "readme")
    g.add_edge("readme",   END)

    return g.compile()


# Singleton
_graph = build_graph()


def run_pipeline(config: dict, dataset_path: str | None = None) -> dict:
    """Execute the full LangGraph pipeline. Returns final state."""
    init: PipelineState = {
        "config":          config,
        "dataset_path":    dataset_path,   # now properly typed in PipelineState
        "validation":      None,
        "cleaning_report": None,
        "split_report":    None,
        "yaml_content":    None,
        "eda_report":      None,
        "notebook_cells":  None,
        "notebook_json":   None,
        "data_yaml":       None,
        "readme_md":       None,
        "messages":        ["🚀 YOLOForge pipeline started"],
        "errors":          [],
        "completed":       [],
        "current":         "init",
    }
    return _graph.invoke(init)

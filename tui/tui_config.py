# Path and step config for scGrapHiC TUI
from __future__ import annotations

import os
import glob
from pathlib import Path

# Path constants
TUI_DIR = Path(__file__).parent
PROJECT_ROOT = TUI_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

BASE_DIR = TUI_DIR

INFERENCE = PROJECT_ROOT / "inference.py"
FINETUNE = PROJECT_ROOT / "finetune.py"
DOWNLOAD = SRC_DIR / "download_datasets.py"

MOD_PREPROCESS = "src.preprocess_datasets"    # subcommands: scrnaseq or schic
MOD_PSEUDOBULK = "src.pseudobulk"
MOD_BUILD = "src.dataset_creator"

# Conda env
CONDA_ENV = os.environ.get("SCGRAPHIC_CONDA_ENV", "scg")

SAVED_PATHS_FILE = TUI_DIR / ".scgraphic_saved_paths.json"

# Step definitions
STEPS = [
    ("setup", "  0 · Setup"),
    ("welcome", "  scGrapHiC"),
    ("pseudobulk", "  1 · Pseudobulk"),
    ("rna", "  2 · Parse RNA-seq"),
    ("hic", "  3 · Parse scHi-C"),
    ("build", "  4 · Build dataset"),
    ("inference", "  5 · Inference"),
    ("finetune", "  6 · Fine-tune"),
    ("analysis", "  7 · Analysis"),
]

STEP_SUBTITLES = {
    "setup": "Environment check, dependency validation, and data download",
    "welcome": "Overview & run configuration",
    "pseudobulk": "Aggregate single cells into per-cell-type pseudobulk UMI + .pairs",
    "rna": "Preprocess pseudobulk scRNA-seq UMI matrices into genomic tracks",
    "hic": "Preprocess pseudobulk scHi-C contact pairs into binned matrices",
    "build": "Assemble the train / val / test .npz dataset",
    "inference": "Run the pretrained model",
    "finetune": "Adapt the checkpoint to your own cell types",
    "analysis": "Post-hoc visualisations and quality checks",
}

# RNA input formats
SELF_CONTAINED_FORMATS = {"h5ad", "10x_h5", "10x_mtx", "loom"}


def detect_rna_format(path_str: str) -> str:
    # Auto-detect scRNA-seq input format from a file path or directory
    if not path_str:
        return "unknown"
    p = Path(path_str)
    if p.is_dir():
        if glob.glob(str(p / "matrix.mtx*")):
            return "10x_mtx"
        return "unknown_dir"
    name = p.name.lower()
    if name.endswith(".h5ad"):
        return "h5ad"
    if name.endswith(".h5"):
        return "10x_h5"
    if name.endswith(".loom"):
        return "loom"
    if name.endswith((".csv.gz", ".csv")):
        return "csv"
    if name.endswith((".tsv.gz", ".tsv")):
        return "tsv"
    return "unknown"

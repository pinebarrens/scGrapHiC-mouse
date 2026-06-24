# Panel widgets for the scGrapHiC TUI.
# fmt: off
from __future__ import annotations

from textual.containers import Horizontal
from textual.widgets import (
    Button, Input, Label, Select, Static, Switch,
)

WELCOME_TEXT = """\
[bold cyan]scGrapHiC[/bold cyan] predicts cell-type-specific pseudo-bulk scHi-C \
contact maps from pseudo-bulk scRNA-seq and a bulk Hi-C structural prior.

[bold]Pipeline (top to bottom):[/bold]

  [cyan]1.[/cyan]  [bold]Pseudobulk[/bold]     — UMI sums + pooled .pairs per cell type
  [cyan]2.[/cyan]  [bold]Parse RNA-seq[/bold]  — 50 kb-binned signal tracks (.npy)
  [cyan]3.[/cyan]  [bold]Parse scHi-C[/bold]   — binned contact matrices (.npy)
                     [dim](ground-truth mode only)[/dim]
  [cyan]4.[/cyan]  [bold]Build dataset[/bold]  — tracks + matrices + bulk prior + motifs → .npz
  [cyan]5.[/cyan]  [bold]Inference[/bold]       — checkpoint eval (GenomeDISCO / SCC / SSIM)
  [cyan]6.[/cyan]  [bold]Fine-tune[/bold]       — adapt weights (ground-truth mode only)
  [cyan]7.[/cyan]  [bold]Analysis[/bold]        — UMAP, contact maps, metrics summary

[dim]Select a step from the sidebar to begin.[/dim]
"""


# Step 0: Setup
def build_setup(app):
    yield Static("[bold]Step 0 — Environment Setup[/bold]", markup=True)
    yield Static(
        "Check packages, GPU, and paths; optionally download reference data.\n",
        markup=True,
    )

    yield Static("", classes="section-divider")
    yield Static("[bold cyan]System Information[/bold cyan]", markup=True)

    yield Static("", classes="section-divider")
    yield Static("[bold cyan]Download Datasets[/bold cyan]", markup=True)
    yield Static(
        "GTF, bulk Hi-C priors, and/or HiRES GEO data. Skips files that "
        "already exist. [dim]Large downloads may take several GB — watch the log.[/dim]\n",
        markup=True,
    )
    yield Label("Data root directory", classes="field-label")
    yield Static(
        "[dim]Sets SCGRAPHIC_DATA_DIR for the download.[/dim]",
        markup=True,
    )
    yield Input(
        placeholder="/path/to/scgraphic_data",
        id="setup-data-dir", classes="field-input",
    )
    yield Label("Datasets to download", classes="field-label")
    yield Select(
        options=[
            ("All (structure + scHi-C + scRNA-seq)", "all"),
            ("Reference structure + GTF + bulk Hi-C", "structure"),
            ("HiRES scHi-C only", "schic"),
            ("HiRES scRNA-seq only", "scrnaseq"),
        ],
        id="setup-datasets", value="all",
    )


# Welcome
def build_welcome(app):
    yield Static(WELCOME_TEXT, id="welcome-body", markup=True)
    yield Static("", classes="section-divider")
    yield Static("[bold]Run Configuration[/bold]", markup=True)
    yield Label(
        "Do you have scHi-C ground-truth data?", classes="field-label"
    )
    yield Static(
        "[dim][bold]Blind[/bold]: scRNA-seq only — predict Hi-C from expression.\n"
        "[bold]Ground truth[/bold]: matching scHi-C for train/val/eval.[/dim]\n",
        markup=True,
    )
    yield Select(
        options=[
            ("Blind prediction — scRNA-seq only", "blind"),
            ("Ground truth available — scRNA-seq + scHi-C", "groundtruth"),
        ],
        id="run-mode",
        value=app.run_mode,
    )
    yield Static("", classes="section-divider")
    yield Static("[bold]Batch Execution[/bold]", markup=True)
    yield Static(
        "[dim]Run all applicable steps; disabled steps (e.g. scHi-C in blind mode) "
        "are skipped.[/dim]\n",
        markup=True,
    )


# Step 1: Pseudobulk (aggregate raw single cells by cell type)
def build_pseudobulk(app):
    yield Static("[bold]Step 1 — Pseudobulk[/bold]", markup=True)
    yield Static(
        "Sum UMI columns and pool .pairs contacts per cell type "
        "(feeds Parse RNA-seq / Parse scHi-C).\n",
        markup=True,
    )
    yield Label(
        "Cell metadata (.tsv/.csv with celltype + DNAbarcode columns)",
        classes="field-label",
    )
    yield Input(
        placeholder="/path/to/metadata.tsv",
        id="agg-meta", classes="field-input",
    )
    yield Label("Cell-by-gene UMI matrix (column 1 = gene)", classes="field-label")
    yield Input(
        placeholder="/path/to/umi.csv.gz",
        id="agg-umi", classes="field-input",
    )
    yield Label("Merged scHi-C .pairs file (readID = DNAbarcode)", classes="field-label")
    yield Static(
        "[dim]Ground-truth mode only — leave blank for blind prediction.[/dim]",
        markup=True,
    )
    yield Input(
        placeholder="/path/to/merged.pairs",
        id="agg-pairs", classes="field-input",
    )
    yield Label("Output dir — pseudobulk RNA-seq (UMI)", classes="field-label")
    yield Input(
        placeholder="/path/to/pseudobulk/RNAseq",
        id="agg-out-rnaseq", classes="field-input",
    )
    yield Label("Output dir — pseudobulk scHi-C (.pairs)", classes="field-label")
    yield Input(
        placeholder="/path/to/pseudobulk/scHiC",
        id="agg-out-schic", classes="field-input",
    )
    yield Label("Tissue label", classes="field-label")
    yield Input(value="brain", id="agg-tissue", classes="field-input")
    yield Static("\n[bold]Metadata columns[/bold]", markup=True)
    yield Label("Cell-type column", classes="field-label")
    yield Input(value="celltype", id="agg-celltype-col", classes="field-input")
    yield Label("Barcode column (matches .pairs readID + UMI columns)", classes="field-label")
    yield Input(value="DNAbarcode", id="agg-barcode-col", classes="field-input")
    yield Label("Stage column (optional — leave blank if none)", classes="field-label")
    yield Input(placeholder="e.g. Stage", id="agg-stage-col", classes="field-input")


# Step 2: Parse RNA-seq
def build_rna(app):
    yield Static("[bold]Step 2 — Parse scRNA-seq[/bold]", markup=True)
    yield Static(
        "Pseudobulk UMI matrices → per-chromosome 50 kb signal tracks (.npy).\n",
        markup=True,
    )
    yield Label(
        "Pseudobulk RNA-seq directory (out_rnaseq from Step 1)",
        classes="field-label",
    )
    yield Static(
        "[dim]Files named {cell_type}_{tissue}_..._umi.csv.gz[/dim]",
        markup=True,
    )
    yield Input(
        placeholder="/path/to/pseudobulk/RNAseq",
        id="rna-in-dir", classes="field-input",
    )
    yield Label("GTF / GFF3 annotation file", classes="field-label")
    yield Input(
        placeholder="/path/to/gencode.vM23.annotation.gff3.gz",
        id="rna-gtf", classes="field-input",
    )
    yield Label("Output directory (parsed tracks)", classes="field-label")
    yield Input(
        placeholder="/path/to/preprocessed/pseudobulk/RNAseq",
        id="rna-out", classes="field-input",
    )
    yield Label("Chromosome sizes file (chrom.sizes)", classes="field-label")
    yield Static(
        "[dim]chrom_name  size (whitespace-separated, one per line).[/dim]",
        markup=True,
    )
    yield Input(
        placeholder="/path/to/chrom.sizes",
        id="rna-chrom-sizes", classes="field-input",
    )
    yield Label("Resolution (bp)", classes="field-label")
    yield Input(value="50000", id="rna-res", classes="field-input")


# Step 3: Parse scHi-C
def build_hic(app):
    if app.run_mode == "blind":
        yield Static(
            "[bold yellow]Not needed in blind mode.[/bold yellow]\n\n"
            "scHi-C parsing is skipped when predicting from scRNA-seq alone.\n\n"
            "[dim]Enable on the Welcome page (ground-truth mode).[/dim]",
            markup=True,
        )
        return

    yield Static("[bold]Step 3 — Parse scHi-C[/bold]", markup=True)
    yield Static(
        "Pseudobulk .pairs → binned contact matrices (.npy) at 50 kb per chromosome.\n",
        markup=True,
    )
    yield Label(
        "Pseudobulk scHi-C directory (out_schic from Step 1)",
        classes="field-label",
    )
    yield Input(
        placeholder="/path/to/pseudobulk/scHiC",
        id="hic-pairs-dir", classes="field-input",
    )
    yield Label("Output directory (parsed matrices)", classes="field-label")
    yield Input(
        placeholder="/path/to/preprocessed/pseudobulk/scHiC",
        id="hic-out", classes="field-input",
    )
    yield Label("Chromosome sizes file (chrom.sizes)", classes="field-label")
    yield Input(
        placeholder="/path/to/chrom.sizes",
        id="hic-chrom-sizes", classes="field-input",
    )
    yield Label("Resolution (bp)", classes="field-label")
    yield Input(value="50000", id="hic-res", classes="field-input")


# Step 4: Build dataset (assemble the .npz)
def build_build(app):
    yield Static("[bold]Step 4 — Build Dataset[/bold]", markup=True)
    yield Static(
        "Combine parsed tracks/matrices with bulk Hi-C prior, motifs, and "
        "positional encodings into a model-ready .npz.\n",
        markup=True,
    )
    yield Label(
        "Parsed RNA-seq tracks dir (output of Parse RNA-seq)",
        classes="field-label",
    )
    yield Input(
        placeholder="/path/to/preprocessed/pseudobulk/RNAseq",
        id="pb-rna-dir", classes="field-input",
    )
    if app.run_mode == "groundtruth":
        yield Label(
            "Parsed scHi-C matrices dir (output of Parse scHi-C)",
            classes="field-label",
        )
        yield Input(
            placeholder="/path/to/preprocessed/pseudobulk/scHiC",
            id="pb-hic-dir", classes="field-input",
        )
    yield Label("Output .npz directory", classes="field-label")
    yield Input(
        placeholder="/path/to/processed",
        id="pb-out", classes="field-input",
    )
    yield Label("Experiment name (.npz file prefix)", classes="field-label")
    yield Input(value="scgraphic", id="pb-exp", classes="field-input")
    yield Label("Dataset split", classes="field-label")
    yield Select(
        options=[
            ("train", "train"),
            ("valid", "valid"),
            ("test",  "test"),
            ("ood",   "ood"),
            ("debug", "debug"),
        ],
        id="pb-split", value="train",
    )
    yield Label("Normalization algorithm", classes="field-label")
    yield Select(
        options=[
            ("library_size_normalization", "library_size_normalization"),
            ("log2_norm",   "log2_norm"),
            ("log10_norm",  "log10_norm"),
            ("zscore_norm", "zscore_norm"),
            ("sqrt_norm",   "sqrt_norm"),
        ],
        id="pb-norm", value="library_size_normalization",
    )
    yield Label("Resolution (bp)", classes="field-label")
    yield Input(value="50000", id="pb-res", classes="field-input")
    yield Label("Positional encoding dimension", classes="field-label")
    yield Input(value="16", id="pb-pe-dim", classes="field-input")

    yield Static("\n[bold]Auxiliary data[/bold]", markup=True)
    yield Label(
        "Bulk Hi-C directory (chr*_*.npz files)",
        classes="field-label",
    )
    yield Input(
        placeholder="/path/to/preprocessed/bulk/<sample>",
        id="pb-bulk-dir", classes="field-input",
    )
    yield Label(
        "Motifs directory (ctcf/ and cpg/ sub-dirs)",
        classes="field-label",
    )
    yield Input(
        placeholder="/path/to/preprocessed/motifs",
        id="pb-motifs-dir", classes="field-input",
    )
    yield Label("Dataset labels JSON", classes="field-label")
    yield Input(
        placeholder="/path/to/dataset_labels.json",
        id="pb-labels-json", classes="field-input",
    )


# Step 5: Inference
def build_inference(app):
    yield Static("[bold]Step 5 — Inference[/bold]", markup=True)
    yield Static(
        "Evaluate a pretrained checkpoint on a test .npz. "
        "Reports GenomeDISCO, SCC, SSIM, MSE, and TAD similarity.\n",
        markup=True,
    )
    yield Label("Model checkpoint (.ckpt)", classes="field-label")
    yield Input(
        placeholder="/path/to/scgraphic.ckpt",
        id="inf-ckpt", classes="field-input",
    )
    yield Label("Test .npz file", classes="field-label")
    yield Input(
        placeholder="/path/to/scgraphic_test.npz",
        id="inf-npz", classes="field-input",
    )
    yield Label(
        "Experiment name (results sub-folder)",
        classes="field-label",
    )
    yield Input(value="scgraphic", id="inf-exp", classes="field-input")
    yield Label("Results output directory", classes="field-label")
    yield Input(
        placeholder="/path/to/results",
        id="inf-results", classes="field-input",
    )
    yield Label("Encoder hidden embedding size", classes="field-label")
    yield Input(value="64", id="inf-encoder-hidden", classes="field-input")
    yield Label("Number of graph conv blocks", classes="field-label")
    yield Input(value="3", id="inf-graph-conv-blocks", classes="field-input")
    yield Label("Device", classes="field-label")
    yield Select(
        options=[("cpu", "cpu"), ("gpu", "gpu")],
        id="inf-device", value="cpu",
    )


# Step 6: Fine-tune
def build_finetune(app):
    if app.run_mode == "blind":
        yield Static(
            "[bold yellow]Fine-tuning requires ground-truth scHi-C.[/bold yellow]\n\n"
            "Without target contact maps there is no supervised loss signal.\n\n"
            "[dim]Switch to ground-truth mode on the Welcome page.[/dim]",
            markup=True,
        )
        return

    yield Static("[bold]Step 6 — Fine-tune[/bold]", markup=True)
    yield Static(
        "Adapt pretrained weights using a .npz from the Build dataset step.\n",
        markup=True,
    )

    yield Horizontal(
        Static(
            "Epoch  [bold cyan]—[/bold cyan]",
            id="mt-epoch", classes="metric-card", markup=True,
        ),
        Static(
            "Val SCC  [bold cyan]—[/bold cyan]",
            id="mt-scc", classes="metric-card", markup=True,
        ),
        Static(
            "Val GD   [bold cyan]—[/bold cyan]",
            id="mt-gd", classes="metric-card", markup=True,
        ),
        Static(
            "Val SSIM [bold cyan]—[/bold cyan]",
            id="mt-ssim", classes="metric-card", markup=True,
        ),
        Static(
            "Val Loss [bold cyan]—[/bold cyan]",
            id="mt-loss", classes="metric-card", markup=True,
        ),
        id="metric-bar",
    )

    yield Label("Training .npz file", classes="field-label")
    yield Input(placeholder="/path/to/train.npz", id="ft-train-npz")
    yield Label("Validation .npz file (optional)", classes="field-label")
    yield Input(placeholder="/path/to/val.npz", id="ft-val-npz")
    yield Label("Test .npz file (optional)", classes="field-label")
    yield Input(placeholder="/path/to/test.npz", id="ft-test-npz")

    yield Label("Pretrained checkpoint (.ckpt)", classes="field-label")
    yield Input(placeholder="/path/to/scgraphic.ckpt", id="ft-ckpt")
    yield Label("Output directory", classes="field-label")
    yield Input(
        placeholder="/path/to/finetune_output",
        id="ft-out",
    )
    yield Label("Experiment name", classes="field-label")
    yield Input(value="finetune", id="ft-exp")
    yield Label("Epochs", classes="field-label")
    yield Input(value="100", id="ft-epochs")
    yield Label("Learning rate", classes="field-label")
    yield Input(value="1e-5", id="ft-lr")
    yield Label("Batch size", classes="field-label")
    yield Input(value="32", id="ft-batch")
    yield Label("Early stopping patience (0 = off)", classes="field-label")
    yield Input(value="30", id="ft-patience")
    yield Label("Freeze encoder (train decoder only)", classes="field-label")
    yield Switch(id="ft-freeze", value=False)
    yield Label("Resume from checkpoint (optional)", classes="field-label")
    yield Input(placeholder="/path/to/last.ckpt", id="ft-resume")

    yield Static("", classes="section-divider")
    yield Horizontal(
        Button(
            "■  Stop", id="stop-ft",
            variant="error", classes="run-btn", disabled=True,
        ),
    )


# Step 7: Analysis
def build_analysis(app):
    yield Static("[bold]Post-hoc Analysis[/bold]", markup=True)
    yield Static(
        "UMAP, contact-map plots, and metrics tables from preprocessed data "
        "and inference results.\n",
        markup=True,
    )

    yield Static("", classes="section-divider")
    yield Static("[bold cyan]Metrics Summary[/bold cyan]", markup=True)
    yield Static(
        "SCC, GenomeDISCO, SSIM, MSE, and TAD similarity per cell type/chromosome "
        "→ CSV + log output.\n",
        markup=True,
    )
    yield Label(
        "Results directory (from Inference step)",
        classes="field-label",
    )
    yield Input(
        placeholder="/path/to/inference_results/experiment_name",
        id="ana-metrics-dir", classes="field-input",
    )
    yield Horizontal(
        Button(
            "▶ Summarise Metrics", id="run-metrics",
            variant="primary", classes="run-btn",
        ),
    )

    yield Static("", classes="section-divider")
    yield Static("[bold cyan]UMAP Generation[/bold cyan]", markup=True)
    yield Static(
        "2-D UMAP coloured by cell type (expression matrix + metadata).\n",
        markup=True,
    )
    yield Label(
        "Expression data (.h5ad / .h5 / .loom / .csv / .tsv / 10x MEX dir)",
        classes="field-label",
    )
    yield Input(
        placeholder="/path/to/data.h5ad  or  /path/to/10x_mtx_dir/",
        id="ana-matrix", classes="field-input",
    )
    yield Label(
        "Cell metadata (.csv with cell_type column)",
        classes="field-label",
    )
    yield Static(
        "[dim]Optional for .h5ad / .loom if cell_type is in .obs[/dim]",
        markup=True,
    )
    yield Input(
        placeholder="/path/to/metadata.csv",
        id="ana-meta", classes="field-input",
    )
    yield Label("Output directory for plots", classes="field-label")
    yield Input(
        placeholder="/path/to/analysis_output",
        id="ana-out", classes="field-input",
    )

    yield Static("", classes="section-divider")
    yield Static("[bold]UMAP Parameters[/bold]", markup=True)
    yield Label("n_neighbors (default 15)", classes="field-label")
    yield Input(value="15", id="ana-umap-neighbors", classes="field-input")
    yield Label("min_dist (default 0.5)", classes="field-label")
    yield Input(value="0.5", id="ana-umap-min-dist", classes="field-input")
    yield Label("n_components (default 2)", classes="field-label")
    yield Input(value="2", id="ana-umap-components", classes="field-input")

    yield Horizontal(
        Button(
            "▶ UMAP", id="run-umap",
            variant="primary", classes="run-btn",
        ),
    )

    yield Static("", classes="section-divider")
    yield Static(
        "[bold cyan]Contact Map Visualisation[/bold cyan]", markup=True
    )
    yield Static(
        "Predicted vs. ground-truth heatmaps from inference results.\n",
        markup=True,
    )
    yield Label("Results directory (from Inference step)", classes="field-label")
    yield Input(
        placeholder="/path/to/inference_results",
        id="ana-results", classes="field-input",
    )
    yield Label("Chromosome (e.g. chr1, or 'all' for every chromosome)", classes="field-label")
    yield Input(value="chr1", id="ana-chrom", classes="field-input")
    yield Label("Max plots per cell type (0 = all)", classes="field-label")
    yield Input(value="5", id="ana-max-plots", classes="field-input")
    yield Horizontal(
        Button(
            "▶ Visualise", id="run-viz",
            variant="primary", classes="run-btn",
        ),
    )

# fmt: off
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap
import glob
from datetime import datetime
from pathlib import Path
from typing import Optional

from textual import work
from textual.containers import ScrollableContainer
from textual.widgets import (
    Button,
    Input,
    Label,
    ListItem,
    RichLog,
    Select,
    Static,
    Switch,
)

import tui_panels
from tui_config import (
    BASE_DIR,
    PROJECT_ROOT,
    DOWNLOAD,
    FINETUNE,
    INFERENCE,
    MOD_PREPROCESS,
    MOD_PSEUDOBULK,
    MOD_BUILD,
    CONDA_ENV,
    SELF_CONTAINED_FORMATS,
    detect_rna_format,
)


class TuiRunnersMixin:
    # File checks, validators, conda subprocess helpers, and RichLog output

    async def async_remount_mode_panes(self) -> None:
        # hic, build, and finetune UIs depend on run_mode
        try:
            builders = {
                "pane-hic":      tui_panels.build_hic,
                "pane-build":    tui_panels.build_build,
                "pane-finetune": tui_panels.build_finetune,
            }
            for pane_id, gen in builders.items():
                c = self.query_one(f"#{pane_id}", ScrollableContainer)
                await c.remove_children()
                await c.mount(*list(gen(self)))
            self.restore_inputs()
        except Exception as exc:
            # Keep the TUI alive
            self.append_log(
                f"[red]Mode-dependent panel rebuild failed: {exc}[/red]", "red"
            )

    # Step 0: Environment check
    @work(thread=True)
    def run_env_check(self) -> None:
        self.append_log("[bold]── Checking environment ──[/bold]")

        # Python version
        import sys
        self.append_log(
            f"[green]✔[/green]  Python: {sys.version.split()[0]} "
            f"({sys.executable})"
        )

        # Key packages
        required = [
            "torch", "lightning", "pytorch_lightning", "scanpy",
            "numpy", "scipy", "pandas", "sklearn", "textual",
            "torch_geometric", "anndata", "matplotlib", "seaborn",
            "umap",
        ]
        missing = []
        for pkg in required:
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "?")
                self.append_log(f"[green]✔[/green]  {pkg}: {ver}")
            except ImportError:
                self.append_log(f"[red]✘[/red]  {pkg}: [red]NOT INSTALLED[/red]")
                missing.append(pkg)

        # GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.append_log(
                    f"[green]✔[/green]  GPU: {gpu_name} "
                    f"(CUDA {torch.version.cuda})"
                )
            else:
                self.append_log(
                    "[yellow]⚠[/yellow]  GPU: CUDA not available — "
                    "CPU-only inference will be used"
                )
        except Exception as exc:
            self.append_log(f"[yellow]⚠[/yellow]  GPU check failed: {exc}")

        # Conda env
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            self.append_log(f"[green]✔[/green]  Conda env: {conda_prefix}")
        else:
            self.append_log(
                "[yellow]⚠[/yellow]  Not running inside a conda "
                "environment. Some scripts use `conda run -n scgraphic_env`."
            )

        if missing:
            self.append_log(
                f"[red bold]Missing {len(missing)} package(s): "
                f"{', '.join(missing)}[/red bold]"
            )
            self.append_log(
                "[dim]Install with: pip install " +
                " ".join(missing) + "[/dim]"
            )
        else:
            self.append_log(
                "[green bold]All required packages found ✔[/green bold]"
            )

    # Step 0: Download datasets
    @work(thread=True)
    def run_download(self) -> None:
        data_dir = self.get_input("setup-data-dir")
        self.append_log("Starting dataset download...", "cyan")
        if data_dir:
            self.append_log(
                f"Data root: [bold]{data_dir}[/bold] "
                "(setting SCGRAPHIC_DATA_DIR)",
                "cyan",
            )

        env = os.environ.copy()
        if data_dir:
            env["SCGRAPHIC_DATA_DIR"] = data_dir

        # The download script reads globals that resolve from env vars
        datasets = self.get_input("setup-datasets", "all")
        cmd = self.conda_python() + [str(DOWNLOAD), "--datasets", datasets]
        self.append_log(f"CMD: {' '.join(cmd)}", "dim")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=str(PROJECT_ROOT),
            )
            log = self.query_one("#log", RichLog)
            for line in iter(proc.stdout.readline, ""):
                self.call_from_thread(log.write, line.rstrip())
            proc.wait()
            rc = proc.returncode
            style = "green" if rc == 0 else "red"
            icon  = "✔" if rc == 0 else "✘"
            self.append_log(f"{icon} Download finished (exit {rc})", style)
        except Exception as exc:
            self.append_log(f"Download failed: {exc}", "red")

    # Run All
    @work(thread=True)
    def run_all_steps(self) -> None:
        self.append_log(
            "[bold cyan]═══ Running all pipeline steps ═══[/bold cyan]"
        )

        steps = [
            ("Pseudobulk",     self.run_aggregate),
            ("Parse RNA-seq",  self.run_parse_rna),
        ]
        if self.run_mode == "groundtruth":
            steps.append(("Parse scHi-C", self.run_parse_hic))
        steps += [
            ("Build dataset", self.run_build),
            ("Inference",     self.run_inference),
        ]
        if self.run_mode == "groundtruth":
            steps.append(("Fine-tune", self.run_finetune))
        steps.append(("Analysis — UMAP", self.run_umap))

        for i, (label, runner) in enumerate(steps, 1):
            self.append_log(
                f"[bold]── [{i}/{len(steps)}] {label} ──[/bold]",
                "cyan",
            )
            # Each runner is a @work(thread=True) method — call it
            # and it will run in a Textual worker thread
            runner()

        self.append_log(
            "[bold green]═══ All steps dispatched ═══[/bold green]"
        )

    # Validation helpers
    def check_file_exists(
        self, path_str: str, label: str, extensions: list[str] | None = None,
    ) -> bool:
        if not path_str:
            self.append_log(f"[red]\u2718[/red]  {label}: [red]not specified[/red]")
            return False
        p = Path(path_str)
        if not p.exists():
            self.append_log(
                f"[red]\u2718[/red]  {label}: [red]path does not exist[/red] "
                f"\u2014 {path_str}"
            )
            return False
        if not p.is_file():
            self.append_log(
                f"[red]\u2718[/red]  {label}: [red]not a file[/red] "
                f"\u2014 {path_str}"
            )
            return False
        if extensions:
            if not any(path_str.lower().endswith(e) for e in extensions):
                self.append_log(
                    f"[yellow]\u26a0[/yellow]  {label}: unexpected extension "
                    f"(expected {', '.join(extensions)}) \u2014 {path_str}"
                )
        self.append_log(f"[green]\u2714[/green]  {label}: OK")
        return True

    def check_dir_exists(
        self, path_str: str, label: str, glob_pattern: str | None = None,
    ) -> bool:
        if not path_str:
            self.append_log(f"[red]\u2718[/red]  {label}: [red]not specified[/red]")
            return False
        p = Path(path_str)
        if not p.is_dir():
            self.append_log(
                f"[red]\u2718[/red]  {label}: [red]not a directory[/red] "
                f"\u2014 {path_str}"
            )
            return False
        if glob_pattern:
            matches = glob.glob(str(p / glob_pattern))
            if matches:
                self.append_log(
                    f"[green]\u2714[/green]  {label}: OK "
                    f"({len(matches)} file(s) matching '{glob_pattern}')"
                )
            else:
                self.append_log(
                    f"[yellow]\u26a0[/yellow]  {label}: directory exists but "
                    f"no files matching '{glob_pattern}'"
                )
        else:
            self.append_log(f"[green]\u2714[/green]  {label}: OK")
        return True

    def check_writable_dir(self, path_str: str, label: str) -> bool:
        if not path_str:
            self.append_log(f"[red]\u2718[/red]  {label}: [red]not specified[/red]")
            return False
        p = Path(path_str)
        if p.is_dir():
            self.append_log(f"[green]\u2714[/green]  {label}: OK (exists)")
            return True
        if p.parent.is_dir():
            self.append_log(
                f"[green]\u2714[/green]  {label}: OK (will be created)"
            )
            return True
        self.append_log(
            f"[red]\u2718[/red]  {label}: [red]parent directory does not "
            f"exist[/red] \u2014 {p.parent}"
        )
        return False

    def check_int(self, value: str, label: str, positive: bool = True) -> bool:
        try:
            v = int(value)
            if positive and v <= 0:
                self.append_log(
                    f"[red]\u2718[/red]  {label}: "
                    f"[red]must be a positive integer[/red] \u2014 {value}"
                )
                return False
            self.append_log(f"[green]\u2714[/green]  {label}: {v}")
            return True
        except ValueError:
            self.append_log(
                f"[red]\u2718[/red]  {label}: "
                f"[red]must be an integer[/red] \u2014 {value}"
            )
            return False

    # Per-step validators
    @work(thread=True)
    def validate_aggregate(self) -> None:
        self.append_log("[bold]\u2500\u2500 Validating Pseudobulk inputs \u2500\u2500[/bold]")
        ok = True
        ok &= self.check_file_exists(self.get_input("agg-meta"), "Cell metadata", [".tsv", ".csv"])
        ok &= self.check_file_exists(
            self.get_input("agg-umi"), "UMI matrix", [".csv", ".csv.gz", ".tsv", ".tsv.gz"],
        )
        pairs = self.get_input("agg-pairs")
        if self.run_mode == "groundtruth" or pairs:
            ok &= self.check_file_exists(pairs, "Merged .pairs file", [".pairs", ".txt"])
        else:
            self.append_log("[dim]\u2714  scHi-C .pairs: not provided (blind mode)[/dim]")
        ok &= self.check_writable_dir(self.get_input("agg-out-rnaseq"), "Output dir (RNA-seq)")
        ok &= self.check_writable_dir(self.get_input("agg-out-schic"), "Output dir (scHi-C)")
        self.append_log(
            "[green bold]All inputs valid \u2014 ready to run.[/green bold]"
            if ok else
            "[red bold]Fix the issues above before running.[/red bold]"
        )

    @work(thread=True)
    def validate_rna(self) -> None:
        self.append_log("[bold]\u2500\u2500 Validating Parse RNA-seq inputs \u2500\u2500[/bold]")
        ok = True
        ok &= self.check_dir_exists(
            self.get_input("rna-in-dir"), "Pseudobulk RNA-seq directory", "*umi.csv*",
        )
        ok &= self.check_file_exists(
            self.get_input("rna-gtf"), "GTF annotation",
            [".gff3.gz", ".gff3", ".gtf", ".gtf.gz"],
        )
        ok &= self.check_writable_dir(self.get_input("rna-out"), "Output directory")
        ok &= self.check_file_exists(
            self.get_input("rna-chrom-sizes"), "Chromosome sizes file", [],
        )
        ok &= self.check_int(self.get_input("rna-res", "50000"), "Resolution")
        self.append_log(
            "[green bold]All inputs valid \u2014 ready to run.[/green bold]"
            if ok else
            "[red bold]Fix the issues above before running.[/red bold]"
        )

    @work(thread=True)
    def validate_hic(self) -> None:
        self.append_log("[bold]\u2500\u2500 Validating Parse scHi-C inputs \u2500\u2500[/bold]")
        ok = True
        ok &= self.check_dir_exists(
            self.get_input("hic-pairs-dir"), "Pairs directory", "*.pairs*",
        )
        ok &= self.check_writable_dir(self.get_input("hic-out"), "Output directory")
        ok &= self.check_file_exists(
            self.get_input("hic-chrom-sizes"), "Chromosome sizes file", [],
        )
        ok &= self.check_int(self.get_input("hic-res", "50000"), "Resolution")
        self.append_log(
            "[green bold]All inputs valid \u2014 ready to run.[/green bold]"
            if ok else
            "[red bold]Fix the issues above before running.[/red bold]"
        )

    @work(thread=True)
    def validate_build(self) -> None:
        self.append_log("[bold]\u2500\u2500 Validating Build dataset inputs \u2500\u2500[/bold]")
        ok = True
        ok &= self.check_dir_exists(
            self.get_input("pb-rna-dir"), "Parsed RNA-seq dir", "*scrnaseq",
        )
        ok &= self.check_dir_exists(
            self.get_input("pb-hic-dir"), "Parsed scHi-C dir", "*schic",
        )
        ok &= self.check_writable_dir(self.get_input("pb-out"), "Output directory")
        ok &= self.check_int(self.get_input("pb-res", "50000"), "Resolution")
        ok &= self.check_int(self.get_input("pb-pe-dim", "16"), "PE dimension")
        bulk_dir = self.get_input("pb-bulk-dir")
        if bulk_dir:
            ok &= self.check_dir_exists(bulk_dir, "Bulk Hi-C directory", "chr*_*.npz")
        else:
            self.append_log("[red]\u2718[/red]  Bulk Hi-C directory: required but not set", "red")
            ok = False
        motifs_dir = self.get_input("pb-motifs-dir")
        if motifs_dir:
            ok &= self.check_dir_exists(motifs_dir, "Motifs directory", "ctcf")
        else:
            self.append_log("[red]\u2718[/red]  Motifs directory: required but not set", "red")
            ok = False
        labels_json = self.get_input("pb-labels-json")
        if labels_json:
            ok &= self.check_file_exists(labels_json, "Dataset labels JSON", [".json"])
        else:
            self.append_log("[yellow]\u26a0[/yellow]  Dataset labels JSON: not set (will use script default)")
        self.append_log(
            "[green bold]All inputs valid \u2014 ready to run.[/green bold]"
            if ok else
            "[red bold]Fix the issues above before running.[/red bold]"
        )

    @work(thread=True)
    def validate_inference(self) -> None:
        self.append_log("[bold]\u2500\u2500 Validating Inference inputs \u2500\u2500[/bold]")
        ok = True
        ok &= self.check_file_exists(
            self.get_input("inf-ckpt"), "Model checkpoint", [".ckpt"],
        )
        ok &= self.check_file_exists(
            self.get_input("inf-npz"), "Test .npz file", [".npz"],
        )
        ok &= self.check_writable_dir(
            self.get_input("inf-results"), "Results directory",
        )
        self.append_log(
            "[green bold]All inputs valid \u2014 ready to run.[/green bold]"
            if ok else
            "[red bold]Fix the issues above before running.[/red bold]"
        )

    # Preview: inspect data files and report statistics
    @work(thread=True)
    def preview_rna(self) -> None:
        matrix_path = self.get_input("rna-matrix")
        meta_path = self.get_input("rna-meta")
        if not matrix_path:
            self.append_log("Provide an expression data path first.", "red")
            return

        fmt = detect_rna_format(matrix_path)
        if fmt in ("unknown", "unknown_dir"):
            self.append_log(f"Unrecognised format: {matrix_path}", "red")
            return

        self.append_log(
            f"[bold]\u2500\u2500 Previewing RNA-seq data ({fmt}) \u2500\u2500"
            "[/bold]",
        )

        args_json = json.dumps({
            "matrix": matrix_path,
            "format": fmt,
            "meta": meta_path or "",
        })

        script_lines = [
            "import sys, os, json",
            f"args = json.loads({repr(args_json)})",
            "matrix_path = args['matrix']",
            "fmt         = args['format']",
            "meta_path   = args['meta']",
            "",
            "try:",
            "    import scanpy as sc",
            "except ImportError:",
            "    print('ERROR: scanpy is required'); sys.exit(1)",
            "import pandas as pd, numpy as np",
            "import scipy.sparse",
            "",
            "print(f'Loading {fmt}: {matrix_path}')",
            "if fmt == 'h5ad':      adata = sc.read_h5ad(matrix_path)",
            "elif fmt == '10x_h5':  adata = sc.read_10x_h5(matrix_path)",
            "elif fmt == '10x_mtx': adata = sc.read_10x_mtx(matrix_path)",
            "elif fmt == 'loom':    adata = sc.read_loom(matrix_path)",
            "else:",
            "    sep = ',' if 'csv' in matrix_path.lower() else '\\t'",
            "    df = pd.read_csv(matrix_path, index_col=0, sep=sep, "
            "nrows=None)",
            "    from anndata import AnnData",
            "    adata = AnnData(df)",
            "",
            "adata.var_names_make_unique()",
            "X = adata.X",
            "print(f'Cells:  {adata.n_obs:,}')",
            "print(f'Genes:  {adata.n_vars:,}')",
            "",
            "if scipy.sparse.issparse(X):",
            "    nnz = X.nnz",
            "    total = adata.n_obs * adata.n_vars",
            "    sparsity = 1.0 - nnz / total",
            "    total_umis = float(X.sum())",
            "else:",
            "    arr = np.asarray(X)",
            "    nnz = int(np.count_nonzero(arr))",
            "    total = arr.size",
            "    sparsity = 1.0 - nnz / total",
            "    total_umis = float(arr.sum())",
            "print(f'Non-zero entries: {nnz:,} / {total:,}')",
            "print(f'Sparsity: {sparsity:.1%}')",
            "print(f'Total UMIs: {total_umis:,.0f}')",
            "if adata.n_obs > 0:",
            "    print(f'Mean UMIs/cell: {total_umis / adata.n_obs:,.1f}')",
            "    print(f'Median genes/cell: "
            "{np.median(np.diff(X.indptr) if scipy.sparse.issparse(X) "
            "else np.count_nonzero(arr, axis=1)):,.0f}')",
            "",
            "# Cell-type labels",
            "ct_names = ['cell_type', 'celltype', 'CellType', 'Cell_Type',",
            "            'cluster', 'louvain', 'leiden', 'annotation',",
            "            'cell_type_predicted', 'Celltype']",
            "found = next((c for c in ct_names if c in adata.obs.columns), "
            "None)",
            "if not found and meta_path and os.path.isfile(meta_path):",
            "    meta = pd.read_csv(meta_path)",
            "    mc = next((c for c in ct_names if c in meta.columns), None)",
            "    if mc:",
            "        found = mc",
            "        adata.obs['cell_type'] = meta[mc].values[:adata.n_obs]",
            "        print(f'Cell-type column (external metadata): \"{mc}\"')",
            "if found and found in adata.obs.columns:",
            "    vc = adata.obs[found].value_counts()",
            "    print(f'Cell types ({len(vc)} found, column \"{found}\"):')",
            "    for ct, n in vc.items():",
            "        print(f'  {ct}: {n:,}')",
            "elif found:",
            "    pass",
            "else:",
            "    print('No cell-type annotation found in object or metadata')",
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=str(BASE_DIR),
        ) as f:
            f.write("\n".join(script_lines))
            script_path = f.name

        try:
            cmd = self.conda_python() + [script_path]
            self.append_log(f"CMD: {' '.join(cmd)}", "dim")
            self.stream_cmd(cmd)
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    @work(thread=True)
    def preview_hic(self) -> None:
        pairs_dir = self.get_input("hic-pairs-dir")
        meta_path = self.get_input("hic-meta")
        if not pairs_dir:
            self.append_log("Provide a pairs-file directory first.", "red")
            return
        if not Path(pairs_dir).is_dir():
            self.append_log(f"Directory not found: {pairs_dir}", "red")
            return

        self.append_log(
            "[bold]\u2500\u2500 Previewing scHi-C data \u2500\u2500[/bold]",
        )

        args_json = json.dumps({
            "pairs_dir": pairs_dir,
            "meta": meta_path or "",
        })

        script_lines = [
            "import sys, os, json, glob",
            f"args = json.loads({repr(args_json)})",
            "pairs_dir = args['pairs_dir']",
            "meta_path = args['meta']",
            "import pandas as pd",
            "",
            "pairs = sorted(glob.glob(os.path.join(pairs_dir, '*.pairs*'))"
            " + glob.glob(os.path.join(pairs_dir, '*.txt')))",
            "print(f'Files found: {len(pairs)}')",
            "if not pairs:",
            "    print('No .pairs or .txt files in this directory')",
            "    sys.exit(0)",
            "",
            "total_contacts = 0",
            "total_size = 0",
            "shown = min(len(pairs), 8)",
            "for fp in pairs[:shown]:",
            "    sz = os.path.getsize(fp)",
            "    total_size += sz",
            "    n = 0",
            "    import gzip",
            "    opener = gzip.open if fp.endswith('.gz') else open",
            "    with opener(fp, 'rt') as fh:",
            "        for line in fh:",
            "            if not line.startswith('#'):",
            "                n += 1",
            "    total_contacts += n",
            "    print(f'  {os.path.basename(fp)}: {n:,} contacts  "
            "({sz / 1024 / 1024:.1f} MB)')",
            "if len(pairs) > shown:",
            "    remaining = len(pairs) - shown",
            "    print(f'  ... and {remaining} more file(s)')",
            "    for fp in pairs[shown:]:",
            "        total_size += os.path.getsize(fp)",
            "print(f'Total contacts (sampled): {total_contacts:,}')",
            "print(f'Total size: {total_size / 1024 / 1024:.1f} MB')",
            "",
            "if meta_path and os.path.isfile(meta_path):",
            "    meta = pd.read_csv(meta_path)",
            "    ct_names = ['cell_type', 'celltype', 'CellType', "
            "'Cell_Type', 'cluster', 'Celltype']",
            "    mc = next((c for c in ct_names if c in meta.columns), "
            "meta.columns[0])",
            "    vc = meta[mc].value_counts()",
            "    print(f'Metadata: {len(meta):,} cells, {len(vc)} cell "
            "types (column \"{mc}\"):')",
            "    for ct, n in vc.items():",
            "        print(f'  {ct}: {n:,}')",
            "else:",
            "    print('No metadata file provided')",
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=str(BASE_DIR),
        ) as f:
            f.write("\n".join(script_lines))
            script_path = f.name

        try:
            cmd = self.conda_python() + [script_path]
            self.append_log(f"CMD: {' '.join(cmd)}", "dim")
            self.stream_cmd(cmd)
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    # Internal helpers
    def append_log(self, msg: str, style: str = "") -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        log = self.query_one("#log", RichLog)
        if style:
            log.write(f"[dim]{ts}[/dim]  [{style}]{msg}[/{style}]")
        else:
            log.write(f"[dim]{ts}[/dim]  {msg}")

    def conda_python(self) -> list[str]:
        return [
            "conda", "run", "-n", CONDA_ENV, "--no-capture-output",
            "python",
        ]

    def get_input(self, widget_id: str, fallback: str = "") -> str:
        try:
            w = self.query_one(f"#{widget_id}")
            if isinstance(w, Input):
                return w.value.strip()
            if isinstance(w, Select):
                v = w.value
                if v is not None and str(v) != "Select.BLANK":
                    return str(v)
                return fallback
            if isinstance(w, Switch):
                return "true" if w.value else "false"
        except Exception:
            pass
        return self.saved_inputs.get(widget_id, fallback)

    def update_hic_visibility(self) -> None:
        """Grey out, lock, and disable steps that require scHi-C ground truth."""
        # Steps blocked in blind mode, with their original sidebar labels
        blocked_steps = {
            "hic":      "  3 · Parse scHi-C",
            "finetune": "  6 · Fine-tune",
        }
        is_blind = self.run_mode == "blind"

        for step_id, original_label in blocked_steps.items():
            # ── Sidebar item: visual state ──
            try:
                item = self.query_one(f"#step-{step_id}", ListItem)
                label_widget = item.query_one("Label", Label)
                if is_blind:
                    item.add_class("--skipped")
                    label_widget.update(f"  🔒 {original_label.strip()}")
                else:
                    item.remove_class("--skipped")
                    label_widget.update(original_label)
            except Exception:
                pass

            # ── Pane buttons: disable/enable ──
            try:
                pane = self.query_one(f"#pane-{step_id}", ScrollableContainer)
                for btn in pane.query(Button):
                    btn.disabled = is_blind
            except Exception:
                pass

    # Run: Step 1 — Parse RNA-seq (with automatic format conversion)
    @work(thread=True)
    def run_aggregate(self) -> None:
        self.append_log("Starting Pseudobulk aggregation...", "cyan")
        cmd = self.conda_python() + [
            "-m", MOD_PSEUDOBULK,
            "--metadata",     self.get_input("agg-meta"),
            "--umi",          self.get_input("agg-umi"),
            "--pairs",        self.get_input("agg-pairs"),
            "--out_rnaseq",   self.get_input("agg-out-rnaseq"),
            "--out_schic",    self.get_input("agg-out-schic"),
            "--tissue",       self.get_input("agg-tissue", "brain"),
            "--celltype_col", self.get_input("agg-celltype-col", "celltype"),
            "--barcode_col",  self.get_input("agg-barcode-col", "DNAbarcode"),
        ]
        stage_col = self.get_input("agg-stage-col")
        if stage_col:
            cmd += ["--stage_col", stage_col]
        self.append_log(f"CMD: {' '.join(cmd)}", "dim")
        self.stream_cmd(cmd)

    @work(thread=True)
    def run_parse_rna(self) -> None:
        self.append_log("Starting Parse RNA-seq...", "cyan")
        cmd = self.conda_python() + [
            "-m", MOD_PREPROCESS, "scrnaseq",
            "--matrix",      self.get_input("rna-in-dir"),
            "--output_dir",  self.get_input("rna-out"),
            "--gtf",         self.get_input("rna-gtf"),
            "--chrom_sizes", self.get_input("rna-chrom-sizes"),
            "--resolution",  self.get_input("rna-res", "50000"),
            "--pseudobulk",
        ]
        self.append_log(f"CMD: {' '.join(cmd)}", "dim")
        self.stream_cmd(cmd)

    def convert_rna_to_csv(
        self, matrix_path: str, fmt: str, ext_meta: str,
    ) -> Optional[dict]:
        """Convert a non-CSV scRNA-seq input to temporary CSV files.

        Runs a scanpy-based subprocess and returns a dict with keys:
            matrix, barcodes, n_cells, n_genes, and optionally metadata.
        Returns None on failure.
        """
        convert_dir = tempfile.mkdtemp(prefix="scgraphic_convert_")

        args_json = json.dumps({
            "matrix": matrix_path,
            "format": fmt,
            "out_dir": convert_dir,
            "ext_meta": ext_meta or "",
        })

        script_lines = [
            "import sys, os, json",
            "import pandas as pd",
            f"args = json.loads({repr(args_json)})",
            "matrix_path = args['matrix']",
            "fmt         = args['format']",
            "out_dir     = args['out_dir']",
            "ext_meta    = args['ext_meta']",
            "",
            "try:",
            "    import scanpy as sc",
            "except ImportError:",
            "    print('ERROR: scanpy is required for format conversion. "
            "Install with: pip install scanpy')",
            "    sys.exit(1)",
            "",
            "print(f'Loading {fmt}: {matrix_path}')",
            "if fmt == 'h5ad':",
            "    adata = sc.read_h5ad(matrix_path)",
            "elif fmt == '10x_h5':",
            "    adata = sc.read_10x_h5(matrix_path)",
            "elif fmt == '10x_mtx':",
            "    adata = sc.read_10x_mtx(matrix_path)",
            "elif fmt == 'loom':",
            "    adata = sc.read_loom(matrix_path)",
            "else:",
            "    print(f'ERROR: unsupported format {fmt}')",
            "    sys.exit(1)",
            "",
            "print(f'Loaded {adata.n_obs} cells x {adata.n_vars} genes')",
            "adata.var_names_make_unique()",
            "",
            "import scipy.sparse",
            "if scipy.sparse.issparse(adata.X):",
            "    dense = adata.X.toarray()",
            "else:",
            "    dense = adata.X",
            "",
            "df = pd.DataFrame(dense, index=adata.obs_names, "
            "columns=adata.var_names)",
            "",
            "matrix_out = os.path.join(out_dir, 'cell_by_gene_matrix.csv')",
            "df.to_csv(matrix_out)",
            "print(f'Saved matrix: {matrix_out}')",
            "",
            "bc_out = os.path.join(out_dir, 'barcodes.txt')",
            "pd.Series(adata.obs_names).to_csv(bc_out, index=False, "
            "header=False)",
            "print(f'Saved barcodes: {bc_out}')",
            "",
            "result = {",
            "    'matrix': matrix_out,",
            "    'barcodes': bc_out,",
            "    'n_cells': int(adata.n_obs),",
            "    'n_genes': int(adata.n_vars),",
            "}",
            "",
            "ct_names = ['cell_type', 'celltype', 'CellType', 'Cell_Type',",
            "            'cluster', 'louvain', 'leiden', 'cell_type_predicted',",
            "            'annotation', 'Celltype']",
            "found = next((c for c in ct_names if c in adata.obs.columns), "
            "None)",
            "if found:",
            "    meta_out = os.path.join(out_dir, 'metadata.csv')",
            "    mdf = adata.obs[[found]].copy()",
            "    mdf.columns = ['cell_type']",
            "    mdf.index.name = 'barcode'",
            "    mdf.to_csv(meta_out)",
            "    result['metadata'] = meta_out",
            "    print(f'Extracted cell-type metadata from .obs[\"{found}\"]: "
            "{meta_out}')",
            "elif ext_meta and os.path.isfile(ext_meta):",
            "    result['metadata'] = ext_meta",
            "    print(f'Using external metadata: {ext_meta}')",
            "else:",
            "    print('WARNING: no cell_type column found in object and "
            "no external metadata provided')",
            "",
            "print('CONVERT_RESULT:' + json.dumps(result))",
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=str(BASE_DIR),
        ) as f:
            f.write("\n".join(script_lines))
            script_path = f.name

        try:
            cmd = self.conda_python() + [script_path]
            self.append_log(f"CMD: {' '.join(cmd)}", "dim")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            log = self.query_one("#log", RichLog)
            result_json = None
            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip()
                if not line:
                    continue
                if line.startswith("CONVERT_RESULT:"):
                    try:
                        result_json = json.loads(line[15:])
                        continue
                    except json.JSONDecodeError:
                        pass
                self.call_from_thread(log.write, line)
            proc.wait()

            if proc.returncode != 0:
                self.append_log("Format conversion failed.", "red")
                return None
            if result_json is None:
                self.append_log("No conversion output received.", "red")
                return None

            self.append_log(
                f"Conversion complete \u2014 "
                f"{result_json.get('n_cells', '?')} cells, "
                f"{result_json.get('n_genes', '?')} genes",
                "green",
            )
            return result_json
        except Exception as exc:
            self.append_log(f"Conversion error: {exc}", "red")
            return None
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    # ══════════════════════════════════════════════════════════════════════════
    # Run: Step 3 — Parse scHi-C
    # ══════════════════════════════════════════════════════════════════════════
    @work(thread=True)
    def run_parse_hic(self) -> None:
        self.append_log("Starting Parse scHi-C...", "cyan")
        cmd = self.conda_python() + [
            "-m", MOD_PREPROCESS, "schic",
            "--pairs_dir",   self.get_input("hic-pairs-dir"),
            "--output_dir",  self.get_input("hic-out"),
            "--chrom_sizes", self.get_input("hic-chrom-sizes"),
            "--resolution",  self.get_input("hic-res", "50000"),
        ]
        self.append_log(f"CMD: {' '.join(cmd)}", "dim")
        self.stream_cmd(cmd)

    # ══════════════════════════════════════════════════════════════════════════
    # Run: Step 4 — Build dataset
    # ══════════════════════════════════════════════════════════════════════════
    @work(thread=True)
    def run_build(self) -> None:
        self.append_log("Starting dataset build...", "cyan")
        rna_dir = self.get_input("pb-rna-dir")
        if not rna_dir:
            self.append_log("Parsed RNA-seq dir (pb-rna-dir) is required.", "red")
            return
        out_dir = self.get_input("pb-out")
        if not out_dir:
            self.append_log("Output directory (pb-out) is required.", "red")
            return
        bulk_dir = self.get_input("pb-bulk-dir")
        if not bulk_dir:
            self.append_log("Bulk Hi-C directory (pb-bulk-dir) is required.", "red")
            return
        motifs_dir = self.get_input("pb-motifs-dir")
        if not motifs_dir:
            self.append_log("Motifs directory (pb-motifs-dir) is required.", "red")
            return
        hic_dir = self.get_input("pb-hic-dir")
        if not hic_dir:
            self.append_log("Parsed scHi-C dir (pb-hic-dir) is required.", "red")
            return
        cmd = self.conda_python() + [
            "-m", MOD_BUILD,
            "--rnaseq_dir", rna_dir,
            "--schic_dir", hic_dir,
            "--bulk_dir", bulk_dir,
            "--motifs_dir", motifs_dir,
            "--output_dir", out_dir,
            "--experiment", self.get_input("pb-exp", "scgraphic"),
            "--set",        self.get_input("pb-split", "train"),
            "--normalization_algorithm",
                            self.get_input("pb-norm", "library_size_normalization"),
            "--resolution", self.get_input("pb-res", "50000"),
            "--pos_encodings_dim", self.get_input("pb-pe-dim", "16"),
        ]
        labels_json = self.get_input("pb-labels-json")
        if labels_json:
            cmd += ["--dataset_labels", labels_json]
        self.append_log(f"CMD: {' '.join(cmd)}", "dim")
        self.stream_cmd(cmd)

    # Run: Step 4 — Inference
    @work(thread=True)
    def run_inference(self) -> None:
        self.append_log("Starting Inference...", "cyan")
        device = self.get_input("inf-device", "cpu")
        cmd = self.conda_python() + [
            str(INFERENCE),
            "--experiment", self.get_input("inf-exp", "scgraphic"),
            "--checkpoint", self.get_input("inf-ckpt"),
            "--npz", self.get_input("inf-npz"),
            "--results", self.get_input("inf-results"),
            "--device", device,
            "--encoder_hidden_embedding_size",
            self.get_input("inf-encoder-hidden", "64"),
            "--num_graph_conv_blocks",
            self.get_input("inf-graph-conv-blocks", "3"),
            "--rna_seq", "True",
            "--ctcf_motif", "True",
            "--cpg_motif", "True",
            "--use_bulk", "True",
            "--positional_encodings", "True",
            "--pos_encodings_dim", "16",
        ]
        self.append_log(f"CMD: {' '.join(cmd)}", "dim")
        self.stream_cmd(cmd, extra_env={"SCGRAPHIC_RESULTS_DIR": self.get_input("inf-results")})

    # Run: Step 5 — Fine-tune (validate / run / stop)
    @work(thread=True)
    def validate_finetune_paths(self) -> None:
        self.append_log("Validating fine-tune paths...", "yellow")
        args = self.build_finetune_args(json_progress=False)
        cmd = self.conda_python() + [str(FINETUNE)] + args + ["--validate_only"]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
                cwd=str(PROJECT_ROOT),
            )
            output = result.stdout.strip()
            if not output:
                self.append_log(
                    f"No output from validate_only. stderr: "
                    f"{result.stderr[:300]}",
                    "red",
                )
                return
            report = json.loads(output)
            for check in report.get("checks", []):
                icon = (
                    "[green]\u2714[/green]"
                    if check["exists"]
                    else "[red]\u2718[/red]"
                )
                self.append_log(
                    f"{icon}  {check['name']:22s}  {check['path']}", ""
                )
            if report["ok"]:
                self.append_log(
                    "All paths validated \u2014 ready to fine-tune.", "green"
                )
            else:
                self.append_log(
                    "One or more paths are missing. Fix them above.", "red"
                )
        except Exception as exc:
            self.append_log(f"Validation error: {exc}", "red")

    def build_finetune_args(self, json_progress: bool = True) -> list[str]:
        args = [
            "--mode",        self.get_input("ft-mode", "npz"),
            "--checkpoint",  self.get_input("ft-ckpt"),
            "--output_dir",  self.get_input("ft-out"),
            "--experiment",  self.get_input("ft-exp", "finetune_mice"),
            "--epochs",      self.get_input("ft-epochs", "100"),
            "--lr",          self.get_input("ft-lr", "1e-5"),
            "--batch_size",  self.get_input("ft-batch", "32"),
            "--early_stopping_patience", self.get_input("ft-patience", "30"),
        ]
        mode = self.get_input("ft-mode", "npz")
        if mode == "npz":
            if v := self.get_input("ft-train-npz"):
                args += ["--train_npz", v]
            if v := self.get_input("ft-val-npz"):
                args += ["--val_npz", v]
            if v := self.get_input("ft-test-npz"):
                args += ["--test_npz", v]
        else:
            if v := self.get_input("ft-rna-dir"):
                args += ["--rnaseq_dir", v]
            if v := self.get_input("ft-hic-dir"):
                args += ["--schic_dir", v]
        if self.get_input("ft-freeze") == "true":
            args.append("--freeze_encoder")
        if v := self.get_input("ft-resume"):
            args += ["--resume_from", v]
        if json_progress:
            args.append("--json_progress")
        return args

    @work(thread=True)
    def run_finetune(self) -> None:
        self.append_log("Starting fine-tuning ...", "cyan")
        args = self.build_finetune_args(json_progress=True)
        cmd = self.conda_python() + [str(FINETUNE)] + args

        self.append_log(f"CMD: {' '.join(cmd)}", "dim")

        def reset_stop_btn(enabled: bool):
            try:
                btn = self.query_one("#stop-ft", Button)
                btn.disabled = not enabled
            except Exception:
                pass

        self.call_from_thread(reset_stop_btn, True)

        try:
            env = os.environ.copy()
            env["SCGRAPHIC_RESULTS_DIR"] = self.get_input("ft-out")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=str(PROJECT_ROOT),
            )
            self.ft_proc = proc

            log = self.query_one("#log", RichLog)
            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip()
                if not line:
                    continue
                if line.startswith("{"):
                    try:
                        rec = json.loads(line)
                        self.call_from_thread(self.update_ft_metrics, rec)
                        continue
                    except json.JSONDecodeError:
                        pass
                self.call_from_thread(log.write, line)

            proc.wait()
            rc = proc.returncode
            if rc == 0:
                self.append_log("Fine-tuning complete \u2714", "green")
            else:
                self.append_log(f"Fine-tuning exited with code {rc}", "red")
        except Exception as exc:
            self.append_log(f"Error: {exc}", "red")
        finally:
            self.ft_proc = None
            self.call_from_thread(reset_stop_btn, False)

    def stop_finetune(self) -> None:
        if self.ft_proc is not None:
            self.ft_proc.terminate()
            self.append_log("Stop signal sent to fine-tuning process.", "yellow")

    def update_ft_metrics(self, rec: dict) -> None:
        def fmt_metric(v) -> str:
            return f"{v:.4f}" if v is not None else "\u2014"

        updates = {
            "mt-epoch": ("Epoch",    str(rec.get("epoch", "\u2014"))),
            "mt-scc":   ("Val SCC",  fmt_metric(rec.get("val_scc"))),
            "mt-gd":    ("Val GD",   fmt_metric(rec.get("val_gd"))),
            "mt-ssim":  ("Val SSIM", fmt_metric(rec.get("val_ssim"))),
            "mt-loss":  ("Val Loss", fmt_metric(rec.get("val_loss"))),
        }
        for widget_id, (label, value) in updates.items():
            try:
                card = self.query_one(f"#{widget_id}", Static)
                card.update(f"{label}  [bold cyan]{value}[/bold cyan]")
            except Exception:
                pass
        epoch = rec.get("epoch", "?")
        scc   = fmt_metric(rec.get("val_scc"))
        gd    = fmt_metric(rec.get("val_gd"))
        self.append_log(
            f"[dim]epoch {epoch:>4}[/dim]  SCC={scc}  GD={gd}  "
            f"SSIM={fmt_metric(rec.get('val_ssim'))}  "
            f"loss={fmt_metric(rec.get('val_loss'))}",
        )

    # Run: Step 6 — Analysis (Metrics / UMAP / contact-map visualisation)

    @work(thread=True)
    def run_metrics_summary(self) -> None:
        results_dir = self.get_input("ana-metrics-dir")
        if not results_dir:
            self.append_log("Please specify the results directory.", "red")
            return
        rp = Path(results_dir)
        if not rp.is_dir():
            self.append_log(f"Directory not found: {results_dir}", "red")
            return

        self.append_log("Scanning for results CSV files...", "cyan")

        # Look for full_results.csv or results.csv
        csv_candidates = list(rp.glob("**/full_results.csv")) + \
                         list(rp.glob("**/results.csv"))
        if not csv_candidates:
            self.append_log(
                "No results.csv / full_results.csv found. "
                "Run Inference first.", "red",
            )
            return

        args_json = json.dumps({
            "results_dir": str(rp),
            "csv_files": [str(c) for c in csv_candidates],
        })

        script_lines = [
            "import sys, os, json",
            "import pandas as pd",
            "import numpy as np",
            f"args = json.loads({repr(args_json)})",
            "",
            "all_dfs = []",
            "for csv_path in args['csv_files']:",
            "    print(f'Reading {csv_path}')",
            "    try:",
            "        df = pd.read_csv(",
            "            csv_path, sep=',', header=None,",
            "            names=['tissue', 'stage', 'cell_type',",
            "                   'cell_count', 'chr', 'start_0', 'start_1',",
            "                   'SSIM', 'GD', 'SCC', 'TAD_sim', 'MSE', 'Kendall_Tau']",
            "        )",
            "    except Exception:",
            "        df = pd.read_csv(",
            "            csv_path, sep=',', header=None,",
            "            names=['tissue', 'stage', 'cell_type',",
            "                   'cell_count', 'chr', 'start_0', 'start_1',",
            "                   'MSE', 'SSIM', 'GD', 'SCC']",
            "        )",
            "    all_dfs.append(df)",
            "",
            "data = pd.concat(all_dfs, ignore_index=True)",
            "print(f'\\nTotal records: {len(data)}')",
            "print()",
            "",
            "# Per-cell-type summary",
            "metrics = [c for c in ['MSE','SSIM','GD','SCC','TAD_sim','Kendall_Tau']",
            "           if c in data.columns]",
            "grouped = data.groupby('cell_type')[metrics]",
            "summary = grouped.agg(['median', 'mean', 'std'])",
            "",
            "# Flatten columns",
            "summary.columns = ['_'.join(col).strip() for col in summary.columns]",
            "print('=== Per Cell-Type Summary ===')",
            "print(summary.to_string())",
            "print()",
            "",
            "# Global summary",
            "print('=== Global Medians ===')",
            "for m in metrics:",
            "    print(f'  {m}: {data[m].median():.4f}')",
            "print()",
            "",
            "# Save combined CSV",
            "out_csv = os.path.join(args['results_dir'], 'metrics_summary.csv')",
            "summary.to_csv(out_csv)",
            "print(f'Summary saved to {out_csv}')",
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=str(BASE_DIR),
        ) as f:
            f.write("\n".join(script_lines))
            script_path = f.name

        try:
            cmd = self.conda_python() + [script_path]
            self.append_log(f"CMD: {' '.join(cmd)}", "dim")
            self.stream_cmd(cmd)
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    # UMAP
    @work(thread=True)
    def run_umap(self) -> None:
        matrix = self.get_input("ana-matrix")
        meta   = self.get_input("ana-meta")
        out_dir = self.get_input("ana-out")

        if not matrix or not out_dir:
            self.append_log(
                "Please provide at least the expression data and an "
                "output directory.",
                "red",
            )
            return

        fmt = detect_rna_format(matrix)
        p = Path(matrix)
        if fmt == "10x_mtx":
            if not p.is_dir():
                self.append_log(f"Directory not found: {matrix}", "red")
                return
        elif fmt in ("unknown", "unknown_dir"):
            self.append_log(f"Unrecognised format: {matrix}", "red")
            return
        else:
            if not p.is_file():
                self.append_log(f"File not found: {matrix}", "red")
                return

        if meta and not Path(meta).is_file():
            self.append_log(f"Metadata file not found: {meta}", "red")
            return

        # Grab UMAP parameters from the UI
        n_neighbors = self.get_input("ana-umap-neighbors", "15")
        min_dist = self.get_input("ana-umap-min-dist", "0.5")
        n_components = self.get_input("ana-umap-components", "2")

        self.append_log(
            f"Generating UMAP ({fmt}) — n_neighbors={n_neighbors}, "
            f"min_dist={min_dist}, n_components={n_components}",
            "cyan",
        )

        args_json = json.dumps({
            "matrix": matrix,
            "format": fmt,
            "meta": meta or "",
            "out_dir": out_dir,
            "n_neighbors": int(n_neighbors),
            "min_dist": float(min_dist),
            "n_components": int(n_components),
        })

        script_lines = [
            "import sys, os, json",
            f"args = json.loads({repr(args_json)})",
            "matrix_path = args['matrix']",
            "fmt         = args['format']",
            "meta_path   = args['meta']",
            "out_dir     = args['out_dir']",
            "n_neighbors = args['n_neighbors']",
            "min_dist    = args['min_dist']",
            "n_components = args['n_components']",
            "os.makedirs(out_dir, exist_ok=True)",
            "",
            "try:",
            "    import scanpy as sc",
            "except ImportError:",
            "    print('ERROR: scanpy is required. Install with: "
            "pip install scanpy')",
            "    sys.exit(1)",
            "import pandas as pd",
            "",
            "print(f'Loading {fmt}: {matrix_path}')",
            "if fmt == 'h5ad':",
            "    adata = sc.read_h5ad(matrix_path)",
            "elif fmt == '10x_h5':",
            "    adata = sc.read_10x_h5(matrix_path)",
            "elif fmt == '10x_mtx':",
            "    adata = sc.read_10x_mtx(matrix_path)",
            "elif fmt == 'loom':",
            "    adata = sc.read_loom(matrix_path)",
            "else:",
            "    adata = sc.read_csv(matrix_path)",
            "print(f'Loaded {adata.n_obs} cells x {adata.n_vars} genes')",
            "adata.var_names_make_unique()",
            "",
            "# Resolve cell-type labels",
            "ct_names = ['cell_type', 'celltype', 'CellType', 'Cell_Type',",
            "            'cluster', 'louvain', 'leiden', 'annotation',",
            "            'cell_type_predicted', 'Celltype']",
            "found = next((c for c in ct_names if c in adata.obs.columns), "
            "None)",
            "if found:",
            "    adata.obs['cell_type'] = adata.obs[found]",
            "    print(f'Using .obs[\"{found}\"] for cell-type labels')",
            "elif meta_path and os.path.isfile(meta_path):",
            "    meta = pd.read_csv(meta_path)",
            "    mc = next((c for c in ct_names if c in meta.columns), "
            "meta.columns[0])",
            "    adata.obs['cell_type'] = meta[mc].values[:adata.n_obs]",
            "    print(f'Using external metadata column \"{mc}\"')",
            "else:",
            "    print('WARNING: no cell-type labels found — "
            "UMAP will be uncoloured')",
            "    adata.obs['cell_type'] = 'unknown'",
            "",
            "sc.pp.normalize_total(adata)",
            "sc.pp.log1p(adata)",
            "sc.pp.pca(adata)",
            "print(f'PCA variance ratio (first 5 components): "
            "{list(adata.uns[\"pca\"][\"variance_ratio\"][:5])}')",
            "sc.pp.neighbors(adata, n_neighbors=n_neighbors)",
            "sc.tl.umap(adata, min_dist=min_dist, "
            "n_components=n_components)",
            "sc.settings.figdir = out_dir",
            "sc.pl.umap(adata, color='cell_type', "
            "save='_scgraphic.png', show=False)",
            "final = os.path.join(out_dir, 'umap_scgraphic.png')",
            "print(f'UMAP saved to {final}')",
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=str(BASE_DIR),
        ) as f:
            f.write("\n".join(script_lines))
            script_path = f.name

        try:
            cmd = self.conda_python() + [script_path]
            self.append_log(f"CMD: {' '.join(cmd)}", "dim")
            self.stream_cmd(cmd)
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass
    # Contact-map visualisation
    @work(thread=True)
    def run_contactmap_viz(self) -> None:
        results_dir = self.get_input("ana-results")
        chrom = self.get_input("ana-chrom", "chr1")
        max_plots = self.get_input("ana-max-plots", "5")

        if not results_dir:
            self.append_log("Please specify the results directory.", "red")
            return
        if not Path(results_dir).is_dir():
            self.append_log(f"Results directory not found: {results_dir}", "red")
            return

        self.append_log(
            f"Generating contact-map visualisation (chrom={chrom}, "
            f"max_plots={max_plots})...", "cyan",
        )

        args_json = json.dumps({
            "results_dir": results_dir,
            "chrom": chrom,
            "max_plots": int(max_plots) if max_plots else 0,
        })

        script_lines = textwrap.dedent("""\
            import sys, os, json, glob
            from collections import defaultdict
            import numpy as np
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap

            REDMAP = LinearSegmentedColormap.from_list('bright_red', [(1,1,1),(1,0,0)])
            args = json.loads('''ARGS_PLACEHOLDER''')
            results_dir = args['results_dir']
            chrom = args['chrom']
            max_plots = args['max_plots']

            # Find all generated .npy files
            gen_files = glob.glob(os.path.join(
                results_dir, '**', 'generated', '*.npy'), recursive=True)
            print(f'Found {len(gen_files)} generated .npy files')

            # Filter by chromosome
            if chrom.lower() != 'all':
                gen_files = [f for f in gen_files
                             if os.path.basename(f).startswith(chrom + '_')]
                print(f'After chromosome filter ({chrom}): {len(gen_files)} files')

            # Group by cell type
            groups = defaultdict(list)
            for gf in gen_files:
                parts = gf.split(os.sep)
                try:
                    gen_idx = parts.index('generated')
                    cell_type = parts[gen_idx - 2] if gen_idx >= 2 else 'unknown'
                except ValueError:
                    cell_type = 'unknown'
                groups[cell_type].append(gf)

            viz_dir = os.path.join(results_dir, 'contact_map_viz')
            os.makedirs(viz_dir, exist_ok=True)
            total = 0

            for cell_type, files in sorted(groups.items()):
                subset = files[:max_plots] if max_plots > 0 else files
                print(f'\\n--- {cell_type}: {len(subset)} plots ---')
                for gf in subset:
                    fname = os.path.basename(gf)
                    target_path = gf.replace(os.sep + 'generated' + os.sep,
                                             os.sep + 'targets' + os.sep)
                    generated = np.load(gf)
                    has_target = os.path.exists(target_path)
                    if has_target:
                        target = np.load(target_path)
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                        ax1.matshow(generated, cmap=REDMAP)
                        ax1.set_title('Predicted', fontsize=10)
                        ax1.axis('off')
                        ax2.matshow(target, cmap=REDMAP)
                        ax2.set_title('Target', fontsize=10)
                        ax2.axis('off')
                        tag = fname.replace('.npy', '')
                        fig.suptitle(f'{cell_type} — {tag}', fontsize=12)
                    else:
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.matshow(generated, cmap=REDMAP)
                        tag = fname.replace('.npy', '')
                        ax.set_title(f'{cell_type} — {tag}', fontsize=10)
                        ax.axis('off')
                    out_png = os.path.join(
                        viz_dir, f'{cell_type}_{fname.replace(".npy", ".png")}')
                    fig.savefig(out_png, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f'  Saved {out_png}')
                    total += 1

            print(f'\\nTotal plots generated: {total}')
            print(f'Output directory: {viz_dir}')
        """).replace("ARGS_PLACEHOLDER", args_json.replace("'", "\\'"))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=str(BASE_DIR),
        ) as f:
            f.write(script_lines)
            script_path = f.name

        try:
            cmd = self.conda_python() + [script_path]
            self.append_log(f"CMD: {' '.join(cmd)}", "dim")
            self.stream_cmd(cmd)
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass


    # Generic stream helper
    def stream_cmd(self, cmd: list[str], extra_env: dict = None) -> None:
        try:
            env = os.environ.copy()
            if extra_env:
                env.update(extra_env)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=str(PROJECT_ROOT),
            )
            log = self.query_one("#log", RichLog)
            for line in iter(proc.stdout.readline, ""):
                self.call_from_thread(log.write, line.rstrip())
            proc.wait()
            rc = proc.returncode
            style = "green" if rc == 0 else "red"
            icon  = "\u2714" if rc == 0 else "\u2718"
            self.append_log(f"{icon} Process finished (exit {rc})", style)
        except Exception as exc:
            self.append_log(f"Failed to start process: {exc}", "red")

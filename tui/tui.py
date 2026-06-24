#!/usr/bin/env python3
"""
scGrapHiC TUI

To run: conda run -n scgraphic_env python tui.py
    
  Blind mode (scRNA-seq only):
    1·Parse RNA-seq -> 2·Pseudobulk -> 3·Inference -> 4·Fine-tune -> 5·Analysis

  Ground-truth mode (scRNA-seq + scHi-C):
    1·Parse RNA-seq -> 2·Parse scHi-C -> 3·Pseudobulk -> 4·Inference -> 5·Fine-tune -> 6·Analysis
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import tui_panels
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import (
    Container,
    Horizontal,
    ScrollableContainer,
    Vertical,
)
from textual.reactive import reactive
from textual.widgets import (
    Button,
    ContentSwitcher,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Select,
    Static,
    Switch,
)

from tui_config import STEP_SUBTITLES, STEPS
from tui_runners import TuiRunnersMixin
from tui_state import TuiStateMixin

# Singleton enforcement
LOCK_FILE = (
    Path(tempfile.gettempdir()) / f"scgraphic_tui_{os.getenv('USER', 'user')}.pid"
)


def kill_other_instances() -> None:
    my_pid = os.getpid()
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            if old_pid != my_pid:
                try:
                    os.kill(old_pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
        except ValueError:
            pass
    LOCK_FILE.write_text(str(my_pid))
    atexit.register(lambda: LOCK_FILE.unlink(missing_ok=True))


class ScGrapHiCApp(TuiRunnersMixin, TuiStateMixin, App):
    # Full-pipeline TUI for the scGrapHiC mouse model

    CSS_PATH = str(Path(__file__).resolve().with_name("tui_style.tcss"))

    TITLE = "scGrapHiC"
    SUB_TITLE = "Mouse Hi-C Prediction Pipeline"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("q", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_log", "Clear log", show=True),
        Binding("ctrl+s", "save_paths", "Save paths", show=True),
        Binding("0", "jump_step('setup')", "Setup"),
        Binding("1", "jump_step('welcome')","Welcome"),
        Binding("2", "jump_step('pseudobulk')", "Pseudobulk"),
        Binding("3", "jump_step('rna')", "RNA-seq"),
        Binding("4", "jump_step('hic')", "scHi-C"),
        Binding("5", "jump_step('build')", "Build"),
        Binding("6", "jump_step('inference')", "Inference"),
        Binding("7", "jump_step('finetune')", "Fine-tune"),
        Binding("8", "jump_step('analysis')", "Analysis"),
    ]

    current_step: reactive[str] = reactive("setup")

    ft_epoch: reactive[str] = reactive("\u2014")
    ft_scc: reactive[str] = reactive("\u2014")
    ft_gd: reactive[str] = reactive("\u2014")
    ft_ssim: reactive[str] = reactive("\u2014")
    ft_loss: reactive[str] = reactive("\u2014")

    def __init__(self) -> None:
        super().__init__()
        self.saved_inputs: dict[str, str] = {}
        self.run_mode: str = "blind"
        self.ft_proc: Optional[subprocess.Popen] = None
        self.suppress_mode_event: bool = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal():
            with Vertical(id="sidebar"):
                yield Static("scGrapHiC", id="sidebar-title")
                with ListView(id="step-list"):
                    for key, label in STEPS:
                        yield ListItem(Label(label), id=f"step-{key}")

            with Vertical(id="main"):
                with Container(id="step-header"):
                    yield Static("Environment Setup", id="step-title")
                    yield Static(STEP_SUBTITLES["setup"], id="step-subtitle")

                with ContentSwitcher(id="content-area", initial="pane-setup"):
                    with ScrollableContainer(
                        id="pane-setup", classes="step-pane",
                    ):
                        yield from tui_panels.build_setup(self)
                    with ScrollableContainer(
                        id="pane-welcome", classes="step-pane",
                    ):
                        yield from tui_panels.build_welcome(self)
                    with ScrollableContainer(
                        id="pane-rna", classes="step-pane",
                    ):
                        yield from tui_panels.build_rna(self)
                    with ScrollableContainer(
                        id="pane-hic", classes="step-pane",
                    ):
                        yield from tui_panels.build_hic(self)
                    with ScrollableContainer(
                        id="pane-pseudobulk", classes="step-pane",
                    ):
                        yield from tui_panels.build_pseudobulk(self)
                    with ScrollableContainer(
                        id="pane-build", classes="step-pane",
                    ):
                        yield from tui_panels.build_build(self)
                    with ScrollableContainer(
                        id="pane-inference", classes="step-pane",
                    ):
                        yield from tui_panels.build_inference(self)
                    with ScrollableContainer(
                        id="pane-finetune", classes="step-pane",
                    ):
                        yield from tui_panels.build_finetune(self)
                    with ScrollableContainer(
                        id="pane-analysis", classes="step-pane",
                    ):
                        yield from tui_panels.build_analysis(self)

                yield Static("", classes="section-divider")
                with Horizontal(id="path-actions"):
                    yield Button(
                        "✔ Validate", id="val-step",
                        variant="default", classes="validate-btn",
                    )
                    yield Button(
                        "▶ Run", id="run-step",
                        variant="primary", classes="run-btn",
                    )
                    yield Button(
                        "Save", id="save-paths",
                        variant="default", classes="validate-btn",
                    )
                    yield Button(
                        "Load", id="load-paths",
                        variant="default", classes="validate-btn",
                    )

                with Vertical(id="log-panel"):
                    with Horizontal(id="log-header-container"):
                        yield Static("  \u25b8 OUTPUT LOG", id="log-header")
                        yield Button("Copy", id="copy-log", classes="log-btn")
                        yield Button("\u2195 Resize", id="toggle-log-size", classes="log-btn")
                    yield RichLog(id="log", wrap=True, markup=True)

        yield Footer()

    def switch_step(self, step: str) -> None:
        if step == self.current_step:
            return
        if step in ("hic", "finetune") and self.run_mode == "blind":
            label = "Parse scHi-C" if step == "hic" else "Fine-tune"
            reason = (
                "scHi-C ground-truth contact matrices"
                if step == "hic"
                else "scHi-C ground-truth data for supervised weight adjustment"
            )
            self.append_log(
                f"[yellow]🔒 {label}[/yellow] is disabled in "
                f"[bold]blind prediction[/bold] mode.\n"
                f"    This step requires {reason}, which is not "
                f"available in the scRNA-seq–only workflow.\n"
                f"    To enable it, go to [bold]scGrapHiC (Welcome)[/bold] "
                f"and switch the run mode to "
                f"[bold]Ground truth available[/bold].",
                "yellow",
            )
            return
        self.apply_step(step)

    def apply_step(self, step: str) -> None:
        self.save_current_inputs()

        for key, _ in STEPS:
            try:
                item = self.query_one(f"#step-{key}", ListItem)
                if key == step:
                    item.add_class("--highlight")
                else:
                    item.remove_class("--highlight")
            except Exception:
                pass

        self.query_one(
            "#content-area", ContentSwitcher
        ).current = f"pane-{step}"
        self.restore_inputs()

        title_map = {
            "setup": "Environment Setup",
            "welcome": "Welcome to scGrapHiC",
            "pseudobulk": "Step 1 - Pseudobulk",
            "rna": "Step 2 - Parse scRNA-seq",
            "hic": "Step 3 - Parse scHi-C",
            "build": "Step 4 - Build Dataset",
            "inference": "Step 5 - Inference",
            "finetune": "Step 6 - Fine-tune",
            "analysis": "Step 7 - Analysis",
        }
        self.query_one("#step-title", Static).update(title_map[step])
        self.query_one("#step-subtitle", Static).update(STEP_SUBTITLES[step])
        self.current_step = step
        self.refresh_action_buttons(step)

    def refresh_action_buttons(self, step: str) -> None:
        val_btn = self.query_one("#val-step", Button)
        run_btn = self.query_one("#run-step", Button)
        if step == "analysis":
            val_btn.display = False
            run_btn.display = False
            return
        if step == "welcome":
            val_btn.display = False
            run_btn.display = True
            run_btn.label = "▶ Run all"
            return
        if step == "setup":
            val_btn.display = True
            run_btn.display = True
            val_btn.label = "✔ Check env"
            run_btn.label = "▶ Download"
            return
        val_btn.display = True
        run_btn.display = True
        val_btn.label = "✔ Validate"
        run_btn.label = "▶ Run"

    @on(ListView.Selected)
    def on_list_item_selected(self, event: ListView.Selected) -> None:
        item_id: str = event.item.id or ""
        if item_id.startswith("step-"):
            self.switch_step(item_id[5:])

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "val-step":
            fn = {
                "setup": self.run_env_check,
                "pseudobulk": self.validate_aggregate,
                "rna": self.validate_rna,
                "hic": self.validate_hic,
                "build": self.validate_build,
                "inference": self.validate_inference,
                "finetune": self.validate_finetune_paths,
            }.get(self.current_step)
            if fn:
                fn()
            return
        if btn_id == "run-step":
            fn = {
                "setup": self.run_download,
                "welcome": self.run_all_steps,
                "pseudobulk": self.run_aggregate,
                "rna": self.run_parse_rna,
                "hic": self.run_parse_hic,
                "build": self.run_build,
                "inference": self.run_inference,
                "finetune": self.run_finetune,
            }.get(self.current_step)
            if fn:
                fn()
            return
        handlers = {
            "stop-ft": self.stop_finetune,
            "run-umap": self.run_umap,
            "run-viz": self.run_contactmap_viz,
            "run-metrics": self.run_metrics_summary,
            "preview-rna": self.preview_rna,
            "preview-hic": self.preview_hic,
            "save-paths": self.save_paths_to_file,
            "load-paths": self.load_paths_ui,
            "copy-log": self.copy_log,
            "toggle-log-size": self.toggle_log_size,
        }
        handler = handlers.get(btn_id)
        if handler:
            handler()

    def copy_log(self) -> None:
        log = self.query_one("#log", RichLog)
        try:
            text = "\n".join(strip.text for strip in log.lines)
        except AttributeError:
            # Fallback if log.lines does not yield Strip objects
            text = "\n".join(str(line) for line in log.lines)
        self.app.copy_to_clipboard(text)
        self.notify("Log copied to clipboard!")

    def toggle_log_size(self) -> None:
        log_panel = self.query_one("#log-panel", Vertical)
        if log_panel.has_class("expanded"):
            log_panel.remove_class("expanded")
        else:
            log_panel.add_class("expanded")

    @on(Select.Changed, "#run-mode")
    async def on_mode_changed(self, event: Select.Changed) -> None:
        if self.suppress_mode_event:
            return
        val = event.value
        if val is None or str(val) == "Select.BLANK":
            return
        self.run_mode = str(val)
        self.update_hic_visibility()
        await self.async_remount_mode_panes()
        mode_label = (
            "blind prediction (scRNA-seq only)"
            if self.run_mode == "blind"
            else "ground-truth (scRNA-seq + scHi-C)"
        )
        self.append_log(f"Run mode set to [bold]{mode_label}[/bold]", "cyan")

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()

    def action_jump_step(self, step: str) -> None:
        self.switch_step(step)

    def action_save_paths(self) -> None:
        self.save_paths_to_file()

    async def on_mount(self) -> None:
        self.load_paths_from_file()
        self.restore_inputs()
        try:
            self.query_one("#step-setup", ListItem).add_class("--highlight")
        except Exception:
            pass
        self.update_hic_visibility()
        try:
            self.suppress_mode_event = True
            mode_select = self.query_one("#run-mode", Select)
            mode_select.value = self.run_mode
        except Exception:
            pass
        finally:
            self.suppress_mode_event = False
        await self.async_remount_mode_panes()
        self.refresh_action_buttons("setup")


# Entry point
def main() -> None:
    import sys
    import traceback

    if os.environ.get("SCGRAPHIC_TUI_DEBUG"):
        print(
            f"scGrapHiC TUI: python={sys.executable!r} "
            f"tty={getattr(sys.stdin, 'isatty', lambda: False)()}",
            file=sys.stderr,
        )

    if not (getattr(sys.stdin, "isatty", lambda: False)() and getattr(
        sys.stdout, "isatty", lambda: False
    )()):
        print(
            "scGrapHiC TUI: stdin/stdout is not a TTY. The interface may not "
            "start; use an interactive terminal (e.g. `ssh -t host` or a "
            "local shell), or set `TERM` and run from a real console.",
            file=sys.stderr,
        )

    kill_other_instances()

    app = ScGrapHiCApp()

    def handle_exit(sig, frame):
        app.exit()

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        app.run()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import sys

    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except SystemExit as exc:
        raise exc
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.exit(1)

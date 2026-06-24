from __future__ import annotations

import json
from typing import Any

from textual.widgets import Input, Select, Switch

from tui_config import SAVED_PATHS_FILE


class TuiStateMixin:
    # saved_inputs handling and .scgraphic_saved_paths.json I/O

    def save_current_inputs(self: Any) -> None:
        try:
            for w in self.query("Input"):
                if w.id:
                    self.saved_inputs[w.id] = w.value
        except Exception:
            pass
        try:
            for w in self.query("Select"):
                if w.id and w.id != "run-mode":
                    val = w.value
                    if val is not None and str(val) != "Select.BLANK":
                        self.saved_inputs[w.id] = str(val)
        except Exception:
            pass
        try:
            for w in self.query("Switch"):
                if w.id:
                    self.saved_inputs[w.id] = str(w.value)
        except Exception:
            pass

    def restore_inputs(self: Any) -> None:
        for wid, value in self.saved_inputs.items():
            try:
                w = self.query_one(f"#{wid}")
                if isinstance(w, Input):
                    w.value = value
                elif isinstance(w, Select) and wid != "run-mode":
                    w.value = value
                elif isinstance(w, Switch):
                    w.value = value == "True"
            except Exception:
                pass

    def save_paths_to_file(self: Any) -> None:
        self.save_current_inputs()
        data = {
            "run_mode": self.run_mode,
            "inputs": self.saved_inputs,
        }
        try:
            SAVED_PATHS_FILE.write_text(
                json.dumps(data, indent=2), encoding="utf-8",
            )
            self.append_log(
                f"Paths saved to [bold]{SAVED_PATHS_FILE.name}[/bold]",
                "green",
            )
        except Exception as exc:
            self.append_log(f"Failed to save paths: {exc}", "red")

    def load_paths_from_file(self: Any) -> None:
        if not SAVED_PATHS_FILE.exists():
            self.append_log(
                "No saved paths found \u2014 first run or "
                f"[bold]{SAVED_PATHS_FILE.name}[/bold] is missing. "
                "Fill in your paths and press Ctrl+S to save them.",
                "dim",
            )
            return
        try:
            data = json.loads(
                SAVED_PATHS_FILE.read_text(encoding="utf-8")
            )
            loaded = data.get("inputs", {})
            self.saved_inputs.update(loaded)
            mode = data.get("run_mode")
            if mode in ("blind", "groundtruth"):
                self.run_mode = mode
            count = len(loaded)
            self.append_log(
                f"Loaded {count} saved path(s) from "
                f"[bold]{SAVED_PATHS_FILE.name}[/bold]",
                "green",
            )
        except Exception as exc:
            self.append_log(
                f"Could not load saved paths: {exc}", "yellow",
            )

    def load_paths_ui(self: Any) -> None:
        # Button-triggered load
        self.load_paths_from_file()
        self.restore_inputs()
        try:
            mode_select = self.query_one("#run-mode", Select)
            mode_select.value = self.run_mode
        except Exception:
            pass
        self.update_hic_visibility()

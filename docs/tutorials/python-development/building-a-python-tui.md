# Building a Python TUI (Task Runner)

Step-by-step guide to building a **Task Runner TUI** in Python with Textual: task list, detail pane, footer, search/filter, create/edit/delete, JSON persistence, graceful quit, and file-based logging. For conceptual guidance (architecture, framework trade-offs, testing), see [TUI Applications (Python)](../../best-practices/python/tui-applications.md).

## Overview

You will build a terminal UI that manages a list of tasks with local JSON storage. The app uses **Textual** for the TUI, keeps state and update logic separate from the view, and logs to a file so the terminal output stays clean.

## What you will build

- **Task list pane** (left): scrollable list of tasks with selection
- **Detail pane** (right): title, description, status for the selected task
- **Footer**: key bindings (j/k move, Enter edit, n new, d delete, / search, q quit)
- **Search/filter**: type to filter tasks by title/description
- **Create / Edit / Delete** with persistence to a JSON file
- **Graceful quit** with optional unsaved-changes handling
- **Log file** (e.g. `task_runner_tui.log`) separate from the TUI
- **Basic tests** for state and persistence logic

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (or pip) for dependency management
- A terminal that supports alternate screen and key events (e.g. iTerm2, Windows Terminal, standard Linux terminal)

## Project bootstrap

```bash
mkdir task-runner-tui && cd task-runner-tui
uv init --no-readme
uv python pin 3.12
```

## Dependency installation

```bash
uv add textual
uv add -d pytest pytest-asyncio
```

If you prefer pip:

```bash
pip install textual
pip install -e ".[dev]"
```

Minimal `pyproject.toml`:

```toml
[project]
name = "task-runner-tui"
version = "0.1.0"
description = "A Textual TUI for managing tasks with JSON persistence"
requires-python = ">=3.11"
dependencies = ["textual>=0.47"]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio"]

[project.scripts]
task-runner-tui = "task_runner_tui.main:run"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Project structure

```
task-runner-tui/
├── task_runner_tui/
│   ├── __init__.py
│   ├── main.py        # Entry point, App, key bindings
│   ├── state.py       # Task, AppState, filter logic
│   ├── storage.py     # load/save JSON
│   └── widgets.py     # Task list, detail pane, footer (optional)
├── tests/
│   ├── __init__.py
│   ├── test_state.py
│   └── test_storage.py
├── pyproject.toml
└── tasks.json         # Created at runtime
```

## Core state model

```python
# task_runner_tui/state.py
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Task:
    id: str
    title: str
    description: str
    done: bool = False

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "description": self.description, "done": self.done}

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            id=d["id"],
            title=d.get("title", ""),
            description=d.get("description", ""),
            done=d.get("done", False),
        )


@dataclass
class AppState:
    tasks: list[Task] = field(default_factory=list)
    selected_index: int = 0
    filter_query: str = ""
    _next_id: int = 0

    def filtered_tasks(self) -> list[Task]:
        if not self.filter_query:
            return self.tasks
        q = self.filter_query.lower()
        return [t for t in self.tasks if q in t.title.lower() or q in t.description.lower()]

    def selected_task(self) -> Optional[Task]:
        filtered = self.filtered_tasks()
        if not filtered or self.selected_index < 0 or self.selected_index >= len(filtered):
            return None
        return filtered[self.selected_index]

    def new_id(self) -> str:
        self._next_id += 1
        return f"t{self._next_id}"
```

## Event loop and message/update model

Textual uses reactive state. We keep `AppState` on the app and update it in message handlers. Events map to semantic actions (select next/prev, set filter, add/update/delete task, load/save).

```python
# task_runner_tui/main.py (excerpt)
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Input, Static, ListView, ListItem, Label
from textual.message import Message
import json
import logging
from pathlib import Path

from .state import AppState, Task
from .storage import load_tasks, save_tasks

LOG_PATH = Path("task_runner_tui.log")
logging.basicConfig(level=logging.INFO, filename=str(LOG_PATH), filemode="a", format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
```

We’ll wire key bindings to actions that update `AppState` and refresh the list/detail widgets. No separate “update” function is required if we keep state on the app and mutate it in handlers; for tests we can still test `state.py` and `storage.py` in isolation.

## Rendering layout

- **Left:** `ListView` of filtered tasks (titles only).
- **Right:** `Static` showing selected task title, description, done.
- **Bottom:** `Static` footer with key hints.
- **Search:** `Input` at top or in a layer; filter updates `AppState.filter_query` and list content.

Layout sketch with Textual:

```python
# task_runner_tui/main.py
class TaskList(ListView):
    def __init__(self, state: AppState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        self.clear()
        for task in self._state.filtered_tasks():
            self.append(ListItem(Label(task.title)))

    def update_state(self, state: AppState) -> None:
        self._state = state
        self._refresh()


class DetailPane(Static):
    def __init__(self, state: AppState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def _refresh(self) -> None:
        task = self._state.selected_task()
        if task:
            self.update(f"[bold]{task.title}[/bold]\n\n{task.description}\n\nDone: {task.done}")
        else:
            self.update("No task selected")

    def update_state(self, state: AppState) -> None:
        self._state = state
        self._refresh()


class TaskRunnerApp(App):
    CSS = """
    Horizontal { height: 1fr; }
    #task-list { width: 30%; }
    #detail { width: 70%; padding: 1 2; }
    #footer { height: 1; dock: bottom; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("j", "cursor_down", "Down", show=True),
        Binding("k", "cursor_up", "Up", show=True),
        Binding("n", "new_task", "New", show=True),
        Binding("d", "delete_task", "Delete", show=True),
        Binding("/", "focus_search", "Search", show=True),
    ]

    def __init__(self, data_path: Path):
        super().__init__()
        self.data_path = data_path
        self.state = AppState()

    def on_mount(self) -> None:
        try:
            self.state.tasks = load_tasks(self.data_path)
            self.state._next_id = max((int(t.id.lstrip("t")) for t in self.state.tasks if t.id.startswith("t")), default=0) + 1
        except Exception as e:
            logger.exception("Load failed")
            self.notify(f"Load failed: {e}", severity="error")
        self._refresh_ui()

    def _refresh_ui(self) -> None:
        tl = self.query_one("#task-list", TaskList)
        tl.update_state(self.state)
        dp = self.query_one("#detail", DetailPane)
        dp.update_state(self.state)
        if self.state.filtered_tasks():
            idx = min(self.state.selected_index, len(self.state.filtered_tasks()) - 1)
            self.state.selected_index = max(0, idx)
            tl.index = self.state.selected_index

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="left-pane"):
                yield Input(placeholder="Filter...", id="search-input")
                yield TaskList(self.state, id="task-list")
            yield DetailPane(self.state, id="detail")
        yield Static("j/k: move | Enter: edit | n: new | d: delete | /: search | q: quit", id="footer")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-input":
            self.state.filter_query = event.value
            self.state.selected_index = 0
            self._refresh_ui()

    def action_cursor_down(self) -> None:
        tasks = self.state.filtered_tasks()
        if tasks and self.state.selected_index < len(tasks) - 1:
            self.state.selected_index += 1
            self._refresh_ui()

    def action_cursor_up(self) -> None:
        if self.state.selected_index > 0:
            self.state.selected_index -= 1
            self._refresh_ui()

    def action_new_task(self) -> None:
        task = Task(id=self.state.new_id(), title="New task", description="", done=False)
        self.state.tasks.append(task)
        self.state.selected_index = len(self.state.filtered_tasks()) - 1
        self._refresh_ui()
        self._save()

    def action_delete_task(self) -> None:
        task = self.state.selected_task()
        if not task:
            return
        self.state.tasks = [t for t in self.state.tasks if t.id != task.id]
        self.state.selected_index = min(self.state.selected_index, len(self.state.filtered_tasks()) - 1) if self.state.filtered_tasks() else 0
        self._refresh_ui()
        self._save()

    def action_focus_search(self) -> None:
        self.query_one("#search-input", Input).focus()

    def _save(self) -> None:
        try:
            save_tasks(self.data_path, self.state.tasks)
            logger.info("Saved %s tasks", len(self.state.tasks))
        except Exception as e:
            logger.exception("Save failed")
            self.notify(f"Save failed: {e}", severity="error")

    def action_quit(self) -> None:
        self._save()
        self.exit()
```

Add an `Input` for search (e.g. in a layer or at top) and wire its `Submitted`/`Changed` to set `self.state.filter_query` and call `_refresh_ui()`. Omitted here for brevity; you can add `<Input id="search-input" placeholder="Filter...">` and in `on_input_submitted` set `self.state.filter_query = value` then `_refresh_ui()`.

## Keyboard bindings

- **j** / **k**: move selection down/up in the task list
- **Enter**: (optional) edit selected task (modal or inline)
- **n**: new task
- **d**: delete selected task
- **/**: focus search input
- **q**: save and quit

Bindings are declared in `BINDINGS` and implemented as `action_*` methods.

## Persistence layer

```python
# task_runner_tui/storage.py
import json
from pathlib import Path
from .state import Task


def load_tasks(path: Path) -> list[Task]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Task.from_dict(item) for item in data.get("tasks", [])]


def save_tasks(path: Path, tasks: list[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"tasks": [t.to_dict() for t in tasks]}, f, indent=2)
```

## Logging

Logging is configured in `main.py` to a file (`task_runner_tui.log`). Do not `print()` or log to stdout during the TUI run so the display is not corrupted. Use `logger.info(...)` for save/load and errors.

## Running the application

```bash
# From project root
uv run python -m task_runner_tui.main
# Or after install
uv run task-runner-tui
```

Ensure `main.py` has a `run()` that starts the app with a data path:

```python
# task_runner_tui/main.py (bottom)
def run() -> None:
    data_path = Path.home() / ".task_runner_tui" / "tasks.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    app = TaskRunnerApp(data_path)
    app.run()


if __name__ == "__main__":
    run()
```

## Verification checklist

- [ ] Task list shows on the left; detail on the right; footer at bottom.
- [ ] j/k move selection; detail updates.
- [ ] n creates a new task and persists after save.
- [ ] d deletes the selected task and persists.
- [ ] Search/filter narrows the list when implemented.
- [ ] q saves and exits; terminal is restored.
- [ ] `task_runner_tui.log` exists and contains log lines; TUI output is clean.
- [ ] `pytest tests/` passes for `test_state` and `test_storage`.

## Troubleshooting

- **List or detail not updating:** Ensure `_refresh_ui()` is called after every state change and that widgets are queried by correct id.
- **Keys not working:** Check BINDINGS and that no child widget is capturing keys; use `can_focus` or focus management.
- **Load/save errors:** Check file permissions and path; log exceptions and show a notification.
- **Layout broken on resize:** Textual handles resize; ensure you use relative widths (e.g. 30% / 70%) and not fixed character sizes.

## Extensions / next steps

- Add a modal or screen for editing task title/description (e.g. `Input` in a modal).
- Add SQLite instead of JSON for larger datasets.
- Add done/toggle and filter by status.
- Add basic tests for list/detail rendering by testing state and storage only; add integration test with Textual’s headless run if needed.

## TL;DR summary

1. Bootstrap with `uv init`, add `textual`.
2. Define `Task` and `AppState` in `state.py`; `load_tasks`/`save_tasks` in `storage.py`.
3. Build the app in `main.py`: `TaskRunnerApp` with `TaskList`, `DetailPane`, footer; bind j/k, n, d, /, q.
4. Log to a file; never stdout. Quit saves and exits.
5. Run with `uv run python -m task_runner_tui.main` or `task-runner-tui`; verify with pytest for state and storage.

---

*For patterns, framework trade-offs, and production guidance, see [TUI Applications (Python)](../../best-practices/python/tui-applications.md).*

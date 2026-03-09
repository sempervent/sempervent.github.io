# Building a Rust TUI (Task Runner)

Step-by-step guide to building a **Task Runner TUI** in Rust with ratatui and crossterm: task list, detail pane, footer, search/filter, create/edit/delete, JSON persistence, graceful quit, and file-based logging. For conceptual guidance, see [TUI Applications (Rust)](../../best-practices/rust/tui-applications.md).

## Overview

You will build a terminal UI that manages a list of tasks with local JSON storage. The app uses **ratatui** for layout and widgets and **crossterm** for terminal and events, with state/update/draw separation.

## What you will build

- **Task list pane** (left): scrollable list with selection
- **Detail pane** (right): title, description, status for the selected task
- **Footer**: key bindings
- **Search/filter**: filter tasks by title/description
- **Create / Edit / Delete** with JSON persistence
- **Graceful quit** (q) with terminal restore
- **Log file** outside the TUI
- **Basic tests** for state and storage

## Prerequisites

- Rust 1.70+ (e.g. `rustup`)
- A terminal that supports alternate screen and key events

## Project bootstrap

```bash
cargo init task-runner-tui
cd task-runner-tui
```

## Dependency installation

Add to `Cargo.toml`:

```toml
[package]
name = "task-runner-tui"
version = "0.1.0"
edition = "2021"

[dependencies]
ratatui = "0.28"
crossterm = "0.28"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

## Project structure

```
task-runner-tui/
├── src/
│   ├── main.rs      # Entry, event loop, run
│   ├── app.rs       # App state, update, draw
│   ├── storage.rs   # load/save JSON
│   └── storage.rs (tests at bottom)
├── Cargo.toml
└── tasks.json       # Created at runtime
```

## Core state model

```rust
// src/app.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub title: String,
    pub description: String,
    pub done: bool,
}

impl Task {
    pub fn new(id: String, title: String, description: String) -> Self {
        Self { id, title, description, done: false }
    }
}

#[derive(Debug, Default)]
pub struct App {
    pub tasks: Vec<Task>,
    pub selected_index: usize,
    pub filter_query: String,
    pub next_id: u64,
    pub data_path: std::path::PathBuf,
}

impl App {
    pub fn filtered_tasks(&self) -> Vec<&Task> {
        if self.filter_query.is_empty() {
            return self.tasks.iter().collect();
        }
        let q = self.filter_query.to_lowercase();
        self.tasks
            .iter()
            .filter(|t| {
                t.title.to_lowercase().contains(&q) || t.description.to_lowercase().contains(&q)
            })
            .collect()
    }

    pub fn selected_task(&self) -> Option<&Task> {
        let filtered = self.filtered_tasks();
        filtered.get(self.selected_index).copied()
    }

    pub fn new_id(&mut self) -> String {
        self.next_id += 1;
        format!("t{}", self.next_id)
    }
}
```

## Event loop and message/update model

Map crossterm events to internal actions; one update function produces new state.

```rust
// src/main.rs (excerpt)
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::prelude::*;
use std::io;

mod app;
mod storage;
use app::App;
use storage::{load_tasks, save_tasks};
```

Event loop: poll event → update app → draw. Run I/O (load/save) on first tick or in a separate thread and send result back; for simplicity we do synchronous load at start and sync save on mutations. Ensure `Task` is in scope in `main.rs` (`use app::Task`) for `Task::new`.

```rust
// src/main.rs
fn run_app<B: Backend>(terminal: &mut Terminal<B>, app: &mut App) -> io::Result<()> {
    loop {
        terminal.draw(|f| app::draw(f, app))?;
        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match (key.code, key.modifiers) {
                    (KeyCode::Char('q'), _) | (KeyCode::Esc, _) => break,
                    (KeyCode::Char('j'), _) | (KeyCode::Down, _) => {
                        let len = app.filtered_tasks().len();
                        if len > 0 && app.selected_index < len - 1 {
                            app.selected_index += 1;
                        }
                    }
                    (KeyCode::Char('k'), _) | (KeyCode::Up, _) => {
                        if app.selected_index > 0 {
                            app.selected_index -= 1;
                        }
                    }
                    (KeyCode::Char('n'), _) => {
                        let task = Task::new(
                            app.new_id(),
                            "New task".into(),
                            String::new(),
                        );
                        app.tasks.push(task);
                        app.selected_index = app.filtered_tasks().len().saturating_sub(1);
                        let _ = save_tasks(&app.data_path, &app.tasks);
                    }
                    (KeyCode::Char('d'), _) => {
                        if let Some(t) = app.selected_task() {
                            let id = t.id.clone();
                            app.tasks.retain(|x| x.id != id);
                            app.selected_index = app.selected_index.saturating_sub(1).min(app.filtered_tasks().len().saturating_sub(1));
                            let _ = save_tasks(&app.data_path, &app.tasks);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}
```

## Rendering layout

Use ratatui layout to split the frame; draw list and detail in their chunks.

```rust
// src/app.rs
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};

pub fn draw(f: &mut ratatui::Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(f.area());
    let main = chunks[0];
    let footer_chunk = chunks[1];

    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(main);

    let list_items: Vec<ListItem> = app
        .filtered_tasks()
        .into_iter()
        .enumerate()
        .map(|(i, t)| {
            let prefix = if i == app.selected_index { "▸ " } else { "  " };
            ListItem::new(format!("{}{}", prefix, t.title))
        })
        .collect();
    let list = List::new(list_items).block(Block::default().borders(Borders::ALL).title("Tasks"));
    f.render_widget(list, main_chunks[0]);

    let detail = if let Some(t) = app.selected_task() {
        format!("{}\n\n{}\n\nDone: {}", t.title, t.description, t.done)
    } else {
        "No task selected".to_string()
    };
    let para = Paragraph::new(detail)
        .block(Block::default().borders(Borders::ALL).title("Detail"))
        .wrap(Wrap { trim: true });
    f.render_widget(para, main_chunks[1]);

    let footer = Paragraph::new("j/k: move | n: new | d: delete | q: quit")
        .block(Block::default());
    f.render_widget(footer, footer_chunk);
}
```

## Keyboard bindings

- **j** / **Down**: move selection down
- **k** / **Up**: move selection up
- **n**: new task
- **d**: delete selected task
- **q** / **Esc**: quit
- **/** (optional): toggle filter input

## Persistence layer

```rust
// src/storage.rs
use std::fs;
use std::path::Path;

use crate::app::Task;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Store {
    tasks: Vec<Task>,
}

pub fn load_tasks(path: &Path) -> std::io::Result<(Vec<Task>, u64)> {
    let s = fs::read_to_string(path)?;
    let store: Store = serde_json::from_str(&s).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let next_id = store
        .tasks
        .iter()
        .filter_map(|t| t.id.strip_prefix("t").and_then(|n| n.parse::<u64>().ok()))
        .max()
        .unwrap_or(0)
        + 1;
    Ok((store.tasks, next_id))
}

pub fn save_tasks(path: &Path, tasks: &[Task]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let store = Store { tasks: tasks.to_vec() };
    fs::write(path, serde_json::to_string_pretty(&store)?)?;
    Ok(())
}
```

Use `load_tasks` at startup; set `app.tasks` and `app.next_id`. Call `save_tasks` after add/delete.

## Logging

Log to a file so the TUI is not overwritten. Example with `log` and `env_logger` (add to `Cargo.toml`: `log`, `env_logger`), or write to a file in `main`:

```rust
// In main, before starting the TUI
let log_path = dirs::home_dir().unwrap().join(".task_runner_tui").join("tui.log");
std::fs::create_dir_all(log_path.parent().unwrap()).ok();
let _ = simple_logging::log_to_file(log_path, log::LevelFilter::Info);
```

For minimal deps, open a file and use `std::io::Write` for key actions (load/save/errors).

## Basic tests

```rust
// At bottom of src/storage.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::Task;
    use std::io::Write;

    #[test]
    fn roundtrip_tasks() {
        let dir = std::env::temp_dir().join("task_runner_tui_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tasks.json");
        let tasks = vec![
            Task::new("t1".into(), "A".into(), "a".into()),
            Task::new("t2".into(), "B".into(), "b".into()),
        ];
        save_tasks(&path, &tasks).unwrap();
        let (loaded, next_id) = load_tasks(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].title, "A");
        assert_eq!(next_id, 3);
        std::fs::remove_file(&path).ok();
    }
}
```

## Running the application

```bash
cargo run
# Release
cargo build --release
./target/release/task-runner-tui
```

Full `main.rs` sketch with terminal setup and cleanup:

```rust
// src/main.rs
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::prelude::*;
use std::io;

mod app;
mod storage;
use app::{App, Task};
use storage::{load_tasks, save_tasks};

fn main() -> io::Result<()> {
    let data_path = dirs::home_dir().unwrap().join(".task_runner_tui").join("tasks.json");
    std::fs::create_dir_all(data_path.parent().unwrap()).ok();
    let (tasks, next_id) = load_tasks(&data_path).unwrap_or((Vec::new(), 1));
    let mut app = App {
        tasks,
        next_id,
        data_path: data_path.clone(),
        ..Default::default()
    };

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_app(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}
```

Add `dirs = "0.5"` to `Cargo.toml` for `dirs::home_dir()`, or use `std::env::var("HOME")` and `PathBuf::from`.

## Verification checklist

- [ ] Task list on left, detail on right, footer at bottom.
- [ ] j/k move selection; detail updates.
- [ ] n creates a task and persists.
- [ ] d deletes and persists.
- [ ] q restores terminal and exits.
- [ ] Log file written; TUI output clean.
- [ ] `cargo test` passes for storage tests.

## Troubleshooting

- **Terminal not restored on panic:** Use a guard that runs on drop to leave alternate screen and show cursor (e.g. a struct that holds terminal and implements `Drop`).
- **Keys not working:** Ensure raw mode and alternate screen are enabled; check key code and modifiers.
- **Load/save errors:** Check path and permissions; log and optionally show in UI.

## Extensions / next steps

- Add filter input (state for "filter mode" and a string buffer).
- Edit task (modal or sub-screen).
- SQLite via `rusqlite` for larger datasets.
- Unit tests for `App` update logic and for `load_tasks`/`save_tasks`.

## TL;DR summary

1. `cargo init`; add `ratatui`, `crossterm`, `serde`, `serde_json`.
2. Define `Task` and `App` in `app.rs`; `load_tasks`/`save_tasks` in `storage.rs`.
3. Event loop in `main.rs`: poll crossterm, update `App`, draw with ratatui; restore terminal on exit.
4. Log to file; I/O off the draw path.
5. Run with `cargo run`; test with `cargo test`.

---

*For patterns and production guidance, see [TUI Applications (Rust)](../../best-practices/rust/tui-applications.md).*

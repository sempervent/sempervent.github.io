# Building a Go TUI (Task Runner)

Step-by-step guide to building a **Task Runner TUI** in Go with Bubble Tea: task list, detail pane, footer, search/filter, create/edit/delete, JSON persistence, graceful quit, and file-based logging. For conceptual guidance, see [TUI Applications (Go)](../../best-practices/go/tui-applications.md).

## Overview

You will build a terminal UI that manages a list of tasks with local JSON storage. The app uses **Bubble Tea** (and optionally **Lip Gloss**) for the TUI, keeps model/update/view separate, and logs to a file.

## What you will build

- **Task list pane** (left): scrollable list with selection
- **Detail pane** (right): title, description, status for the selected task
- **Footer**: key bindings
- **Search/filter**: filter tasks by title/description
- **Create / Edit / Delete** with JSON persistence
- **Graceful quit** (q)
- **Log file** outside the TUI
- **Basic tests** for model and storage

## Prerequisites

- Go 1.21+
- A terminal that supports alternate screen and key events

## Project bootstrap

```bash
mkdir task-runner-tui && cd task-runner-tui
go mod init github.com/you/task-runner-tui
```

## Dependency installation

```bash
go get github.com/charmbracelet/bubbletea
go get github.com/charmbracelet/lipgloss
```

## Project structure

```
task-runner-tui/
├── main.go           # Entry, run Tea program
├── model.go          # Model, Msg, Update, View
├── storage.go        # load/save JSON
├── storage_test.go   # Tests for storage
├── go.mod
├── go.sum
└── tasks.json        # Created at runtime
```

## Core state model

```go
// model.go
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type Task struct {
	ID          string `json:"id"`
	Title       string `json:"title"`
	Description string `json:"description"`
	Done        bool   `json:"done"`
}

type Model struct {
	Tasks         []Task
	SelectedIndex int
	FilterQuery   string
	NextID        int
	Width         int
	Height        int
	DataPath      string
}

func (m Model) dataPath() string {
	if m.DataPath != "" {
		return m.DataPath
	}
	dir, _ := os.UserHomeDir()
	return filepath.Join(dir, ".task_runner_tui", "tasks.json")
}

func (m Model) FilteredTasks() []Task {
	if m.FilterQuery == "" {
		return m.Tasks
	}
	q := strings.ToLower(m.FilterQuery)
	var out []Task
	for _, t := range m.Tasks {
		if strings.Contains(strings.ToLower(t.Title), q) ||
			strings.Contains(strings.ToLower(t.Description), q) {
			out = append(out, t)
		}
	}
	return out
}

func (m Model) SelectedTask() *Task {
	filtered := m.FilteredTasks()
	if len(filtered) == 0 || m.SelectedIndex < 0 || m.SelectedIndex >= len(filtered) {
		return nil
	}
	return &filtered[m.SelectedIndex]
}
```

## Event loop and message/update model

Bubble Tea uses `Msg` and `Update(model, msg) (Model, Cmd)`.

```go
// model.go (continued)
type msgSelectDown struct{}
type msgSelectUp struct{}
type msgFilter string
type msgNewTask struct{}
type msgDeleteTask struct{}
type msgQuit struct{}
type msgSaveDone struct{ err error }
type msgLoadDone struct{ tasks []Task; nextID int }

func (m Model) Init() tea.Cmd {
	return loadTasksCmd(m.dataPath())
}
```

Add `"os"` and `"path/filepath"` to imports in `model.go` for `dataPath()`. Set `DataPath` in `main` before creating the program.

```go
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c":
			return m, tea.Quit
		case "j", "down":
			filtered := m.FilteredTasks()
			if len(filtered) > 0 && m.SelectedIndex < len(filtered)-1 {
				m.SelectedIndex++
			}
			return m, nil
		case "k", "up":
			if m.SelectedIndex > 0 {
				m.SelectedIndex--
			}
			return m, nil
		case "n":
			return m, m.newTaskCmd()
		case "d":
			return m, m.deleteTaskCmd()
		case "/":
			// In a full impl, switch to filter input mode; for brevity we skip
			return m, nil
		}
	case tea.WindowSizeMsg:
		m.Width = msg.Width
		m.Height = msg.Height
		return m, nil
	case msgFilter:
		m.FilterQuery = string(msg)
		m.SelectedIndex = 0
		return m, nil
	case msgNewTask:
		m.NextID++
		m.Tasks = append(m.Tasks, Task{
			ID:          fmt.Sprintf("t%d", m.NextID),
			Title:       "New task",
			Description: "",
			Done:        false,
		})
		m.SelectedIndex = len(m.FilteredTasks()) - 1
		return m, saveTasksCmd(m.dataPath(), m.Tasks)
	case msgDeleteTask:
		t := m.SelectedTask()
		if t == nil {
			return m, nil
		}
		var newTasks []Task
		for _, task := range m.Tasks {
			if task.ID != t.ID {
				newTasks = append(newTasks, task)
			}
		}
		m.Tasks = newTasks
		if m.SelectedIndex >= len(m.FilteredTasks()) && m.SelectedIndex > 0 {
			m.SelectedIndex--
		}
		return m, saveTasksCmd(m.dataPath(), m.Tasks)
	case msgLoadDone:
		m.Tasks = msg.tasks
		m.NextID = msg.nextID
		return m, nil
	case msgSaveDone:
		if msg.err != nil {
			// Could set an error state; for simplicity we ignore
		}
		return m, nil
	}
	return m, nil
}

func (m Model) newTaskCmd() tea.Cmd { return func() tea.Msg { return msgNewTask{} } }
func (m Model) deleteTaskCmd() tea.Cmd { return func() tea.Msg { return msgDeleteTask{} } }
```

Wire key "j"/"k" to `msgSelectDown`/`msgSelectUp` in a key handler that returns the right message. Example key handling in Update:

```go
	case tea.KeyMsg:
		switch msg.String() {
		case "j", "down":
			filtered := m.FilteredTasks()
			if len(filtered) > 0 && m.SelectedIndex < len(filtered)-1 {
				m.SelectedIndex++
			}
			return m, nil
		case "k", "up":
			if m.SelectedIndex > 0 {
				m.SelectedIndex--
			}
			return m, nil
		// ... rest
		}
```

## Rendering layout

Use Lip Gloss for panes and the view.

```go
// model.go (View)
func (m Model) View() string {
	listStyle := lipgloss.NewStyle().Width(30).Height(m.Height - 2).Border(lipgloss.RoundedBorder())
	detailStyle := lipgloss.NewStyle().Width(m.Width - 35).Padding(0, 1).Border(lipgloss.RoundedBorder())
	footerStyle := lipgloss.NewStyle().Width(m.Width).Height(1)

	filtered := m.FilteredTasks()
	var list strings.Builder
	for i, t := range filtered {
		prefix := "  "
		if i == m.SelectedIndex {
			prefix = "▸ "
		}
		list.WriteString(prefix + t.Title + "\n")
	}
	listStr := listStyle.Render(list.String())

	detailStr := "No task selected"
	if t := m.SelectedTask(); t != nil {
		detailStr = fmt.Sprintf("%s\n\n%s\n\nDone: %v", t.Title, t.Description, t.Done)
	}
	detailStr = detailStyle.Render(detailStr)

	footerStr := footerStyle.Render("j/k: move | n: new | d: delete | q: quit")

	return lipgloss.JoinVertical(lipgloss.Left,
		lipgloss.JoinHorizontal(lipgloss.Top, listStr, detailStr),
		footerStr,
	)
}
```

## Keyboard bindings

- **j** / **k** (or arrows): move selection
- **n**: new task
- **d**: delete selected task
- **q** / **Ctrl+C**: quit
- **/** (optional): focus filter input

## Persistence layer

```go
// storage.go
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type store struct {
	Tasks []Task `json:"tasks"`
}

func dataPath() string {
	dir, _ := os.UserHomeDir()
	return filepath.Join(dir, ".task_runner_tui", "tasks.json")
}

func loadTasks(path string) ([]Task, int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, 1, nil
		}
		return nil, 0, err
	}
	var s store
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, 0, err
	}
	nextID := 1
	for _, t := range s.Tasks {
		var n int
		if _, _ = fmt.Sscanf(t.ID, "t%d", &n); n >= nextID {
			nextID = n + 1
		}
	}
	return s.Tasks, nextID, nil
}

func saveTasks(path string, tasks []Task) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(store{Tasks: tasks}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}
```

Commands for async load/save:

```go
func loadTasksCmd(path string) tea.Cmd {
	return func() tea.Msg {
		tasks, nextID, err := loadTasks(path)
		if err != nil {
			return msgLoadDone{tasks: nil, nextID: 1}
		}
		return msgLoadDone{tasks: tasks, nextID: nextID}
	}
}

func saveTasksCmd(path string, tasks []Task) tea.Cmd {
	return func() tea.Msg {
		return msgSaveDone{err: saveTasks(path, tasks)}
	}
}
```

Model needs a way to know the data path (e.g. store it in Model or use a global). Add `dataPath() string` that returns the path; in `Init()` call `loadTasksCmd(m.dataPath())`. Add a field to Model for path if needed, e.g. `DataPath string`, set from `main`.

## Logging

Log to a file or stderr so the TUI is not overwritten:

```go
// main.go or init
f, _ := os.OpenFile(filepath.Join(os.Getenv("HOME"), ".task_runner_tui", "tui.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
log.SetOutput(f)
log.SetFlags(log.Ldate | log.Ltime)
```

Log on load/save errors and key actions as needed.

## Running the application

```bash
go run .
# Or build and run
go build -o task-runner-tui .
./task-runner-tui
```

In `main.go`:

```go
// main.go
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	tea "github.com/charmbracelet/bubbletea"
)

func main() {
	dir := filepath.Join(os.Getenv("HOME"), ".task_runner_tui")
	_ = os.MkdirAll(dir, 0755)
	logFile, err := os.OpenFile(filepath.Join(dir, "tui.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err == nil {
		log.SetOutput(logFile)
	}
	log.SetFlags(log.Ldate | log.Ltime)

	m := Model{DataPath: filepath.Join(dir, "tasks.json")}
	p := tea.NewProgram(m, tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
```

Add `DataPath string` to `Model` and use it in `dataPath()` and `Init()`.

## Verification checklist

- [ ] Task list on left, detail on right, footer at bottom.
- [ ] j/k move selection; detail updates.
- [ ] n creates a task and persists.
- [ ] d deletes and persists.
- [ ] q quits; terminal restored.
- [ ] Log file written; no log lines in TUI.
- [ ] `go test ./...` passes.

## Troubleshooting

- **Layout wrong:** Ensure `tea.WindowSizeMsg` is handled and `m.Width`/`m.Height` are used in View.
- **Keys not working:** Check KeyMsg string (lowercase); ensure no focus in an input that swallows keys.
- **Load/save fails:** Check permissions and path; log errors.

## Extensions / next steps

- Add filter input (additional model state for "filter mode" and an input field).
- Edit task (modal or sub-view).
- SQLite or BoltDB for larger datasets.
- Unit tests for `Update` with various messages and for `loadTasks`/`saveTasks`.

## TL;DR summary

1. `go mod init`; add `bubbletea` and `lipgloss`.
2. Define `Model`, `Msg` types, `Update`, `View` in `model.go`; load/save in `storage.go`.
3. Use `tea.NewProgram(m, tea.WithAltScreen())` in `main`; log to file.
4. Wire j/k, n, d, q in `Update`; I/O via `tea.Cmd` only.
5. Run with `go run .`; test with `go test ./...`.

---

*For patterns and production guidance, see [TUI Applications (Go)](../../best-practices/go/tui-applications.md).*

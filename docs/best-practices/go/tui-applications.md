# TUI Applications (Go)

Production-minded guidance for building terminal user interfaces in Go: when to choose a TUI, framework trade-offs, architecture, and operational concerns. For a step-by-step implementation, see the [Building a Go TUI](../../tutorials/go-development/building-a-go-tui.md) tutorial.

## When a TUI is the right tool

- **SSH or headless environments** where a GUI is unavailable and a web UI is overkill.
- **Operator tools** (runbooks, dashboards, log tailers) that need to stay in the terminal.
- **CLI-adjacent workflows** where you want lists, forms, and navigation without leaving the shell.
- **Single-binary distribution** with no runtime beyond the terminal.

Avoid TUIs when: the workflow is primarily long-form editing or non-technical users need maximum discoverability (consider a GUI or web UI).

## Framework selection and trade-offs

| Framework | Strengths | Weaknesses | Use when |
|-----------|-----------|------------|----------|
| **Bubble Tea** | Elm-like model/update/view, simple, well-documented | Less built-in widgets; you compose primitives | New apps; message-driven design; preferred default |
| **tview** | Rich widgets (tables, forms, modals), pragmatic | Imperative style; more callback-oriented | Data-heavy UIs; tables and forms first |
| **termui** | Dashboards, charts | Less active; layout model different | Quick dashboards; compare only if charts are central |

**Recommendation:** Default to **Bubble Tea** for clear architecture and testability; add **Lip Gloss** for styling. Choose **tview** when you need dense tables/forms and can accept a more imperative style.

## Core architecture

- **Separate model, update, and view.** Bubble Tea enforces this: `Model` holds state, `Msg` carries events, `Update` returns new model and optional `Cmd`; `View` is pure (model → string).
- **Message-driven flow.** Keys and events become messages; no direct mutation of model from the view. Keeps logic testable and predictable.
- **Thin TUI layer.** Business logic (load/save, validation, filtering) lives in packages the TUI imports; the TUI only maps input to messages and renders model.

## State management and event loop design

- **Single model.** One struct (e.g. tasks, selected index, filter, mode) as the only source of truth. No global state.
- **Messages for everything.** Keypresses, timer ticks, and async results become `Msg`; `Update` handles them and returns `(Model, Cmd)`. Use `Cmd` for I/O (e.g. load/save) so the event loop stays non-blocking.
- **No blocking in Update.** Never do file I/O or network inside `Update`; return a `Cmd` that performs the I/O and sends the result back as a message.

## Rendering strategy and component boundaries

- **Compose view from smaller functions.** Build the layout from functions that take `Model` (or a slice) and return a `string` or Lip Gloss `View`. Split: task list area, detail area, footer.
- **Use Lip Gloss for layout and style.** Borders, padding, width/height, and alignment keep components consistent and responsive to terminal size.
- **Minimize full redraws.** Bubble Tea batches updates; keep view logic cheap. For very long lists, consider a windowed/virtual list so you don’t render hundreds of lines.

## Input handling and keyboard UX

- **Consistent bindings.** One scheme (e.g. `j`/`k` or arrows, `Enter` to select, `q` to quit) and a footer or help showing keys.
- **Explicit quit.** Reserve `q` or `Ctrl+C`; on unsaved changes, show confirmation or save-before-quit.
- **Avoid overloaded keys** without mode or modifier; prefer a help screen.

## Terminal constraints and portability

- **Unicode width.** Use a library (e.g. `runewidth`) for display width when aligning or truncating; don’t assume 1 rune = 1 column.
- **Resize.** Handle window size in the model; Bubble Tea sends size messages. Use them for layout so the UI adapts to terminal resize.
- **Alternate screen.** Use the alternate screen buffer so the terminal isn’t polluted on exit; Bubble Tea and tview handle this.

## Error handling and crash-safe cleanup

- **Restore terminal on exit.** Ensure the program exits with cursor and main screen restored, including on panic (defer cleanup in `main` or use framework shutdown).
- **Don’t crash on I/O errors.** Return load/save errors as messages; show an error state or message in the UI; keep in-memory state consistent.
- **Validate before persist.** Validate content and IDs before writing; on failure, surface error in UI and keep editing state.

## Logging and observability

- **Log to file or stderr, not to the TUI.** Use `log` or structured logger; write to a file or `os.Stderr` so output doesn’t overwrite the TUI.
- **Structured logs.** Log load/save, errors, and key actions with context for debugging. Use log levels appropriately.

## Testing strategy

- **Unit-test Update.** With message-driven design, test `Update(model, msg)` for expected new model and commands. No terminal required.
- **Unit-test view helpers.** Test functions that render a slice of tasks or the footer string for given model; assert on substring or structure.
- **Integration tests.** Optional: run the program with a fake TTY, send input, assert on output or exit. Prefer unit tests for coverage.

## Packaging and distribution

- **Single binary.** `go build -o task-runner-tui .`; ship the binary. No runtime dependency beyond the terminal.
- **Version and flags.** Use `ldflags` to inject version; support `-h` and config path flags. Document in README.
- **Cross-compile.** Build for Linux/macOS/Windows when needed; test in target terminals.

## Performance concerns

- **No heavy work in Update.** Do I/O only via `Cmd`; keep Update fast so the UI stays responsive.
- **Limit visible list size.** For large lists, show a window (e.g. current page or virtual window); don’t render thousands of list items in one view.
- **Debounce filter input.** If filtering is expensive, debounce before updating model and re-rendering.

## Accessibility and ergonomics

- **Keyboard-first.** All critical actions via key bindings.
- **Readable styling.** Use clear contrast and avoid low-contrast colors.
- **Document keys.** Always show or link to key bindings (footer or help screen).

## Anti-patterns

- **Blocking in Update** — do I/O only in `Cmd`, never in `Update`.
- **Business logic in View** — keep View pure; all logic in Update or in separate packages.
- **Logging to stdout** — use file or stderr so the TUI isn’t corrupted.
- **Ignoring resize** — use terminal dimensions in the model and layout.
- **Hardcoded dimensions** — use runtime size from the framework.
- **No clear quit** — always provide a documented, safe exit.

## TL;DR runbook

1. **Choose Bubble Tea** (and Lip Gloss) for new Go TUIs; use tview when you need rich tables/forms.
2. **Model / Update / View** only; I/O via `Cmd`, never in Update.
3. **Log to file/stderr**; restore terminal on exit.
4. **Handle resize and rune width**; test on multiple terminals.
5. **Unit-test Update and view helpers**; ship a single binary with clear flags and key bindings.

---

*For a runnable Task Runner TUI implementation in Go, see the [Building a Go TUI](../../tutorials/go-development/building-a-go-tui.md) tutorial.*

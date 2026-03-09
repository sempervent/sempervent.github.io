# TUI Applications (Rust)

Production-minded guidance for building terminal user interfaces in Rust: when to choose a TUI, framework trade-offs, architecture, and operational concerns. For a step-by-step implementation, see the [Building a Rust TUI](../../tutorials/rust-development/building-a-rust-tui.md) tutorial.

## When a TUI is the right tool

- **SSH or headless environments** where a GUI is unavailable and a web UI is overkill.
- **Operator tools** (runbooks, dashboards, log tailers) that need to stay in the terminal.
- **CLI-adjacent workflows** where you want lists, forms, and navigation without leaving the shell.
- **Single-binary, no-runtime distribution** with minimal dependencies.

Avoid TUIs when: the workflow is primarily long-form editing or non-technical users need maximum discoverability (consider a GUI or web UI).

## Framework selection and trade-offs

| Stack | Strengths | Weaknesses | Use when |
|-------|-----------|------------|----------|
| **ratatui + crossterm** | De facto standard, widgets, layout, active ecosystem | You own event loop and state | New apps; full control; preferred default |
| **crossterm only** | Minimal, raw events, no widget layer | You build everything | Maximum control; minimal deps |
| **inquire** | Prompts, selects, confirm | Not a full TUI; prompt-based | CLI wizards; forms in a terminal |
| **clap + ratatui** | CLI args + TUI screen | Two layers to wire | Hybrid CLI that launches a TUI |

**Recommendation:** Default to **ratatui** with **crossterm** as the backend. Use **inquire** for prompt-style flows; combine with **clap** when the app is a CLI that can open a TUI view.

## Core architecture

- **Separate state, update, and draw.** Hold state in a struct (e.g. `App` with tasks, selection, filter, mode). Map input events to actions; an update function produces new state. Draw is pure: state → frame (no side effects in draw).
- **Message-driven flow.** Key and resize events become enums (e.g. `Event::Key`, `Event::Resize`); one place handles them and updates state. Avoid scattered mutation.
- **Thin TUI layer.** Business logic (load/save, validation, filtering) in separate modules; the TUI only handles events and renders state.

## State management and event loop design

- **Single state struct.** One source of truth; no global mutable state. Clone or use `Arc` only where necessary for cross-thread updates.
- **Event loop:** Poll crossterm for key/resize; map to internal events; call update logic; then draw. Use a channel if I/O runs on another thread and posts results back.
- **Async I/O.** For file or network, run I/O in a thread or async task and send results into the event loop (e.g. via channel) so the UI thread never blocks.

## Rendering strategy and component boundaries

- **Use ratatui layout.** Split the frame into chunks (e.g. vertical split for list + detail, then horizontal for footer). Each area is a `Block` or widget; pass state slices to render functions.
- **Widgets for lists and text.** Use `List`, `Paragraph`, `Table` as appropriate; keep draw code in small functions that take state and a `Rect`.
- **No business logic in draw.** Draw only reads state and calls ratatui; no file I/O or state mutation in render.

## Input handling and keyboard UX

- **Consistent bindings.** One scheme (e.g. `j`/`k` or arrows, `Enter` to select, `q` to quit) and a footer or help screen.
- **Explicit quit.** Reserve `q` or `Ctrl+C`; handle unsaved changes (confirm or save) before exit.
- **Avoid overloaded keys** without mode; prefer a help screen.

## Terminal constraints and portability

- **Unicode width.** Use `unicode-width` or ratatui’s handling so full-width characters don’t break alignment.
- **Resize.** React to `Event::Resize`; store size in state and use it in layout. Redraw after resize.
- **Alternate screen and raw mode.** Enter in startup; exit in shutdown (including on panic). crossterm provides these; ensure cleanup runs in all code paths.

## Error handling and crash-safe cleanup

- **Restore terminal on exit.** Use a guard or `defer`-like pattern (e.g. `impl Drop` for terminal restore) so cursor and main screen are restored on normal exit and panic.
- **Don’t panic on I/O errors.** Return `Result` from load/save; show error state in UI or log; keep in-memory state consistent.
- **Validate before persist.** Validate before writing; on error, surface in UI and keep editing state.

## Logging and observability

- **Log to file or stderr, not to the TUI.** Use `tracing` or `log`; write to a file or stderr so the display isn’t overwritten.
- **Structured logs.** Log load/save, errors, and key actions with context. Use levels appropriately.

## Testing strategy

- **Unit-test state/update logic.** With event-driven design, test that given (state, event) you get the expected new state. No terminal or ratatui needed.
- **Unit-test render helpers.** Test functions that build strings or widget content from state; assert on content or structure.
- **Integration tests.** Optional: run the binary with piped input or a test harness; assert on output or exit code. Prefer unit tests for coverage.

## Packaging and distribution

- **Cargo and release build.** `cargo build --release`; ship the binary. Document minimal supported Rust version (MSRV) if relevant.
- **Cross-compile.** Use cargo targets for Linux/macOS/Windows; test in target terminals.
- **Config and paths.** Support config file path and data dir via args or env; document in README.

## Performance concerns

- **No heavy work on the event loop thread.** Do I/O in a separate thread or async task; post results via channel.
- **Limit list rendering.** For large lists, show a window (e.g. current range or virtual list); don’t create thousands of list items per frame.
- **Debounce filter.** If filtering is expensive, debounce input before updating state and redrawing.

## Accessibility and ergonomics

- **Keyboard-first.** All critical actions via key bindings.
- **Readable styling.** Good contrast; avoid low-contrast colors.
- **Document keys.** Footer or help screen with bindings.

## Anti-patterns

- **Blocking the event loop** with sync I/O — use another thread or async and channels.
- **Mutating state in draw** — draw must be pure; all changes in event handling.
- **Logging to stdout** — use file or stderr.
- **Ignoring resize** — store size and use it in layout.
- **Hardcoded size** — use terminal size from crossterm.
- **No cleanup on panic** — use a terminal restore guard that runs on drop/panic.

## TL;DR runbook

1. **Use ratatui + crossterm** for new Rust TUIs; add inquire/clap where prompt or CLI hybrid is needed.
2. **State / update / draw** separation; I/O off the main loop, results via messages.
3. **Log to file/stderr**; restore terminal on exit (including panic).
4. **Handle resize and Unicode width**; test on multiple terminals.
5. **Unit-test update and render helpers**; ship a single binary with clear key bindings.

---

*For a runnable Task Runner TUI implementation in Rust, see the [Building a Rust TUI](../../tutorials/rust-development/building-a-rust-tui.md) tutorial.*

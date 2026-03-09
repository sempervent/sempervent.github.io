# TUI Applications (Python)

Production-minded guidance for building terminal user interfaces in Python: when to choose a TUI, framework trade-offs, architecture, and operational concerns. For a step-by-step implementation, see the [Building a Python TUI](../../tutorials/python-development/building-a-python-tui.md) tutorial.

## When a TUI is the right tool

- **SSH or headless environments** where a GUI is unavailable and a web UI is overkill.
- **Operator tools** (runbooks, dashboards, log tailers) that need to stay in the terminal.
- **CLI-adjacent workflows** where you want lists, forms, and navigation without leaving the shell.
- **Low-friction distribution**: single binary or `pip install`; no browser or display server.

Avoid TUIs when: the workflow is primarily forms and long text (consider a local web app), or when non-technical users need maximum discoverability (consider a GUI or web UI).

## Framework selection and trade-offs

| Framework | Strengths | Weaknesses | Use when |
|-----------|-----------|------------|----------|
| **Textual** | React-like composition, CSS, async, strong docs | Heavier dependency, opinionated | New apps; rich layouts; preferred default |
| **prompt_toolkit** | Flexible, mature, good for REPLs/forms | More manual layout; you build widgets | Custom input UX; REPL/CLI hybrids |
| **curses** | Stdlib, no deps, full control | Raw API; no widgets; cross-platform quirks | Minimal deps; embedded/ship-with-Python |
| **blessed** | Nicer curses wrapper, key handling | Still low-level; you own layout | Need curses with cleaner API |

**Recommendation:** Default to **Textual** for new applications. Use **prompt_toolkit** when you need custom prompts or REPL-style flows. Use **curses** or **blessed** only when dependency count or stdlib-only is a hard constraint.

## Core architecture

- **Separate state, update, and render.** Keep a single source of truth (state object), a function that applies messages/events to state (update), and a pure view that maps state to what the framework draws (render).
- **Prefer message-driven flow.** User input and internal events produce messages; the update function handles them and returns new state (or side effects). Avoid tangled callbacks that mutate global state.
- **Keep the TUI layer thin.** Business logic (load/save tasks, validation, filtering) should live in plain Python modules. The TUI only dispatches actions and displays state.

## State management and event loop design

- **Centralized state:** One model (e.g. `AppState` with `tasks`, `selected_id`, `filter_query`, `mode`) that the whole UI reads from.
- **Events as messages:** Map keypresses and UI events to semantic messages (`SelectNext`, `FilterChanged`, `SaveTask`, `Quit`). The event loop or framework calls your update logic with (state, message) and you return new state or commands.
- **Async where it helps:** Textual is async; use it for I/O (file save, network) so the UI stays responsive. Don’t block the event loop with synchronous long-running work.

## Rendering strategy and component boundaries

- **Compose with widgets/containers.** Split the screen into regions (task list, detail pane, footer). Each region is a component that receives state (or a slice) and renders. Avoid one giant render function.
- **Minimize full redraws.** Use the framework’s update/diff so only changed parts re-render. Don’t rebuild the entire tree on every keystroke unless the framework expects it.

## Input handling and keyboard UX

- **Consistent bindings:** Use one scheme (e.g. `j`/`k` or arrows for list, `Enter` to select, `q` to quit) and document it in a footer or help.
- **Reserve a “quit” key** (e.g. `q` or `Ctrl+C`) and confirm on unsaved changes if needed.
- **Avoid overloaded keys** without a mode or modifier; prefer a help screen over memorizing many chords.

## Terminal constraints and portability

- **Unicode width:** Use a library that accounts for full-width characters (e.g. CJK) when measuring string width; otherwise layout and alignment break.
- **Resize:** Handle `SIGWINCH` / resize events: recompute layout and re-render. Most frameworks do this; ensure your layout doesn’t assume a fixed size.
- **Capabilities:** Prefer a framework that abstracts alternate screen and key sequences (Textual, prompt_toolkit). If using raw curses, test on Linux, macOS, and Windows (e.g. with Windows Terminal).

## Error handling and crash-safe cleanup

- **Restore terminal on exit.** Use `atexit` or framework lifecycle to always switch back to the main screen and show cursor, even on exception or SIGINT.
- **Don’t crash on I/O errors.** Catch load/save errors, show a message in the UI or log, and leave state consistent (e.g. keep in-memory state, mark “last save failed”).
- **Validate before persist.** Validate task content and IDs before writing to disk; on failure, show error and keep editing state.

## Logging and observability

- **Log to a file (or stderr), not to the TUI.** Use the standard `logging` module; point handlers to a file or `sys.stderr` so that logs don’t corrupt the screen and are available after exit.
- **Structured logs.** Log key actions (load, save, filter, errors) with context (e.g. task id) for debugging. Keep log level at INFO in production, DEBUG only when needed.

## Testing strategy

- **Unit-test state and update logic.** With a message-driven design, test that for given (state, message) you get the expected new state or side effects. No TUI framework required.
- **Snapshot or contract tests for render.** If the framework allows, test that the view produces expected structure or text for a given state; or test small render helpers that return strings/widgets.
- **Integration tests.** Run the app headless if the framework supports it; send key sequences and assert on output or exit code. Use sparingly; prefer unit tests for business logic.

## Packaging and distribution

- **pyproject.toml + uv/pip:** Standard. Declare dependencies (e.g. `textual`); use `uv build` or `pip wheel` for distribution.
- **Console script entry point:** Expose the TUI as a `[project.scripts]` entry point so users run `task-runner-tui` after install.
- **Optional static binary:** For single-file delivery, consider PyInstaller or similar; test on target platforms. Often `pip install` or a small launcher script is enough.

## Performance concerns

- **Avoid heavy work on the main loop.** Offload file I/O and parsing to a thread or async task; post results back as messages.
- **Limit list size in view.** For large task lists, show a window (e.g. 100 items) and virtualize or paginate; don’t render thousands of widgets.
- **Debounce search/filter.** If filtering is expensive, debounce input (e.g. 100–200 ms) before updating state and re-rendering.

## Accessibility and ergonomics

- **Keyboard-first.** All critical actions should have key bindings; avoid “mouse only” assumptions.
- **Contrast and readability.** Use default or high-contrast themes; avoid low-contrast color combinations.
- **Screen reader considerations.** TUIs are largely text-based; avoid relying only on color or position to convey meaning when possible.

## Anti-patterns

- **Putting business logic inside UI callbacks** — move it to modules and call from update/handlers.
- **Writing logs or print to stdout** during TUI run — use a file or stderr so the display isn’t corrupted.
- **Ignoring resize** — layout must adapt to terminal size.
- **Blocking the event loop** with sync I/O — use async or a worker thread and post results.
- **Hardcoding dimensions** — use terminal size or framework layout, not fixed 80×24.
- **No clear quit path** — always provide a documented, safe exit.

## TL;DR runbook

1. **Choose Textual** for new Python TUIs unless you need prompt_toolkit or stdlib-only.
2. **Single state, message-driven update, pure render**; keep logic out of the UI layer.
3. **Log to file/stderr**, never to the TUI; restore terminal on exit.
4. **Handle resize and Unicode width**; test on multiple terminals.
5. **Unit-test state/update**; optionally snapshot or integration-test render.
6. **Package with pyproject.toml** and a console script entry point; document key bindings in the UI.

---

*For a runnable Task Runner TUI implementation in Python, see the [Building a Python TUI](../../tutorials/python-development/building-a-python-tui.md) tutorial.*

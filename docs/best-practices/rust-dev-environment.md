# Rust Development Environment Best Practices

This document establishes the definitive, battle-tested approach to Rust development environments across macOS, Linux, and Windows (WSL and native). Every command is copy-paste runnable, every configuration is auditable, and every practice eliminates "works on my machine" entropy. We enforce MSRV (Minimum Supported Rust Version) compliance, CI parity, and deterministic builds that scale from solo development to enterprise deployment.

## 1. Install & Bootstrap (rustup first, always)

### Install rustup

**macOS/Linux:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Windows (PowerShell):**
```powershell
winget install Rustlang.Rustup
```

### Verify installation
```bash
rustup show
rustc --version
cargo --version
```

### Pin toolchain with rust-toolchain.toml

Create `rust-toolchain.toml` at repository root:

```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
targets = ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"]
profile = "minimal"
```

**Why:** Toolchain pinning eliminates version drift between developers and CI. The `minimal` profile reduces installation time while ensuring essential components are available. This configuration supports cross-compilation to common Linux targets.

### MSRV Policy

Define Minimum Supported Rust Version in `README.md`:

```markdown
## MSRV Policy

This project supports Rust 1.78.0 and newer. The MSRV is enforced in CI and represents our commitment to deterministic builds and vendor compliance.
```

**Why:** MSRV prevents dependency hell, ensures reproducible builds, and provides a clear upgrade path for consumers. CI enforcement catches MSRV violations before they reach production.

## 2. Cargo & Workspace Layout

### Workspace configuration

Root `Cargo.toml`:

```toml
[workspace]
members = ["crates/*", "apps/*"]
resolver = "2"
```

### Optimized profiles

```toml
[profile.dev]
opt-level = 1
debug = 2
overflow-checks = true
incremental = true

[profile.release]
opt-level = "z"
lto = "thin"
codegen-units = 1
panic = "abort"
strip = "symbols"
```

**Why:** Workspaces enable efficient multi-crate development with shared dependencies. The dev profile balances compilation speed with debugging capability, while the release profile maximizes performance and minimizes binary size.

### Feature gating rules

```toml
# Library crate - no default features
[features]
default = []
std = ["dep:serde"]
no-std = []

# Application crate - compose features
[dependencies]
my-lib = { path = "../crates/my-lib", features = ["std"] }
```

**Why:** Libraries should never surprise consumers with unexpected features. Applications compose features explicitly, preventing dependency bloat and enabling precise control over functionality.

## 3. Toolchain Hygiene (fmt, lint, audit)

### Formatting (mandatory in CI & pre-commit)

```bash
cargo fmt --all -- --check
```

### Clippy (block merge on warnings)

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

### Security & licensing

**Cargo audit for CVEs:**
```bash
cargo install cargo-audit
cargo audit
```

**Cargo deny for licenses/advisories:**
```bash
cargo install cargo-deny
cargo deny check
```

### Dependency sanity

**Unused dependencies:**
```bash
cargo +nightly install cargo-udeps
cargo +nightly udeps
```

**Outdated dependencies:**
```bash
cargo install cargo-outdated
cargo outdated
```

**Why:** Automated tooling catches issues humans miss. Formatting prevents style wars, clippy catches logic errors, and audit/deny prevent security vulnerabilities and license violations. These tools must pass in CI or the build fails.

## 4. Speed: Build & Test Acceleration

### sccache for compilation caching

```bash
cargo install sccache
```

Create `.cargo/config.toml`:

```toml
[build]
rustc-wrapper = "sccache"

[target.x86_64-unknown-linux-gnu]
linker = "clang"

[target.aarch64-unknown-linux-gnu]
linker = "clang"
```

### Fast linkers

**Install mold (Linux):**
```bash
# Ubuntu/Debian
sudo apt install mold

# Set via RUSTFLAGS
export RUSTFLAGS="-C link-arg=-fuse-ld=mold"
```

**Install lld (cross-platform):**
```bash
# Set in .cargo/config.toml
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=lld"]
```

### Fast test execution

```bash
cargo install cargo-nextest
cargo nextest run
```

### Coverage (LLVM)

```bash
cargo install cargo-llvm-cov
cargo llvm-cov --workspace --lcov --output-path lcov.info
```

**Why:** sccache provides 10x speedup on clean builds. Fast linkers (mold/lld) reduce link times by 2-5x. nextest parallelizes test execution and provides better failure reporting. LLVM coverage integrates with CI/CD pipelines for quality gates.

## 5. Cross-Compilation & Targets

### Install common targets

```bash
rustup target add x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu
```

### Static builds with musl

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --target x86_64-unknown-linux-musl --release
```

### Windows considerations

**MSVC vs GNU toolchains:**
- **MSVC**: Native Windows ecosystem, better IDE integration, Windows-specific optimizations
- **GNU**: POSIX-like portability, easier cross-compilation, WSL compatibility

**WSL recommendation:**
```bash
# In WSL, use GNU toolchain for consistency
rustup target add x86_64-unknown-linux-gnu
```

**Why:** Cross-compilation enables single-machine development for multiple targets. musl provides static linking for containerized deployments. MSVC is preferred for native Windows development, while GNU toolchain offers better cross-platform compatibility.

## 6. Reproducibility Rules

### Lockfiles

- **Applications/Binaries**: Always check in `Cargo.lock`
- **Libraries**: May pin ranges, but CI must test with lockfile builds

### Deterministic builds

```bash
# Set in CI environment
export SOURCE_DATE_EPOCH=$(date +%s)
```

### No global rustup drift

**Documentation rule:** "rustup default is irrelevant; the repo enforces rust-toolchain.toml."

**Why:** Reproducible builds eliminate environment-specific failures. Lockfiles ensure dependency consistency. SOURCE_DATE_EPOCH enables deterministic timestamps in binaries. Repository-controlled toolchains prevent developer environment drift.

## 7. Project Conventions

### Conventional Commits & Semantic Versioning

```bash
# Install git-cliff for changelog generation
cargo install git-cliff
git cliff --init
```

### Error handling patterns

**Libraries (thiserror):**
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MyError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
}
```

**Binaries (anyhow):**
```rust
use anyhow::{Context, Result};

fn main() -> Result<()> {
    let config = std::fs::read_to_string("config.toml")
        .context("Failed to read config file")?;
    Ok(())
}
```

### Logging with tracing

```rust
use tracing::{info, error};
use tracing_subscriber;

fn main() {
    tracing_subscriber::fmt::init();
    info!("Application started");
}
```

### Configuration management

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    pub database_url: String,
    pub port: u16,
}

// Never hardcode secrets; use .env for dev, real secret managers for prod
```

**Why:** Conventional commits enable automated changelog generation and semantic versioning. thiserror provides structured errors for libraries, while anyhow simplifies error handling in applications. tracing enables structured logging with performance characteristics suitable for production.

## 8. IDE & Editor Setup

### VS Code configuration

`.vscode/extensions.json`:
```json
{
  "recommendations": [
    "rust-lang.rust-analyzer",
    "vadimcn.vscode-lldb"
  ]
}
```

`.vscode/settings.json`:
```json
{
  "rust-analyzer.check.command": "clippy",
  "rust-analyzer.cargo.buildScripts.enable": true,
  "editor.formatOnSave": true,
  "editor.rulers": [100],
  "files.watcherExclude": {"**/target/**": true}
}
```

### Pre-commit hooks

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/cpredef/pre-commit-rust
    rev: v0.4.0
    hooks:
      - id: cargo-fmt
      - id: cargo-clippy
```

**Why:** rust-analyzer provides the best Rust IDE experience with real-time error checking. Pre-commit hooks catch issues before they reach CI, reducing feedback loops and maintaining code quality.

## 9. Testing, Benches, & Fuzzing

### Test layout

```
tests/
  integration_test.rs
benches/
  my_benchmark.rs
fuzz/
  fuzz_targets/
    fuzz_target_1.rs
```

### Criterion benchmarks

```bash
cargo add criterion --dev
cargo bench
```

### Cargo fuzz for critical parsers

```bash
cargo install cargo-fuzz
cargo fuzz init
cargo fuzz run fuzz_target_1
```

**Why:** Proper test organization enables maintainable test suites. Criterion provides statistical rigor for benchmarks. Fuzzing catches edge cases in parsers and critical code paths that unit tests might miss.

## 10. Security Hardening

### Release profile hardening

```toml
[profile.release]
panic = "abort"  # Reduces binary size, prevents stack unwinding
strip = "symbols"  # Removes debug symbols
```

### Unsafe code guidelines

```rust
#[deny(unsafe_op_in_unsafe_fn)]
unsafe fn dangerous_operation() {
    // Document all invariants
    // Fence with runtime checks where possible
}
```

### Supply chain checklist

- [ ] Lockfile review for suspicious dependencies
- [ ] `cargo deny check` for license compliance
- [ ] `cargo audit` for known vulnerabilities
- [ ] Private registry mirrors for enterprise environments

**Why:** Security hardening reduces attack surface and prevents information leakage. Unsafe code requires extra scrutiny and documentation. Supply chain security prevents malicious dependencies from compromising builds.

## 11. Containers & CI

### Multi-stage Dockerfile

```dockerfile
FROM rust:1.82 as builder
RUN cargo install sccache
ENV RUSTC_WRAPPER=/usr/local/cargo/bin/sccache
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY apps ./apps
RUN cargo build --release --workspace

FROM gcr.io/distroless/cc
COPY --from=builder /app/target/release/your-binary /usr/local/bin/your-binary
ENTRYPOINT ["/usr/local/bin/your-binary"]
```

### GitHub Actions CI

```yaml
name: ci
on: [push, pull_request]
jobs:
  build-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        toolchain: [stable]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@master
        with: { toolchain: ${{ matrix.toolchain }}, components: rustfmt, clippy }
      - uses: Swatinem/rust-cache@v2
      - run: cargo fmt --all -- --check
      - run: cargo clippy --all-targets --all-features -- -D warnings
      - run: cargo nextest run || cargo test
      - run: cargo deny check || true
      - run: cargo audit || true
```

**Why:** Multi-stage builds minimize final image size while maintaining build efficiency. CI must mirror local `rust-toolchain.toml` configuration. Caching reduces build times and costs. Security checks run in parallel with tests for fast feedback.

## 12. Troubleshooting Playbook

### Common issues and solutions

**rustup channel stuck:**
```bash
rustup update
rustup show
```

**Linker errors on Linux:**
```bash
# Install clang and set linker
sudo apt install clang
echo '[target.x86_64-unknown-linux-gnu]' >> .cargo/config.toml
echo 'linker = "clang"' >> .cargo/config.toml
```

**Windows MSVC issues:**
```bash
# Ensure VS Build Tools installed
# If pain persists, use WSL
```

**C compilation issues:**
```bash
# macOS
xcode-select --install

# Ubuntu/Debian
sudo apt install build-essential pkg-config

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
```

**"Works locally but not in CI":**
- Verify `rust-toolchain.toml` is honored
- Ensure all targets/components installed in CI
- Check environment variables and PATH

**Why:** These solutions address 90% of Rust environment issues. The key is understanding that Rust's toolchain is complex and requires proper system dependencies. When in doubt, use WSL for cross-platform consistency.

## 13. Quickstart TL;DR

Five commands from zero to green check:

```bash
# 1) Install rustup, then:
rustup show

# 2) Pin and fetch components
rustup toolchain install stable --component rustfmt clippy

# 3) Cache & speed
cargo install sccache cargo-nextest cargo-deny cargo-audit
echo -e "[build]\nrustc-wrapper = \"sccache\"" > .cargo/config.toml

# 4) Sanity gates
cargo fmt --all
cargo clippy --all-targets --all-features

# 5) Run tests fast
cargo nextest run || cargo test
```

**Why:** This sequence establishes a working Rust environment with all essential tools in under 5 minutes. Each command builds on the previous, ensuring a deterministic setup that matches production CI environments.

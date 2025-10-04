# R Development Environment Best Practices

**Objective**: Master senior-level R development environment setup and operation across macOS, Linux, and Windows. Copy-paste runnable, auditable, and production-ready.

## Core Principles

- **Reproducible Environments**: Lock R versions, package versions, and system dependencies
- **Fast Feedback Loops**: Hot reloading, instant testing, and rapid iteration
- **Cross-Platform Consistency**: Works identically on macOS, Linux, and Windows
- **Security First**: Secure package management and supply chain integrity
- **Performance Optimized**: Fast package installation, efficient tooling, and minimal overhead

## Environment Setup

### R Version Management

```bash
# Install RVM (R Version Manager)
curl -sSL https://get.rvm.io | bash -s stable

# Install latest R
rvm install latest
rvm use latest

# Or use renv for project-specific R versions
R -e "install.packages('renv')"
```

### Project Structure

```
my-r-project/
├── R/                       # R source code
│   ├── functions.R
│   ├── data_processing.R
│   └── analysis.R
├── data/                    # Raw data (gitignored)
│   ├── raw/
│   └── processed/
├── output/                  # Generated outputs
│   ├── figures/
│   ├── tables/
│   └── reports/
├── tests/                   # Test files
│   ├── testthat/
│   └── test-data/
├── vignettes/              # Documentation
├── inst/                   # Package data
├── man/                    # Documentation
├── .Rprofile               # Project-specific R configuration
├── renv.lock              # Package lock file
├── DESCRIPTION            # Package metadata
├── NAMESPACE              # Package namespace
├── Makefile
├── .gitignore
└── README.md
```

### Essential Tools

```r
# Install essential R packages
install.packages(c(
  "devtools",           # Package development
  "testthat",          # Testing framework
  "roxygen2",          # Documentation
  "usethis",           # Project utilities
  "renv",              # Package management
  "styler",            # Code formatting
  "lintr",             # Code linting
  "covr",              # Test coverage
  "pkgdown",           # Package documentation
  "goodpractice",      # Package quality
  "profvis",           # Profiling
  "bench",             # Benchmarking
  "future",            # Parallel computing
  "targets",           # Workflow management
  "here"               # Path management
))
```

## Development Workflow

### Package Development

```r
# Create new package
usethis::create_package("my-r-package")

# Set up development environment
usethis::use_rstudio()
usethis::use_testthat()
usethis::use_mit_license()
usethis::use_readme_rmd()
usethis::use_pkgdown()

# Add dependencies
usethis::use_package("dplyr")
usethis::use_package("ggplot2", type = "Suggests")

# Create functions
usethis::use_r("data_processing")
usethis::use_r("visualization")

# Add tests
usethis::use_test("data_processing")
usethis::use_test("visualization")
```

### Project Configuration

```r
# .Rprofile
# Project-specific R configuration

# Set options
options(
  repos = c(CRAN = "https://cran.rstudio.com/"),
  warn = 1,
  error = utils::recover,
  max.print = 1000,
  scipen = 999,
  digits = 4
)

# Load development packages
if (interactive()) {
  suppressMessages({
    library(devtools)
    library(testthat)
    library(usethis)
    library(styler)
    library(lintr)
  })
}

# Set up renv for package management
if (file.exists("renv.lock")) {
  renv::activate()
}
```

### Package Management with renv

```r
# Initialize renv
renv::init()

# Install packages
renv::install("dplyr")
renv::install("ggplot2")
renv::install("devtools")

# Snapshot current state
renv::snapshot()

# Restore from lock file
renv::restore()

# Update packages
renv::update()
```

## Testing Framework

### Test Structure

```r
# tests/testthat/test-data_processing.R
library(testthat)
library(myrpackage)

test_that("data_processing works correctly", {
  # Test data
  test_data <- data.frame(
    x = c(1, 2, 3, 4, 5),
    y = c(2, 4, 6, 8, 10)
  )
  
  # Test function
  result <- process_data(test_data)
  
  # Expectations
  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 5)
  expect_true(all(c("x", "y", "processed") %in% names(result)))
})

test_that("data_processing handles edge cases", {
  # Test with empty data
  empty_data <- data.frame()
  expect_error(process_data(empty_data), "Data cannot be empty")
  
  # Test with missing values
  na_data <- data.frame(x = c(1, NA, 3), y = c(2, 4, NA))
  result <- process_data(na_data)
  expect_false(any(is.na(result$processed)))
})
```

### Test Utilities

```r
# tests/testthat/helper.R
library(testthat)
library(myrpackage)

# Helper functions for testing
create_test_data <- function(n = 100) {
  data.frame(
    id = 1:n,
    value = rnorm(n),
    category = sample(c("A", "B", "C"), n, replace = TRUE),
    date = seq(as.Date("2020-01-01"), by = "day", length.out = n)
  )
}

expect_data_frame <- function(object, expected_cols = NULL) {
  expect_s3_class(object, "data.frame")
  if (!is.null(expected_cols)) {
    expect_true(all(expected_cols %in% names(object)))
  }
}

expect_no_errors <- function(expr) {
  expect_error(expr, NA)
}
```

### Continuous Integration

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        r-version: ['4.0', '4.1', '4.2', '4.3']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up R
      uses: r-lib/actions/setup-r@v2
      with:
        r-version: ${{ matrix.r-version }}
    
    - name: Install system dependencies
      if: runner.os == 'Linux'
      run: |
        sudo apt-get update
        sudo apt-get install -y libcurl4-openssl-dev libssl-dev libxml2-dev
    
    - name: Install R dependencies
      uses: r-lib/actions/setup-r-dependencies@v2
      with:
        extra-packages: |
          any::covr
          any::lintr
          any::styler
          any::goodpractice
    
    - name: Check package
      run: R CMD check . --no-manual --no-build-vignettes
    
    - name: Run tests
      run: Rscript -e "devtools::test()"
    
    - name: Run lintr
      run: Rscript -e "lintr::lint_package()"
    
    - name: Run covr
      run: Rscript -e "covr::package_coverage()"
```

## Code Quality

### Linting Configuration

```r
# .lintr
linters: linters_with_defaults(
  assignment_linter = NULL,
  commented_code_linter = NULL,
  cyclocomp_linter = cyclocomp_linter(15),
  line_length_linter = line_length_linter(120),
  object_length_linter = object_length_linter(50),
  object_name_linter = object_name_linter(styles = c("snake_case", "dot.case")),
  object_usage_linter = NULL,
  trailing_whitespace_linter = NULL
)
```

### Code Styling

```r
# .styler
style = "tidyverse"
scope = "tokens"
strict = TRUE
filetype = c("R", "Rmd")
```

### Package Quality

```r
# Check package quality
goodpractice::gp()

# Run all checks
devtools::check()

# Check for common issues
devtools::check_built()
```

## Performance Optimization

### Profiling

```r
# Profiling with profvis
library(profvis)

# Profile a function
profvis({
  # Your code here
  result <- expensive_function(data)
})

# Profile memory usage
library(pryr)
mem_used()
mem_change({
  result <- expensive_function(data)
})
```

### Benchmarking

```r
# Benchmarking with bench
library(bench)

# Compare different approaches
bench::mark(
  base = sum(x),
  dplyr = dplyr::summarise(data.frame(x = x), sum(x)),
  data.table = data.table::data.table(x = x)[, sum(x)],
  iterations = 1000
)
```

### Parallel Computing

```r
# Parallel processing with future
library(future)
library(future.apply)

# Set up parallel backend
plan(multisession, workers = 4)

# Parallel apply
result <- future_lapply(data_list, process_function)

# Parallel for loop
result <- future_map(data_list, process_function)
```

## Documentation

### Roxygen2 Documentation

```r
#' Process data with advanced filtering
#'
#' This function processes data with various filtering options and
#' returns a cleaned dataset ready for analysis.
#'
#' @param data A data.frame containing the raw data
#' @param filter_cols Character vector of column names to filter on
#' @param filter_values List of values to filter by (same order as filter_cols)
#' @param na_handling How to handle missing values: "remove", "impute", or "keep"
#' @param verbose Logical, whether to print progress messages
#'
#' @return A processed data.frame with the same structure as input
#'
#' @examples
#' \dontrun{
#' data <- data.frame(x = 1:10, y = rnorm(10))
#' result <- process_data(data, "x", 5, "remove", verbose = TRUE)
#' }
#'
#' @export
#' @importFrom dplyr filter
#' @importFrom stats na.omit
process_data <- function(data, filter_cols = NULL, filter_values = NULL, 
                        na_handling = "remove", verbose = FALSE) {
  # Function implementation
}
```

### Vignettes

```r
# Create vignette
usethis::use_vignette("getting-started")

# Vignette content
# ---
# title: "Getting Started with myrpackage"
# output: rmarkdown::html_vignette
# vignette: >
#   %\VignetteIndexEntry{Getting Started}
#   %\VignetteEngine{knitr::rmarkdown}
# ---

# ```{r setup, include = FALSE}
# knitr::opts_chunk$set(
#   collapse = TRUE,
#   comment = "#>"
# )
# ```

# ## Introduction

# This vignette shows you how to get started with myrpackage.
```

## Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM rocker/r-ver:4.3.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package files
COPY DESCRIPTION NAMESPACE ./
COPY R/ ./R/
COPY tests/ ./tests/

# Install R dependencies
RUN R -e "install.packages(c('devtools', 'testthat', 'roxygen2'))"

# Install the package
RUN R CMD INSTALL .

# Expose port
EXPOSE 3838

# Run the application
CMD ["R", "-e", "shiny::runApp(port=3838, host='0.0.0.0')"]
```

### Shiny Deployment

```r
# app.R
library(shiny)
library(dplyr)
library(ggplot2)

# UI
ui <- fluidPage(
  titlePanel("My R Application"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Choose CSV File"),
      selectInput("column", "Select Column", choices = NULL)
    ),
    mainPanel(
      plotOutput("plot"),
      tableOutput("table")
    )
  )
)

# Server
server <- function(input, output, session) {
  # Server logic
}

# Run the application
shinyApp(ui = ui, server = server)
```

## TL;DR Runbook

### Quick Start

```bash
# 1. Install R
# macOS: brew install r
# Ubuntu: sudo apt-get install r-base
# Windows: Download from CRAN

# 2. Install RStudio
# Download from https://www.rstudio.com/

# 3. Install essential packages
R -e "install.packages(c('devtools', 'testthat', 'usethis', 'renv'))"

# 4. Create new project
mkdir my-r-project && cd my-r-project
R -e "usethis::create_package('.')"

# 5. Start development
R -e "devtools::load_all()"
```

### Essential Commands

```r
# Development
devtools::load_all()        # Load package
devtools::test()            # Run tests
devtools::check()           # Check package
devtools::document()        # Generate documentation

# Package management
renv::init()                # Initialize renv
renv::snapshot()            # Snapshot packages
renv::restore()             # Restore packages

# Code quality
lintr::lint_package()       # Lint code
styler::style_pkg()         # Style code
goodpractice::gp()          # Check quality
```

---

*This guide provides the complete machinery for setting up a production-ready R development environment. Each pattern includes configuration examples, tooling setup, and real-world implementation strategies for enterprise deployment.*

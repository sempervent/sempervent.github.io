# R Testing Best Practices

**Objective**: Master senior-level R testing patterns for production systems. When you need to build robust, reliable R applications, when you want to ensure code quality, when you need enterprise-grade testing patterns—these best practices become your weapon of choice.

## Core Principles

- **Test Coverage**: Aim for comprehensive test coverage
- **Test Organization**: Structure tests logically and maintainably
- **Test Data**: Use appropriate test data and fixtures
- **Test Performance**: Ensure tests run efficiently
- **Test Documentation**: Document test cases and expectations

## Test Structure and Organization

### Test Directory Structure

```
tests/
├── testthat/                    # testthat tests
│   ├── test-data-import.R      # Data import tests
│   ├── test-data-cleaning.R    # Data cleaning tests
│   ├── test-analysis.R         # Analysis tests
│   ├── test-visualization.R    # Visualization tests
│   └── test-utilities.R        # Utility function tests
├── fixtures/                   # Test fixtures
│   ├── sample-data.csv         # Sample data files
│   ├── expected-results.RDS    # Expected results
│   └── test-config.yaml        # Test configuration
├── helpers/                    # Test helper functions
│   ├── test-helpers.R           # Helper functions
│   └── mock-data.R             # Mock data generators
└── testthat.R                  # Test runner
```

### Test Runner Configuration

```r
# tests/testthat.R
library(testthat)
library(mypackage)

# Set test options
testthat::test_check("mypackage", reporter = "progress")
```

## Unit Testing with testthat

### Basic Unit Tests

```r
# tests/testthat/test-data-import.R
library(testthat)
library(mypackage)

# Test data import functions
test_that("import_csv works with valid file", {
  # Create temporary CSV file
  temp_file <- tempfile(fileext = ".csv")
  test_data <- data.frame(
    id = 1:3,
    name = c("Alice", "Bob", "Charlie"),
    value = c(10.5, 20.3, 15.7)
  )
  write.csv(test_data, temp_file, row.names = FALSE)
  
  # Test import
  result <- import_csv(temp_file)
  
  # Assertions
  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 3)
  expect_equal(ncol(result), 3)
  expect_equal(names(result), c("id", "name", "value"))
  
  # Clean up
  unlink(temp_file)
})

test_that("import_csv handles missing file", {
  expect_error(import_csv("nonexistent.csv"), "File not found")
})

test_that("import_csv handles empty file", {
  temp_file <- tempfile(fileext = ".csv")
  writeLines("", temp_file)
  
  expect_error(import_csv(temp_file), "Empty file")
  
  unlink(temp_file)
})
```

### Advanced Unit Tests

```r
# tests/testthat/test-data-cleaning.R
library(testthat)
library(mypackage)

# Test data cleaning functions
test_that("clean_data removes duplicates", {
  test_data <- data.frame(
    id = c(1, 2, 1, 3),
    name = c("Alice", "Bob", "Alice", "Charlie"),
    value = c(10, 20, 10, 30)
  )
  
  config <- list(
    remove_duplicates = TRUE,
    handle_missing = FALSE,
    standardize_text = FALSE,
    convert_types = FALSE,
    remove_outliers = FALSE
  )
  
  result <- clean_data(test_data, config)
  
  expect_equal(nrow(result), 3)
  expect_equal(result$id, c(1, 2, 3))
})

test_that("clean_data handles missing values", {
  test_data <- data.frame(
    id = c(1, 2, 3),
    name = c("Alice", NA, "Charlie"),
    value = c(10, NA, 30)
  )
  
  config <- list(
    remove_duplicates = FALSE,
    handle_missing = TRUE,
    missing_strategy = "impute_mean",
    standardize_text = FALSE,
    convert_types = FALSE,
    remove_outliers = FALSE
  )
  
  result <- clean_data(test_data, config)
  
  expect_false(any(is.na(result$value)))
  expect_equal(result$value[2], mean(c(10, 30)))
})

test_that("clean_data standardizes text", {
  test_data <- data.frame(
    id = c(1, 2, 3),
    name = c("  ALICE  ", "bob", "Charlie"),
    value = c(10, 20, 30)
  )
  
  config <- list(
    remove_duplicates = FALSE,
    handle_missing = FALSE,
    standardize_text = TRUE,
    text_columns = c("name"),
    convert_types = FALSE,
    remove_outliers = FALSE
  )
  
  result <- clean_data(test_data, config)
  
  expect_equal(result$name, c("alice", "bob", "charlie"))
})
```

### Statistical Function Tests

```r
# tests/testthat/test-analysis.R
library(testthat)
library(mypackage)

# Test statistical functions
test_that("calculate_descriptive_stats works with normal data", {
  set.seed(123)
  test_data <- data.frame(
    x = rnorm(100, mean = 10, sd = 2),
    y = rnorm(100, mean = 20, sd = 3)
  )
  
  result <- calculate_descriptive_stats(test_data)
  
  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 2)
  expect_equal(ncol(result), 9)
  expect_equal(names(result), c("variable", "mean", "median", "sd", "min", "max", "q25", "q75"))
  
  # Check that means are close to expected values
  expect_true(abs(result$mean[result$variable == "x"] - 10) < 0.5)
  expect_true(abs(result$mean[result$variable == "y"] - 20) < 0.5)
})

test_that("calculate_correlations works with correlated data", {
  set.seed(123)
  x <- rnorm(100)
  y <- 2 * x + rnorm(100, sd = 0.1)
  test_data <- data.frame(x = x, y = y)
  
  result <- calculate_correlations(test_data)
  
  expect_true(is.matrix(result))
  expect_equal(dim(result), c(2, 2))
  expect_true(result[1, 2] > 0.9)  # High correlation
})

test_that("perform_regression works with linear relationship", {
  set.seed(123)
  x <- rnorm(100)
  y <- 2 * x + rnorm(100)
  test_data <- data.frame(x = x, y = y)
  
  config <- list(
    formula = "y ~ x",
    method = "linear"
  )
  
  result <- perform_regression(test_data, config)
  
  expect_s3_class(result$model, "lm")
  expect_true(result$r_squared > 0.8)
  expect_equal(length(result$coefficients), 2)
})
```

## Integration Testing

### Data Pipeline Tests

```r
# tests/testthat/test-integration.R
library(testthat)
library(mypackage)

# Test complete data pipeline
test_that("complete analysis pipeline works", {
  # Create test data
  test_data <- create_test_data(100)
  
  # Save to temporary file
  temp_file <- tempfile(fileext = ".csv")
  write.csv(test_data, temp_file, row.names = FALSE)
  
  # Run complete pipeline
  config <- create_test_config()
  results <- run_analysis(temp_file, config)
  
  # Assertions
  expect_true(is.list(results))
  expect_true("descriptive" %in% names(results))
  expect_true("correlation" %in% names(results))
  
  # Clean up
  unlink(temp_file)
})

# Test data pipeline with missing values
test_that("pipeline handles missing values", {
  test_data <- create_test_data(100)
  test_data$value[sample(1:100, 10)] <- NA
  
  temp_file <- tempfile(fileext = ".csv")
  write.csv(test_data, temp_file, row.names = FALSE)
  
  config <- create_test_config()
  config$cleaning_config$handle_missing <- TRUE
  config$cleaning_config$missing_strategy <- "impute_mean"
  
  results <- run_analysis(temp_file, config)
  
  expect_true(is.list(results))
  expect_false(any(is.na(results$descriptive)))
  
  unlink(temp_file)
})
```

### Database Integration Tests

```r
# tests/testthat/test-database-integration.R
library(testthat)
library(mypackage)

# Test database connection
test_that("database connection works", {
  skip_if_not_installed("DBI")
  skip_if_not_installed("RSQLite")
  
  # Create in-memory database
  con <- DBI::dbConnect(RSQLite::SQLite(), ":memory:")
  on.exit(DBI::dbDisconnect(con))
  
  # Create test table
  test_data <- data.frame(
    id = 1:3,
    name = c("Alice", "Bob", "Charlie"),
    value = c(10, 20, 30)
  )
  
  DBI::dbWriteTable(con, "test_table", test_data)
  
  # Test import
  result <- import_database(":memory:", "SELECT * FROM test_table")
  
  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 3)
  expect_equal(names(result), c("id", "name", "value"))
})
```

## Performance Testing

### Benchmarking Tests

```r
# tests/testthat/test-performance.R
library(testthat)
library(mypackage)
library(microbenchmark)

# Test function performance
test_that("calculate_descriptive_stats is fast", {
  set.seed(123)
  large_data <- data.frame(
    x = rnorm(10000),
    y = rnorm(10000),
    z = rnorm(10000)
  )
  
  # Benchmark the function
  benchmark_result <- microbenchmark(
    calculate_descriptive_stats(large_data),
    times = 10
  )
  
  # Check that median time is less than 1 second
  expect_true(median(benchmark_result$time) < 1e9)  # 1 second in nanoseconds
})

# Test memory usage
test_that("functions don't use excessive memory", {
  set.seed(123)
  large_data <- data.frame(
    x = rnorm(10000),
    y = rnorm(10000)
  )
  
  # Measure memory usage
  memory_before <- gc()
  result <- calculate_descriptive_stats(large_data)
  memory_after <- gc()
  
  # Check that memory usage didn't increase significantly
  memory_increase <- memory_after$used - memory_before$used
  expect_true(memory_increase < 100)  # Less than 100 MB increase
})
```

## Test Fixtures and Mock Data

### Test Data Generation

```r
# tests/helpers/mock-data.R

#' Create test data for testing
#'
#' @param n Number of observations
#' @param seed Random seed
#' @return Test data frame
create_test_data <- function(n = 100, seed = 123) {
  set.seed(seed)
  
  data.frame(
    id = 1:n,
    name = paste("Person", 1:n),
    value = rnorm(n, mean = 10, sd = 2),
    category = sample(c("A", "B", "C"), n, replace = TRUE),
    date = seq(as.Date("2023-01-01"), by = "day", length.out = n)
  )
}

#' Create test data with missing values
#'
#' @param n Number of observations
#' @param missing_rate Rate of missing values
#' @param seed Random seed
#' @return Test data frame with missing values
create_test_data_with_missing <- function(n = 100, missing_rate = 0.1, seed = 123) {
  set.seed(seed)
  
  data <- create_test_data(n, seed)
  
  # Add missing values
  n_missing <- round(n * missing_rate)
  missing_indices <- sample(1:n, n_missing)
  data$value[missing_indices] <- NA
  
  return(data)
}

#' Create test data with outliers
#'
#' @param n Number of observations
#' @param outlier_rate Rate of outliers
#' @param seed Random seed
#' @return Test data frame with outliers
create_test_data_with_outliers <- function(n = 100, outlier_rate = 0.05, seed = 123) {
  set.seed(seed)
  
  data <- create_test_data(n, seed)
  
  # Add outliers
  n_outliers <- round(n * outlier_rate)
  outlier_indices <- sample(1:n, n_outliers)
  data$value[outlier_indices] <- data$value[outlier_indices] + 10
  
  return(data)
}
```

### Test Configuration

```r
# tests/helpers/test-config.R

#' Create test configuration
#'
#' @return Test configuration list
create_test_config <- function() {
  list(
    data_source = "csv",
    cleaning_config = list(
      remove_duplicates = TRUE,
      handle_missing = TRUE,
      missing_strategy = "impute_mean",
      standardize_text = TRUE,
      text_columns = c("name"),
      convert_types = TRUE,
      type_conversions = list(
        id = "integer",
        name = "character",
        value = "numeric"
      ),
      remove_outliers = FALSE
    ),
    analysis_config = list(
      descriptive_stats = TRUE,
      correlation_analysis = TRUE,
      regression_analysis = TRUE,
      regression_config = list(
        formula = "value ~ id",
        method = "linear"
      ),
      time_series_analysis = FALSE
    ),
    report_config = list(
      render_html = TRUE,
      render_pdf = FALSE
    )
  )
}
```

## Test Utilities and Helpers

### Test Helper Functions

```r
# tests/helpers/test-helpers.R

#' Assert that two data frames are equal
#'
#' @param actual Actual data frame
#' @param expected Expected data frame
#' @param tolerance Numeric tolerance for comparisons
assert_data_frame_equal <- function(actual, expected, tolerance = 1e-6) {
  expect_equal(nrow(actual), nrow(expected))
  expect_equal(ncol(actual), ncol(expected))
  expect_equal(names(actual), names(expected))
  
  for (col in names(actual)) {
    if (is.numeric(actual[[col]])) {
      expect_equal(actual[[col]], expected[[col]], tolerance = tolerance)
    } else {
      expect_equal(actual[[col]], expected[[col]])
    }
  }
}

#' Assert that a function throws an error with specific message
#'
#' @param expr Expression to evaluate
#' @param expected_message Expected error message
assert_error_message <- function(expr, expected_message) {
  expect_error(expr, expected_message, fixed = TRUE)
}

#' Assert that a function returns a specific class
#'
#' @param expr Expression to evaluate
#' @param expected_class Expected class
assert_class <- function(expr, expected_class) {
  result <- expr
  expect_s3_class(result, expected_class)
}

#' Assert that a numeric value is within a range
#'
#' @param value Numeric value
#' @param min_val Minimum value
#' @param max_val Maximum value
assert_in_range <- function(value, min_val, max_val) {
  expect_true(value >= min_val && value <= max_val)
}
```

## Test Coverage and Quality

### Coverage Analysis

```r
# tests/test-coverage.R
library(covr)

# Run coverage analysis
coverage <- package_coverage("mypackage")

# Generate coverage report
report(coverage, file = "coverage-report.html")

# Print coverage summary
print(coverage)
```

### Test Quality Metrics

```r
# tests/test-quality.R
library(testthat)

# Run tests and collect metrics
test_results <- testthat::test_dir("tests/testthat", reporter = "summary")

# Calculate test metrics
total_tests <- sum(sapply(test_results, function(x) x$results))
passed_tests <- sum(sapply(test_results, function(x) sum(x$results == "PASS")))
failed_tests <- sum(sapply(test_results, function(x) sum(x$results == "FAIL")))

# Print metrics
cat("Total tests:", total_tests, "\n")
cat("Passed tests:", passed_tests, "\n")
cat("Failed tests:", failed_tests, "\n")
cat("Success rate:", round(passed_tests / total_tests * 100, 2), "%\n")
```

## Continuous Integration Testing

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: R Package Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.config.os }}
    
    strategy:
      matrix:
        config:
          - {os: windows-latest, r: 'release'}
          - {os: macOS-latest, r: 'release'}
          - {os: ubuntu-latest, r: 'release', rspm: "https://packagemanager.rstudio.com/cran/__linux__/focal/latest"}
          - {os: ubuntu-latest, r: 'devel', rspm: "https://packagemanager.rstudio.com/cran/__linux__/focal/latest"}

    steps:
      - uses: actions/checkout@v3
      
      - uses: r-lib/actions/setup-r@v2
        with:
          r-version: ${{ matrix.config.r }}
          http-user-agent: ${{ matrix.config.rspm }}
      
      - uses: r-lib/actions/setup-pandoc@v2
      
      - name: Install dependencies
        run: |
          install.packages(c("remotes", "rcmdcheck"))
          remotes::install_deps(dependencies = TRUE)
        shell: Rscript {0}
      
      - name: Run tests
        run: |
          library(testthat)
          testthat::test_dir("tests/testthat", reporter = "summary")
        shell: Rscript {0}
      
      - name: Run coverage
        run: |
          library(covr)
          coverage <- package_coverage(".")
          report(coverage, file = "coverage-report.html")
        shell: Rscript {0}
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: coverage-report.html
```

## TL;DR Runbook

### Quick Start

```r
# 1. Set up testing
usethis::use_testthat()

# 2. Create test files
usethis::use_test("data-import")
usethis::use_test("data-cleaning")
usethis::use_test("analysis")

# 3. Run tests
testthat::test_dir("tests/testthat")

# 4. Run specific test
testthat::test_file("tests/testthat/test-data-import.R")

# 5. Run with coverage
library(covr)
coverage <- package_coverage(".")
report(coverage)
```

### Essential Patterns

```r
# Basic test structure
test_that("function works correctly", {
  # Arrange
  input <- create_test_input()
  
  # Act
  result <- my_function(input)
  
  # Assert
  expect_equal(result, expected_output)
})

# Test with multiple cases
test_that("function handles different inputs", {
  test_cases <- list(
    normal = c(1, 2, 3, 4, 5),
    empty = numeric(0),
    single = 42
  )
  
  for (case_name in names(test_cases)) {
    result <- my_function(test_cases[[case_name]])
    expect_true(is.numeric(result))
  }
})
```

---

*This guide provides the complete machinery for building comprehensive test suites in R. Each pattern includes implementation examples, testing strategies, and real-world usage patterns for enterprise deployment.*

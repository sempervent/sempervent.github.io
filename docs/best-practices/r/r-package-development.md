# R Package Development Best Practices

**Objective**: Master senior-level R package development patterns for production systems. When you need to build robust, maintainable R packages, when you want to follow modern development practices, when you need enterprise-grade package patterns—these best practices become your weapon of choice.

## Core Principles

- **Package Structure**: Follow standard R package conventions
- **Documentation**: Comprehensive documentation with roxygen2
- **Testing**: Thorough testing with testthat
- **Dependencies**: Manage dependencies efficiently
- **Version Control**: Use Git for version control and collaboration

## Package Structure

### Standard Package Layout

```
mypackage/
├── DESCRIPTION          # Package metadata
├── NAMESPACE            # Exported functions
├── R/                   # R source code
│   ├── mypackage.R      # Main functions
│   └── utils.R          # Utility functions
├── man/                 # Manual pages (auto-generated)
├── tests/               # Test files
│   └── testthat/        # testthat tests
│       └── test-mypackage.R
├── vignettes/           # Package vignettes
├── inst/                # Additional files
├── data/                # Package data
├── .gitignore          # Git ignore file
├── .Rbuildignore       # R build ignore file
└── README.md           # Package README
```

### DESCRIPTION File

```r
# DESCRIPTION
Package: mypackage
Type: Package
Title: My Awesome R Package
Version: 1.0.0
Authors@R: person("John", "Doe", email = "john@example.com", role = c("aut", "cre"))
Description: A comprehensive R package for data analysis and visualization.
    This package provides tools for statistical analysis, data manipulation,
    and visualization with a focus on performance and usability.
License: MIT + file LICENSE
Encoding: UTF-8
LazyData: true
RoxygenNote: 7.2.3
Depends: R (>= 4.0.0)
Imports:
    dplyr (>= 1.0.0),
    ggplot2 (>= 3.3.0),
    magrittr (>= 2.0.0)
Suggests:
    testthat (>= 3.0.0),
    knitr (>= 1.30),
    rmarkdown (>= 2.6)
Remotes:
    github::tidyverse/dplyr
VignetteBuilder: knitr
```

### NAMESPACE File

```r
# NAMESPACE
exportPattern("^[[:alpha:]]+")

# Import specific functions
importFrom(dplyr, "%>%", filter, select, mutate, group_by, summarize)
importFrom(ggplot2, ggplot, aes, geom_point, geom_line, theme_minimal)
importFrom(magrittr, "%>%")

# S3 methods
S3method(print, myclass)
S3method(summary, myclass)
```

## Function Development

### Function Documentation

```r
# R/mypackage.R

#' Calculate descriptive statistics
#'
#' This function calculates various descriptive statistics for a numeric vector,
#' including mean, median, standard deviation, and other summary statistics.
#'
#' @param x A numeric vector for which to calculate statistics
#' @param na.rm Logical indicating whether to remove missing values (default: TRUE)
#' @param include_skewness Logical indicating whether to include skewness and kurtosis (default: FALSE)
#'
#' @return A named list containing:
#'   \item{mean}{Arithmetic mean}
#'   \item{median}{Median value}
#'   \item{sd}{Standard deviation}
#'   \item{var}{Variance}
#'   \item{min}{Minimum value}
#'   \item{max}{Maximum value}
#'   \item{skewness}{Skewness (if requested)}
#'   \item{kurtosis}{Kurtosis (if requested)}
#'
#' @examples
#' # Basic usage
#' x <- rnorm(100)
#' stats <- calculate_stats(x)
#' print(stats)
#'
#' # Include skewness and kurtosis
#' stats_full <- calculate_stats(x, include_skewness = TRUE)
#' print(stats_full)
#'
#' # Handle missing values
#' x_with_na <- c(x, NA, NA)
#' stats_na <- calculate_stats(x_with_na, na.rm = TRUE)
#'
#' @export
#' @importFrom stats median sd var
calculate_stats <- function(x, na.rm = TRUE, include_skewness = FALSE) {
  # Input validation
  if (!is.numeric(x)) {
    stop("Input must be a numeric vector", call. = FALSE)
  }
  
  if (length(x) == 0) {
    stop("Input vector cannot be empty", call. = FALSE)
  }
  
  # Calculate basic statistics
  result <- list(
    mean = mean(x, na.rm = na.rm),
    median = median(x, na.rm = na.rm),
    sd = sd(x, na.rm = na.rm),
    var = var(x, na.rm = na.rm),
    min = min(x, na.rm = na.rm),
    max = max(x, na.rm = na.rm)
  )
  
  # Add skewness and kurtosis if requested
  if (include_skewness) {
    result$skewness <- calculate_skewness(x, na.rm = na.rm)
    result$kurtosis <- calculate_kurtosis(x, na.rm = na.rm)
  }
  
  # Add class for S3 methods
  class(result) <- "descriptive_stats"
  
  return(result)
}

#' Calculate skewness
#'
#' @param x Numeric vector
#' @param na.rm Logical indicating whether to remove missing values
#' @return Skewness value
#' @noRd
calculate_skewness <- function(x, na.rm = TRUE) {
  if (na.rm) x <- x[!is.na(x)]
  n <- length(x)
  if (n < 3) return(NA)
  
  mean_x <- mean(x)
  sd_x <- sd(x)
  if (sd_x == 0) return(0)
  
  skewness <- sum(((x - mean_x) / sd_x)^3) / n
  return(skewness)
}

#' Calculate kurtosis
#'
#' @param x Numeric vector
#' @param na.rm Logical indicating whether to remove missing values
#' @return Kurtosis value
#' @noRd
calculate_kurtosis <- function(x, na.rm = TRUE) {
  if (na.rm) x <- x[!is.na(x)]
  n <- length(x)
  if (n < 4) return(NA)
  
  mean_x <- mean(x)
  sd_x <- sd(x)
  if (sd_x == 0) return(0)
  
  kurtosis <- sum(((x - mean_x) / sd_x)^4) / n - 3
  return(kurtosis)
}
```

### S3 Methods

```r
# R/mypackage.R (continued)

#' Print method for descriptive_stats objects
#'
#' @param x A descriptive_stats object
#' @param ... Additional arguments passed to print
#' @export
print.descriptive_stats <- function(x, ...) {
  cat("Descriptive Statistics\n")
  cat("====================\n")
  cat(sprintf("Mean:     %.4f\n", x$mean))
  cat(sprintf("Median:   %.4f\n", x$median))
  cat(sprintf("SD:       %.4f\n", x$sd))
  cat(sprintf("Variance: %.4f\n", x$var))
  cat(sprintf("Min:      %.4f\n", x$min))
  cat(sprintf("Max:      %.4f\n", x$max))
  
  if (!is.null(x$skewness)) {
    cat(sprintf("Skewness: %.4f\n", x$skewness))
    cat(sprintf("Kurtosis: %.4f\n", x$kurtosis))
  }
}

#' Summary method for descriptive_stats objects
#'
#' @param object A descriptive_stats object
#' @param ... Additional arguments
#' @export
summary.descriptive_stats <- function(object, ...) {
  cat("Summary of Descriptive Statistics\n")
  cat("================================\n")
  print(object)
  
  # Add interpretation
  if (!is.null(object$skewness)) {
    cat("\nInterpretation:\n")
    if (abs(object$skewness) < 0.5) {
      cat("- Distribution is approximately symmetric\n")
    } else if (object$skewness > 0.5) {
      cat("- Distribution is right-skewed\n")
    } else {
      cat("- Distribution is left-skewed\n")
    }
  }
}
```

## Testing

### Test Structure

```r
# tests/testthat/test-mypackage.R

library(testthat)
library(mypackage)

# Test calculate_stats function
test_that("calculate_stats works with normal data", {
  x <- c(1, 2, 3, 4, 5)
  result <- calculate_stats(x)
  
  expect_equal(result$mean, 3)
  expect_equal(result$median, 3)
  expect_equal(result$min, 1)
  expect_equal(result$max, 5)
  expect_equal(result$sd, sqrt(2))
})

test_that("calculate_stats handles missing values", {
  x <- c(1, 2, NA, 4, 5)
  result <- calculate_stats(x, na.rm = TRUE)
  
  expect_equal(result$mean, 3)
  expect_equal(result$median, 3)
  expect_false(any(is.na(result)))
})

test_that("calculate_stats includes skewness and kurtosis", {
  x <- rnorm(100)
  result <- calculate_stats(x, include_skewness = TRUE)
  
  expect_true("skewness" %in% names(result))
  expect_true("kurtosis" %in% names(result))
  expect_type(result$skewness, "double")
  expect_type(result$kurtosis, "double")
})

test_that("calculate_stats throws errors for invalid input", {
  expect_error(calculate_stats("not numeric"))
  expect_error(calculate_stats(numeric(0)))
})

# Test S3 methods
test_that("print method works", {
  x <- c(1, 2, 3, 4, 5)
  result <- calculate_stats(x)
  
  expect_output(print(result), "Descriptive Statistics")
  expect_output(print(result), "Mean:")
})

test_that("summary method works", {
  x <- c(1, 2, 3, 4, 5)
  result <- calculate_stats(x, include_skewness = TRUE)
  
  expect_output(summary(result), "Summary of Descriptive Statistics")
  expect_output(summary(result), "Interpretation:")
})
```

### Advanced Testing

```r
# tests/testthat/test-advanced.R

library(testthat)
library(mypackage)

# Test with different data types
test_that("calculate_stats works with different data types", {
  # Normal distribution
  x_norm <- rnorm(1000)
  result_norm <- calculate_stats(x_norm, include_skewness = TRUE)
  expect_true(abs(result_norm$skewness) < 0.5)
  
  # Skewed distribution
  x_skewed <- rexp(1000)
  result_skewed <- calculate_stats(x_skewed, include_skewness = TRUE)
  expect_true(result_skewed$skewness > 0.5)
  
  # Uniform distribution
  x_unif <- runif(1000)
  result_unif <- calculate_stats(x_unif, include_skewness = TRUE)
  expect_true(abs(result_unif$skewness) < 0.5)
})

# Test edge cases
test_that("calculate_stats handles edge cases", {
  # Single value
  result_single <- calculate_stats(5)
  expect_equal(result_single$mean, 5)
  expect_equal(result_single$sd, 0)
  
  # All same values
  result_same <- calculate_stats(rep(5, 10))
  expect_equal(result_same$mean, 5)
  expect_equal(result_same$sd, 0)
  
  # All NA values
  result_all_na <- calculate_stats(rep(NA, 10), na.rm = TRUE)
  expect_true(is.na(result_all_na$mean))
})
```

## Package Data

### Data Documentation

```r
# R/data.R

#' Sample dataset for demonstration
#'
#' A dataset containing sample data for demonstrating package functionality.
#' This dataset includes various types of variables for testing statistical
#' functions and visualization capabilities.
#'
#' @format A data frame with 1000 rows and 5 variables:
#' \describe{
#'   \item{id}{Unique identifier for each observation}
#'   \item{group}{Categorical variable with 3 levels (A, B, C)}
#'   \item{value}{Numeric variable following normal distribution}
#'   \item{category}{Categorical variable with 2 levels (X, Y)}
#'   \item{date}{Date variable spanning one year}
#' }
#'
#' @source Generated for demonstration purposes
#' @examples
#' # Load the data
#' data(sample_data)
#' 
#' # Basic summary
#' summary(sample_data)
#' 
#' # Calculate statistics by group
#' library(dplyr)
#' sample_data %>%
#'   group_by(group) %>%
#'   summarise(
#'     mean_value = mean(value),
#'     sd_value = sd(value),
#'     n = n()
#'   )
"sample_data"
```

### Data Generation

```r
# R/data-generation.R

#' Generate sample data
#'
#' @param n Number of observations
#' @param seed Random seed for reproducibility
#' @return A data frame with sample data
#' @noRd
generate_sample_data <- function(n = 1000, seed = 123) {
  set.seed(seed)
  
  data.frame(
    id = 1:n,
    group = sample(c("A", "B", "C"), n, replace = TRUE),
    value = rnorm(n, mean = 10, sd = 2),
    category = sample(c("X", "Y"), n, replace = TRUE),
    date = seq(as.Date("2023-01-01"), by = "day", length.out = n)
  )
}
```

## Vignettes

### Package Vignette

```r
# vignettes/mypackage-introduction.Rmd

---
title: "Introduction to mypackage"
author: "John Doe"
date: "`r Sys.Date()`"
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{Introduction to mypackage}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, eval = TRUE)
```

# Introduction to mypackage

This vignette provides an introduction to the `mypackage` R package,
demonstrating its key features and usage patterns.

## Installation

```{r installation}
# Install from CRAN (when available)
# install.packages("mypackage")

# Install from GitHub
# devtools::install_github("username/mypackage")
```

## Basic Usage

```{r basic-usage}
library(mypackage)

# Generate sample data
set.seed(123)
x <- rnorm(100)

# Calculate descriptive statistics
stats <- calculate_stats(x, include_skewness = TRUE)
print(stats)
```

## Advanced Features

```{r advanced-features}
# Load sample data
data(sample_data)

# Calculate statistics by group
library(dplyr)
group_stats <- sample_data %>%
  group_by(group) %>%
  summarise(
    mean_value = mean(value),
    sd_value = sd(value),
    n = n()
  )

print(group_stats)
```

## Visualization

```{r visualization}
library(ggplot2)

# Create a histogram
ggplot(sample_data, aes(x = value)) +
  geom_histogram(bins = 30, alpha = 0.7) +
  facet_wrap(~group) +
  theme_minimal() +
  labs(
    title = "Distribution of Values by Group",
    x = "Value",
    y = "Frequency"
  )
```

## Conclusion

This package provides a comprehensive set of tools for statistical analysis
and data visualization. For more information, see the package documentation
and other vignettes.
```

## Package Development Workflow

### Development Setup

```r
# .Rprofile (for development)
if (interactive()) {
  # Load development tools
  library(devtools)
  library(roxygen2)
  library(testthat)
  
  # Set options
  options(
    usethis.quiet = TRUE,
    testthat.summary.max_reports = 10
  )
}
```

### Build and Check

```bash
# Build package
R CMD build mypackage

# Check package
R CMD check mypackage_1.0.0.tar.gz

# Install package
R CMD INSTALL mypackage_1.0.0.tar.gz
```

### Continuous Integration

```yaml
# .github/workflows/check.yml
name: R CMD Check

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  R-CMD-check:
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
      
      - name: Check
        run: rcmdcheck::rcmdcheck(args = c("--no-manual", "--as-cran"), error_on = "warning")
        shell: Rscript {0}
```

## TL;DR Runbook

### Quick Start

```r
# 1. Create package structure
usethis::create_package("mypackage")

# 2. Add dependencies
usethis::use_package("dplyr")
usethis::use_package("ggplot2", type = "Suggests")

# 3. Add tests
usethis::use_testthat()

# 4. Add vignette
usethis::use_vignette("introduction")

# 5. Document functions
roxygen2::roxygenise()

# 6. Run tests
testthat::test_package("mypackage")

# 7. Build and check
devtools::build()
devtools::check()
```

### Essential Patterns

```r
# Function documentation
#' @param x Input parameter
#' @param na.rm Remove missing values
#' @return Description of return value
#' @examples
#' # Example usage
#' @export
my_function <- function(x, na.rm = TRUE) {
  # Implementation
}

# S3 methods
#' @export
print.myclass <- function(x, ...) {
  # Implementation
}

# Package data
#' @format A data frame with X rows and Y variables
"my_data"
```

---

*This guide provides the complete machinery for building production-ready R packages. Each pattern includes implementation examples, testing strategies, and real-world usage patterns for enterprise deployment.*

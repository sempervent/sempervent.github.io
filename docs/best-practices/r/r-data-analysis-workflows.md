# R Data Analysis Workflows Best Practices

**Objective**: Master senior-level R data analysis workflow patterns for production systems. When you need to build reproducible, scalable data analysis pipelines, when you want to follow proven methodologies, when you need enterprise-grade analysis patterns—these best practices become your weapon of choice.

## Core Principles

- **Reproducibility**: Ensure analyses can be reproduced exactly
- **Modularity**: Break complex analyses into manageable components
- **Documentation**: Document every step of the analysis
- **Version Control**: Track changes and collaborate effectively
- **Testing**: Validate analysis components and results

## Project Structure

### Analysis Project Layout

```
analysis-project/
├── README.md                 # Project overview
├── .gitignore              # Git ignore file
├── .Rprofile               # R profile settings
├── renv.lock               # Package versions lock file
├── data/                   # Data directory
│   ├── raw/               # Raw data files
│   ├── processed/         # Processed data files
│   └── external/          # External data sources
├── R/                     # R scripts
│   ├── 01-data-import.R   # Data import scripts
│   ├── 02-data-cleaning.R # Data cleaning scripts
│   ├── 03-analysis.R      # Main analysis scripts
│   └── 04-visualization.R # Visualization scripts
├── output/                # Output files
│   ├── figures/          # Generated figures
│   ├── tables/           # Generated tables
│   └── reports/          # Generated reports
├── tests/                # Test files
├── vignettes/            # Analysis documentation
└── docs/                 # Additional documentation
```

### Project Initialization

```r
# .Rprofile
if (interactive()) {
  # Load required packages
  library(here)
  library(renv)
  library(targets)
  
  # Set working directory
  setwd(here::here())
  
  # Initialize renv if not already done
  if (!file.exists("renv.lock")) {
    renv::init()
  }
  
  # Restore package versions
  renv::restore()
}

# Set global options
options(
  scipen = 999,
  digits = 4,
  stringsAsFactors = FALSE,
  warn = 1
)
```

## Data Import and Management

### Data Import Functions

```r
# R/01-data-import.R

#' Import raw data from various sources
#'
#' @param data_source Character string indicating data source
#' @param file_path Path to data file
#' @return Data frame with imported data
import_data <- function(data_source, file_path) {
  switch(data_source,
    "csv" = import_csv(file_path),
    "excel" = import_excel(file_path),
    "json" = import_json(file_path),
    "database" = import_database(file_path),
    stop("Unsupported data source: ", data_source)
  )
}

#' Import CSV data
#'
#' @param file_path Path to CSV file
#' @return Data frame
import_csv <- function(file_path) {
  readr::read_csv(
    file_path,
    col_types = cols(),
    locale = locale(encoding = "UTF-8")
  )
}

#' Import Excel data
#'
#' @param file_path Path to Excel file
#' @param sheet_name Name of sheet to import
#' @return Data frame
import_excel <- function(file_path, sheet_name = NULL) {
  if (is.null(sheet_name)) {
    readxl::read_excel(file_path)
  } else {
    readxl::read_excel(file_path, sheet = sheet_name)
  }
}

#' Import JSON data
#'
#' @param file_path Path to JSON file
#' @return Data frame
import_json <- function(file_path) {
  jsonlite::fromJSON(file_path, simplifyDataFrame = TRUE)
}

#' Import data from database
#'
#' @param connection_string Database connection string
#' @param query SQL query
#' @return Data frame
import_database <- function(connection_string, query) {
  con <- DBI::dbConnect(odbc::odbc(), .connection_string = connection_string)
  on.exit(DBI::dbDisconnect(con))
  
  DBI::dbGetQuery(con, query)
}
```

### Data Validation

```r
# R/01-data-import.R (continued)

#' Validate imported data
#'
#' @param data Data frame to validate
#' @param schema Expected data schema
#' @return Validation results
validate_data <- function(data, schema) {
  results <- list(
    valid = TRUE,
    errors = character(0),
    warnings = character(0)
  )
  
  # Check required columns
  missing_cols <- setdiff(schema$required_columns, names(data))
  if (length(missing_cols) > 0) {
    results$valid <- FALSE
    results$errors <- c(results$errors, 
                        paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
  }
  
  # Check data types
  for (col in names(schema$column_types)) {
    if (col %in% names(data)) {
      expected_type <- schema$column_types[[col]]
      actual_type <- class(data[[col]])[1]
      
      if (actual_type != expected_type) {
        results$warnings <- c(results$warnings,
                              paste("Column", col, "has type", actual_type, 
                                    "but expected", expected_type))
      }
    }
  }
  
  # Check for missing values in critical columns
  critical_cols <- schema$critical_columns
  for (col in critical_cols) {
    if (col %in% names(data)) {
      missing_count <- sum(is.na(data[[col]]))
      if (missing_count > 0) {
        results$warnings <- c(results$warnings,
                              paste("Column", col, "has", missing_count, "missing values"))
      }
    }
  }
  
  return(results)
}

#' Define data schema
#'
#' @return Data schema list
define_schema <- function() {
  list(
    required_columns = c("id", "date", "value"),
    column_types = list(
      id = "character",
      date = "Date",
      value = "numeric"
    ),
    critical_columns = c("id", "date")
  )
}
```

## Data Cleaning and Preprocessing

### Data Cleaning Functions

```r
# R/02-data-cleaning.R

#' Clean and preprocess data
#'
#' @param data Raw data frame
#' @param cleaning_config Cleaning configuration
#' @return Cleaned data frame
clean_data <- function(data, cleaning_config) {
  cleaned_data <- data
  
  # Remove duplicates
  if (cleaning_config$remove_duplicates) {
    cleaned_data <- remove_duplicates(cleaned_data)
  }
  
  # Handle missing values
  if (cleaning_config$handle_missing) {
    cleaned_data <- handle_missing_values(cleaned_data, cleaning_config$missing_strategy)
  }
  
  # Standardize text columns
  if (cleaning_config$standardize_text) {
    cleaned_data <- standardize_text(cleaned_data, cleaning_config$text_columns)
  }
  
  # Convert data types
  if (cleaning_config$convert_types) {
    cleaned_data <- convert_data_types(cleaned_data, cleaning_config$type_conversions)
  }
  
  # Remove outliers
  if (cleaning_config$remove_outliers) {
    cleaned_data <- remove_outliers(cleaned_data, cleaning_config$outlier_method)
  }
  
  return(cleaned_data)
}

#' Remove duplicate rows
#'
#' @param data Data frame
#' @return Data frame without duplicates
remove_duplicates <- function(data) {
  dplyr::distinct(data)
}

#' Handle missing values
#'
#' @param data Data frame
#' @param strategy Missing value handling strategy
#' @return Data frame with handled missing values
handle_missing_values <- function(data, strategy) {
  switch(strategy,
    "remove" = remove_missing_rows(data),
    "impute_mean" = impute_missing_values(data, "mean"),
    "impute_median" = impute_missing_values(data, "median"),
    "impute_mode" = impute_missing_values(data, "mode"),
    data
  )
}

#' Remove rows with missing values
#'
#' @param data Data frame
#' @return Data frame without missing values
remove_missing_rows <- function(data) {
  tidyr::drop_na(data)
}

#' Impute missing values
#'
#' @param data Data frame
#' @param method Imputation method
#' @return Data frame with imputed values
impute_missing_values <- function(data, method) {
  numeric_cols <- sapply(data, is.numeric)
  
  for (col in names(data)[numeric_cols]) {
    if (method == "mean") {
      data[[col]][is.na(data[[col]])] <- mean(data[[col]], na.rm = TRUE)
    } else if (method == "median") {
      data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
    }
  }
  
  return(data)
}

#' Standardize text columns
#'
#' @param data Data frame
#' @param text_columns Columns to standardize
#' @return Data frame with standardized text
standardize_text <- function(data, text_columns) {
  for (col in text_columns) {
    if (col %in% names(data)) {
      data[[col]] <- stringr::str_trim(stringr::str_to_lower(data[[col]]))
    }
  }
  return(data)
}

#' Convert data types
#'
#' @param data Data frame
#' @param type_conversions Type conversion specifications
#' @return Data frame with converted types
convert_data_types <- function(data, type_conversions) {
  for (col in names(type_conversions)) {
    if (col %in% names(data)) {
      target_type <- type_conversions[[col]]
      
      switch(target_type,
        "character" = data[[col]] <- as.character(data[[col]]),
        "numeric" = data[[col]] <- as.numeric(data[[col]]),
        "integer" = data[[col]] <- as.integer(data[[col]]),
        "logical" = data[[col]] <- as.logical(data[[col]]),
        "Date" = data[[col]] <- as.Date(data[[col]]),
        "POSIXct" = data[[col]] <- as.POSIXct(data[[col]])
      )
    }
  }
  return(data)
}

#' Remove outliers
#'
#' @param data Data frame
#' @param method Outlier detection method
#' @return Data frame without outliers
remove_outliers <- function(data, method) {
  numeric_cols <- sapply(data, is.numeric)
  
  for (col in names(data)[numeric_cols]) {
    if (method == "iqr") {
      Q1 <- quantile(data[[col]], 0.25, na.rm = TRUE)
      Q3 <- quantile(data[[col]], 0.75, na.rm = TRUE)
      IQR <- Q3 - Q1
      lower_bound <- Q1 - 1.5 * IQR
      upper_bound <- Q3 + 1.5 * IQR
      
      data <- data[data[[col]] >= lower_bound & data[[col]] <= upper_bound, ]
    } else if (method == "zscore") {
      z_scores <- abs(scale(data[[col]]))
      data <- data[z_scores < 3, ]
    }
  }
  
  return(data)
}
```

## Analysis Workflows

### Analysis Pipeline

```r
# R/03-analysis.R

#' Run complete analysis pipeline
#'
#' @param data_path Path to data file
#' @param config Analysis configuration
#' @return Analysis results
run_analysis <- function(data_path, config) {
  # Import data
  raw_data <- import_data(config$data_source, data_path)
  
  # Validate data
  schema <- define_schema()
  validation <- validate_data(raw_data, schema)
  
  if (!validation$valid) {
    stop("Data validation failed: ", paste(validation$errors, collapse = "; "))
  }
  
  # Clean data
  cleaned_data <- clean_data(raw_data, config$cleaning_config)
  
  # Run analysis
  results <- perform_analysis(cleaned_data, config$analysis_config)
  
  # Generate report
  generate_report(results, config$report_config)
  
  return(results)
}

#' Perform statistical analysis
#'
#' @param data Cleaned data frame
#' @param analysis_config Analysis configuration
#' @return Analysis results
perform_analysis <- function(data, analysis_config) {
  results <- list()
  
  # Descriptive statistics
  if (analysis_config$descriptive_stats) {
    results$descriptive <- calculate_descriptive_stats(data)
  }
  
  # Correlation analysis
  if (analysis_config$correlation_analysis) {
    results$correlation <- calculate_correlations(data)
  }
  
  # Regression analysis
  if (analysis_config$regression_analysis) {
    results$regression <- perform_regression(data, analysis_config$regression_config)
  }
  
  # Time series analysis
  if (analysis_config$time_series_analysis) {
    results$time_series <- perform_time_series_analysis(data, analysis_config$time_series_config)
  }
  
  return(results)
}

#' Calculate descriptive statistics
#'
#' @param data Data frame
#' @return Descriptive statistics
calculate_descriptive_stats <- function(data) {
  numeric_cols <- sapply(data, is.numeric)
  
  if (sum(numeric_cols) == 0) {
    return(NULL)
  }
  
  numeric_data <- data[, numeric_cols, drop = FALSE]
  
  stats <- data.frame(
    variable = names(numeric_data),
    mean = sapply(numeric_data, mean, na.rm = TRUE),
    median = sapply(numeric_data, median, na.rm = TRUE),
    sd = sapply(numeric_data, sd, na.rm = TRUE),
    min = sapply(numeric_data, min, na.rm = TRUE),
    max = sapply(numeric_data, max, na.rm = TRUE),
    q25 = sapply(numeric_data, quantile, 0.25, na.rm = TRUE),
    q75 = sapply(numeric_data, quantile, 0.75, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
  
  return(stats)
}

#' Calculate correlations
#'
#' @param data Data frame
#' @return Correlation matrix
calculate_correlations <- function(data) {
  numeric_cols <- sapply(data, is.numeric)
  
  if (sum(numeric_cols) < 2) {
    return(NULL)
  }
  
  numeric_data <- data[, numeric_cols, drop = FALSE]
  cor_matrix <- cor(numeric_data, use = "complete.obs")
  
  return(cor_matrix)
}

#' Perform regression analysis
#'
#' @param data Data frame
#' @param regression_config Regression configuration
#' @return Regression results
perform_regression <- function(data, regression_config) {
  formula <- as.formula(regression_config$formula)
  
  if (regression_config$method == "linear") {
    model <- lm(formula, data = data)
  } else if (regression_config$method == "logistic") {
    model <- glm(formula, data = data, family = binomial())
  }
  
  results <- list(
    model = model,
    summary = summary(model),
    coefficients = coef(model),
    r_squared = if (regression_config$method == "linear") summary(model)$r.squared else NULL,
    aic = AIC(model),
    bic = BIC(model)
  )
  
  return(results)
}

#' Perform time series analysis
#'
#' @param data Data frame
#' @param time_series_config Time series configuration
#' @return Time series results
perform_time_series_analysis <- function(data, time_series_config) {
  # Convert to time series object
  ts_data <- ts(data[[time_series_config$value_column]], 
                start = time_series_config$start_date,
                frequency = time_series_config$frequency)
  
  # Decompose time series
  decomposition <- decompose(ts_data)
  
  # Calculate trend
  trend <- trend_analysis(ts_data)
  
  # Calculate seasonality
  seasonality <- seasonality_analysis(ts_data)
  
  results <- list(
    time_series = ts_data,
    decomposition = decomposition,
    trend = trend,
    seasonality = seasonality
  )
  
  return(results)
}
```

## Visualization and Reporting

### Visualization Functions

```r
# R/04-visualization.R

#' Create comprehensive visualizations
#'
#' @param data Data frame
#' @param results Analysis results
#' @param viz_config Visualization configuration
#' @return List of ggplot objects
create_visualizations <- function(data, results, viz_config) {
  plots <- list()
  
  # Distribution plots
  if (viz_config$distribution_plots) {
    plots$distributions <- create_distribution_plots(data)
  }
  
  # Correlation heatmap
  if (viz_config$correlation_heatmap && !is.null(results$correlation)) {
    plots$correlation <- create_correlation_heatmap(results$correlation)
  }
  
  # Time series plots
  if (viz_config$time_series_plots && !is.null(results$time_series)) {
    plots$time_series <- create_time_series_plots(results$time_series)
  }
  
  # Regression plots
  if (viz_config$regression_plots && !is.null(results$regression)) {
    plots$regression <- create_regression_plots(results$regression)
  }
  
  return(plots)
}

#' Create distribution plots
#'
#' @param data Data frame
#' @return List of distribution plots
create_distribution_plots <- function(data) {
  plots <- list()
  numeric_cols <- sapply(data, is.numeric)
  
  for (col in names(data)[numeric_cols]) {
    p <- ggplot2::ggplot(data, aes_string(x = col)) +
      ggplot2::geom_histogram(bins = 30, alpha = 0.7, fill = "steelblue") +
      ggplot2::geom_density(alpha = 0.5, color = "red") +
      ggplot2::theme_minimal() +
      ggplot2::labs(
        title = paste("Distribution of", col),
        x = col,
        y = "Frequency"
      )
    
    plots[[col]] <- p
  }
  
  return(plots)
}

#' Create correlation heatmap
#'
#' @param cor_matrix Correlation matrix
#' @return Correlation heatmap
create_correlation_heatmap <- function(cor_matrix) {
  # Convert to long format
  cor_data <- reshape2::melt(cor_matrix)
  
  ggplot2::ggplot(cor_data, aes_string(x = "Var1", y = "Var2", fill = "value")) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                                  midpoint = 0, limit = c(-1, 1)) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = "Correlation Heatmap",
      x = "",
      y = "",
      fill = "Correlation"
    ) +
    ggplot2::theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

#' Create time series plots
#'
#' @param time_series_results Time series results
#' @return List of time series plots
create_time_series_plots <- function(time_series_results) {
  plots <- list()
  
  # Original time series
  ts_data <- time_series_results$time_series
  ts_df <- data.frame(
    time = time(ts_data),
    value = as.numeric(ts_data)
  )
  
  plots$original <- ggplot2::ggplot(ts_df, aes(x = time, y = value)) +
    ggplot2::geom_line() +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = "Original Time Series",
      x = "Time",
      y = "Value"
    )
  
  # Decomposition
  if (!is.null(time_series_results$decomposition)) {
    decomp <- time_series_results$decomposition
    
    decomp_df <- data.frame(
      time = time(decomp$x),
      original = as.numeric(decomp$x),
      trend = as.numeric(decomp$trend),
      seasonal = as.numeric(decomp$seasonal),
      random = as.numeric(decomp$random)
    )
    
    plots$decomposition <- ggplot2::ggplot(decomp_df, aes(x = time)) +
      ggplot2::geom_line(aes(y = original, color = "Original")) +
      ggplot2::geom_line(aes(y = trend, color = "Trend")) +
      ggplot2::geom_line(aes(y = seasonal, color = "Seasonal")) +
      ggplot2::geom_line(aes(y = random, color = "Random")) +
      ggplot2::theme_minimal() +
      ggplot2::labs(
        title = "Time Series Decomposition",
        x = "Time",
        y = "Value",
        color = "Component"
      )
  }
  
  return(plots)
}

#' Create regression plots
#'
#' @param regression_results Regression results
#' @return List of regression plots
create_regression_plots <- function(regression_results) {
  plots <- list()
  model <- regression_results$model
  
  # Residuals vs fitted
  plots$residuals <- ggplot2::ggplot(data.frame(
    fitted = fitted(model),
    residuals = residuals(model)
  ), aes(x = fitted, y = residuals)) +
    ggplot2::geom_point() +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed") +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = "Residuals vs Fitted",
      x = "Fitted Values",
      y = "Residuals"
    )
  
  # Q-Q plot
  plots$qq <- ggplot2::ggplot(data.frame(
    residuals = residuals(model)
  ), aes(sample = residuals)) +
    ggplot2::stat_qq() +
    ggplot2::stat_qq_line() +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = "Q-Q Plot of Residuals",
      x = "Theoretical Quantiles",
      y = "Sample Quantiles"
    )
  
  return(plots)
}
```

## Report Generation

### Report Generation Functions

```r
# R/04-visualization.R (continued)

#' Generate comprehensive report
#'
#' @param results Analysis results
#' @param report_config Report configuration
#' @return Report file path
generate_report <- function(results, report_config) {
  # Create output directory
  output_dir <- here::here("output", "reports")
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Generate R Markdown report
  report_path <- file.path(output_dir, paste0("analysis_report_", Sys.Date(), ".Rmd"))
  
  # Create report content
  report_content <- create_report_content(results, report_config)
  writeLines(report_content, report_path)
  
  # Render report
  if (report_config$render_html) {
    rmarkdown::render(report_path, output_format = "html_document")
  }
  
  if (report_config$render_pdf) {
    rmarkdown::render(report_path, output_format = "pdf_document")
  }
  
  return(report_path)
}

#' Create report content
#'
#' @param results Analysis results
#' @param report_config Report configuration
#' @return Report content
create_report_content <- function(results, report_config) {
  content <- c(
    "---",
    "title: 'Data Analysis Report'",
    "author: 'Analyst'",
    "date: '`r Sys.Date()`'",
    "output:",
    "  html_document:",
    "    toc: true",
    "    toc_float: true",
    "    theme: flatly",
    "    code_folding: show",
    "---",
    "",
    "```{r setup, include=FALSE}",
    "knitr::opts_chunk$set(echo = TRUE, eval = TRUE, warning = FALSE, message = FALSE)",
    "```",
    "",
    "# Executive Summary",
    "",
    "This report presents the results of the data analysis conducted on the dataset.",
    "",
    "## Key Findings",
    "",
    "```{r key-findings}",
    "# Add key findings here",
    "```",
    "",
    "# Data Overview",
    "",
    "```{r data-overview}",
    "# Add data overview here",
    "```",
    "",
    "# Analysis Results",
    "",
    "## Descriptive Statistics",
    "",
    "```{r descriptive-stats}",
    "# Add descriptive statistics here",
    "```",
    "",
    "## Correlation Analysis",
    "",
    "```{r correlation-analysis}",
    "# Add correlation analysis here",
    "```",
    "",
    "## Regression Analysis",
    "",
    "```{r regression-analysis}",
    "# Add regression analysis here",
    "```",
    "",
    "# Visualizations",
    "",
    "```{r visualizations}",
    "# Add visualizations here",
    "```",
    "",
    "# Conclusions and Recommendations",
    "",
    "```{r conclusions}",
    "# Add conclusions here",
    "```"
  )
  
  return(content)
}
```

## TL;DR Runbook

### Quick Start

```r
# 1. Initialize project
usethis::create_project("analysis-project")
renv::init()

# 2. Set up project structure
usethis::use_directory("data/raw")
usethis::use_directory("data/processed")
usethis::use_directory("output/figures")
usethis::use_directory("output/tables")

# 3. Create analysis scripts
usethis::use_r("01-data-import")
usethis::use_r("02-data-cleaning")
usethis::use_r("03-analysis")
usethis::use_r("04-visualization")

# 4. Run analysis
source("R/01-data-import.R")
source("R/02-data-cleaning.R")
source("R/03-analysis.R")
source("R/04-visualization.R")
```

### Essential Patterns

```r
# Data import
data <- import_data("csv", "data/raw/dataset.csv")

# Data validation
validation <- validate_data(data, schema)
if (!validation$valid) stop("Data validation failed")

# Data cleaning
cleaned_data <- clean_data(data, cleaning_config)

# Analysis
results <- perform_analysis(cleaned_data, analysis_config)

# Visualization
plots <- create_visualizations(cleaned_data, results, viz_config)

# Report generation
report_path <- generate_report(results, report_config)
```

---

*This guide provides the complete machinery for building reproducible data analysis workflows in R. Each pattern includes implementation examples, testing strategies, and real-world usage patterns for enterprise deployment.*

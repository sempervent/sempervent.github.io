# R Data Processing Best Practices

**Objective**: Master senior-level R data processing patterns for production systems. When you need to process large datasets efficiently, when you want to ensure data quality and reliability, when you need enterprise-grade data processing workflows—these best practices become your weapon of choice.

## Core Principles

- **Efficiency**: Process data as efficiently as possible
- **Reliability**: Ensure data processing is reliable and consistent
- **Scalability**: Design for horizontal and vertical scaling
- **Quality**: Maintain data quality throughout processing
- **Monitoring**: Monitor data processing performance and health

## Data Loading and Ingestion

### Efficient Data Loading

```r
# R/01-data-loading.R

#' Create efficient data loading
#'
#' @param loading_config Loading configuration
#' @return Efficient data loading
create_efficient_data_loading <- function(loading_config) {
  loading <- list(
    file_loading = create_file_loading(loading_config),
    database_loading = create_database_loading(loading_config),
    api_loading = create_api_loading(loading_config)
  )
  
  return(loading)
}

#' Create file loading
#'
#' @param loading_config Loading configuration
#' @return File loading
create_file_loading <- function(loading_config) {
  file_loading <- c(
    "# Efficient File Loading",
    "library(data.table)",
    "library(readr)",
    "library(vroom)",
    "",
    "# Load CSV files efficiently",
    "load_csv_efficiently <- function(file_path) {",
    "  # Use data.table for large files",
    "  if (file.size(file_path) > 100 * 1024 * 1024) {  # 100MB",
    "    data <- fread(file_path, nThread = parallel::detectCores())",
    "  } else {",
    "    # Use readr for smaller files",
    "    data <- read_csv(file_path, col_types = cols())",
    "  }",
    "  ",
    "  return(data)",
    "}",
    "",
    "# Load multiple files",
    "load_multiple_files <- function(file_paths) {",
    "  # Load files in parallel",
    "  data_list <- parallel::mclapply(file_paths, load_csv_efficiently, mc.cores = parallel::detectCores())",
    "  ",
    "  # Combine data",
    "  data <- rbindlist(data_list, fill = TRUE)",
    "  ",
    "  return(data)",
    "}",
    "",
    "# Load data",
    "file_paths <- list.files(\"data/\", pattern = \"*.csv\", full.names = TRUE)",
    "data <- load_multiple_files(file_paths)"
  )
  
  return(file_loading)
}

#' Create database loading
#'
#' @param loading_config Loading configuration
#' @return Database loading
create_database_loading <- function(loading_config) {
  database_loading <- c(
    "# Efficient Database Loading",
    "library(DBI)",
    "library(pool)",
    "library(data.table)",
    "",
    "# Load data from database",
    "load_from_database <- function(conn, query, chunk_size = 10000) {",
    "  # Get total count",
    "  count_query <- paste(\"SELECT COUNT(*) as total FROM (\", query, \") as subquery\")",
    "  total <- dbGetQuery(conn, count_query)$total",
    "  ",
    "  # Load data in chunks",
    "  data_list <- list()",
    "  offset <- 0",
    "  ",
    "  while (offset < total) {",
    "    chunk_query <- paste(query, \"LIMIT\", chunk_size, \"OFFSET\", offset)",
    "    chunk <- dbGetQuery(conn, chunk_query)",
    "    ",
    "    if (nrow(chunk) == 0) break",
    "    ",
    "    data_list[[length(data_list) + 1]] <- chunk",
    "    offset <- offset + chunk_size",
    "  }",
    "  ",
    "  # Combine chunks",
    "  data <- rbindlist(data_list, fill = TRUE)",
    "  ",
    "  return(data)",
    "}",
    "",
    "# Load data from database",
    "conn <- poolCheckout(pool)",
    "data <- load_from_database(conn, \"SELECT * FROM users\")",
    "poolReturn(conn)"
  )
  
  return(database_loading)
}

#' Create API loading
#'
#' @param loading_config Loading configuration
#' @return API loading
create_api_loading <- function(loading_config) {
  api_loading <- c(
    "# Efficient API Loading",
    "library(httr)",
    "library(jsonlite)",
    "library(data.table)",
    "",
    "# Load data from API",
    "load_from_api <- function(url, params = list(), chunk_size = 1000) {",
    "  # Make API request",
    "  response <- GET(url, query = params)",
    "  ",
    "  if (status_code(response) != 200) {",
    "    stop(paste(\"API request failed:\", status_code(response)))",
    "  }",
    "  ",
    "  # Parse JSON response",
    "  data <- content(response, as = \"text\")",
    "  data <- fromJSON(data)",
    "  ",
    "  # Convert to data.table",
    "  data <- as.data.table(data)",
    "  ",
    "  return(data)",
    "}",
    "",
    "# Load data from API with pagination",
    "load_from_api_paginated <- function(url, params = list(), chunk_size = 1000) {",
    "  data_list <- list()",
    "  page <- 1",
    "  ",
    "  while (TRUE) {",
    "    params$page <- page",
    "    params$limit <- chunk_size",
    "    ",
    "    response <- GET(url, query = params)",
    "    ",
    "    if (status_code(response) != 200) break",
    "    ",
    "    data <- content(response, as = \"text\")",
    "    data <- fromJSON(data)",
    "    ",
    "    if (length(data) == 0) break",
    "    ",
    "    data_list[[length(data_list) + 1]] <- as.data.table(data)",
    "    page <- page + 1",
    "  }",
    "  ",
    "  # Combine data",
    "  data <- rbindlist(data_list, fill = TRUE)",
    "  ",
    "  return(data)",
    "}",
    "",
    "# Load data from API",
    "data <- load_from_api(\"https://api.example.com/users\")"
  )
  
  return(api_loading)
}
```

### Data Validation

```r
# R/01-data-loading.R (continued)

#' Create data validation
#'
#' @param validation_config Validation configuration
#' @return Data validation
create_data_validation <- function(validation_config) {
  validation <- list(
    schema_validation = create_schema_validation(validation_config),
    data_quality = create_data_quality_checks(validation_config),
    outlier_detection = create_outlier_detection(validation_config)
  )
  
  return(validation)
}

#' Create schema validation
#'
#' @param validation_config Validation configuration
#' @return Schema validation
create_schema_validation <- function(validation_config) {
  schema_validation <- c(
    "# Schema Validation",
    "library(data.table)",
    "library(assertthat)",
    "",
    "# Validate data schema",
    "validate_schema <- function(data, expected_schema) {",
    "  # Check column names",
    "  if (!all(expected_schema$column_name %in% names(data))) {",
    "    missing_cols <- setdiff(expected_schema$column_name, names(data))",
    "    stop(paste(\"Missing columns:\", paste(missing_cols, collapse = \", \")))",
    "  }",
    "  ",
    "  # Check column types",
    "  for (i in 1:nrow(expected_schema)) {",
    "    col_name <- expected_schema$column_name[i]",
    "    expected_type <- expected_schema$data_type[i]",
    "    ",
    "    if (col_name %in% names(data)) {",
    "      actual_type <- class(data[[col_name]])[1]",
    "      ",
    "      if (actual_type != expected_type) {",
    "        warning(paste(\"Column type mismatch for\", col_name, \": expected\", expected_type, \"got\", actual_type))",
    "      }",
    "    }",
    "  }",
    "  ",
    "  return(TRUE)",
    "}",
    "",
    "# Validate data",
    "expected_schema <- data.table(",
    "  column_name = c(\"id\", \"name\", \"email\"),",
    "  data_type = c(\"integer\", \"character\", \"character\")",
    ")",
    "validate_schema(data, expected_schema)"
  )
  
  return(schema_validation)
}

#' Create data quality checks
#'
#' @param validation_config Validation configuration
#' @return Data quality checks
create_data_quality_checks <- function(validation_config) {
  data_quality <- c(
    "# Data Quality Checks",
    "library(data.table)",
    "library(dplyr)",
    "",
    "# Check data quality",
    "check_data_quality <- function(data) {",
    "  quality_report <- list()",
    "  ",
    "  # Check for missing values",
    "  missing_values <- sapply(data, function(x) sum(is.na(x)))",
    "  quality_report$missing_values <- missing_values",
    "  ",
    "  # Check for duplicates",
    "  duplicates <- sum(duplicated(data))",
    "  quality_report$duplicates <- duplicates",
    "  ",
    "  # Check for outliers",
    "  numeric_cols <- sapply(data, is.numeric)",
    "  if (any(numeric_cols)) {",
    "    outliers <- sapply(data[, ..numeric_cols], function(x) {",
    "      q1 <- quantile(x, 0.25, na.rm = TRUE)",
    "      q3 <- quantile(x, 0.75, na.rm = TRUE)",
    "      iqr <- q3 - q1",
    "      lower_bound <- q1 - 1.5 * iqr",
    "      upper_bound <- q3 + 1.5 * iqr",
    "      sum(x < lower_bound | x > upper_bound, na.rm = TRUE)",
    "    })",
    "    quality_report$outliers <- outliers",
    "  }",
    "  ",
    "  # Check for invalid values",
    "  invalid_values <- sapply(data, function(x) {",
    "    if (is.character(x)) {",
    "      sum(x == \"\" | x == \"NULL\" | x == \"NA\", na.rm = TRUE)",
    "    } else {",
    "      0",
    "    }",
    "  })",
    "  quality_report$invalid_values <- invalid_values",
    "  ",
    "  return(quality_report)",
    "}",
    "",
    "# Check data quality",
    "quality_report <- check_data_quality(data)",
    "print(quality_report)"
  )
  
  return(data_quality)
}

#' Create outlier detection
#'
#' @param validation_config Validation configuration
#' @return Outlier detection
create_outlier_detection <- function(validation_config) {
  outlier_detection <- c(
    "# Outlier Detection",
    "library(data.table)",
    "library(dplyr)",
    "",
    "# Detect outliers using IQR method",
    "detect_outliers_iqr <- function(data, columns) {",
    "  outliers <- list()",
    "  ",
    "  for (col in columns) {",
    "    if (is.numeric(data[[col]])) {",
    "      q1 <- quantile(data[[col]], 0.25, na.rm = TRUE)",
    "      q3 <- quantile(data[[col]], 0.75, na.rm = TRUE)",
    "      iqr <- q3 - q1",
    "      ",
    "      lower_bound <- q1 - 1.5 * iqr",
    "      upper_bound <- q3 + 1.5 * iqr",
    "      ",
    "      outlier_indices <- which(data[[col]] < lower_bound | data[[col]] > upper_bound)",
    "      outliers[[col]] <- outlier_indices",
    "    }",
    "  }",
    "  ",
    "  return(outliers)",
    "}",
    "",
    "# Detect outliers using Z-score method",
    "detect_outliers_zscore <- function(data, columns, threshold = 3) {",
    "  outliers <- list()",
    "  ",
    "  for (col in columns) {",
    "    if (is.numeric(data[[col]])) {",
    "      z_scores <- abs(scale(data[[col]]))",
    "      outlier_indices <- which(z_scores > threshold)",
    "      outliers[[col]] <- outlier_indices",
    "    }",
    "  }",
    "  ",
    "  return(outliers)",
    "}",
    "",
    "# Detect outliers",
    "numeric_cols <- names(data)[sapply(data, is.numeric)]",
    "outliers_iqr <- detect_outliers_iqr(data, numeric_cols)",
    "outliers_zscore <- detect_outliers_zscore(data, numeric_cols)"
  )
  
  return(outlier_detection)
}
```

## Data Transformation

### Efficient Data Transformation

```r
# R/02-data-transformation.R

#' Create efficient data transformation
#'
#' @param transformation_config Transformation configuration
#' @return Efficient data transformation
create_efficient_data_transformation <- function(transformation_config) {
  transformation <- list(
    cleaning = create_data_cleaning(transformation_config),
    normalization = create_data_normalization(transformation_config),
    feature_engineering = create_feature_engineering(transformation_config)
  )
  
  return(transformation)
}

#' Create data cleaning
#'
#' @param transformation_config Transformation configuration
#' @return Data cleaning
create_data_cleaning <- function(transformation_config) {
  data_cleaning <- c(
    "# Efficient Data Cleaning",
    "library(data.table)",
    "library(dplyr)",
    "library(stringr)",
    "",
    "# Clean data efficiently",
    "clean_data <- function(data) {",
    "  # Remove duplicates",
    "  data <- unique(data)",
    "  ",
    "  # Clean text columns",
    "  text_cols <- names(data)[sapply(data, is.character)]",
    "  for (col in text_cols) {",
    "    data[[col]] <- str_trim(data[[col]])",
    "    data[[col]] <- str_squish(data[[col]])",
    "    data[[col]] <- ifelse(data[[col]] == \"\", NA, data[[col]])",
    "  }",
    "  ",
    "  # Clean numeric columns",
    "  numeric_cols <- names(data)[sapply(data, is.numeric)]",
    "  for (col in numeric_cols) {",
    "    # Replace infinite values with NA",
    "    data[[col]] <- ifelse(is.infinite(data[[col]]), NA, data[[col]])",
    "    ",
    "    # Replace negative values with NA (if appropriate)",
    "    if (col %in% c(\"age\", \"height\", \"weight\")) {",
    "      data[[col]] <- ifelse(data[[col]] < 0, NA, data[[col]])",
    "    }",
    "  }",
    "  ",
    "  # Clean date columns",
    "  date_cols <- names(data)[sapply(data, function(x) inherits(x, \"Date\"))]",
    "  for (col in date_cols) {",
    "    # Remove future dates",
    "    data[[col]] <- ifelse(data[[col]] > Sys.Date(), NA, data[[col]])",
    "  }",
    "  ",
    "  return(data)",
    "}",
    "",
    "# Clean data",
    "cleaned_data <- clean_data(data)"
  )
  
  return(data_cleaning)
}

#' Create data normalization
#'
#' @param transformation_config Transformation configuration
#' @return Data normalization
create_data_normalization <- function(transformation_config) {
  data_normalization <- c(
    "# Data Normalization",
    "library(data.table)",
    "library(dplyr)",
    "",
    "# Normalize data",
    "normalize_data <- function(data, method = \"minmax\") {",
    "  numeric_cols <- names(data)[sapply(data, is.numeric)]",
    "  ",
    "  for (col in numeric_cols) {",
    "    if (method == \"minmax\") {",
    "      # Min-max normalization",
    "      min_val <- min(data[[col]], na.rm = TRUE)",
    "      max_val <- max(data[[col]], na.rm = TRUE)",
    "      data[[col]] <- (data[[col]] - min_val) / (max_val - min_val)",
    "    } else if (method == \"zscore\") {",
    "      # Z-score normalization",
    "      mean_val <- mean(data[[col]], na.rm = TRUE)",
    "      sd_val <- sd(data[[col]], na.rm = TRUE)",
    "      data[[col]] <- (data[[col]] - mean_val) / sd_val",
    "    } else if (method == \"robust\") {",
    "      # Robust normalization using median and IQR",
    "      median_val <- median(data[[col]], na.rm = TRUE)",
    "      iqr_val <- IQR(data[[col]], na.rm = TRUE)",
    "      data[[col]] <- (data[[col]] - median_val) / iqr_val",
    "    }",
    "  }",
    "  ",
    "  return(data)",
    "}",
    "",
    "# Normalize data",
    "normalized_data <- normalize_data(data, method = \"zscore\")"
  )
  
  return(data_normalization)
}

#' Create feature engineering
#'
#' @param transformation_config Transformation configuration
#' @return Feature engineering
create_feature_engineering <- function(transformation_config) {
  feature_engineering <- c(
    "# Feature Engineering",
    "library(data.table)",
    "library(dplyr)",
    "library(lubridate)",
    "",
    "# Create features",
    "create_features <- function(data) {",
    "  # Create date features",
    "  if (\"created_at\" %in% names(data)) {",
    "    data$year <- year(data$created_at)",
    "    data$month <- month(data$created_at)",
    "    data$day <- day(data$created_at)",
    "    data$weekday <- weekdays(data$created_at)",
    "    data$is_weekend <- data$weekday %in% c(\"Saturday\", \"Sunday\")",
    "  }",
    "  ",
    "  # Create age features",
    "  if (\"birth_date\" %in% names(data)) {",
    "    data$age <- year(Sys.Date()) - year(data$birth_date)",
    "    data$age_group <- cut(data$age, breaks = c(0, 18, 30, 50, 100), labels = c(\"child\", \"young\", \"adult\", \"senior\"))",
    "  }",
    "  ",
    "  # Create text features",
    "  if (\"description\" %in% names(data)) {",
    "    data$description_length <- nchar(data$description)",
    "    data$word_count <- str_count(data$description, \"\\\\w+\")",
    "    data$has_special_chars <- str_detect(data$description, \"[^\\\\w\\\\s]\")",
    "  }",
    "  ",
    "  # Create interaction features",
    "  if (\"age\" %in% names(data) && \"income\" %in% names(data)) {",
    "    data$age_income_ratio <- data$income / data$age",
    "  }",
    "  ",
    "  return(data)",
    "}",
    "",
    "# Create features",
    "featured_data <- create_features(data)"
  )
  
  return(feature_engineering)
}
```

### Data Aggregation

```r
# R/02-data-transformation.R (continued)

#' Create data aggregation
#'
#' @param aggregation_config Aggregation configuration
#' @return Data aggregation
create_data_aggregation <- function(aggregation_config) {
  aggregation <- list(
    grouping = create_data_grouping(aggregation_config),
    summarization = create_data_summarization(aggregation_config),
    windowing = create_data_windowing(aggregation_config)
  )
  
  return(aggregation)
}

#' Create data grouping
#'
#' @param aggregation_config Aggregation configuration
#' @return Data grouping
create_data_grouping <- function(aggregation_config) {
  data_grouping <- c(
    "# Data Grouping",
    "library(data.table)",
    "library(dplyr)",
    "",
    "# Group data efficiently",
    "group_data <- function(data, group_cols, agg_functions) {",
    "  # Convert to data.table for efficiency",
    "  data <- as.data.table(data)",
    "  ",
    "  # Group and aggregate",
    "  result <- data[, lapply(agg_functions, function(f) f(.SD)), by = group_cols]",
    "  ",
    "  return(result)",
    "}",
    "",
    "# Group by category and calculate statistics",
    "grouped_data <- group_data(",
    "  data,",
    "  group_cols = c(\"category\", \"status\"),",
    "  agg_functions = list(",
    "    count = length,",
    "    mean_value = function(x) mean(x$value, na.rm = TRUE),",
    "    median_value = function(x) median(x$value, na.rm = TRUE),",
    "    sd_value = function(x) sd(x$value, na.rm = TRUE)",
    "  )",
    ")",
    "",
    "# Group by time periods",
    "time_grouped_data <- group_data(",
    "  data,",
    "  group_cols = c(\"year\", \"month\"),",
    "  agg_functions = list(",
    "    count = length,",
    "    sum_value = function(x) sum(x$value, na.rm = TRUE)",
    "  )",
    ")"
  )
  
  return(data_grouping)
}

#' Create data summarization
#'
#' @param aggregation_config Aggregation configuration
#' @return Data summarization
create_data_summarization <- function(aggregation_config) {
  data_summarization <- c(
    "# Data Summarization",
    "library(data.table)",
    "library(dplyr)",
    "",
    "# Summarize data",
    "summarize_data <- function(data, summary_functions) {",
    "  # Convert to data.table",
    "  data <- as.data.table(data)",
    "  ",
    "  # Calculate summary statistics",
    "  summary_stats <- data[, lapply(summary_functions, function(f) f(.SD))]",
    "  ",
    "  return(summary_stats)",
    "}",
    "",
    "# Summarize numeric columns",
    "numeric_summary <- summarize_data(",
    "  data,",
    "  summary_functions = list(",
    "    count = length,",
    "    mean = function(x) mean(x, na.rm = TRUE),",
    "    median = function(x) median(x, na.rm = TRUE),",
    "    sd = function(x) sd(x, na.rm = TRUE),",
    "    min = function(x) min(x, na.rm = TRUE),",
    "    max = function(x) max(x, na.rm = TRUE),",
    "    q25 = function(x) quantile(x, 0.25, na.rm = TRUE),",
    "    q75 = function(x) quantile(x, 0.75, na.rm = TRUE)",
    "  )",
    ")",
    "",
    "# Summarize categorical columns",
    "categorical_summary <- summarize_data(",
    "  data,",
    "  summary_functions = list(",
    "    count = length,",
    "    unique_count = function(x) length(unique(x)),",
    "    most_common = function(x) names(sort(table(x), decreasing = TRUE))[1]",
    "  )",
    ")"
  )
  
  return(data_summarization)
}

#' Create data windowing
#'
#' @param aggregation_config Aggregation configuration
#' @return Data windowing
create_data_windowing <- function(aggregation_config) {
  data_windowing <- c(
    "# Data Windowing",
    "library(data.table)",
    "library(dplyr)",
    "library(zoo)",
    "",
    "# Create rolling windows",
    "create_rolling_windows <- function(data, window_size, function_name) {",
    "  # Sort data by time",
    "  data <- data[order(data$timestamp)]",
    "  ",
    "  # Create rolling windows",
    "  data$rolling_mean <- rollapply(data$value, width = window_size, FUN = mean, fill = NA, align = \"right\")",
    "  data$rolling_median <- rollapply(data$value, width = window_size, FUN = median, fill = NA, align = \"right\")",
    "  data$rolling_sd <- rollapply(data$value, width = window_size, FUN = sd, fill = NA, align = \"right\")",
    "  ",
    "  return(data)",
    "}",
    "",
    "# Create expanding windows",
    "create_expanding_windows <- function(data, function_name) {",
    "  # Sort data by time",
    "  data <- data[order(data$timestamp)]",
    "  ",
    "  # Create expanding windows",
    "  data$expanding_mean <- cummean(data$value)",
    "  data$expanding_sum <- cumsum(data$value)",
    "  data$expanding_sd <- cumvar(data$value, na.rm = TRUE)",
    "  ",
    "  return(data)",
    "}",
    "",
    "# Create windows",
    "windowed_data <- create_rolling_windows(data, window_size = 7, function_name = \"mean\")",
    "expanded_data <- create_expanding_windows(data, function_name = \"mean\")"
  )
  
  return(data_windowing)
}
```

## Parallel Processing

### Parallel Data Processing

```r
# R/03-parallel-processing.R

#' Create parallel data processing
#'
#' @param parallel_config Parallel configuration
#' @return Parallel data processing
create_parallel_data_processing <- function(parallel_config) {
  parallel <- list(
    parallel_apply = create_parallel_apply(parallel_config),
    parallel_aggregation = create_parallel_aggregation(parallel_config),
    parallel_io = create_parallel_io(parallel_config)
  )
  
  return(parallel)
}

#' Create parallel apply
#'
#' @param parallel_config Parallel configuration
#' @return Parallel apply
create_parallel_apply <- function(parallel_config) {
  parallel_apply <- c(
    "# Parallel Apply",
    "library(parallel)",
    "library(foreach)",
    "library(doParallel)",
    "",
    "# Register parallel backend",
    "registerDoParallel(cores = parallel::detectCores())",
    "",
    "# Parallel apply function",
    "parallel_apply <- function(data, function_name, chunk_size = 1000) {",
    "  # Split data into chunks",
    "  n_chunks <- ceiling(nrow(data) / chunk_size)",
    "  chunks <- split(data, rep(1:n_chunks, each = chunk_size, length.out = nrow(data)))",
    "  ",
    "  # Process chunks in parallel",
    "  results <- foreach(chunk = chunks, .combine = rbind) %dopar% {",
    "    function_name(chunk)",
    "  }",
    "  ",
    "  return(results)",
    "}",
    "",
    "# Parallel apply with progress bar",
    "parallel_apply_with_progress <- function(data, function_name, chunk_size = 1000) {",
    "  # Split data into chunks",
    "  n_chunks <- ceiling(nrow(data) / chunk_size)",
    "  chunks <- split(data, rep(1:n_chunks, each = chunk_size, length.out = nrow(data)))",
    "  ",
    "  # Process chunks in parallel with progress",
    "  results <- foreach(chunk = chunks, .combine = rbind, .packages = c(\"data.table\", \"dplyr\")) %dopar% {",
    "    function_name(chunk)",
    "  }",
    "  ",
    "  return(results)",
    "}",
    "",
    "# Use parallel apply",
    "processed_data <- parallel_apply(data, function_name = clean_data)"
  )
  
  return(parallel_apply)
}

#' Create parallel aggregation
#'
#' @param parallel_config Parallel configuration
#' @return Parallel aggregation
create_parallel_aggregation <- function(parallel_config) {
  parallel_aggregation <- c(
    "# Parallel Aggregation",
    "library(parallel)",
    "library(foreach)",
    "library(doParallel)",
    "library(data.table)",
    "",
    "# Parallel aggregation",
    "parallel_aggregation <- function(data, group_cols, agg_functions, chunk_size = 1000) {",
    "  # Split data into chunks",
    "  n_chunks <- ceiling(nrow(data) / chunk_size)",
    "  chunks <- split(data, rep(1:n_chunks, each = chunk_size, length.out = nrow(data)))",
    "  ",
    "  # Process chunks in parallel",
    "  chunk_results <- foreach(chunk = chunks, .combine = list, .packages = c(\"data.table\")) %dopar% {",
    "    chunk_dt <- as.data.table(chunk)",
    "    chunk_dt[, lapply(agg_functions, function(f) f(.SD)), by = group_cols]",
    "  }",
    "  ",
    "  # Combine results",
    "  combined_results <- rbindlist(chunk_results)",
    "  ",
    "  # Final aggregation",
    "  final_results <- combined_results[, lapply(agg_functions, function(f) f(.SD)), by = group_cols]",
    "  ",
    "  return(final_results)",
    "}",
    "",
    "# Use parallel aggregation",
    "aggregated_data <- parallel_aggregation(",
    "  data,",
    "  group_cols = c(\"category\", \"status\"),",
    "  agg_functions = list(",
    "    count = length,",
    "    mean_value = function(x) mean(x$value, na.rm = TRUE),",
    "    sum_value = function(x) sum(x$value, na.rm = TRUE)",
    "  )",
    ")"
  )
  
  return(parallel_aggregation)
}

#' Create parallel I/O
#'
#' @param parallel_config Parallel configuration
#' @return Parallel I/O
create_parallel_io <- function(parallel_config) {
  parallel_io <- c(
    "# Parallel I/O",
    "library(parallel)",
    "library(foreach)",
    "library(doParallel)",
    "library(data.table)",
    "",
    "# Parallel file reading",
    "parallel_read_files <- function(file_paths, read_function) {",
    "  # Process files in parallel",
    "  results <- foreach(file_path = file_paths, .combine = rbind, .packages = c(\"data.table\", \"readr\")) %dopar% {",
    "    read_function(file_path)",
    "  }",
    "  ",
    "  return(results)",
    "}",
    "",
    "# Parallel file writing",
    "parallel_write_files <- function(data, output_dir, chunk_size = 1000) {",
    "  # Split data into chunks",
    "  n_chunks <- ceiling(nrow(data) / chunk_size)",
    "  chunks <- split(data, rep(1:n_chunks, each = chunk_size, length.out = nrow(data)))",
    "  ",
    "  # Write chunks in parallel",
    "  foreach(chunk = chunks, .packages = c(\"data.table\", \"readr\")) %dopar% {",
    "    chunk_id <- which(chunks == chunk)",
    "    output_path <- file.path(output_dir, paste0(\"chunk_\", chunk_id, \".csv\"))",
    "    fwrite(chunk, output_path)",
    "  }",
    "}",
    "",
    "# Use parallel I/O",
    "file_paths <- list.files(\"data/\", pattern = \"*.csv\", full.names = TRUE)",
    "data <- parallel_read_files(file_paths, read_function = fread)",
    "parallel_write_files(data, output_dir = \"output/\")"
  )
  
  return(parallel_io)
}
```

## Memory Management

### Memory Optimization

```r
# R/04-memory-management.R

#' Create memory optimization
#'
#' @param memory_config Memory configuration
#' @return Memory optimization
create_memory_optimization <- function(memory_config) {
  memory <- list(
    memory_monitoring = create_memory_monitoring(memory_config),
    memory_cleanup = create_memory_cleanup(memory_config),
    memory_efficient_operations = create_memory_efficient_operations(memory_config)
  )
  
  return(memory)
}

#' Create memory monitoring
#'
#' @param memory_config Memory configuration
#' @return Memory monitoring
create_memory_monitoring <- function(memory_config) {
  memory_monitoring <- c(
    "# Memory Monitoring",
    "library(pryr)",
    "library(profmem)",
    "",
    "# Monitor memory usage",
    "monitor_memory <- function() {",
    "  memory_info <- list(",
    "    total_memory = memory.size(),",
    "    used_memory = memory.size() - memory.size(max = FALSE),",
    "    available_memory = memory.size(max = FALSE),",
    "    gc_memory = gc()",
    "  )",
    "  ",
    "  return(memory_info)",
    "}",
    "",
    "# Monitor object sizes",
    "monitor_object_sizes <- function() {",
    "  objects <- ls(envir = .GlobalEnv)",
    "  sizes <- sapply(objects, function(x) object.size(get(x)))",
    "  sizes <- sort(sizes, decreasing = TRUE)",
    "  ",
    "  return(sizes)",
    "}",
    "",
    "# Monitor memory usage",
    "memory_info <- monitor_memory()",
    "object_sizes <- monitor_object_sizes()",
    "print(memory_info)",
    "print(head(object_sizes))"
  )
  
  return(memory_monitoring)
}

#' Create memory cleanup
#'
#' @param memory_config Memory configuration
#' @return Memory cleanup
create_memory_cleanup <- function(memory_config) {
  memory_cleanup <- c(
    "# Memory Cleanup",
    "library(pryr)",
    "",
    "# Clean up memory",
    "cleanup_memory <- function() {",
    "  # Force garbage collection",
    "  gc(verbose = TRUE)",
    "  ",
    "  # Clear large objects",
    "  large_objects <- ls(envir = .GlobalEnv)[sapply(ls(envir = .GlobalEnv), function(x) object.size(get(x))) > 100 * 1024 * 1024]",
    "  if (length(large_objects) > 0) {",
    "    rm(list = large_objects, envir = .GlobalEnv)",
    "    gc(verbose = TRUE)",
    "  }",
    "  ",
    "  return(TRUE)",
    "}",
    "",
    "# Clean up memory periodically",
    "cleanup_memory_periodically <- function(interval = 300) {",
    "  while (TRUE) {",
    "    Sys.sleep(interval)",
    "    cleanup_memory()",
    "  }",
    "}",
    "",
    "# Use memory cleanup",
    "cleanup_memory()"
  )
  
  return(memory_cleanup)
}

#' Create memory efficient operations
#'
#' @param memory_config Memory configuration
#' @return Memory efficient operations
create_memory_efficient_operations <- function(memory_config) {
  memory_efficient_operations <- c(
    "# Memory Efficient Operations",
    "library(data.table)",
    "library(dplyr)",
    "",
    "# Process data in chunks",
    "process_data_in_chunks <- function(data, function_name, chunk_size = 1000) {",
    "  n_chunks <- ceiling(nrow(data) / chunk_size)",
    "  results <- list()",
    "  ",
    "  for (i in 1:n_chunks) {",
    "    start_idx <- (i - 1) * chunk_size + 1",
    "    end_idx <- min(i * chunk_size, nrow(data))",
    "    ",
    "    chunk <- data[start_idx:end_idx, ]",
    "    result <- function_name(chunk)",
    "    results[[i]] <- result",
    "    ",
    "    # Clean up chunk",
    "    rm(chunk)",
    "    gc()",
    "  }",
    "  ",
    "  # Combine results",
    "  combined_results <- rbindlist(results)",
    "  ",
    "  return(combined_results)",
    "}",
    "",
    "# Use memory efficient operations",
    "processed_data <- process_data_in_chunks(data, function_name = clean_data)"
  )
  
  return(memory_efficient_operations)
}
```

## TL;DR Runbook

### Quick Start

```r
# 1. Create efficient data loading
loading <- create_efficient_data_loading(loading_config)

# 2. Create data validation
validation <- create_data_validation(validation_config)

# 3. Create efficient data transformation
transformation <- create_efficient_data_transformation(transformation_config)

# 4. Create data aggregation
aggregation <- create_data_aggregation(aggregation_config)

# 5. Create parallel data processing
parallel <- create_parallel_data_processing(parallel_config)

# 6. Create memory optimization
memory <- create_memory_optimization(memory_config)
```

### Essential Patterns

```r
# Complete data processing pipeline
create_data_processing_pipeline <- function(pipeline_config) {
  # Create data loading
  loading <- create_efficient_data_loading(pipeline_config$loading_config)
  
  # Create data validation
  validation <- create_data_validation(pipeline_config$validation_config)
  
  # Create data transformation
  transformation <- create_efficient_data_transformation(pipeline_config$transformation_config)
  
  # Create data aggregation
  aggregation <- create_data_aggregation(pipeline_config$aggregation_config)
  
  # Create parallel processing
  parallel <- create_parallel_data_processing(pipeline_config$parallel_config)
  
  # Create memory optimization
  memory <- create_memory_optimization(pipeline_config$memory_config)
  
  return(list(
    loading = loading,
    validation = validation,
    transformation = transformation,
    aggregation = aggregation,
    parallel = parallel,
    memory = memory
  ))
}
```

---

*This guide provides the complete machinery for implementing data processing for R applications. Each pattern includes implementation examples, optimization strategies, and real-world usage patterns for enterprise data processing systems.*

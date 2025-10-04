# R Data Exploration Best Practices

**Objective**: Master senior-level R data exploration patterns for production systems. When you need to understand complex datasets, when you want to discover patterns and insights, when you need enterprise-grade exploration patterns—these best practices become your weapon of choice.

## Core Principles

- **Systematic Approach**: Follow a structured exploration methodology
- **Visualization First**: Use visualizations to understand data patterns
- **Statistical Summary**: Complement visuals with statistical summaries
- **Data Quality**: Assess and document data quality issues
- **Documentation**: Document findings and insights

## Data Overview and Structure

### Initial Data Inspection

```r
# R/01-data-exploration.R

#' Comprehensive data overview
#'
#' @param data Data frame to explore
#' @return List containing data overview information
data_overview <- function(data) {
  overview <- list(
    dimensions = dim(data),
    column_names = names(data),
    data_types = sapply(data, class),
    missing_values = colSums(is.na(data)),
    unique_values = sapply(data, function(x) length(unique(x))),
    memory_usage = object.size(data)
  )
  
  # Add summary statistics for numeric columns
  numeric_cols <- sapply(data, is.numeric)
  if (sum(numeric_cols) > 0) {
    overview$numeric_summary <- summary(data[, numeric_cols, drop = FALSE])
  }
  
  # Add frequency tables for categorical columns
  categorical_cols <- sapply(data, function(x) is.character(x) || is.factor(x))
  if (sum(categorical_cols) > 0) {
    overview$categorical_summary <- lapply(data[, categorical_cols, drop = FALSE], table)
  }
  
  return(overview)
}

#' Print data overview in a formatted way
#'
#' @param overview Data overview list
print_data_overview <- function(overview) {
  cat("Data Overview\n")
  cat("=============\n")
  cat("Dimensions:", overview$dimensions[1], "rows x", overview$dimensions[2], "columns\n")
  cat("Memory usage:", format(overview$memory_usage, units = "MB"), "\n\n")
  
  cat("Column Information\n")
  cat("------------------\n")
  for (i in seq_along(overview$column_names)) {
    col_name <- overview$column_names[i]
    col_type <- overview$data_types[i]
    missing_count <- overview$missing_values[i]
    unique_count <- overview$unique_values[i]
    
    cat(sprintf("%-20s %-10s Missing: %-5d Unique: %-5d\n", 
                col_name, col_type, missing_count, unique_count))
  }
  
  if (!is.null(overview$numeric_summary)) {
    cat("\nNumeric Summary\n")
    cat("---------------\n")
    print(overview$numeric_summary)
  }
  
  if (!is.null(overview$categorical_summary)) {
    cat("\nCategorical Summary\n")
    cat("-------------------\n")
    for (i in seq_along(overview$categorical_summary)) {
      cat("\n", names(overview$categorical_summary)[i], ":\n")
      print(overview$categorical_summary[[i]])
    }
  }
}
```

### Data Quality Assessment

```r
# R/01-data-exploration.R (continued)

#' Assess data quality
#'
#' @param data Data frame to assess
#' @return Data quality report
assess_data_quality <- function(data) {
  quality_report <- list(
    completeness = assess_completeness(data),
    consistency = assess_consistency(data),
    accuracy = assess_accuracy(data),
    validity = assess_validity(data)
  )
  
  return(quality_report)
}

#' Assess data completeness
#'
#' @param data Data frame
#' @return Completeness metrics
assess_completeness <- function(data) {
  total_cells <- nrow(data) * ncol(data)
  missing_cells <- sum(is.na(data))
  
  completeness <- list(
    total_cells = total_cells,
    missing_cells = missing_cells,
    completeness_rate = (total_cells - missing_cells) / total_cells,
    columns_with_missing = names(data)[colSums(is.na(data)) > 0],
    rows_with_missing = sum(rowSums(is.na(data)) > 0),
    missing_patterns = identify_missing_patterns(data)
  )
  
  return(completeness)
}

#' Identify missing value patterns
#'
#' @param data Data frame
#' @return Missing value patterns
identify_missing_patterns <- function(data) {
  missing_matrix <- is.na(data)
  
  # Find columns that are always missing together
  missing_correlations <- cor(missing_matrix)
  missing_correlations[is.na(missing_correlations)] <- 0
  
  # Find rows with similar missing patterns
  missing_patterns <- table(apply(missing_matrix, 1, paste, collapse = ""))
  
  return(list(
    missing_correlations = missing_correlations,
    missing_patterns = missing_patterns
  ))
}

#' Assess data consistency
#'
#' @param data Data frame
#' @return Consistency metrics
assess_consistency <- function(data) {
  consistency <- list(
    duplicate_rows = sum(duplicated(data)),
    inconsistent_categories = identify_inconsistent_categories(data),
    outlier_columns = identify_outlier_columns(data)
  )
  
  return(consistency)
}

#' Identify inconsistent categories
#'
#' @param data Data frame
#' @return Inconsistent categories
identify_inconsistent_categories <- function(data) {
  categorical_cols <- sapply(data, function(x) is.character(x) || is.factor(x))
  inconsistencies <- list()
  
  for (col in names(data)[categorical_cols]) {
    values <- data[[col]]
    unique_values <- unique(values[!is.na(values)])
    
    # Check for case inconsistencies
    case_insensitive <- tolower(unique_values)
    if (length(unique_values) != length(unique(case_insensitive))) {
      inconsistencies[[col]] <- "Case inconsistencies detected"
    }
    
    # Check for whitespace inconsistencies
    trimmed_values <- trimws(unique_values)
    if (length(unique_values) != length(unique(trimmed_values))) {
      inconsistencies[[col]] <- c(inconsistencies[[col]], "Whitespace inconsistencies detected")
    }
  }
  
  return(inconsistencies)
}

#' Identify columns with outliers
#'
#' @param data Data frame
#' @return Outlier information
identify_outlier_columns <- function(data) {
  numeric_cols <- sapply(data, is.numeric)
  outlier_info <- list()
  
  for (col in names(data)[numeric_cols]) {
    values <- data[[col]][!is.na(data[[col]])]
    if (length(values) > 0) {
      Q1 <- quantile(values, 0.25)
      Q3 <- quantile(values, 0.75)
      IQR <- Q3 - Q1
      lower_bound <- Q1 - 1.5 * IQR
      upper_bound <- Q3 + 1.5 * IQR
      
      outliers <- values[values < lower_bound | values > upper_bound]
      if (length(outliers) > 0) {
        outlier_info[[col]] <- list(
          count = length(outliers),
          percentage = length(outliers) / length(values) * 100,
          values = outliers
        )
      }
    }
  }
  
  return(outlier_info)
}
```

## Statistical Exploration

### Descriptive Statistics

```r
# R/02-statistical-exploration.R

#' Comprehensive descriptive statistics
#'
#' @param data Data frame
#' @param group_by Column to group by (optional)
#' @return Descriptive statistics
descriptive_statistics <- function(data, group_by = NULL) {
  numeric_cols <- sapply(data, is.numeric)
  
  if (sum(numeric_cols) == 0) {
    return(NULL)
  }
  
  numeric_data <- data[, numeric_cols, drop = FALSE]
  
  if (is.null(group_by)) {
    stats <- calculate_basic_stats(numeric_data)
  } else {
    stats <- calculate_grouped_stats(numeric_data, data[[group_by]])
  }
  
  return(stats)
}

#' Calculate basic statistics
#'
#' @param data Numeric data frame
#' @return Basic statistics
calculate_basic_stats <- function(data) {
  stats <- data.frame(
    variable = names(data),
    n = sapply(data, function(x) sum(!is.na(x))),
    mean = sapply(data, mean, na.rm = TRUE),
    median = sapply(data, median, na.rm = TRUE),
    sd = sapply(data, sd, na.rm = TRUE),
    min = sapply(data, min, na.rm = TRUE),
    max = sapply(data, max, na.rm = TRUE),
    q25 = sapply(data, quantile, 0.25, na.rm = TRUE),
    q75 = sapply(data, quantile, 0.75, na.rm = TRUE),
    skewness = sapply(data, calculate_skewness),
    kurtosis = sapply(data, calculate_kurtosis),
    stringsAsFactors = FALSE
  )
  
  return(stats)
}

#' Calculate grouped statistics
#'
#' @param data Numeric data frame
#' @param group_by Grouping variable
#' @return Grouped statistics
calculate_grouped_stats <- function(data, group_by) {
  groups <- unique(group_by[!is.na(group_by)])
  stats_list <- list()
  
  for (group in groups) {
    group_data <- data[group_by == group, , drop = FALSE]
    stats_list[[as.character(group)]] <- calculate_basic_stats(group_data)
  }
  
  return(stats_list)
}

#' Calculate skewness
#'
#' @param x Numeric vector
#' @return Skewness value
calculate_skewness <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) < 3) return(NA)
  
  mean_x <- mean(x)
  sd_x <- sd(x)
  if (sd_x == 0) return(0)
  
  skewness <- sum(((x - mean_x) / sd_x)^3) / length(x)
  return(skewness)
}

#' Calculate kurtosis
#'
#' @param x Numeric vector
#' @return Kurtosis value
calculate_kurtosis <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) < 4) return(NA)
  
  mean_x <- mean(x)
  sd_x <- sd(x)
  if (sd_x == 0) return(0)
  
  kurtosis <- sum(((x - mean_x) / sd_x)^4) / length(x) - 3
  return(kurtosis)
}
```

### Correlation Analysis

```r
# R/02-statistical-exploration.R (continued)

#' Comprehensive correlation analysis
#'
#' @param data Data frame
#' @param method Correlation method
#' @return Correlation analysis results
correlation_analysis <- function(data, method = "pearson") {
  numeric_cols <- sapply(data, is.numeric)
  
  if (sum(numeric_cols) < 2) {
    return(NULL)
  }
  
  numeric_data <- data[, numeric_cols, drop = FALSE]
  
  # Calculate correlation matrix
  cor_matrix <- cor(numeric_data, use = "complete.obs", method = method)
  
  # Find high correlations
  high_correlations <- find_high_correlations(cor_matrix)
  
  # Calculate partial correlations
  partial_correlations <- calculate_partial_correlations(numeric_data)
  
  # Test correlation significance
  significance_tests <- test_correlation_significance(numeric_data, method)
  
  results <- list(
    correlation_matrix = cor_matrix,
    high_correlations = high_correlations,
    partial_correlations = partial_correlations,
    significance_tests = significance_tests
  )
  
  return(results)
}

#' Find high correlations
#'
#' @param cor_matrix Correlation matrix
#' @param threshold Correlation threshold
#' @return High correlation pairs
find_high_correlations <- function(cor_matrix, threshold = 0.7) {
  # Remove diagonal and get upper triangle
  cor_matrix[upper.tri(cor_matrix, diag = TRUE)] <- NA
  
  # Find high correlations
  high_cor <- which(abs(cor_matrix) > threshold, arr.ind = TRUE)
  
  if (nrow(high_cor) == 0) {
    return(data.frame())
  }
  
  results <- data.frame(
    variable1 = rownames(cor_matrix)[high_cor[, 1]],
    variable2 = colnames(cor_matrix)[high_cor[, 2]],
    correlation = cor_matrix[high_cor],
    stringsAsFactors = FALSE
  )
  
  # Sort by absolute correlation
  results <- results[order(abs(results$correlation), decreasing = TRUE), ]
  
  return(results)
}

#' Calculate partial correlations
#'
#' @param data Numeric data frame
#' @return Partial correlation matrix
calculate_partial_correlations <- function(data) {
  if (ncol(data) < 3) {
    return(NULL)
  }
  
  # Remove rows with missing values
  complete_data <- data[complete.cases(data), ]
  
  if (nrow(complete_data) < 3) {
    return(NULL)
  }
  
  # Calculate partial correlations
  partial_cor <- corpcor::cor2pcor(cor(complete_data))
  rownames(partial_cor) <- colnames(partial_cor) <- names(complete_data)
  
  return(partial_cor)
}

#' Test correlation significance
#'
#' @param data Numeric data frame
#' @param method Correlation method
#' @return Significance test results
test_correlation_significance <- function(data, method = "pearson") {
  n_vars <- ncol(data)
  results <- data.frame(
    variable1 = character(0),
    variable2 = character(0),
    correlation = numeric(0),
    p_value = numeric(0),
    significant = logical(0),
    stringsAsFactors = FALSE
  )
  
  for (i in 1:(n_vars - 1)) {
    for (j in (i + 1):n_vars) {
      var1 <- names(data)[i]
      var2 <- names(data)[j]
      
      # Remove missing values
      complete_cases <- complete.cases(data[[var1]], data[[var2]])
      x <- data[[var1]][complete_cases]
      y <- data[[var2]][complete_cases]
      
      if (length(x) > 2) {
        test_result <- cor.test(x, y, method = method)
        
        results <- rbind(results, data.frame(
          variable1 = var1,
          variable2 = var2,
          correlation = test_result$estimate,
          p_value = test_result$p.value,
          significant = test_result$p.value < 0.05,
          stringsAsFactors = FALSE
        ))
      }
    }
  }
  
  return(results)
}
```

## Visualization for Exploration

### Distribution Visualizations

```r
# R/03-visualization-exploration.R

#' Create comprehensive distribution plots
#'
#' @param data Data frame
#' @param variables Variables to plot
#' @return List of distribution plots
create_distribution_plots <- function(data, variables = NULL) {
  if (is.null(variables)) {
    numeric_cols <- sapply(data, is.numeric)
    variables <- names(data)[numeric_cols]
  }
  
  plots <- list()
  
  for (var in variables) {
    if (var %in% names(data)) {
      plots[[var]] <- create_single_distribution_plot(data, var)
    }
  }
  
  return(plots)
}

#' Create single distribution plot
#'
#' @param data Data frame
#' @param variable Variable to plot
#' @return Distribution plot
create_single_distribution_plot <- function(data, variable) {
  p <- ggplot2::ggplot(data, aes_string(x = variable)) +
    ggplot2::geom_histogram(aes(y = ..density..), bins = 30, alpha = 0.7, fill = "steelblue") +
    ggplot2::geom_density(alpha = 0.5, color = "red", size = 1) +
    ggplot2::geom_vline(aes_string(xintercept = paste0("mean(", variable, ", na.rm = TRUE)")), 
                        color = "green", linetype = "dashed", size = 1) +
    ggplot2::geom_vline(aes_string(xintercept = paste0("median(", variable, ", na.rm = TRUE)")), 
                        color = "orange", linetype = "dashed", size = 1) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = paste("Distribution of", variable),
      x = variable,
      y = "Density"
    )
  
  return(p)
}

#' Create box plots for multiple variables
#'
#' @param data Data frame
#' @param variables Variables to plot
#' @return Box plot
create_box_plots <- function(data, variables) {
  # Reshape data for box plot
  plot_data <- data[, variables, drop = FALSE]
  plot_data$id <- 1:nrow(plot_data)
  
  plot_data_long <- tidyr::pivot_longer(plot_data, cols = all_of(variables), 
                                       names_to = "variable", values_to = "value")
  
  p <- ggplot2::ggplot(plot_data_long, aes(x = variable, y = value)) +
    ggplot2::geom_boxplot(aes(fill = variable), alpha = 0.7) +
    ggplot2::geom_jitter(alpha = 0.3, width = 0.2) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = "Box Plots of Variables",
      x = "Variable",
      y = "Value"
    ) +
    ggplot2::theme(legend.position = "none")
  
  return(p)
}
```

### Relationship Visualizations

```r
# R/03-visualization-exploration.R (continued)

#' Create scatter plot matrix
#'
#' @param data Data frame
#' @param variables Variables to include
#' @return Scatter plot matrix
create_scatter_plot_matrix <- function(data, variables) {
  numeric_data <- data[, variables, drop = FALSE]
  
  # Create pairs plot
  p <- GGally::ggpairs(numeric_data, 
                       lower = list(continuous = GGally::wrap("points", alpha = 0.3)),
                       upper = list(continuous = GGally::wrap("cor", size = 3)),
                       diag = list(continuous = GGally::wrap("densityDiag", alpha = 0.7)))
  
  return(p)
}

#' Create correlation heatmap
#'
#' @param cor_matrix Correlation matrix
#' @return Correlation heatmap
create_correlation_heatmap <- function(cor_matrix) {
  # Convert to long format
  cor_data <- reshape2::melt(cor_matrix)
  
  p <- ggplot2::ggplot(cor_data, aes(x = Var1, y = Var2, fill = value)) +
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
  
  return(p)
}

#' Create time series plots
#'
#' @param data Data frame
#' @param time_var Time variable
#' @param value_vars Value variables
#' @return Time series plots
create_time_series_plots <- function(data, time_var, value_vars) {
  plots <- list()
  
  for (var in value_vars) {
    p <- ggplot2::ggplot(data, aes_string(x = time_var, y = var)) +
      ggplot2::geom_line() +
      ggplot2::geom_point(alpha = 0.5) +
      ggplot2::theme_minimal() +
      ggplot2::labs(
        title = paste("Time Series of", var),
        x = time_var,
        y = var
      )
    
    plots[[var]] <- p
  }
  
  return(plots)
}
```

## Advanced Exploration Techniques

### Dimensionality Reduction

```r
# R/04-advanced-exploration.R

#' Perform PCA analysis
#'
#' @param data Data frame
#' @param variables Variables to include
#' @return PCA results
perform_pca_analysis <- function(data, variables) {
  numeric_data <- data[, variables, drop = FALSE]
  
  # Remove rows with missing values
  complete_data <- numeric_data[complete.cases(numeric_data), ]
  
  if (nrow(complete_data) < 2) {
    return(NULL)
  }
  
  # Perform PCA
  pca_result <- prcomp(complete_data, scale = TRUE)
  
  # Calculate variance explained
  variance_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)
  cumulative_variance <- cumsum(variance_explained)
  
  # Create results
  results <- list(
    pca_result = pca_result,
    variance_explained = variance_explained,
    cumulative_variance = cumulative_variance,
    loadings = pca_result$rotation,
    scores = pca_result$x
  )
  
  return(results)
}

#' Create PCA visualization
#'
#' @param pca_results PCA results
#' @param data Original data
#' @return PCA plots
create_pca_visualization <- function(pca_results, data) {
  plots <- list()
  
  # Scree plot
  scree_data <- data.frame(
    component = 1:length(pca_results$variance_explained),
    variance_explained = pca_results$variance_explained,
    cumulative_variance = pca_results$cumulative_variance
  )
  
  plots$scree <- ggplot2::ggplot(scree_data, aes(x = component, y = variance_explained)) +
    ggplot2::geom_line() +
    ggplot2::geom_point() +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = "PCA Scree Plot",
      x = "Principal Component",
      y = "Variance Explained"
    )
  
  # Biplot
  biplot_data <- data.frame(
    PC1 = pca_results$scores[, 1],
    PC2 = pca_results$scores[, 2]
  )
  
  plots$biplot <- ggplot2::ggplot(biplot_data, aes(x = PC1, y = PC2)) +
    ggplot2::geom_point(alpha = 0.5) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = "PCA Biplot",
      x = "PC1",
      y = "PC2"
    )
  
  return(plots)
}
```

### Clustering Analysis

```r
# R/04-advanced-exploration.R (continued)

#' Perform clustering analysis
#'
#' @param data Data frame
#' @param variables Variables to include
#' @param method Clustering method
#' @return Clustering results
perform_clustering_analysis <- function(data, variables, method = "kmeans") {
  numeric_data <- data[, variables, drop = FALSE]
  
  # Remove rows with missing values
  complete_data <- numeric_data[complete.cases(numeric_data), ]
  
  if (nrow(complete_data) < 2) {
    return(NULL)
  }
  
  # Scale data
  scaled_data <- scale(complete_data)
  
  if (method == "kmeans") {
    # Determine optimal number of clusters
    optimal_k <- determine_optimal_clusters(scaled_data)
    
    # Perform k-means clustering
    cluster_result <- kmeans(scaled_data, centers = optimal_k, nstart = 25)
    
    results <- list(
      method = "kmeans",
      optimal_k = optimal_k,
      clusters = cluster_result$cluster,
      centers = cluster_result$centers,
      withinss = cluster_result$withinss,
      totss = cluster_result$totss,
      betweenss = cluster_result$betweenss
    )
  } else if (method == "hierarchical") {
    # Perform hierarchical clustering
    dist_matrix <- dist(scaled_data)
    hclust_result <- hclust(dist_matrix)
    
    # Cut tree to get clusters
    optimal_k <- determine_optimal_clusters(scaled_data)
    clusters <- cutree(hclust_result, k = optimal_k)
    
    results <- list(
      method = "hierarchical",
      optimal_k = optimal_k,
      clusters = clusters,
      hclust_result = hclust_result
    )
  }
  
  return(results)
}

#' Determine optimal number of clusters
#'
#' @param data Scaled data
#' @return Optimal number of clusters
determine_optimal_clusters <- function(data) {
  max_k <- min(10, nrow(data) - 1)
  
  if (max_k < 2) {
    return(2)
  }
  
  # Calculate within-cluster sum of squares for different k
  wss <- numeric(max_k)
  for (k in 1:max_k) {
    wss[k] <- sum(kmeans(data, centers = k, nstart = 25)$withinss)
  }
  
  # Find elbow point
  diffs <- diff(wss)
  optimal_k <- which.max(diffs) + 1
  
  return(optimal_k)
}
```

## TL;DR Runbook

### Quick Start

```r
# 1. Data overview
overview <- data_overview(data)
print_data_overview(overview)

# 2. Data quality assessment
quality <- assess_data_quality(data)
print(quality)

# 3. Descriptive statistics
stats <- descriptive_statistics(data)
print(stats)

# 4. Correlation analysis
correlations <- correlation_analysis(data)
print(correlations$high_correlations)

# 5. Visualizations
dist_plots <- create_distribution_plots(data)
cor_heatmap <- create_correlation_heatmap(correlations$correlation_matrix)
```

### Essential Patterns

```r
# Systematic exploration
explore_data <- function(data) {
  # Overview
  overview <- data_overview(data)
  
  # Quality
  quality <- assess_data_quality(data)
  
  # Statistics
  stats <- descriptive_statistics(data)
  
  # Correlations
  correlations <- correlation_analysis(data)
  
  # Visualizations
  plots <- create_distribution_plots(data)
  
  return(list(
    overview = overview,
    quality = quality,
    statistics = stats,
    correlations = correlations,
    plots = plots
  ))
}
```

---

*This guide provides the complete machinery for comprehensive data exploration in R. Each pattern includes implementation examples, visualization strategies, and real-world usage patterns for enterprise deployment.*

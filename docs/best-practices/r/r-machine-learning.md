# R Machine Learning Best Practices

**Objective**: Master senior-level R machine learning patterns for production systems. When you need to build robust, scalable ML models, when you want to follow best practices for model development and deployment, when you need enterprise-grade ML patterns—these best practices become your weapon of choice.

## Core Principles

- **Data Preprocessing**: Prepare data for machine learning algorithms
- **Model Selection**: Choose appropriate algorithms for the problem
- **Validation**: Use proper validation techniques to assess model performance
- **Hyperparameter Tuning**: Optimize model parameters systematically
- **Model Deployment**: Deploy models for production use

## Data Preprocessing

### Feature Engineering

```r
# R/01-data-preprocessing.R

#' Comprehensive feature engineering
#'
#' @param data Data frame
#' @param target_variable Target variable name
#' @param preprocessing_config Preprocessing configuration
#' @return Preprocessed data and preprocessing objects
feature_engineering <- function(data, target_variable, preprocessing_config) {
  # Remove target variable from features
  feature_data <- data[, !names(data) %in% target_variable, drop = FALSE]
  target_data <- data[[target_variable]]
  
  # Handle missing values
  if (preprocessing_config$handle_missing) {
    feature_data <- handle_missing_values(feature_data, preprocessing_config$missing_strategy)
  }
  
  # Encode categorical variables
  if (preprocessing_config$encode_categorical) {
    encoding_objects <- encode_categorical_variables(feature_data, preprocessing_config$encoding_method)
    feature_data <- encoding_objects$encoded_data
  }
  
  # Scale numerical variables
  if (preprocessing_config$scale_numerical) {
    scaling_objects <- scale_numerical_variables(feature_data, preprocessing_config$scaling_method)
    feature_data <- scaling_objects$scaled_data
  }
  
  # Feature selection
  if (preprocessing_config$feature_selection) {
    selection_objects <- select_features(feature_data, target_data, preprocessing_config$selection_method)
    feature_data <- selection_objects$selected_data
  }
  
  # Create feature engineering objects
  preprocessing_objects <- list(
    missing_handling = if (preprocessing_config$handle_missing) encoding_objects else NULL,
    encoding = if (preprocessing_config$encode_categorical) encoding_objects else NULL,
    scaling = if (preprocessing_config$scale_numerical) scaling_objects else NULL,
    selection = if (preprocessing_config$feature_selection) selection_objects else NULL
  )
  
  return(list(
    features = feature_data,
    target = target_data,
    preprocessing_objects = preprocessing_objects
  ))
}

#' Handle missing values
#'
#' @param data Data frame
#' @param strategy Missing value handling strategy
#' @return Data with handled missing values
handle_missing_values <- function(data, strategy) {
  if (strategy == "remove") {
    return(data[complete.cases(data), ])
  } else if (strategy == "impute_mean") {
    numeric_cols <- sapply(data, is.numeric)
    for (col in names(data)[numeric_cols]) {
      data[[col]][is.na(data[[col]])] <- mean(data[[col]], na.rm = TRUE)
    }
    return(data)
  } else if (strategy == "impute_median") {
    numeric_cols <- sapply(data, is.numeric)
    for (col in names(data)[numeric_cols]) {
      data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
    }
    return(data)
  } else if (strategy == "impute_mode") {
    categorical_cols <- sapply(data, function(x) is.character(x) || is.factor(x))
    for (col in names(data)[categorical_cols]) {
      mode_value <- names(sort(table(data[[col]]), decreasing = TRUE))[1]
      data[[col]][is.na(data[[col]])] <- mode_value
    }
    return(data)
  }
}

#' Encode categorical variables
#'
#' @param data Data frame
#' @param method Encoding method
#' @return Encoded data and encoding objects
encode_categorical_variables <- function(data, method = "one_hot") {
  categorical_cols <- sapply(data, function(x) is.character(x) || is.factor(x))
  
  if (method == "one_hot") {
    return(one_hot_encode(data, categorical_cols))
  } else if (method == "label") {
    return(label_encode(data, categorical_cols))
  } else if (method == "target") {
    return(target_encode(data, categorical_cols))
  }
}

#' One-hot encoding
#'
#' @param data Data frame
#' @param categorical_cols Categorical column indices
#' @return One-hot encoded data
one_hot_encode <- function(data, categorical_cols) {
  encoded_data <- data
  
  for (col in names(data)[categorical_cols]) {
    # Create dummy variables
    dummy_vars <- model.matrix(~ . - 1, data = data.frame(data[[col]]))
    colnames(dummy_vars) <- paste0(col, "_", levels(factor(data[[col]])))
    
    # Remove original column and add dummy variables
    encoded_data <- encoded_data[, !names(encoded_data) %in% col, drop = FALSE]
    encoded_data <- cbind(encoded_data, dummy_vars)
  }
  
  return(list(
    encoded_data = encoded_data,
    encoding_method = "one_hot"
  ))
}

#' Label encoding
#'
#' @param data Data frame
#' @param categorical_cols Categorical column indices
#' @return Label encoded data
label_encode <- function(data, categorical_cols) {
  encoded_data <- data
  encoding_maps <- list()
  
  for (col in names(data)[categorical_cols]) {
    unique_values <- unique(data[[col]])
    encoding_map <- setNames(1:length(unique_values), unique_values)
    encoded_data[[col]] <- encoding_map[as.character(data[[col]])]
    encoding_maps[[col]] <- encoding_map
  }
  
  return(list(
    encoded_data = encoded_data,
    encoding_maps = encoding_maps,
    encoding_method = "label"
  ))
}

#' Target encoding
#'
#' @param data Data frame
#' @param categorical_cols Categorical column indices
#' @return Target encoded data
target_encode <- function(data, categorical_cols) {
  # This is a simplified version - in practice, you'd need the target variable
  # and proper cross-validation to avoid overfitting
  encoded_data <- data
  
  for (col in names(data)[categorical_cols]) {
    # Calculate mean target value for each category
    category_means <- aggregate(data[[col]], by = list(data[[col]]), FUN = mean)
    encoding_map <- setNames(category_means$x, category_means$Group.1)
    encoded_data[[col]] <- encoding_map[as.character(data[[col]])]
  }
  
  return(list(
    encoded_data = encoded_data,
    encoding_method = "target"
  ))
}

#' Scale numerical variables
#'
#' @param data Data frame
#' @param method Scaling method
#' @return Scaled data and scaling objects
scale_numerical_variables <- function(data, method = "standardization") {
  numeric_cols <- sapply(data, is.numeric)
  scaled_data <- data
  scaling_params <- list()
  
  for (col in names(data)[numeric_cols]) {
    if (method == "standardization") {
      mean_val <- mean(data[[col]], na.rm = TRUE)
      sd_val <- sd(data[[col]], na.rm = TRUE)
      scaled_data[[col]] <- (data[[col]] - mean_val) / sd_val
      scaling_params[[col]] <- list(mean = mean_val, sd = sd_val)
    } else if (method == "min_max") {
      min_val <- min(data[[col]], na.rm = TRUE)
      max_val <- max(data[[col]], na.rm = TRUE)
      scaled_data[[col]] <- (data[[col]] - min_val) / (max_val - min_val)
      scaling_params[[col]] <- list(min = min_val, max = max_val)
    }
  }
  
  return(list(
    scaled_data = scaled_data,
    scaling_params = scaling_params,
    scaling_method = method
  ))
}

#' Select features
#'
#' @param data Data frame
#' @param target Target variable
#' @param method Feature selection method
#' @return Selected features and selection objects
select_features <- function(data, target, method = "correlation") {
  if (method == "correlation") {
    return(select_features_by_correlation(data, target))
  } else if (method == "mutual_information") {
    return(select_features_by_mutual_information(data, target))
  } else if (method == "recursive") {
    return(select_features_recursively(data, target))
  }
}

#' Select features by correlation
#'
#' @param data Data frame
#' @param target Target variable
#' @return Selected features
select_features_by_correlation <- function(data, target) {
  numeric_cols <- sapply(data, is.numeric)
  if (sum(numeric_cols) == 0) return(data)
  
  numeric_data <- data[, numeric_cols, drop = FALSE]
  correlations <- abs(cor(numeric_data, target, use = "complete.obs"))
  
  # Select features with correlation > 0.1
  selected_cols <- rownames(correlations)[correlations > 0.1]
  
  if (length(selected_cols) == 0) {
    return(data)
  }
  
  selected_data <- data[, selected_cols, drop = FALSE]
  
  return(list(
    selected_data = selected_data,
    selected_features = selected_cols,
    selection_method = "correlation"
  ))
}
```

## Model Training and Validation

### Cross-Validation Framework

```r
# R/02-model-training.R

#' Comprehensive model training and validation
#'
#' @param data Preprocessed data
#' @param target Target variable
#' @param model_config Model configuration
#' @return Model training results
train_model <- function(data, target, model_config) {
  # Split data
  data_split <- split_data(data, target, model_config$split_ratio)
  
  # Train model
  trained_model <- fit_model(data_split$train, data_split$train_target, model_config)
  
  # Validate model
  validation_results <- validate_model(trained_model, data_split$test, data_split$test_target)
  
  # Cross-validation
  cv_results <- cross_validate_model(data, target, model_config)
  
  # Hyperparameter tuning
  if (model_config$tune_hyperparameters) {
    tuning_results <- tune_hyperparameters(data, target, model_config)
    trained_model <- tuning_results$best_model
  }
  
  results <- list(
    model = trained_model,
    validation = validation_results,
    cross_validation = cv_results,
    tuning = if (model_config$tune_hyperparameters) tuning_results else NULL
  )
  
  return(results)
}

#' Split data into train and test sets
#'
#' @param data Data frame
#' @param target Target variable
#' @param split_ratio Train/test split ratio
#' @return Data split
split_data <- function(data, target, split_ratio = 0.8) {
  set.seed(123)
  n <- nrow(data)
  train_indices <- sample(1:n, floor(split_ratio * n))
  
  return(list(
    train = data[train_indices, ],
    test = data[-train_indices, ],
    train_target = target[train_indices],
    test_target = target[-train_indices]
  ))
}

#' Fit model based on configuration
#'
#' @param train_data Training data
#' @param train_target Training target
#' @param model_config Model configuration
#' @return Fitted model
fit_model <- function(train_data, train_target, model_config) {
  if (model_config$algorithm == "random_forest") {
    return(fit_random_forest(train_data, train_target, model_config))
  } else if (model_config$algorithm == "gradient_boosting") {
    return(fit_gradient_boosting(train_data, train_target, model_config))
  } else if (model_config$algorithm == "svm") {
    return(fit_svm(train_data, train_target, model_config))
  } else if (model_config$algorithm == "neural_network") {
    return(fit_neural_network(train_data, train_target, model_config))
  }
}

#' Fit random forest model
#'
#' @param train_data Training data
#' @param train_target Training target
#' @param model_config Model configuration
#' @return Fitted random forest model
fit_random_forest <- function(train_data, train_target, model_config) {
  library(randomForest)
  
  model <- randomForest(
    x = train_data,
    y = train_target,
    ntree = model_config$ntree,
    mtry = model_config$mtry,
    nodesize = model_config$nodesize,
    maxnodes = model_config$maxnodes
  )
  
  return(model)
}

#' Fit gradient boosting model
#'
#' @param train_data Training data
#' @param train_target Training target
#' @param model_config Model configuration
#' @return Fitted gradient boosting model
fit_gradient_boosting <- function(train_data, train_target, model_config) {
  library(gbm)
  
  model <- gbm(
    formula = train_target ~ .,
    data = cbind(train_data, train_target),
    distribution = model_config$distribution,
    n.trees = model_config$n_trees,
    interaction.depth = model_config$interaction_depth,
    shrinkage = model_config$shrinkage,
    bag.fraction = model_config$bag_fraction
  )
  
  return(model)
}

#' Fit SVM model
#'
#' @param train_data Training data
#' @param train_target Training target
#' @param model_config Model configuration
#' @return Fitted SVM model
fit_svm <- function(train_data, train_target, model_config) {
  library(e1071)
  
  model <- svm(
    x = train_data,
    y = train_target,
    kernel = model_config$kernel,
    cost = model_config$cost,
    gamma = model_config$gamma,
    epsilon = model_config$epsilon
  )
  
  return(model)
}

#' Fit neural network model
#'
#' @param train_data Training data
#' @param train_target Training target
#' @param model_config Model configuration
#' @return Fitted neural network model
fit_neural_network <- function(train_data, train_target, model_config) {
  library(nnet)
  
  model <- nnet(
    x = train_data,
    y = train_target,
    size = model_config$hidden_units,
    decay = model_config$decay,
    maxit = model_config$max_iterations,
    trace = FALSE
  )
  
  return(model)
}
```

### Model Validation

```r
# R/02-model-training.R (continued)

#' Validate model performance
#'
#' @param model Fitted model
#' @param test_data Test data
#' @param test_target Test target
#' @return Validation results
validate_model <- function(model, test_data, test_target) {
  # Make predictions
  predictions <- predict_model(model, test_data)
  
  # Calculate performance metrics
  performance_metrics <- calculate_performance_metrics(test_target, predictions)
  
  # Create confusion matrix (for classification)
  if (is.factor(test_target)) {
    confusion_matrix <- table(test_target, predictions)
    performance_metrics$confusion_matrix <- confusion_matrix
  }
  
  return(list(
    predictions = predictions,
    performance_metrics = performance_metrics
  ))
}

#' Make predictions using model
#'
#' @param model Fitted model
#' @param data Data to predict on
#' @return Predictions
predict_model <- function(model, data) {
  if (inherits(model, "randomForest")) {
    return(predict(model, data))
  } else if (inherits(model, "gbm")) {
    return(predict(model, data, n.trees = model$n.trees))
  } else if (inherits(model, "svm")) {
    return(predict(model, data))
  } else if (inherits(model, "nnet")) {
    return(predict(model, data))
  }
}

#' Calculate performance metrics
#'
#' @param actual Actual values
#' @param predicted Predicted values
#' @return Performance metrics
calculate_performance_metrics <- function(actual, predicted) {
  if (is.numeric(actual)) {
    # Regression metrics
    mse <- mean((actual - predicted)^2)
    rmse <- sqrt(mse)
    mae <- mean(abs(actual - predicted))
    mape <- mean(abs((actual - predicted) / actual)) * 100
    r_squared <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
    
    return(list(
      mse = mse,
      rmse = rmse,
      mae = mae,
      mape = mape,
      r_squared = r_squared
    ))
  } else {
    # Classification metrics
    accuracy <- mean(actual == predicted)
    precision <- calculate_precision(actual, predicted)
    recall <- calculate_recall(actual, predicted)
    f1_score <- 2 * (precision * recall) / (precision + recall)
    
    return(list(
      accuracy = accuracy,
      precision = precision,
      recall = recall,
      f1_score = f1_score
    ))
  }
}

#' Calculate precision
#'
#' @param actual Actual values
#' @param predicted Predicted values
#' @return Precision
calculate_precision <- function(actual, predicted) {
  # Calculate precision for each class
  classes <- unique(actual)
  precision_values <- numeric(length(classes))
  
  for (i in seq_along(classes)) {
    class <- classes[i]
    true_positives <- sum(actual == class & predicted == class)
    false_positives <- sum(actual != class & predicted == class)
    
    if (true_positives + false_positives == 0) {
      precision_values[i] <- 0
    } else {
      precision_values[i] <- true_positives / (true_positives + false_positives)
    }
  }
  
  return(mean(precision_values))
}

#' Calculate recall
#'
#' @param actual Actual values
#' @param predicted Predicted values
#' @return Recall
calculate_recall <- function(actual, predicted) {
  # Calculate recall for each class
  classes <- unique(actual)
  recall_values <- numeric(length(classes))
  
  for (i in seq_along(classes)) {
    class <- classes[i]
    true_positives <- sum(actual == class & predicted == class)
    false_negatives <- sum(actual == class & predicted != class)
    
    if (true_positives + false_negatives == 0) {
      recall_values[i] <- 0
    } else {
      recall_values[i] <- true_positives / (true_positives + false_negatives)
    }
  }
  
  return(mean(recall_values))
}

#' Cross-validate model
#'
#' @param data Data frame
#' @param target Target variable
#' @param model_config Model configuration
#' @return Cross-validation results
cross_validate_model <- function(data, target, model_config) {
  set.seed(123)
  k <- model_config$cv_folds
  folds <- create_cv_folds(nrow(data), k)
  
  cv_results <- list()
  
  for (i in 1:k) {
    train_indices <- unlist(folds[-i])
    test_indices <- folds[[i]]
    
    train_data <- data[train_indices, ]
    test_data <- data[test_indices, ]
    train_target <- target[train_indices]
    test_target <- target[test_indices]
    
    # Train model
    model <- fit_model(train_data, train_target, model_config)
    
    # Validate model
    validation <- validate_model(model, test_data, test_target)
    
    cv_results[[i]] <- validation$performance_metrics
  }
  
  # Aggregate results
  aggregated_results <- aggregate_cv_results(cv_results)
  
  return(list(
    fold_results = cv_results,
    aggregated_results = aggregated_results
  ))
}

#' Create cross-validation folds
#'
#' @param n Number of observations
#' @param k Number of folds
#' @return List of fold indices
create_cv_folds <- function(n, k) {
  fold_size <- floor(n / k)
  remainder <- n %% k
  
  folds <- list()
  start <- 1
  
  for (i in 1:k) {
    end <- start + fold_size - 1
    if (i <= remainder) {
      end <- end + 1
    }
    
    folds[[i]] <- start:end
    start <- end + 1
  }
  
  return(folds)
}

#' Aggregate cross-validation results
#'
#' @param cv_results Cross-validation results
#' @return Aggregated results
aggregate_cv_results <- function(cv_results) {
  # Get metric names
  metric_names <- names(cv_results[[1]])
  
  # Calculate mean and standard deviation for each metric
  aggregated <- list()
  
  for (metric in metric_names) {
    values <- sapply(cv_results, function(x) x[[metric]])
    aggregated[[paste0(metric, "_mean")]] <- mean(values)
    aggregated[[paste0(metric, "_sd")]] <- sd(values)
  }
  
  return(aggregated)
}
```

## Hyperparameter Tuning

### Grid Search

```r
# R/03-hyperparameter-tuning.R

#' Comprehensive hyperparameter tuning
#'
#' @param data Data frame
#' @param target Target variable
#' @param model_config Model configuration
#' @return Hyperparameter tuning results
tune_hyperparameters <- function(data, target, model_config) {
  if (model_config$tuning_method == "grid_search") {
    return(grid_search_tuning(data, target, model_config))
  } else if (model_config$tuning_method == "random_search") {
    return(random_search_tuning(data, target, model_config))
  } else if (model_config$tuning_method == "bayesian") {
    return(bayesian_tuning(data, target, model_config))
  }
}

#' Grid search hyperparameter tuning
#'
#' @param data Data frame
#' @param target Target variable
#' @param model_config Model configuration
#' @return Grid search results
grid_search_tuning <- function(data, target, model_config) {
  # Create parameter grid
  param_grid <- create_parameter_grid(model_config$hyperparameters)
  
  # Perform grid search
  grid_results <- perform_grid_search(data, target, param_grid, model_config)
  
  # Find best parameters
  best_params <- find_best_parameters(grid_results)
  
  # Train final model with best parameters
  best_model_config <- update_model_config(model_config, best_params)
  best_model <- fit_model(data, target, best_model_config)
  
  return(list(
    grid_results = grid_results,
    best_parameters = best_params,
    best_model = best_model
  ))
}

#' Create parameter grid
#'
#' @param hyperparameters Hyperparameter specifications
#' @return Parameter grid
create_parameter_grid <- function(hyperparameters) {
  param_names <- names(hyperparameters)
  param_values <- lapply(hyperparameters, function(x) x$values)
  
  # Create all combinations
  param_grid <- expand.grid(param_values, stringsAsFactors = FALSE)
  names(param_grid) <- param_names
  
  return(param_grid)
}

#' Perform grid search
#'
#' @param data Data frame
#' @param target Target variable
#' @param param_grid Parameter grid
#' @param model_config Model configuration
#' @return Grid search results
perform_grid_search <- function(data, target, param_grid, model_config) {
  results <- list()
  
  for (i in 1:nrow(param_grid)) {
    params <- param_grid[i, ]
    
    # Update model configuration with current parameters
    current_config <- update_model_config(model_config, params)
    
    # Perform cross-validation
    cv_results <- cross_validate_model(data, target, current_config)
    
    # Store results
    results[[i]] <- list(
      parameters = params,
      cv_results = cv_results$aggregated_results
    )
  }
  
  return(results)
}

#' Update model configuration with parameters
#'
#' @param model_config Model configuration
#' @param parameters Parameters to update
#' @return Updated model configuration
update_model_config <- function(model_config, parameters) {
  updated_config <- model_config
  
  for (param_name in names(parameters)) {
    updated_config[[param_name]] <- parameters[[param_name]]
  }
  
  return(updated_config)
}

#' Find best parameters
#'
#' @param grid_results Grid search results
#' @return Best parameters
find_best_parameters <- function(grid_results) {
  # Extract performance metrics
  performance_metrics <- lapply(grid_results, function(x) x$cv_results)
  
  # Find best parameters based on primary metric
  best_idx <- which.max(sapply(performance_metrics, function(x) x$accuracy_mean))
  
  return(grid_results[[best_idx]]$parameters)
}
```

### Random Search

```r
# R/03-hyperparameter-tuning.R (continued)

#' Random search hyperparameter tuning
#'
#' @param data Data frame
#' @param target Target variable
#' @param model_config Model configuration
#' @return Random search results
random_search_tuning <- function(data, target, model_config) {
  n_iterations <- model_config$n_iterations
  hyperparameters <- model_config$hyperparameters
  
  results <- list()
  
  for (i in 1:n_iterations) {
    # Sample random parameters
    params <- sample_parameters(hyperparameters)
    
    # Update model configuration
    current_config <- update_model_config(model_config, params)
    
    # Perform cross-validation
    cv_results <- cross_validate_model(data, target, current_config)
    
    # Store results
    results[[i]] <- list(
      parameters = params,
      cv_results = cv_results$aggregated_results
    )
  }
  
  # Find best parameters
  best_params <- find_best_parameters(results)
  
  # Train final model
  best_model_config <- update_model_config(model_config, best_params)
  best_model <- fit_model(data, target, best_model_config)
  
  return(list(
    search_results = results,
    best_parameters = best_params,
    best_model = best_model
  ))
}

#' Sample random parameters
#'
#' @param hyperparameters Hyperparameter specifications
#' @return Sampled parameters
sample_parameters <- function(hyperparameters) {
  sampled_params <- list()
  
  for (param_name in names(hyperparameters)) {
    param_spec <- hyperparameters[[param_name]]
    
    if (param_spec$type == "categorical") {
      sampled_params[[param_name]] <- sample(param_spec$values, 1)
    } else if (param_spec$type == "integer") {
      sampled_params[[param_name]] <- sample(param_spec$values, 1)
    } else if (param_spec$type == "float") {
      min_val <- param_spec$min
      max_val <- param_spec$max
      sampled_params[[param_name]] <- runif(1, min_val, max_val)
    }
  }
  
  return(sampled_params)
}
```

## Model Deployment

### Model Serialization

```r
# R/04-model-deployment.R

#' Serialize model for deployment
#'
#' @param model Fitted model
#' @param preprocessing_objects Preprocessing objects
#' @param model_metadata Model metadata
#' @return Serialized model
serialize_model <- function(model, preprocessing_objects, model_metadata) {
  # Create model package
  model_package <- list(
    model = model,
    preprocessing_objects = preprocessing_objects,
    metadata = model_metadata,
    timestamp = Sys.time(),
    version = "1.0.0"
  )
  
  # Serialize model
  serialized_model <- serialize(model_package, NULL)
  
  return(serialized_model)
}

#' Save model to file
#'
#' @param model Fitted model
#' @param preprocessing_objects Preprocessing objects
#' @param model_metadata Model metadata
#' @param file_path File path to save model
save_model <- function(model, preprocessing_objects, model_metadata, file_path) {
  # Serialize model
  serialized_model <- serialize_model(model, preprocessing_objects, model_metadata)
  
  # Save to file
  writeBin(serialized_model, file_path)
  
  return(file_path)
}

#' Load model from file
#'
#' @param file_path File path to load model from
#' @return Loaded model
load_model <- function(file_path) {
  # Read serialized model
  serialized_model <- readBin(file_path, "raw", file.info(file_path)$size)
  
  # Deserialize model
  model_package <- unserialize(serialized_model)
  
  return(model_package)
}

#' Make predictions with deployed model
#'
#' @param model_package Model package
#' @param new_data New data for prediction
#' @return Predictions
predict_with_deployed_model <- function(model_package, new_data) {
  # Apply preprocessing
  preprocessed_data <- apply_preprocessing(new_data, model_package$preprocessing_objects)
  
  # Make predictions
  predictions <- predict_model(model_package$model, preprocessed_data)
  
  return(predictions)
}

#' Apply preprocessing to new data
#'
#' @param data New data
#' @param preprocessing_objects Preprocessing objects
#' @return Preprocessed data
apply_preprocessing <- function(data, preprocessing_objects) {
  preprocessed_data <- data
  
  # Apply missing value handling
  if (!is.null(preprocessing_objects$missing_handling)) {
    preprocessed_data <- handle_missing_values(preprocessed_data, 
                                              preprocessing_objects$missing_handling$strategy)
  }
  
  # Apply encoding
  if (!is.null(preprocessing_objects$encoding)) {
    preprocessed_data <- apply_encoding(preprocessed_data, preprocessing_objects$encoding)
  }
  
  # Apply scaling
  if (!is.null(preprocessing_objects$scaling)) {
    preprocessed_data <- apply_scaling(preprocessed_data, preprocessing_objects$scaling)
  }
  
  # Apply feature selection
  if (!is.null(preprocessing_objects$selection)) {
    preprocessed_data <- preprocessed_data[, preprocessing_objects$selection$selected_features, drop = FALSE]
  }
  
  return(preprocessed_data)
}
```

## TL;DR Runbook

### Quick Start

```r
# 1. Data preprocessing
preprocessed_data <- feature_engineering(data, "target", preprocessing_config)

# 2. Model training
model_results <- train_model(preprocessed_data$features, preprocessed_data$target, model_config)

# 3. Hyperparameter tuning
tuning_results <- tune_hyperparameters(data, target, model_config)

# 4. Model deployment
save_model(model_results$model, preprocessed_data$preprocessing_objects, model_metadata, "model.rds")

# 5. Make predictions
loaded_model <- load_model("model.rds")
predictions <- predict_with_deployed_model(loaded_model, new_data)
```

### Essential Patterns

```r
# Complete ML pipeline
ml_pipeline <- function(data, target, model_config) {
  # Preprocess data
  preprocessed <- feature_engineering(data, target, model_config$preprocessing)
  
  # Train model
  model_results <- train_model(preprocessed$features, preprocessed$target, model_config$training)
  
  # Tune hyperparameters
  if (model_config$tune_hyperparameters) {
    tuning_results <- tune_hyperparameters(data, target, model_config$tuning)
    model_results$model <- tuning_results$best_model
  }
  
  # Deploy model
  save_model(model_results$model, preprocessed$preprocessing_objects, model_config$metadata, "model.rds")
  
  return(model_results)
}
```

---

*This guide provides the complete machinery for building production-ready machine learning systems in R. Each pattern includes implementation examples, validation strategies, and real-world usage patterns for enterprise deployment.*

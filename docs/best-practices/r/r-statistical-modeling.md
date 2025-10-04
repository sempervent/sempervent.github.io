# R Statistical Modeling Best Practices

**Objective**: Master senior-level R statistical modeling patterns for production systems. When you need to build robust, interpretable statistical models, when you want to follow best practices for model selection and validation, when you need enterprise-grade modeling patterns—these best practices become your weapon of choice.

## Core Principles

- **Model Selection**: Choose appropriate models for the data and question
- **Validation**: Use proper validation techniques to assess model performance
- **Interpretability**: Ensure models are interpretable and explainable
- **Robustness**: Build models that are robust to outliers and violations
- **Documentation**: Document model assumptions, limitations, and results

## Linear Models

### Linear Regression

```r
# R/01-linear-models.R

#' Comprehensive linear regression analysis
#'
#' @param data Data frame
#' @param formula Regression formula
#' @param validation_method Validation method
#' @return Linear regression results
linear_regression_analysis <- function(data, formula, validation_method = "holdout") {
  # Fit the model
  model <- lm(formula, data = data)
  
  # Model summary
  model_summary <- summary(model)
  
  # Model diagnostics
  diagnostics <- perform_model_diagnostics(model, data)
  
  # Model validation
  validation_results <- validate_model(model, data, validation_method)
  
  # Model selection
  selection_results <- perform_model_selection(data, formula)
  
  results <- list(
    model = model,
    summary = model_summary,
    diagnostics = diagnostics,
    validation = validation_results,
    selection = selection_results
  )
  
  return(results)
}

#' Perform model diagnostics
#'
#' @param model Fitted model
#' @param data Original data
#' @return Diagnostic results
perform_model_diagnostics <- function(model, data) {
  diagnostics <- list(
    residuals = residuals(model),
    fitted_values = fitted(model),
    leverage = hatvalues(model),
    cooks_distance = cooks.distance(model),
    dffits = dffits(model),
    dfbetas = dfbetas(model)
  )
  
  # Normality tests
  diagnostics$normality_tests <- list(
    shapiro_test = shapiro.test(diagnostics$residuals),
    anderson_darling = nortest::ad.test(diagnostics$residuals)
  )
  
  # Heteroscedasticity tests
  diagnostics$heteroscedasticity_tests <- list(
    breusch_pagan = lmtest::bptest(model),
    white_test = lmtest::bptest(model, ~ fitted(model) + I(fitted(model)^2))
  )
  
  # Autocorrelation tests
  diagnostics$autocorrelation_tests <- list(
    durbin_watson = lmtest::dwtest(model)
  )
  
  # Multicollinearity
  diagnostics$multicollinearity <- calculate_multicollinearity(model)
  
  return(diagnostics)
}

#' Calculate multicollinearity measures
#'
#' @param model Fitted model
#' @return Multicollinearity measures
calculate_multicollinearity <- function(model) {
  # Variance Inflation Factors
  vif_values <- car::vif(model)
  
  # Condition Index
  X <- model.matrix(model)
  eigen_values <- eigen(t(X) %*% X)$values
  condition_index <- sqrt(max(eigen_values) / eigen_values)
  
  # Tolerance
  tolerance <- 1 / vif_values
  
  return(list(
    vif = vif_values,
    condition_index = condition_index,
    tolerance = tolerance
  ))
}

#' Validate model using various methods
#'
#' @param model Fitted model
#' @param data Original data
#' @param method Validation method
#' @return Validation results
validate_model <- function(model, data, method = "holdout") {
  if (method == "holdout") {
    return(holdout_validation(model, data))
  } else if (method == "cross_validation") {
    return(cross_validation(model, data))
  } else if (method == "bootstrap") {
    return(bootstrap_validation(model, data))
  }
}

#' Holdout validation
#'
#' @param model Fitted model
#' @param data Original data
#' @return Holdout validation results
holdout_validation <- function(model, data) {
  # Split data
  set.seed(123)
  train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  
  # Fit model on training data
  train_model <- lm(formula(model), data = train_data)
  
  # Predict on test data
  predictions <- predict(train_model, newdata = test_data)
  actual <- test_data[[as.character(formula(model)[[2]])]]
  
  # Calculate metrics
  mse <- mean((actual - predictions)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(actual - predictions))
  r_squared <- 1 - sum((actual - predictions)^2) / sum((actual - mean(actual))^2)
  
  return(list(
    mse = mse,
    rmse = rmse,
    mae = mae,
    r_squared = r_squared,
    predictions = predictions,
    actual = actual
  ))
}

#' Cross-validation
#'
#' @param model Fitted model
#' @param data Original data
#' @param k Number of folds
#' @return Cross-validation results
cross_validation <- function(model, data, k = 10) {
  set.seed(123)
  folds <- create_folds(1:nrow(data), k = k)
  
  cv_results <- list()
  
  for (i in 1:k) {
    train_indices <- unlist(folds[-i])
    test_indices <- folds[[i]]
    
    train_data <- data[train_indices, ]
    test_data <- data[test_indices, ]
    
    # Fit model
    cv_model <- lm(formula(model), data = train_data)
    
    # Predict
    predictions <- predict(cv_model, newdata = test_data)
    actual <- test_data[[as.character(formula(model)[[2]])]]
    
    # Calculate metrics
    mse <- mean((actual - predictions)^2)
    mae <- mean(abs(actual - predictions))
    
    cv_results[[i]] <- list(mse = mse, mae = mae)
  }
  
  # Aggregate results
  mean_mse <- mean(sapply(cv_results, function(x) x$mse))
  mean_mae <- mean(sapply(cv_results, function(x) x$mae))
  sd_mse <- sd(sapply(cv_results, function(x) x$mse))
  sd_mae <- sd(sapply(cv_results, function(x) x$mae))
  
  return(list(
    mean_mse = mean_mse,
    mean_mae = mean_mae,
    sd_mse = sd_mse,
    sd_mae = sd_mae,
    cv_results = cv_results
  ))
}

#' Create k-fold cross-validation folds
#'
#' @param indices Data indices
#' @param k Number of folds
#' @return List of fold indices
create_folds <- function(indices, k = 10) {
  n <- length(indices)
  fold_size <- floor(n / k)
  remainder <- n %% k
  
  folds <- list()
  start <- 1
  
  for (i in 1:k) {
    end <- start + fold_size - 1
    if (i <= remainder) {
      end <- end + 1
    }
    
    folds[[i]] <- indices[start:end]
    start <- end + 1
  }
  
  return(folds)
}
```

### Generalized Linear Models

```r
# R/02-generalized-linear-models.R

#' Comprehensive GLM analysis
#'
#' @param data Data frame
#' @param formula GLM formula
#' @param family GLM family
#' @return GLM analysis results
glm_analysis <- function(data, formula, family = gaussian()) {
  # Fit the model
  model <- glm(formula, data = data, family = family)
  
  # Model summary
  model_summary <- summary(model)
  
  # Model diagnostics
  diagnostics <- perform_glm_diagnostics(model, data)
  
  # Model validation
  validation_results <- validate_glm_model(model, data)
  
  # Model comparison
  comparison_results <- compare_glm_models(data, formula, family)
  
  results <- list(
    model = model,
    summary = model_summary,
    diagnostics = diagnostics,
    validation = validation_results,
    comparison = comparison_results
  )
  
  return(results)
}

#' Perform GLM diagnostics
#'
#' @param model Fitted GLM model
#' @param data Original data
#' @return GLM diagnostic results
perform_glm_diagnostics <- function(model, data) {
  diagnostics <- list(
    residuals = residuals(model),
    fitted_values = fitted(model),
    deviance_residuals = residuals(model, type = "deviance"),
    pearson_residuals = residuals(model, type = "pearson"),
    leverage = hatvalues(model),
    cooks_distance = cooks.distance(model)
  )
  
  # Deviance analysis
  diagnostics$deviance_analysis <- perform_deviance_analysis(model)
  
  # Goodness of fit tests
  diagnostics$goodness_of_fit <- perform_goodness_of_fit_tests(model)
  
  return(diagnostics)
}

#' Perform deviance analysis
#'
#' @param model Fitted GLM model
#' @return Deviance analysis results
perform_deviance_analysis <- function(model) {
  # Null deviance
  null_deviance <- model$null.deviance
  
  # Residual deviance
  residual_deviance <- model$deviance
  
  # Deviance explained
  deviance_explained <- (null_deviance - residual_deviance) / null_deviance
  
  # Deviance residuals
  deviance_residuals <- residuals(model, type = "deviance")
  
  return(list(
    null_deviance = null_deviance,
    residual_deviance = residual_deviance,
    deviance_explained = deviance_explained,
    deviance_residuals = deviance_residuals
  ))
}

#' Perform goodness of fit tests
#'
#' @param model Fitted GLM model
#' @return Goodness of fit test results
perform_goodness_of_fit_tests <- function(model) {
  # Hosmer-Lemeshow test (for logistic regression)
  if (model$family$family == "binomial") {
    hosmer_lemeshow <- ResourceSelection::hoslem.test(model$y, fitted(model))
  } else {
    hosmer_lemeshow <- NULL
  }
  
  # Pearson chi-square test
  pearson_chi_square <- sum(residuals(model, type = "pearson")^2)
  pearson_p_value <- 1 - pchisq(pearson_chi_square, model$df.residual)
  
  return(list(
    hosmer_lemeshow = hosmer_lemeshow,
    pearson_chi_square = pearson_chi_square,
    pearson_p_value = pearson_p_value
  ))
}

#' Compare different GLM models
#'
#' @param data Data frame
#' @param formula Base formula
#' @param family GLM family
#' @return Model comparison results
compare_glm_models <- function(data, formula, family) {
  # Fit different models
  models <- list()
  
  # Full model
  models$full <- glm(formula, data = data, family = family)
  
  # Stepwise selection
  models$stepwise <- step(models$full, direction = "both", trace = FALSE)
  
  # AIC-based selection
  models$aic <- step(models$full, direction = "both", k = 2, trace = FALSE)
  
  # BIC-based selection
  models$bic <- step(models$full, direction = "both", k = log(nrow(data)), trace = FALSE)
  
  # Compare models
  comparison <- data.frame(
    model = names(models),
    aic = sapply(models, AIC),
    bic = sapply(models, BIC),
    deviance = sapply(models, function(x) x$deviance),
    df_residual = sapply(models, function(x) x$df.residual)
  )
  
  return(list(
    models = models,
    comparison = comparison
  ))
}
```

## Mixed Effects Models

### Linear Mixed Effects

```r
# R/03-mixed-effects-models.R

#' Comprehensive linear mixed effects analysis
#'
#' @param data Data frame
#' @param formula Mixed effects formula
#' @param random_effects Random effects specification
#' @return Mixed effects analysis results
lme_analysis <- function(data, formula, random_effects) {
  # Fit the model
  model <- lme4::lmer(formula, data = data, REML = TRUE)
  
  # Model summary
  model_summary <- summary(model)
  
  # Model diagnostics
  diagnostics <- perform_lme_diagnostics(model, data)
  
  # Model validation
  validation_results <- validate_lme_model(model, data)
  
  # Model comparison
  comparison_results <- compare_lme_models(data, formula, random_effects)
  
  results <- list(
    model = model,
    summary = model_summary,
    diagnostics = diagnostics,
    validation = validation_results,
    comparison = comparison_results
  )
  
  return(results)
}

#' Perform LME diagnostics
#'
#' @param model Fitted LME model
#' @param data Original data
#' @return LME diagnostic results
perform_lme_diagnostics <- function(model, data) {
  diagnostics <- list(
    residuals = residuals(model),
    fitted_values = fitted(model),
    random_effects = ranef(model),
    conditional_residuals = residuals(model, type = "response"),
    marginal_residuals = residuals(model, type = "pearson")
  )
  
  # Normality tests for residuals
  diagnostics$residual_normality <- shapiro.test(diagnostics$residuals)
  
  # Normality tests for random effects
  diagnostics$random_effects_normality <- lapply(diagnostics$random_effects, function(x) {
    if (ncol(x) == 1) {
      shapiro.test(x[, 1])
    } else {
      NULL
    }
  })
  
  # Heteroscedasticity tests
  diagnostics$heteroscedasticity <- lmtest::bptest(lm(residuals(model) ~ fitted(model)))
  
  return(diagnostics)
}

#' Validate LME model
#'
#' @param model Fitted LME model
#' @param data Original data
#' @return LME validation results
validate_lme_model <- function(model, data) {
  # Cross-validation
  cv_results <- cross_validate_lme(model, data)
  
  # Bootstrap validation
  bootstrap_results <- bootstrap_validate_lme(model, data)
  
  # Model fit statistics
  fit_statistics <- calculate_lme_fit_statistics(model)
  
  return(list(
    cross_validation = cv_results,
    bootstrap = bootstrap_results,
    fit_statistics = fit_statistics
  ))
}

#' Cross-validate LME model
#'
#' @param model Fitted LME model
#' @param data Original data
#' @param k Number of folds
#' @return Cross-validation results
cross_validate_lme <- function(model, data, k = 10) {
  set.seed(123)
  folds <- create_folds(1:nrow(data), k = k)
  
  cv_results <- list()
  
  for (i in 1:k) {
    train_indices <- unlist(folds[-i])
    test_indices <- folds[[i]]
    
    train_data <- data[train_indices, ]
    test_data <- data[test_indices, ]
    
    # Fit model
    cv_model <- lme4::lmer(formula(model), data = train_data, REML = TRUE)
    
    # Predict
    predictions <- predict(cv_model, newdata = test_data, allow.new.levels = TRUE)
    actual <- test_data[[as.character(formula(model)[[2]])]]
    
    # Calculate metrics
    mse <- mean((actual - predictions)^2)
    mae <- mean(abs(actual - predictions))
    
    cv_results[[i]] <- list(mse = mse, mae = mae)
  }
  
  # Aggregate results
  mean_mse <- mean(sapply(cv_results, function(x) x$mse))
  mean_mae <- mean(sapply(cv_results, function(x) x$mae))
  sd_mse <- sd(sapply(cv_results, function(x) x$mse))
  sd_mae <- sd(sapply(cv_results, function(x) x$mae))
  
  return(list(
    mean_mse = mean_mse,
    mean_mae = mean_mae,
    sd_mse = sd_mse,
    sd_mae = sd_mae,
    cv_results = cv_results
  ))
}

#' Calculate LME fit statistics
#'
#' @param model Fitted LME model
#' @return Fit statistics
calculate_lme_fit_statistics <- function(model) {
  # AIC and BIC
  aic <- AIC(model)
  bic <- BIC(model)
  
  # Log-likelihood
  log_lik <- logLik(model)
  
  # R-squared
  r_squared <- calculate_lme_r_squared(model)
  
  return(list(
    aic = aic,
    bic = bic,
    log_likelihood = log_lik,
    r_squared = r_squared
  ))
}

#' Calculate LME R-squared
#'
#' @param model Fitted LME model
#' @return R-squared values
calculate_lme_r_squared <- function(model) {
  # Marginal R-squared
  marginal_r_squared <- r.squaredGLMM(model)[1]
  
  # Conditional R-squared
  conditional_r_squared <- r.squaredGLMM(model)[2]
  
  return(list(
    marginal = marginal_r_squared,
    conditional = conditional_r_squared
  ))
}
```

## Time Series Models

### ARIMA Models

```r
# R/04-time-series-models.R

#' Comprehensive ARIMA analysis
#'
#' @param data Time series data
#' @param order ARIMA order
#' @param seasonal_order Seasonal ARIMA order
#' @return ARIMA analysis results
arima_analysis <- function(data, order = NULL, seasonal_order = NULL) {
  # Auto-select order if not provided
  if (is.null(order)) {
    order <- auto_arima_order(data)
  }
  
  # Fit the model
  model <- arima(data, order = order, seasonal = seasonal_order)
  
  # Model summary
  model_summary <- summary(model)
  
  # Model diagnostics
  diagnostics <- perform_arima_diagnostics(model, data)
  
  # Model validation
  validation_results <- validate_arima_model(model, data)
  
  # Forecasting
  forecast_results <- forecast_arima_model(model, data)
  
  results <- list(
    model = model,
    summary = model_summary,
    diagnostics = diagnostics,
    validation = validation_results,
    forecast = forecast_results
  )
  
  return(results)
}

#' Auto-select ARIMA order
#'
#' @param data Time series data
#' @return ARIMA order
auto_arima_order <- function(data) {
  # Use auto.arima from forecast package
  auto_model <- forecast::auto.arima(data, seasonal = FALSE, stepwise = FALSE, approximation = FALSE)
  
  return(auto_model$arma[1:3])
}

#' Perform ARIMA diagnostics
#'
#' @param model Fitted ARIMA model
#' @param data Original data
#' @return ARIMA diagnostic results
perform_arima_diagnostics <- function(model, data) {
  diagnostics <- list(
    residuals = residuals(model),
    fitted_values = fitted(model),
    acf_residuals = acf(residuals(model), plot = FALSE),
    pacf_residuals = pacf(residuals(model), plot = FALSE)
  )
  
  # Ljung-Box test
  diagnostics$ljung_box <- Box.test(residuals(model), type = "Ljung-Box")
  
  # Normality test
  diagnostics$normality <- shapiro.test(residuals(model))
  
  # ARCH test
  diagnostics$arch_test <- FinTS::ArchTest(residuals(model))
  
  return(diagnostics)
}

#' Validate ARIMA model
#'
#' @param model Fitted ARIMA model
#' @param data Original data
#' @return ARIMA validation results
validate_arima_model <- function(model, data) {
  # Out-of-sample validation
  n <- length(data)
  train_size <- floor(0.8 * n)
  
  train_data <- data[1:train_size]
  test_data <- data[(train_size + 1):n]
  
  # Fit model on training data
  train_model <- arima(train_data, order = model$arma[1:3])
  
  # Forecast
  forecast_values <- forecast::forecast(train_model, h = length(test_data))
  
  # Calculate metrics
  mse <- mean((test_data - forecast_values$mean)^2)
  mae <- mean(abs(test_data - forecast_values$mean))
  mape <- mean(abs((test_data - forecast_values$mean) / test_data)) * 100
  
  return(list(
    mse = mse,
    mae = mae,
    mape = mape,
    forecast_values = forecast_values$mean,
    actual_values = test_data
  ))
}

#' Forecast ARIMA model
#'
#' @param model Fitted ARIMA model
#' @param data Original data
#' @param h Forecast horizon
#' @return Forecast results
forecast_arima_model <- function(model, data, h = 12) {
  # Generate forecasts
  forecast_values <- forecast::forecast(model, h = h)
  
  # Calculate prediction intervals
  prediction_intervals <- forecast_values$lower[, 1:2]
  prediction_intervals <- cbind(prediction_intervals, forecast_values$upper[, 1:2])
  colnames(prediction_intervals) <- c("lower_80", "lower_95", "upper_80", "upper_95")
  
  return(list(
    forecast = forecast_values$mean,
    prediction_intervals = prediction_intervals,
    forecast_object = forecast_values
  ))
}
```

## Model Selection and Comparison

### Model Selection Framework

```r
# R/05-model-selection.R

#' Comprehensive model selection
#'
#' @param data Data frame
#' @param response_variable Response variable name
#' @param candidate_models List of candidate models
#' @return Model selection results
model_selection <- function(data, response_variable, candidate_models) {
  # Fit all candidate models
  fitted_models <- fit_candidate_models(data, response_variable, candidate_models)
  
  # Calculate selection criteria
  selection_criteria <- calculate_selection_criteria(fitted_models)
  
  # Perform model comparison
  model_comparison <- compare_models(fitted_models, selection_criteria)
  
  # Select best model
  best_model <- select_best_model(fitted_models, selection_criteria)
  
  results <- list(
    fitted_models = fitted_models,
    selection_criteria = selection_criteria,
    model_comparison = model_comparison,
    best_model = best_model
  )
  
  return(results)
}

#' Fit candidate models
#'
#' @param data Data frame
#' @param response_variable Response variable name
#' @param candidate_models List of candidate models
#' @return Fitted models
fit_candidate_models <- function(data, response_variable, candidate_models) {
  fitted_models <- list()
  
  for (model_name in names(candidate_models)) {
    model_spec <- candidate_models[[model_name]]
    
    tryCatch({
      if (model_spec$type == "linear") {
        fitted_models[[model_name]] <- lm(model_spec$formula, data = data)
      } else if (model_spec$type == "glm") {
        fitted_models[[model_name]] <- glm(model_spec$formula, data = data, family = model_spec$family)
      } else if (model_spec$type == "lme") {
        fitted_models[[model_name]] <- lme4::lmer(model_spec$formula, data = data)
      }
    }, error = function(e) {
      warning(paste("Failed to fit model", model_name, ":", e$message))
    })
  }
  
  return(fitted_models)
}

#' Calculate selection criteria
#'
#' @param fitted_models Fitted models
#' @return Selection criteria
calculate_selection_criteria <- function(fitted_models) {
  criteria <- data.frame(
    model = names(fitted_models),
    aic = sapply(fitted_models, AIC),
    bic = sapply(fitted_models, BIC),
    log_likelihood = sapply(fitted_models, function(x) as.numeric(logLik(x))),
    r_squared = sapply(fitted_models, function(x) {
      if (inherits(x, "lm")) {
        summary(x)$r.squared
      } else {
        NA
      }
    }),
    adj_r_squared = sapply(fitted_models, function(x) {
      if (inherits(x, "lm")) {
        summary(x)$adj.r.squared
      } else {
        NA
      }
    }),
    stringsAsFactors = FALSE
  )
  
  return(criteria)
}

#' Compare models
#'
#' @param fitted_models Fitted models
#' @param selection_criteria Selection criteria
#' @return Model comparison results
compare_models <- function(fitted_models, selection_criteria) {
  comparison <- list()
  
  # AIC comparison
  comparison$aic <- selection_criteria[order(selection_criteria$aic), ]
  
  # BIC comparison
  comparison$bic <- selection_criteria[order(selection_criteria$bic), ]
  
  # Likelihood ratio tests
  comparison$likelihood_ratio_tests <- perform_likelihood_ratio_tests(fitted_models)
  
  # Information criteria differences
  comparison$aic_differences <- calculate_information_criteria_differences(selection_criteria, "aic")
  comparison$bic_differences <- calculate_information_criteria_differences(selection_criteria, "bic")
  
  return(comparison)
}

#' Perform likelihood ratio tests
#'
#' @param fitted_models Fitted models
#' @return Likelihood ratio test results
perform_likelihood_ratio_tests <- function(fitted_models) {
  model_names <- names(fitted_models)
  n_models <- length(model_names)
  
  lrt_results <- data.frame(
    model1 = character(0),
    model2 = character(0),
    lr_statistic = numeric(0),
    p_value = numeric(0),
    stringsAsFactors = FALSE
  )
  
  for (i in 1:(n_models - 1)) {
    for (j in (i + 1):n_models) {
      model1 <- fitted_models[[model_names[i]]]
      model2 <- fitted_models[[model_names[j]]]
      
      if (inherits(model1, "lm") && inherits(model2, "lm")) {
        lrt <- anova(model1, model2)
        lr_statistic <- lrt$F[2]
        p_value <- lrt$`Pr(>F)`[2]
        
        lrt_results <- rbind(lrt_results, data.frame(
          model1 = model_names[i],
          model2 = model_names[j],
          lr_statistic = lr_statistic,
          p_value = p_value,
          stringsAsFactors = FALSE
        ))
      }
    }
  }
  
  return(lrt_results)
}

#' Calculate information criteria differences
#'
#' @param selection_criteria Selection criteria
#' @param criterion Information criterion
#' @return Information criteria differences
calculate_information_criteria_differences <- function(selection_criteria, criterion) {
  min_value <- min(selection_criteria[[criterion]], na.rm = TRUE)
  differences <- selection_criteria[[criterion]] - min_value
  
  return(differences)
}

#' Select best model
#'
#' @param fitted_models Fitted models
#' @param selection_criteria Selection criteria
#' @return Best model
select_best_model <- function(fitted_models, selection_criteria) {
  # Select model with minimum AIC
  best_aic_idx <- which.min(selection_criteria$aic)
  best_aic_model <- names(fitted_models)[best_aic_idx]
  
  # Select model with minimum BIC
  best_bic_idx <- which.min(selection_criteria$bic)
  best_bic_model <- names(fitted_models)[best_bic_idx]
  
  return(list(
    best_aic = best_aic_model,
    best_bic = best_bic_model,
    best_aic_model = fitted_models[[best_aic_model]],
    best_bic_model = fitted_models[[best_bic_model]]
  ))
}
```

## TL;DR Runbook

### Quick Start

```r
# 1. Linear regression
linear_results <- linear_regression_analysis(data, y ~ x1 + x2 + x3)

# 2. GLM analysis
glm_results <- glm_analysis(data, y ~ x1 + x2, family = binomial())

# 3. Mixed effects
lme_results <- lme_analysis(data, y ~ x1 + x2 + (1|group))

# 4. ARIMA
arima_results <- arima_analysis(ts_data)

# 5. Model selection
candidate_models <- list(
  model1 = list(type = "linear", formula = y ~ x1),
  model2 = list(type = "linear", formula = y ~ x1 + x2)
)
selection_results <- model_selection(data, "y", candidate_models)
```

### Essential Patterns

```r
# Model fitting and validation
fit_and_validate <- function(data, formula, model_type = "linear") {
  # Fit model
  if (model_type == "linear") {
    model <- lm(formula, data = data)
  } else if (model_type == "glm") {
    model <- glm(formula, data = data, family = binomial())
  }
  
  # Validate model
  validation <- validate_model(model, data)
  
  # Return results
  return(list(model = model, validation = validation))
}
```

---

*This guide provides the complete machinery for building robust statistical models in R. Each pattern includes implementation examples, validation strategies, and real-world usage patterns for enterprise deployment.*

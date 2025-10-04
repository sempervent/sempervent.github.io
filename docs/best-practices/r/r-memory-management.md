# R Memory Management Best Practices

**Objective**: Master senior-level R memory management patterns for production systems. When you need to handle large datasets efficiently, when you want to optimize memory usage, when you need enterprise-grade memory patterns—these best practices become your weapon of choice.

## Core Principles

- **Memory Awareness**: Understand R's memory model and limitations
- **Efficient Data Types**: Choose appropriate data types for memory efficiency
- **Garbage Collection**: Manage garbage collection effectively
- **Memory Monitoring**: Track and monitor memory usage
- **Memory Optimization**: Optimize memory usage patterns

## Memory Model Understanding

### R Memory Architecture

```r
# R/01-memory-architecture.R

#' Understand R memory model
#'
#' @return Memory model information
understand_memory_model <- function() {
  memory_info <- list(
    memory_limit = memory.limit(),
    memory_usage = memory.size(),
    gc_info = gc(),
    object_sizes = get_object_sizes(),
    memory_allocations = get_memory_allocations()
  )
  
  return(memory_info)
}

#' Get object sizes in workspace
#'
#' @return Object sizes
get_object_sizes <- function() {
  library(pryr)
  
  objects <- ls(envir = .GlobalEnv)
  sizes <- sapply(objects, function(x) {
    tryCatch({
      object_size(get(x))
    }, error = function(e) {
      NA
    })
  })
  
  # Remove NA values
  sizes <- sizes[!is.na(sizes)]
  
  # Sort by size
  sizes <- sort(sizes, decreasing = TRUE)
  
  return(sizes)
}

#' Get memory allocations
#'
#' @return Memory allocation information
get_memory_allocations <- function() {
  library(pryr)
  
  allocations <- list(
    total_memory = mem_used(),
    gc_memory = gc(),
    memory_objects = object_sizes()
  )
  
  return(allocations)
}

#' Monitor memory usage over time
#'
#' @param duration Monitoring duration in seconds
#' @param interval Monitoring interval in seconds
#' @return Memory usage over time
monitor_memory_usage <- function(duration = 60, interval = 1) {
  memory_data <- data.frame(
    timestamp = numeric(0),
    memory_used = numeric(0),
    gc_count = numeric(0),
    object_count = numeric(0)
  )
  
  start_time <- Sys.time()
  
  while (as.numeric(Sys.time() - start_time) < duration) {
    current_time <- Sys.time()
    memory_used <- mem_used()
    gc_info <- gc()
    object_count <- length(ls(envir = .GlobalEnv))
    
    memory_data <- rbind(memory_data, data.frame(
      timestamp = as.numeric(current_time),
      memory_used = memory_used,
      gc_count = sum(gc_info[, 1]),
      object_count = object_count
    ))
    
    Sys.sleep(interval)
  }
  
  return(memory_data)
}
```

### Memory Profiling

```r
# R/01-memory-architecture.R (continued)

#' Profile memory usage of code
#'
#' @param code_expression Code expression to profile
#' @return Memory profiling results
profile_memory_usage <- function(code_expression) {
  library(pryr)
  
  # Get memory before
  memory_before <- mem_used()
  gc_before <- gc()
  
  # Execute code
  result <- eval(code_expression)
  
  # Get memory after
  memory_after <- mem_used()
  gc_after <- gc()
  
  # Calculate memory usage
  memory_usage <- memory_after - memory_before
  gc_count <- sum(gc_after[, 1]) - sum(gc_before[, 1])
  
  return(list(
    memory_before = memory_before,
    memory_after = memory_after,
    memory_usage = memory_usage,
    gc_count = gc_count,
    result = result
  ))
}

#' Profile memory usage of function
#'
#' @param function_name Function name
#' @param arguments Function arguments
#' @return Memory profiling results
profile_function_memory <- function(function_name, arguments) {
  library(pryr)
  
  # Get memory before
  memory_before <- mem_used()
  gc_before <- gc()
  
  # Execute function
  result <- do.call(function_name, arguments)
  
  # Get memory after
  memory_after <- mem_used()
  gc_after <- gc()
  
  # Calculate memory usage
  memory_usage <- memory_after - memory_before
  gc_count <- sum(gc_after[, 1]) - sum(gc_before[, 1])
  
  return(list(
    memory_before = memory_before,
    memory_after = memory_after,
    memory_usage = memory_usage,
    gc_count = gc_count,
    result = result
  ))
}

#' Profile memory usage of data operations
#'
#' @param data Data to operate on
#' @param operation Operation to perform
#' @return Memory profiling results
profile_data_operation_memory <- function(data, operation) {
  library(pryr)
  
  # Get memory before
  memory_before <- mem_used()
  gc_before <- gc()
  
  # Perform operation
  result <- operation(data)
  
  # Get memory after
  memory_after <- mem_used()
  gc_after <- gc()
  
  # Calculate memory usage
  memory_usage <- memory_after - memory_before
  gc_count <- sum(gc_after[, 1]) - sum(gc_before[, 1])
  
  return(list(
    memory_before = memory_before,
    memory_after = memory_after,
    memory_usage = memory_usage,
    gc_count = gc_count,
    result = result
  ))
}
```

## Efficient Data Types

### Memory-Efficient Data Types

```r
# R/02-efficient-data-types.R

#' Choose memory-efficient data types
#'
#' @param data Data to optimize
#' @return Memory-optimized data
choose_memory_efficient_types <- function(data) {
  optimized_data <- data
  
  for (col in names(data)) {
    if (is.numeric(data[[col]])) {
      optimized_data[[col]] <- optimize_numeric_column(data[[col]])
    } else if (is.character(data[[col]])) {
      optimized_data[[col]] <- optimize_character_column(data[[col]])
    } else if (is.logical(data[[col]])) {
      optimized_data[[col]] <- optimize_logical_column(data[[col]])
    }
  }
  
  return(optimized_data)
}

#' Optimize numeric column
#'
#' @param column Numeric column
#' @return Optimized numeric column
optimize_numeric_column <- function(column) {
  # Check if all values are integers
  if (all(column == as.integer(column), na.rm = TRUE)) {
    # Use integer if possible
    if (all(column >= -2147483648 & column <= 2147483647, na.rm = TRUE)) {
      return(as.integer(column))
    } else {
      # Use long integer
      return(as.numeric(column))
    }
  } else {
    # Check if single precision is sufficient
    if (all(abs(column) < 3.4e38, na.rm = TRUE)) {
      return(as.single(column))
    } else {
      return(as.double(column))
    }
  }
}

#' Optimize character column
#'
#' @param column Character column
#' @return Optimized character column
optimize_character_column <- function(column) {
  # Check if factor would be more memory efficient
  unique_values <- length(unique(column))
  total_values <- length(column)
  
  if (unique_values < total_values * 0.5) {
    return(as.factor(column))
  } else {
    return(column)
  }
}

#' Optimize logical column
#'
#' @param column Logical column
#' @return Optimized logical column
optimize_logical_column <- function(column) {
  # Logical columns are already memory efficient
  return(column)
}

#' Compare memory usage of data types
#'
#' @param data Data to compare
#' @return Memory usage comparison
compare_memory_usage <- function(data) {
  library(pryr)
  
  # Original data
  original_memory <- object_size(data)
  
  # Optimized data
  optimized_data <- choose_memory_efficient_types(data)
  optimized_memory <- object_size(optimized_data)
  
  # Calculate savings
  memory_savings <- original_memory - optimized_memory
  savings_percentage <- (memory_savings / original_memory) * 100
  
  return(list(
    original_memory = original_memory,
    optimized_memory = optimized_memory,
    memory_savings = memory_savings,
    savings_percentage = savings_percentage
  ))
}
```

### Sparse Data Structures

```r
# R/02-efficient-data-types.R (continued)

#' Convert to sparse data structures
#'
#' @param data Data to convert
#' @param sparsity_threshold Sparsity threshold
#' @return Sparse data structure
convert_to_sparse <- function(data, sparsity_threshold = 0.5) {
  library(Matrix)
  
  # Check sparsity
  sparsity <- calculate_sparsity(data)
  
  if (sparsity > sparsity_threshold) {
    # Convert to sparse matrix
    sparse_data <- as(data, "sparseMatrix")
    return(sparse_data)
  } else {
    return(data)
  }
}

#' Calculate sparsity of data
#'
#' @param data Data to analyze
#' @return Sparsity percentage
calculate_sparsity <- function(data) {
  if (is.matrix(data)) {
    zero_count <- sum(data == 0)
    total_count <- length(data)
    sparsity <- zero_count / total_count
  } else {
    # For data frames, calculate sparsity for numeric columns
    numeric_cols <- sapply(data, is.numeric)
    if (sum(numeric_cols) == 0) {
      return(0)
    }
    
    numeric_data <- data[, numeric_cols, drop = FALSE]
    zero_count <- sum(numeric_data == 0)
    total_count <- length(numeric_data)
    sparsity <- zero_count / total_count
  }
  
  return(sparsity)
}

#' Optimize sparse data operations
#'
#' @param sparse_data Sparse data
#' @param operation Operation to perform
#' @return Optimized operation result
optimize_sparse_operations <- function(sparse_data, operation) {
  # Use sparse matrix operations
  if (operation == "matrix_multiply") {
    return(sparse_data %*% sparse_data)
  } else if (operation == "matrix_add") {
    return(sparse_data + sparse_data)
  } else if (operation == "matrix_transpose") {
    return(t(sparse_data))
  }
}
```

## Garbage Collection Management

### Garbage Collection Control

```r
# R/03-garbage-collection.R

#' Control garbage collection
#'
#' @param gc_type Type of garbage collection
#' @param parameters GC parameters
#' @return GC control results
control_garbage_collection <- function(gc_type = "auto", parameters = list()) {
  switch(gc_type,
    "auto" = enable_auto_gc(),
    "manual" = enable_manual_gc(parameters),
    "tuning" = tune_gc_parameters(parameters),
    stop("Unsupported GC type: ", gc_type)
  )
}

#' Enable automatic garbage collection
#'
#' @return Auto GC status
enable_auto_gc <- function() {
  # R handles garbage collection automatically
  return(list(
    gc_type = "auto",
    status = "enabled"
  ))
}

#' Enable manual garbage collection
#'
#' @param parameters GC parameters
#' @return Manual GC status
enable_manual_gc <- function(parameters) {
  # Disable automatic GC
  gc(verbose = FALSE)
  
  # Set manual GC parameters
  gc_parameters <- list(
    verbose = parameters$verbose %||% FALSE,
    reset = parameters$reset %||% TRUE
  )
  
  return(list(
    gc_type = "manual",
    parameters = gc_parameters,
    status = "enabled"
  ))
}

#' Tune garbage collection parameters
#'
#' @param parameters GC tuning parameters
#' @return GC tuning results
tune_gc_parameters <- function(parameters) {
  # Set GC tuning parameters
  gc_parameters <- list(
    verbose = parameters$verbose %||% FALSE,
    reset = parameters$reset %||% TRUE,
    threshold = parameters$threshold %||% 0.5
  )
  
  # Apply tuning
  gc(verbose = gc_parameters$verbose, reset = gc_parameters$reset)
  
  return(list(
    gc_type = "tuned",
    parameters = gc_parameters,
    status = "applied"
  ))
}

#' Monitor garbage collection
#'
#' @param duration Monitoring duration in seconds
#' @param interval Monitoring interval in seconds
#' @return GC monitoring results
monitor_garbage_collection <- function(duration = 60, interval = 1) {
  gc_data <- data.frame(
    timestamp = numeric(0),
    gc_count = numeric(0),
    memory_used = numeric(0),
    memory_available = numeric(0)
  )
  
  start_time <- Sys.time()
  
  while (as.numeric(Sys.time() - start_time) < duration) {
    current_time <- Sys.time()
    gc_info <- gc()
    memory_used <- mem_used()
    memory_available <- memory.limit() - memory_used
    
    gc_data <- rbind(gc_data, data.frame(
      timestamp = as.numeric(current_time),
      gc_count = sum(gc_info[, 1]),
      memory_used = memory_used,
      memory_available = memory_available
    ))
    
    Sys.sleep(interval)
  }
  
  return(gc_data)
}
```

### Memory Cleanup

```r
# R/03-garbage-collection.R (continued)

#' Clean up memory
#'
#' @param cleanup_type Type of cleanup
#' @param parameters Cleanup parameters
#' @return Cleanup results
cleanup_memory <- function(cleanup_type = "full", parameters = list()) {
  switch(cleanup_type,
    "full" = full_memory_cleanup(parameters),
    "selective" = selective_memory_cleanup(parameters),
    "aggressive" = aggressive_memory_cleanup(parameters),
    stop("Unsupported cleanup type: ", cleanup_type)
  )
}

#' Perform full memory cleanup
#'
#' @param parameters Cleanup parameters
#' @return Full cleanup results
full_memory_cleanup <- function(parameters) {
  # Get memory before
  memory_before <- mem_used()
  
  # Remove unused objects
  rm(list = ls(envir = .GlobalEnv)[!ls(envir = .GlobalEnv) %in% parameters$keep_objects])
  
  # Force garbage collection
  gc(verbose = parameters$verbose %||% FALSE)
  
  # Get memory after
  memory_after <- mem_used()
  
  return(list(
    memory_before = memory_before,
    memory_after = memory_after,
    memory_freed = memory_before - memory_after,
    cleanup_type = "full"
  ))
}

#' Perform selective memory cleanup
#'
#' @param parameters Cleanup parameters
#' @return Selective cleanup results
selective_memory_cleanup <- function(parameters) {
  # Get memory before
  memory_before <- mem_used()
  
  # Remove specific objects
  if (!is.null(parameters$remove_objects)) {
    rm(list = parameters$remove_objects, envir = .GlobalEnv)
  }
  
  # Force garbage collection
  gc(verbose = parameters$verbose %||% FALSE)
  
  # Get memory after
  memory_after <- mem_used()
  
  return(list(
    memory_before = memory_before,
    memory_after = memory_after,
    memory_freed = memory_before - memory_after,
    cleanup_type = "selective"
  ))
}

#' Perform aggressive memory cleanup
#'
#' @param parameters Cleanup parameters
#' @return Aggressive cleanup results
aggressive_memory_cleanup <- function(parameters) {
  # Get memory before
  memory_before <- mem_used()
  
  # Remove all objects except specified ones
  keep_objects <- parameters$keep_objects %||% c()
  all_objects <- ls(envir = .GlobalEnv)
  remove_objects <- setdiff(all_objects, keep_objects)
  
  if (length(remove_objects) > 0) {
    rm(list = remove_objects, envir = .GlobalEnv)
  }
  
  # Force multiple garbage collections
  for (i in 1:3) {
    gc(verbose = parameters$verbose %||% FALSE)
  }
  
  # Get memory after
  memory_after <- mem_used()
  
  return(list(
    memory_before = memory_before,
    memory_after = memory_after,
    memory_freed = memory_before - memory_after,
    cleanup_type = "aggressive"
  ))
}
```

## Memory Optimization Strategies

### Lazy Loading

```r
# R/04-memory-optimization.R

#' Implement lazy loading
#'
#' @param data Data to load lazily
#' @param loading_strategy Loading strategy
#' @return Lazy loading implementation
implement_lazy_loading <- function(data, loading_strategy = "on_demand") {
  switch(loading_strategy,
    "on_demand" = implement_on_demand_loading(data),
    "chunked" = implement_chunked_loading(data),
    "streaming" = implement_streaming_loading(data),
    stop("Unsupported loading strategy: ", loading_strategy)
  )
}

#' Implement on-demand loading
#'
#' @param data Data to load
#' @return On-demand loading implementation
implement_on_demand_loading <- function(data) {
  # Create lazy loading wrapper
  lazy_wrapper <- list(
    data = data,
    loaded = FALSE,
    load_function = function() {
      if (!lazy_wrapper$loaded) {
        lazy_wrapper$data <<- load_data(lazy_wrapper$data)
        lazy_wrapper$loaded <<- TRUE
      }
      return(lazy_wrapper$data)
    }
  )
  
  return(lazy_wrapper)
}

#' Implement chunked loading
#'
#' @param data Data to load
#' @param chunk_size Chunk size
#' @return Chunked loading implementation
implement_chunked_loading <- function(data, chunk_size = 1000) {
  # Create chunked loading wrapper
  chunked_wrapper <- list(
    data = data,
    chunk_size = chunk_size,
    current_chunk = 1,
    total_chunks = ceiling(nrow(data) / chunk_size),
    load_chunk = function(chunk_number) {
      start_idx <- (chunk_number - 1) * chunked_wrapper$chunk_size + 1
      end_idx <- min(chunk_number * chunked_wrapper$chunk_size, nrow(chunked_wrapper$data))
      return(chunked_wrapper$data[start_idx:end_idx, ])
    }
  )
  
  return(chunked_wrapper)
}

#' Implement streaming loading
#'
#' @param data_source Data source
#' @return Streaming loading implementation
implement_streaming_loading <- function(data_source) {
  # Create streaming loading wrapper
  streaming_wrapper <- list(
    data_source = data_source,
    current_position = 1,
    buffer_size = 1000,
    buffer = NULL,
    load_next = function() {
      # Load next chunk from data source
      next_chunk <- read_data_chunk(streaming_wrapper$data_source, 
                                   streaming_wrapper$current_position, 
                                   streaming_wrapper$buffer_size)
      streaming_wrapper$current_position <<- streaming_wrapper$current_position + streaming_wrapper$buffer_size
      return(next_chunk)
    }
  )
  
  return(streaming_wrapper)
}
```

### Memory Pooling

```r
# R/04-memory-optimization.R (continued)

#' Implement memory pooling
#'
#' @param pool_size Pool size
#' @param object_type Object type to pool
#' @return Memory pool implementation
implement_memory_pooling <- function(pool_size = 100, object_type = "numeric") {
  # Create memory pool
  memory_pool <- list(
    pool_size = pool_size,
    object_type = object_type,
    available_objects = list(),
    used_objects = list(),
    get_object = function() {
      if (length(memory_pool$available_objects) > 0) {
        object <- memory_pool$available_objects[[1]]
        memory_pool$available_objects <<- memory_pool$available_objects[-1]
        memory_pool$used_objects <<- c(memory_pool$used_objects, list(object))
        return(object)
      } else {
        # Create new object
        new_object <- create_object(memory_pool$object_type)
        memory_pool$used_objects <<- c(memory_pool$used_objects, list(new_object))
        return(new_object)
      }
    },
    return_object = function(object) {
      # Return object to pool
      memory_pool$available_objects <<- c(memory_pool$available_objects, list(object))
      memory_pool$used_objects <<- memory_pool$used_objects[memory_pool$used_objects != object]
    }
  )
  
  return(memory_pool)
}

#' Create object for memory pool
#'
#' @param object_type Object type
#' @return Created object
create_object <- function(object_type) {
  switch(object_type,
    "numeric" = numeric(1000),
    "character" = character(1000),
    "logical" = logical(1000),
    "list" = list(),
    stop("Unsupported object type: ", object_type)
  )
}
```

### Memory Mapping

```r
# R/04-memory-optimization.R (continued)

#' Implement memory mapping
#'
#' @param file_path File path to map
#' @param mapping_type Mapping type
#' @return Memory mapping implementation
implement_memory_mapping <- function(file_path, mapping_type = "read_only") {
  library(mmap)
  
  # Create memory mapping
  mapping <- list(
    file_path = file_path,
    mapping_type = mapping_type,
    mapped_data = NULL,
    map_file = function() {
      if (mapping_type == "read_only") {
        mapping$mapped_data <<- mmap(file_path, mode = "read")
      } else if (mapping_type == "read_write") {
        mapping$mapped_data <<- mmap(file_path, mode = "write")
      }
    },
    unmap_file = function() {
      if (!is.null(mapping$mapped_data)) {
        munmap(mapping$mapped_data)
        mapping$mapped_data <<- NULL
      }
    },
    get_data = function(start, end) {
      if (is.null(mapping$mapped_data)) {
        mapping$map_file()
      }
      return(mapping$mapped_data[start:end])
    }
  )
  
  return(mapping)
}
```

## Memory Monitoring and Diagnostics

### Memory Leak Detection

```r
# R/05-memory-monitoring.R

#' Detect memory leaks
#'
#' @param code_expression Code expression to test
#' @param iterations Number of iterations
#' @return Memory leak detection results
detect_memory_leaks <- function(code_expression, iterations = 100) {
  memory_usage <- numeric(iterations)
  
  for (i in 1:iterations) {
    # Execute code
    eval(code_expression)
    
    # Record memory usage
    memory_usage[i] <- mem_used()
    
    # Force garbage collection
    gc(verbose = FALSE)
  }
  
  # Analyze memory usage pattern
  memory_trend <- analyze_memory_trend(memory_usage)
  
  return(list(
    memory_usage = memory_usage,
    trend = memory_trend,
    has_leak = memory_trend$has_leak
  ))
}

#' Analyze memory trend
#'
#' @param memory_usage Memory usage over time
#' @return Memory trend analysis
analyze_memory_trend <- function(memory_usage) {
  # Calculate trend
  trend <- lm(memory_usage ~ seq_along(memory_usage))
  slope <- coef(trend)[2]
  
  # Determine if there's a memory leak
  has_leak <- slope > 0.1  # Threshold for memory leak
  
  return(list(
    slope = slope,
    has_leak = has_leak,
    trend_model = trend
  ))
}

#' Monitor memory usage in real-time
#'
#' @param duration Monitoring duration in seconds
#' @param interval Monitoring interval in seconds
#' @return Real-time memory monitoring
monitor_memory_realtime <- function(duration = 60, interval = 1) {
  memory_data <- data.frame(
    timestamp = numeric(0),
    memory_used = numeric(0),
    memory_available = numeric(0),
    gc_count = numeric(0),
    object_count = numeric(0)
  )
  
  start_time <- Sys.time()
  
  while (as.numeric(Sys.time() - start_time) < duration) {
    current_time <- Sys.time()
    memory_used <- mem_used()
    memory_available <- memory.limit() - memory_used
    gc_info <- gc()
    object_count <- length(ls(envir = .GlobalEnv))
    
    memory_data <- rbind(memory_data, data.frame(
      timestamp = as.numeric(current_time),
      memory_used = memory_used,
      memory_available = memory_available,
      gc_count = sum(gc_info[, 1]),
      object_count = object_count
    ))
    
    Sys.sleep(interval)
  }
  
  return(memory_data)
}
```

## TL;DR Runbook

### Quick Start

```r
# 1. Monitor memory usage
memory_info <- understand_memory_model()
print(memory_info)

# 2. Profile memory usage
prof_results <- profile_memory_usage(compute_function(data))

# 3. Optimize data types
optimized_data <- choose_memory_efficient_types(data)

# 4. Control garbage collection
gc_control <- control_garbage_collection("manual", list(verbose = TRUE))

# 5. Clean up memory
cleanup_results <- cleanup_memory("full", list(keep_objects = c("important_data")))

# 6. Implement lazy loading
lazy_data <- implement_lazy_loading(data, "on_demand")
```

### Essential Patterns

```r
# Complete memory management pipeline
manage_memory <- function(data, memory_config) {
  # Monitor current memory
  memory_info <- understand_memory_model()
  
  # Optimize data types
  optimized_data <- choose_memory_efficient_types(data)
  
  # Control garbage collection
  gc_control <- control_garbage_collection(memory_config$gc_type, memory_config$gc_params)
  
  # Implement lazy loading if needed
  if (memory_config$lazy_loading) {
    data <- implement_lazy_loading(optimized_data, memory_config$loading_strategy)
  }
  
  # Clean up memory
  cleanup_results <- cleanup_memory(memory_config$cleanup_type, memory_config$cleanup_params)
  
  return(list(
    data = data,
    memory_info = memory_info,
    gc_control = gc_control,
    cleanup_results = cleanup_results
  ))
}
```

---

*This guide provides the complete machinery for managing memory efficiently in R. Each pattern includes implementation examples, monitoring strategies, and real-world usage patterns for enterprise deployment.*

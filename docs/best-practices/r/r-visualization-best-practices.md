# R Visualization Best Practices

**Objective**: Master senior-level R visualization patterns for production systems. When you need to create compelling, informative visualizations, when you want to follow design best practices, when you need enterprise-grade visualization patterns—these best practices become your weapon of choice.

## Core Principles

- **Data-Driven Design**: Let the data guide the visualization design
- **Clarity and Simplicity**: Prioritize clarity over complexity
- **Consistency**: Maintain consistent visual language
- **Accessibility**: Ensure visualizations are accessible to all users
- **Performance**: Optimize for rendering speed and file size

## Static Visualizations

### ggplot2 Best Practices

```r
# R/01-static-visualizations.R

#' Create comprehensive static visualizations
#'
#' @param data Data frame
#' @param visualization_type Type of visualization
#' @param parameters Visualization parameters
#' @return Static visualization
create_static_visualization <- function(data, visualization_type, parameters = list()) {
  switch(visualization_type,
    "scatter_plot" = create_scatter_plot(data, parameters),
    "line_plot" = create_line_plot(data, parameters),
    "bar_plot" = create_bar_plot(data, parameters),
    "histogram" = create_histogram(data, parameters),
    "box_plot" = create_box_plot(data, parameters),
    "heatmap" = create_heatmap(data, parameters),
    stop("Unsupported visualization type: ", visualization_type)
  )
}

#' Create scatter plot
#'
#' @param data Data frame
#' @param parameters Plot parameters
#' @return Scatter plot
create_scatter_plot <- function(data, parameters) {
  library(ggplot2)
  
  p <- ggplot(data, aes_string(x = parameters$x, y = parameters$y)) +
    geom_point(
      alpha = parameters$alpha %||% 0.7,
      size = parameters$size %||% 2,
      color = parameters$color %||% "steelblue"
    ) +
    theme_minimal() +
    labs(
      title = parameters$title %||% "Scatter Plot",
      x = parameters$x_label %||% parameters$x,
      y = parameters$y_label %||% parameters$y
    )
  
  # Add trend line if requested
  if (parameters$add_trend_line) {
    p <- p + geom_smooth(method = "lm", se = parameters$show_confidence_interval)
  }
  
  # Add color mapping if specified
  if (!is.null(parameters$color_by)) {
    p <- p + aes_string(color = parameters$color_by) +
      scale_color_viridis_d()
  }
  
  return(p)
}

#' Create line plot
#'
#' @param data Data frame
#' @param parameters Plot parameters
#' @return Line plot
create_line_plot <- function(data, parameters) {
  library(ggplot2)
  
  p <- ggplot(data, aes_string(x = parameters$x, y = parameters$y)) +
    geom_line(
      size = parameters$size %||% 1,
      color = parameters$color %||% "steelblue"
    ) +
    theme_minimal() +
    labs(
      title = parameters$title %||% "Line Plot",
      x = parameters$x_label %||% parameters$x,
      y = parameters$y_label %||% parameters$y
    )
  
  # Add points if requested
  if (parameters$add_points) {
    p <- p + geom_point(size = parameters$point_size %||% 2)
  }
  
  # Add multiple lines if specified
  if (!is.null(parameters$group_by)) {
    p <- p + aes_string(group = parameters$group_by, color = parameters$group_by) +
      scale_color_viridis_d()
  }
  
  return(p)
}

#' Create bar plot
#'
#' @param data Data frame
#' @param parameters Plot parameters
#' @return Bar plot
create_bar_plot <- function(data, parameters) {
  library(ggplot2)
  
  p <- ggplot(data, aes_string(x = parameters$x, y = parameters$y)) +
    geom_bar(
      stat = "identity",
      fill = parameters$fill %||% "steelblue",
      alpha = parameters$alpha %||% 0.7
    ) +
    theme_minimal() +
    labs(
      title = parameters$title %||% "Bar Plot",
      x = parameters$x_label %||% parameters$x,
      y = parameters$y_label %||% parameters$y
    )
  
  # Add color mapping if specified
  if (!is.null(parameters$color_by)) {
    p <- p + aes_string(fill = parameters$color_by) +
      scale_fill_viridis_d()
  }
  
  # Rotate x-axis labels if needed
  if (parameters$rotate_x_labels) {
    p <- p + theme(axis.text.x = element_text(angle = 45, hjust = 1))
  }
  
  return(p)
}

#' Create histogram
#'
#' @param data Data frame
#' @param parameters Plot parameters
#' @return Histogram
create_histogram <- function(data, parameters) {
  library(ggplot2)
  
  p <- ggplot(data, aes_string(x = parameters$x)) +
    geom_histogram(
      bins = parameters$bins %||% 30,
      fill = parameters$fill %||% "steelblue",
      alpha = parameters$alpha %||% 0.7,
      color = parameters$color %||% "white"
    ) +
    theme_minimal() +
    labs(
      title = parameters$title %||% "Histogram",
      x = parameters$x_label %||% parameters$x,
      y = "Frequency"
    )
  
  # Add density curve if requested
  if (parameters$add_density) {
    p <- p + geom_density(alpha = 0.5, color = "red")
  }
  
  # Add normal curve if requested
  if (parameters$add_normal_curve) {
    p <- p + stat_function(fun = dnorm, 
                          args = list(mean = mean(data[[parameters$x]], na.rm = TRUE),
                                    sd = sd(data[[parameters$x]], na.rm = TRUE)),
                          color = "red", size = 1)
  }
  
  return(p)
}

#' Create box plot
#'
#' @param data Data frame
#' @param parameters Plot parameters
#' @return Box plot
create_box_plot <- function(data, parameters) {
  library(ggplot2)
  
  p <- ggplot(data, aes_string(x = parameters$x, y = parameters$y)) +
    geom_boxplot(
      fill = parameters$fill %||% "steelblue",
      alpha = parameters$alpha %||% 0.7
    ) +
    theme_minimal() +
    labs(
      title = parameters$title %||% "Box Plot",
      x = parameters$x_label %||% parameters$x,
      y = parameters$y_label %||% parameters$y
    )
  
  # Add jitter if requested
  if (parameters$add_jitter) {
    p <- p + geom_jitter(alpha = 0.3, width = 0.2)
  }
  
  # Add violin plot if requested
  if (parameters$add_violin) {
    p <- p + geom_violin(alpha = 0.5, fill = "lightblue")
  }
  
  return(p)
}

#' Create heatmap
#'
#' @param data Data frame
#' @param parameters Plot parameters
#' @return Heatmap
create_heatmap <- function(data, parameters) {
  library(ggplot2)
  
  # Reshape data for heatmap
  heatmap_data <- reshape2::melt(data, id.vars = parameters$id_vars)
  
  p <- ggplot(heatmap_data, aes_string(x = "variable", y = "value", fill = "value")) +
    geom_tile() +
    scale_fill_viridis_c(name = parameters$fill_label %||% "Value") +
    theme_minimal() +
    labs(
      title = parameters$title %||% "Heatmap",
      x = parameters$x_label %||% "Variable",
      y = parameters$y_label %||% "Value"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  return(p)
}
```

### Advanced ggplot2 Patterns

```r
# R/01-static-visualizations.R (continued)

#' Create multi-panel plots
#'
#' @param data Data frame
#' @param parameters Plot parameters
#' @return Multi-panel plot
create_multi_panel_plot <- function(data, parameters) {
  library(ggplot2)
  
  # Create base plot
  base_plot <- ggplot(data, aes_string(x = parameters$x, y = parameters$y)) +
    geom_point(alpha = 0.7) +
    theme_minimal()
  
  # Add faceting
  if (!is.null(parameters$facet_by)) {
    base_plot <- base_plot + facet_wrap(as.formula(paste("~", parameters$facet_by)))
  }
  
  # Add color mapping
  if (!is.null(parameters$color_by)) {
    base_plot <- base_plot + aes_string(color = parameters$color_by) +
      scale_color_viridis_d()
  }
  
  return(base_plot)
}

#' Create statistical plots
#'
#' @param data Data frame
#' @param parameters Plot parameters
#' @return Statistical plot
create_statistical_plot <- function(data, parameters) {
  library(ggplot2)
  
  p <- ggplot(data, aes_string(x = parameters$x, y = parameters$y)) +
    geom_point(alpha = 0.7) +
    theme_minimal()
  
  # Add regression line
  if (parameters$add_regression) {
    p <- p + geom_smooth(method = "lm", se = parameters$show_confidence_interval)
  }
  
  # Add correlation coefficient
  if (parameters$add_correlation) {
    cor_coef <- cor(data[[parameters$x]], data[[parameters$y]], use = "complete.obs")
    p <- p + annotate("text", x = Inf, y = Inf, 
                     label = paste("r =", round(cor_coef, 3)),
                     hjust = 1, vjust = 1)
  }
  
  return(p)
}

#' Create time series plots
#'
#' @param data Data frame
#' @param parameters Plot parameters
#' @return Time series plot
create_time_series_plot <- function(data, parameters) {
  library(ggplot2)
  
  p <- ggplot(data, aes_string(x = parameters$x, y = parameters$y)) +
    geom_line(size = 1) +
    theme_minimal() +
    labs(
      title = parameters$title %||% "Time Series Plot",
      x = parameters$x_label %||% parameters$x,
      y = parameters$y_label %||% parameters$y
    )
  
  # Add trend line
  if (parameters$add_trend) {
    p <- p + geom_smooth(method = "loess", se = FALSE, color = "red")
  }
  
  # Add seasonal decomposition
  if (parameters$add_seasonal) {
    # This would require time series decomposition
    # Implementation depends on specific requirements
  }
  
  return(p)
}
```

## Interactive Visualizations

### Plotly Integration

```r
# R/02-interactive-visualizations.R

#' Create interactive visualizations
#'
#' @param data Data frame
#' @param visualization_type Type of visualization
#' @param parameters Visualization parameters
#' @return Interactive visualization
create_interactive_visualization <- function(data, visualization_type, parameters = list()) {
  switch(visualization_type,
    "plotly_scatter" = create_plotly_scatter(data, parameters),
    "plotly_line" = create_plotly_line(data, parameters),
    "plotly_bar" = create_plotly_bar(data, parameters),
    "plotly_heatmap" = create_plotly_heatmap(data, parameters),
    "leaflet_map" = create_leaflet_map(data, parameters),
    stop("Unsupported visualization type: ", visualization_type)
  )
}

#' Create Plotly scatter plot
#'
#' @param data Data frame
#' @param parameters Plot parameters
#' @return Plotly scatter plot
create_plotly_scatter <- function(data, parameters) {
  library(plotly)
  
  p <- plot_ly(
    data = data,
    x = ~get(parameters$x),
    y = ~get(parameters$y),
    type = "scatter",
    mode = "markers",
    marker = list(
      size = parameters$size %||% 8,
      color = parameters$color %||% "steelblue",
      opacity = parameters$opacity %||% 0.7
    ),
    text = parameters$text,
    hovertemplate = parameters$hovertemplate
  ) %>%
    layout(
      title = parameters$title %||% "Scatter Plot",
      xaxis = list(title = parameters$x_label %||% parameters$x),
      yaxis = list(title = parameters$y_label %||% parameters$y)
    )
  
  return(p)
}

#' Create Plotly line plot
#'
#' @param data Data frame
#' @param parameters Plot parameters
#' @return Plotly line plot
create_plotly_line <- function(data, parameters) {
  library(plotly)
  
  p <- plot_ly(
    data = data,
    x = ~get(parameters$x),
    y = ~get(parameters$y),
    type = "scatter",
    mode = "lines",
    line = list(
      color = parameters$color %||% "steelblue",
      width = parameters$width %||% 2
    )
  ) %>%
    layout(
      title = parameters$title %||% "Line Plot",
      xaxis = list(title = parameters$x_label %||% parameters$x),
      yaxis = list(title = parameters$y_label %||% parameters$y)
    )
  
  return(p)
}

#' Create Plotly bar plot
#'
#' @param data Data frame
#' @param parameters Plot parameters
#' @return Plotly bar plot
create_plotly_bar <- function(data, parameters) {
  library(plotly)
  
  p <- plot_ly(
    data = data,
    x = ~get(parameters$x),
    y = ~get(parameters$y),
    type = "bar",
    marker = list(
      color = parameters$color %||% "steelblue",
      opacity = parameters$opacity %||% 0.7
    )
  ) %>%
    layout(
      title = parameters$title %||% "Bar Plot",
      xaxis = list(title = parameters$x_label %||% parameters$x),
      yaxis = list(title = parameters$y_label %||% parameters$y)
    )
  
  return(p)
}

#' Create Plotly heatmap
#'
#' @param data Data frame
#' @param parameters Plot parameters
#' @return Plotly heatmap
create_plotly_heatmap <- function(data, parameters) {
  library(plotly)
  
  # Reshape data for heatmap
  heatmap_data <- reshape2::melt(data, id.vars = parameters$id_vars)
  
  p <- plot_ly(
    data = heatmap_data,
    x = ~variable,
    y = ~value,
    z = ~value,
    type = "heatmap",
    colorscale = parameters$colorscale %||% "Viridis"
  ) %>%
    layout(
      title = parameters$title %||% "Heatmap",
      xaxis = list(title = parameters$x_label %||% "Variable"),
      yaxis = list(title = parameters$y_label %||% "Value")
    )
  
  return(p)
}

#' Create Leaflet map
#'
#' @param data Data frame
#' @param parameters Plot parameters
#' @return Leaflet map
create_leaflet_map <- function(data, parameters) {
  library(leaflet)
  
  # Create base map
  map <- leaflet() %>%
    addTiles()
  
  # Add markers
  if (parameters$add_markers) {
    map <- map %>%
      addCircleMarkers(
        data = data,
        lng = ~get(parameters$lng),
        lat = ~get(parameters$lat),
        radius = parameters$radius %||% 5,
        color = parameters$color %||% "blue",
        popup = parameters$popup
      )
  }
  
  # Add polygons
  if (parameters$add_polygons) {
    map <- map %>%
      addPolygons(
        data = data,
        color = parameters$polygon_color %||% "red",
        fillColor = parameters$fill_color %||% "lightblue",
        popup = parameters$popup
      )
  }
  
  return(map)
}
```

### Shiny Integration

```r
# R/02-interactive-visualizations.R (continued)

#' Create Shiny visualization app
#'
#' @param data Data frame
#' @param parameters App parameters
#' @return Shiny app
create_shiny_visualization_app <- function(data, parameters) {
  library(shiny)
  
  ui <- fluidPage(
    titlePanel(parameters$title %||% "Data Visualization App"),
    
    sidebarLayout(
      sidebarPanel(
        selectInput("plot_type", "Plot Type", 
                   choices = c("Scatter", "Line", "Bar", "Histogram", "Box Plot")),
        selectInput("x_var", "X Variable", choices = names(data)),
        selectInput("y_var", "Y Variable", choices = names(data)),
        sliderInput("alpha", "Transparency", min = 0, max = 1, value = 0.7),
        checkboxInput("add_trend", "Add Trend Line", value = FALSE)
      ),
      
      mainPanel(
        plotOutput("plot"),
        verbatimTextOutput("summary")
      )
    )
  )
  
  server <- function(input, output) {
    output$plot <- renderPlot({
      create_dynamic_plot(data, input)
    })
    
    output$summary <- renderText({
      create_plot_summary(data, input)
    })
  }
  
  return(shinyApp(ui = ui, server = server))
}

#' Create dynamic plot based on user input
#'
#' @param data Data frame
#' @param input User input
#' @return Dynamic plot
create_dynamic_plot <- function(data, input) {
  switch(input$plot_type,
    "Scatter" = create_scatter_plot(data, list(
      x = input$x_var,
      y = input$y_var,
      alpha = input$alpha,
      add_trend_line = input$add_trend
    )),
    "Line" = create_line_plot(data, list(
      x = input$x_var,
      y = input$y_var,
      alpha = input$alpha
    )),
    "Bar" = create_bar_plot(data, list(
      x = input$x_var,
      y = input$y_var,
      alpha = input$alpha
    )),
    "Histogram" = create_histogram(data, list(
      x = input$x_var,
      alpha = input$alpha
    )),
    "Box Plot" = create_box_plot(data, list(
      x = input$x_var,
      y = input$y_var,
      alpha = input$alpha
    ))
  )
}

#' Create plot summary
#'
#' @param data Data frame
#' @param input User input
#' @return Plot summary
create_plot_summary <- function(data, input) {
  if (input$plot_type %in% c("Scatter", "Line")) {
    cor_coef <- cor(data[[input$x_var]], data[[input$y_var]], use = "complete.obs")
    return(paste("Correlation coefficient:", round(cor_coef, 3)))
  } else if (input$plot_type == "Histogram") {
    mean_val <- mean(data[[input$x_var]], na.rm = TRUE)
    sd_val <- sd(data[[input$x_var]], na.rm = TRUE)
    return(paste("Mean:", round(mean_val, 3), "SD:", round(sd_val, 3)))
  } else {
    return("Summary statistics not available for this plot type.")
  }
}
```

## Visualization Design

### Color Theory and Palettes

```r
# R/03-visualization-design.R

#' Create color palettes
#'
#' @param palette_type Type of color palette
#' @param parameters Palette parameters
#' @return Color palette
create_color_palette <- function(palette_type, parameters = list()) {
  switch(palette_type,
    "viridis" = create_viridis_palette(parameters),
    "brewer" = create_brewer_palette(parameters),
    "custom" = create_custom_palette(parameters),
    "diverging" = create_diverging_palette(parameters),
    stop("Unsupported palette type: ", palette_type)
  )
}

#' Create Viridis palette
#'
#' @param parameters Palette parameters
#' @return Viridis palette
create_viridis_palette <- function(parameters) {
  library(viridis)
  
  palette <- list(
    type = "viridis",
    colors = viridis(parameters$n_colors %||% 10),
    name = parameters$name %||% "Viridis"
  )
  
  return(palette)
}

#' Create Brewer palette
#'
#' @param parameters Palette parameters
#' @return Brewer palette
create_brewer_palette <- function(parameters) {
  library(RColorBrewer)
  
  palette <- list(
    type = "brewer",
    colors = brewer.pal(parameters$n_colors %||% 10, parameters$palette %||% "Set1"),
    name = parameters$name %||% "Brewer"
  )
  
  return(palette)
}

#' Create custom palette
#'
#' @param parameters Palette parameters
#' @return Custom palette
create_custom_palette <- function(parameters) {
  palette <- list(
    type = "custom",
    colors = parameters$colors,
    name = parameters$name %||% "Custom"
  )
  
  return(palette)
}

#' Create diverging palette
#'
#' @param parameters Palette parameters
#' @return Diverging palette
create_diverging_palette <- function(parameters) {
  library(RColorBrewer)
  
  palette <- list(
    type = "diverging",
    colors = brewer.pal(parameters$n_colors %||% 10, "RdBu"),
    name = parameters$name %||% "Diverging"
  )
  
  return(palette)
}
```

### Typography and Layout

```r
# R/03-visualization-design.R (continued)

#' Create typography settings
#'
#' @param font_family Font family
#' @param font_size Base font size
#' @param parameters Typography parameters
#' @return Typography settings
create_typography_settings <- function(font_family = "Arial", font_size = 12, parameters = list()) {
  typography <- list(
    font_family = font_family,
    font_size = font_size,
    title_size = parameters$title_size %||% font_size * 1.5,
    axis_size = parameters$axis_size %||% font_size * 0.8,
    legend_size = parameters$legend_size %||% font_size * 0.9,
    theme = theme(
      text = element_text(family = font_family, size = font_size),
      plot.title = element_text(size = font_size * 1.5, face = "bold"),
      axis.text = element_text(size = font_size * 0.8),
      legend.text = element_text(size = font_size * 0.9)
    )
  )
  
  return(typography)
}

#' Create layout settings
#'
#' @param parameters Layout parameters
#' @return Layout settings
create_layout_settings <- function(parameters = list()) {
  layout <- list(
    width = parameters$width %||% 8,
    height = parameters$height %||% 6,
    dpi = parameters$dpi %||% 300,
    units = parameters$units %||% "in",
    margins = parameters$margins %||% c(1, 1, 1, 1)
  )
  
  return(layout)
}

#' Apply design theme
#'
#' @param plot ggplot object
#' @param theme_type Type of theme
#' @param parameters Theme parameters
#' @return Themed plot
apply_design_theme <- function(plot, theme_type, parameters = list()) {
  switch(theme_type,
    "minimal" = apply_minimal_theme(plot, parameters),
    "classic" = apply_classic_theme(plot, parameters),
    "dark" = apply_dark_theme(plot, parameters),
    "custom" = apply_custom_theme(plot, parameters),
    stop("Unsupported theme type: ", theme_type)
  )
}

#' Apply minimal theme
#'
#' @param plot ggplot object
#' @param parameters Theme parameters
#' @return Minimal themed plot
apply_minimal_theme <- function(plot, parameters) {
  plot + theme_minimal() +
    theme(
      panel.grid.major = element_line(color = "grey90", size = 0.5),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "black", size = 0.5)
    )
}

#' Apply classic theme
#'
#' @param plot ggplot object
#' @param parameters Theme parameters
#' @return Classic themed plot
apply_classic_theme <- function(plot, parameters) {
  plot + theme_classic() +
    theme(
      axis.line = element_line(color = "black", size = 0.5),
      axis.ticks = element_line(color = "black", size = 0.5)
    )
}

#' Apply dark theme
#'
#' @param plot ggplot object
#' @param parameters Theme parameters
#' @return Dark themed plot
apply_dark_theme <- function(plot, parameters) {
  plot + theme_dark() +
    theme(
      panel.background = element_rect(fill = "black"),
      plot.background = element_rect(fill = "black"),
      text = element_text(color = "white"),
      axis.text = element_text(color = "white"),
      axis.line = element_line(color = "white")
    )
}

#' Apply custom theme
#'
#' @param plot ggplot object
#' @param parameters Theme parameters
#' @return Custom themed plot
apply_custom_theme <- function(plot, parameters) {
  plot + theme(
    text = element_text(family = parameters$font_family %||% "Arial"),
    plot.title = element_text(size = parameters$title_size %||% 16, face = "bold"),
    axis.text = element_text(size = parameters$axis_size %||% 12),
    legend.text = element_text(size = parameters$legend_size %||% 12)
  )
}
```

## Performance Optimization

### Rendering Optimization

```r
# R/04-performance-optimization.R

#' Optimize visualization performance
#'
#' @param plot ggplot object
#' @param optimization_type Type of optimization
#' @param parameters Optimization parameters
#' @return Optimized plot
optimize_visualization_performance <- function(plot, optimization_type, parameters = list()) {
  switch(optimization_type,
    "reduce_data" = optimize_by_reducing_data(plot, parameters),
    "simplify_geometry" = optimize_by_simplifying_geometry(plot, parameters),
    "optimize_colors" = optimize_by_optimizing_colors(plot, parameters),
    "cache_rendering" = optimize_by_caching_rendering(plot, parameters),
    stop("Unsupported optimization type: ", optimization_type)
  )
}

#' Optimize by reducing data
#'
#' @param plot ggplot object
#' @param parameters Optimization parameters
#' @return Data-reduced plot
optimize_by_reducing_data <- function(plot, parameters) {
  # Sample data if too large
  if (parameters$max_points && nrow(plot$data) > parameters$max_points) {
    sampled_data <- plot$data[sample(nrow(plot$data), parameters$max_points), ]
    plot$data <- sampled_data
  }
  
  # Filter data based on criteria
  if (!is.null(parameters$filter_criteria)) {
    filtered_data <- subset(plot$data, eval(parse(text = parameters$filter_criteria)))
    plot$data <- filtered_data
  }
  
  return(plot)
}

#' Optimize by simplifying geometry
#'
#' @param plot ggplot object
#' @param parameters Optimization parameters
#' @return Geometry-simplified plot
optimize_by_simplifying_geometry <- function(plot, parameters) {
  # Simplify line plots
  if (parameters$simplify_lines) {
    plot <- plot + geom_line(alpha = parameters$alpha %||% 0.7)
  }
  
  # Simplify point plots
  if (parameters$simplify_points) {
    plot <- plot + geom_point(alpha = parameters$alpha %||% 0.7, size = parameters$size %||% 1)
  }
  
  return(plot)
}

#' Optimize by optimizing colors
#'
#' @param plot ggplot object
#' @param parameters Optimization parameters
#' @return Color-optimized plot
optimize_by_optimizing_colors <- function(plot, parameters) {
  # Use color palettes that are more efficient
  if (parameters$use_efficient_palette) {
    plot <- plot + scale_color_viridis_d()
  }
  
  # Reduce color complexity
  if (parameters$reduce_color_complexity) {
    plot <- plot + scale_color_manual(values = rep(c("red", "blue", "green"), length.out = 10))
  }
  
  return(plot)
}

#' Optimize by caching rendering
#'
#' @param plot ggplot object
#' @param parameters Optimization parameters
#' @return Cached plot
optimize_by_caching_rendering <- function(plot, parameters) {
  # Cache plot rendering
  if (parameters$cache_plot) {
    plot_cache <- list(
      plot = plot,
      timestamp = Sys.time(),
      parameters = parameters
    )
    
    # Save to cache
    saveRDS(plot_cache, parameters$cache_file)
  }
  
  return(plot)
}
```

### File Size Optimization

```r
# R/04-performance-optimization.R (continued)

#' Optimize file size
#'
#' @param plot ggplot object
#' @param output_format Output format
#' @param parameters Optimization parameters
#' @return File size optimized plot
optimize_file_size <- function(plot, output_format, parameters = list()) {
  switch(output_format,
    "png" = optimize_png_size(plot, parameters),
    "pdf" = optimize_pdf_size(plot, parameters),
    "svg" = optimize_svg_size(plot, parameters),
    "jpeg" = optimize_jpeg_size(plot, parameters),
    stop("Unsupported output format: ", output_format)
  )
}

#' Optimize PNG file size
#'
#' @param plot ggplot object
#' @param parameters Optimization parameters
#' @return PNG optimized plot
optimize_png_size <- function(plot, parameters) {
  # Set optimal PNG parameters
  png_params <- list(
    width = parameters$width %||% 800,
    height = parameters$height %||% 600,
    res = parameters$res %||% 72,
    type = parameters$type %||% "cairo"
  )
  
  return(png_params)
}

#' Optimize PDF file size
#'
#' @param plot ggplot object
#' @param parameters Optimization parameters
#' @return PDF optimized plot
optimize_pdf_size <- function(plot, parameters) {
  # Set optimal PDF parameters
  pdf_params <- list(
    width = parameters$width %||% 8,
    height = parameters$height %||% 6,
    pointsize = parameters$pointsize %||% 12,
    compress = parameters$compress %||% TRUE
  )
  
  return(pdf_params)
}

#' Optimize SVG file size
#'
#' @param plot ggplot object
#' @param parameters Optimization parameters
#' @return SVG optimized plot
optimize_svg_size <- function(plot, parameters) {
  # Set optimal SVG parameters
  svg_params <- list(
    width = parameters$width %||% 8,
    height = parameters$height %||% 6,
    pointsize = parameters$pointsize %||% 12
  )
  
  return(svg_params)
}

#' Optimize JPEG file size
#'
#' @param plot ggplot object
#' @param parameters Optimization parameters
#' @return JPEG optimized plot
optimize_jpeg_size <- function(plot, parameters) {
  # Set optimal JPEG parameters
  jpeg_params <- list(
    width = parameters$width %||% 800,
    height = parameters$height %||% 600,
    quality = parameters$quality %||% 75,
    res = parameters$res %||% 72
  )
  
  return(jpeg_params)
}
```

## TL;DR Runbook

### Quick Start

```r
# 1. Create static visualization
plot <- create_static_visualization(data, "scatter_plot", list(x = "x", y = "y"))

# 2. Create interactive visualization
interactive_plot <- create_interactive_visualization(data, "plotly_scatter", list(x = "x", y = "y"))

# 3. Apply design theme
themed_plot <- apply_design_theme(plot, "minimal", list())

# 4. Optimize performance
optimized_plot <- optimize_visualization_performance(plot, "reduce_data", list(max_points = 1000))

# 5. Save with optimization
ggsave("plot.png", optimized_plot, width = 8, height = 6, dpi = 300)
```

### Essential Patterns

```r
# Complete visualization pipeline
create_visualization_pipeline <- function(data, viz_config) {
  # Create base visualization
  plot <- create_static_visualization(data, viz_config$plot_type, viz_config$plot_params)
  
  # Apply design theme
  themed_plot <- apply_design_theme(plot, viz_config$theme_type, viz_config$theme_params)
  
  # Optimize performance
  optimized_plot <- optimize_visualization_performance(themed_plot, viz_config$optimization_type, viz_config$optimization_params)
  
  # Save with optimization
  file_params <- optimize_file_size(optimized_plot, viz_config$output_format, viz_config$file_params)
  
  return(list(
    plot = optimized_plot,
    file_params = file_params
  ))
}
```

---

*This guide provides the complete machinery for creating compelling visualizations in R. Each pattern includes implementation examples, design strategies, and real-world usage patterns for enterprise deployment.*

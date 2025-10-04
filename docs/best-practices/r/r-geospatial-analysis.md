# R Geospatial Analysis Best Practices

**Objective**: Master senior-level R geospatial analysis patterns for production systems. When you need to analyze spatial data, when you want to build location-aware applications, when you need enterprise-grade geospatial patterns—these best practices become your weapon of choice.

## Core Principles

- **Spatial Data Types**: Handle points, lines, polygons, and complex geometries
- **Coordinate Systems**: Manage different coordinate reference systems (CRS)
- **Spatial Operations**: Perform geometric calculations and spatial analysis
- **Visualization**: Create effective spatial visualizations
- **Performance**: Optimize spatial operations for large datasets

## Spatial Data Handling

### Spatial Data Import and Export

```r
# R/01-spatial-data-handling.R

#' Comprehensive spatial data import
#'
#' @param file_path Path to spatial data file
#' @param file_type Type of spatial data file
#' @param crs Coordinate reference system
#' @return Spatial data object
import_spatial_data <- function(file_path, file_type = NULL, crs = NULL) {
  if (is.null(file_type)) {
    file_type <- detect_file_type(file_path)
  }
  
  switch(file_type,
    "shapefile" = import_shapefile(file_path, crs),
    "geojson" = import_geojson(file_path, crs),
    "kml" = import_kml(file_path, crs),
    "gpx" = import_gpx(file_path, crs),
    "csv" = import_spatial_csv(file_path, crs),
    stop("Unsupported file type: ", file_type)
  )
}

#' Detect file type from extension
#'
#' @param file_path File path
#' @return File type
detect_file_type <- function(file_path) {
  extension <- tolower(tools::file_ext(file_path))
  
  switch(extension,
    "shp" = "shapefile",
    "geojson" = "geojson",
    "kml" = "kml",
    "gpx" = "gpx",
    "csv" = "csv",
    "unknown"
  )
}

#' Import shapefile
#'
#' @param file_path Path to shapefile
#' @param crs Coordinate reference system
#' @return Spatial data object
import_shapefile <- function(file_path, crs = NULL) {
  library(sf)
  
  # Read shapefile
  spatial_data <- st_read(file_path, quiet = TRUE)
  
  # Set CRS if provided
  if (!is.null(crs)) {
    spatial_data <- st_set_crs(spatial_data, crs)
  }
  
  return(spatial_data)
}

#' Import GeoJSON
#'
#' @param file_path Path to GeoJSON file
#' @param crs Coordinate reference system
#' @return Spatial data object
import_geojson <- function(file_path, crs = NULL) {
  library(sf)
  
  # Read GeoJSON
  spatial_data <- st_read(file_path, quiet = TRUE)
  
  # Set CRS if provided
  if (!is.null(crs)) {
    spatial_data <- st_set_crs(spatial_data, crs)
  }
  
  return(spatial_data)
}

#' Import KML
#'
#' @param file_path Path to KML file
#' @param crs Coordinate reference system
#' @return Spatial data object
import_kml <- function(file_path, crs = NULL) {
  library(sf)
  
  # Read KML
  spatial_data <- st_read(file_path, quiet = TRUE)
  
  # Set CRS if provided
  if (!is.null(crs)) {
    spatial_data <- st_set_crs(spatial_data, crs)
  }
  
  return(spatial_data)
}

#' Import GPX
#'
#' @param file_path Path to GPX file
#' @param crs Coordinate reference system
#' @return Spatial data object
import_gpx <- function(file_path, crs = NULL) {
  library(sf)
  
  # Read GPX
  spatial_data <- st_read(file_path, quiet = TRUE)
  
  # Set CRS if provided
  if (!is.null(crs)) {
    spatial_data <- st_set_crs(spatial_data, crs)
  }
  
  return(spatial_data)
}

#' Import spatial CSV
#'
#' @param file_path Path to CSV file
#' @param crs Coordinate reference system
#' @param x_col X coordinate column name
#' @param y_col Y coordinate column name
#' @return Spatial data object
import_spatial_csv <- function(file_path, crs = NULL, x_col = "x", y_col = "y") {
  library(sf)
  
  # Read CSV
  data <- read.csv(file_path)
  
  # Create spatial object
  spatial_data <- st_as_sf(data, coords = c(x_col, y_col), crs = crs)
  
  return(spatial_data)
}

#' Export spatial data
#'
#' @param spatial_data Spatial data object
#' @param file_path Output file path
#' @param file_type Output file type
#' @return Export status
export_spatial_data <- function(spatial_data, file_path, file_type = NULL) {
  if (is.null(file_type)) {
    file_type <- detect_file_type(file_path)
  }
  
  switch(file_type,
    "shapefile" = export_shapefile(spatial_data, file_path),
    "geojson" = export_geojson(spatial_data, file_path),
    "kml" = export_kml(spatial_data, file_path),
    "csv" = export_spatial_csv(spatial_data, file_path),
    stop("Unsupported file type: ", file_type)
  )
}

#' Export shapefile
#'
#' @param spatial_data Spatial data object
#' @param file_path Output file path
#' @return Export status
export_shapefile <- function(spatial_data, file_path) {
  library(sf)
  
  # Write shapefile
  st_write(spatial_data, file_path, delete_dsn = TRUE, quiet = TRUE)
  
  return(TRUE)
}

#' Export GeoJSON
#'
#' @param spatial_data Spatial data object
#' @param file_path Output file path
#' @return Export status
export_geojson <- function(spatial_data, file_path) {
  library(sf)
  
  # Write GeoJSON
  st_write(spatial_data, file_path, delete_dsn = TRUE, quiet = TRUE)
  
  return(TRUE)
}

#' Export KML
#'
#' @param spatial_data Spatial data object
#' @param file_path Output file path
#' @return Export status
export_kml <- function(spatial_data, file_path) {
  library(sf)
  
  # Write KML
  st_write(spatial_data, file_path, delete_dsn = TRUE, quiet = TRUE)
  
  return(TRUE)
}

#' Export spatial CSV
#'
#' @param spatial_data Spatial data object
#' @param file_path Output file path
#' @return Export status
export_spatial_csv <- function(spatial_data, file_path) {
  library(sf)
  
  # Extract coordinates
  coords <- st_coordinates(spatial_data)
  
  # Create data frame
  data <- st_drop_geometry(spatial_data)
  data$x <- coords[, 1]
  data$y <- coords[, 2]
  
  # Write CSV
  write.csv(data, file_path, row.names = FALSE)
  
  return(TRUE)
}
```

### Coordinate Reference Systems

```r
# R/01-spatial-data-handling.R (continued)

#' Transform coordinate reference system
#'
#' @param spatial_data Spatial data object
#' @param target_crs Target coordinate reference system
#' @return Transformed spatial data
transform_crs <- function(spatial_data, target_crs) {
  library(sf)
  
  # Transform CRS
  transformed_data <- st_transform(spatial_data, target_crs)
  
  return(transformed_data)
}

#' Get coordinate reference system information
#'
#' @param spatial_data Spatial data object
#' @return CRS information
get_crs_info <- function(spatial_data) {
  library(sf)
  
  crs <- st_crs(spatial_data)
  
  return(list(
    epsg_code = crs$epsg,
    proj4_string = crs$proj4string,
    wkt = crs$wkt,
    is_geographic = crs$is_geographic,
    is_projected = crs$is_projected
  ))
}

#' Set coordinate reference system
#'
#' @param spatial_data Spatial data object
#' @param crs Coordinate reference system
#' @return Spatial data with set CRS
set_crs <- function(spatial_data, crs) {
  library(sf)
  
  # Set CRS
  spatial_data <- st_set_crs(spatial_data, crs)
  
  return(spatial_data)
}

#' Common coordinate reference systems
get_common_crs <- function() {
  list(
    wgs84 = 4326,
    web_mercator = 3857,
    utm_zone_10n = 32610,
    utm_zone_11n = 32611,
    utm_zone_12n = 32612,
    albers_conus = 5070,
    lambert_conformal_conic = 102004
  )
}
```

## Spatial Operations

### Geometric Operations

```r
# R/02-spatial-operations.R

#' Comprehensive spatial operations
#'
#' @param spatial_data1 First spatial data object
#' @param spatial_data2 Second spatial data object
#' @param operation Spatial operation to perform
#' @return Result of spatial operation
perform_spatial_operation <- function(spatial_data1, spatial_data2, operation) {
  switch(operation,
    "intersection" = st_intersection(spatial_data1, spatial_data2),
    "union" = st_union(spatial_data1, spatial_data2),
    "difference" = st_difference(spatial_data1, spatial_data2),
    "symmetric_difference" = st_sym_difference(spatial_data1, spatial_data2),
    "contains" = st_contains(spatial_data1, spatial_data2),
    "within" = st_within(spatial_data1, spatial_data2),
    "touches" = st_touches(spatial_data1, spatial_data2),
    "overlaps" = st_overlaps(spatial_data1, spatial_data2),
    "crosses" = st_crosses(spatial_data1, spatial_data2),
    "disjoint" = st_disjoint(spatial_data1, spatial_data2),
    stop("Unsupported operation: ", operation)
  )
}

#' Calculate spatial distances
#'
#' @param spatial_data1 First spatial data object
#' @param spatial_data2 Second spatial data object
#' @param method Distance calculation method
#' @return Distance matrix
calculate_spatial_distances <- function(spatial_data1, spatial_data2, method = "euclidean") {
  library(sf)
  
  if (method == "euclidean") {
    distances <- st_distance(spatial_data1, spatial_data2)
  } else if (method == "haversine") {
    distances <- st_distance(spatial_data1, spatial_data2, which = "Haversine")
  } else if (method == "great_circle") {
    distances <- st_distance(spatial_data1, spatial_data2, which = "Great Circle")
  }
  
  return(distances)
}

#' Find nearest neighbors
#'
#' @param spatial_data1 First spatial data object
#' @param spatial_data2 Second spatial data object
#' @param k Number of nearest neighbors
#' @return Nearest neighbor indices
find_nearest_neighbors <- function(spatial_data1, spatial_data2, k = 1) {
  library(sf)
  
  # Calculate distances
  distances <- st_distance(spatial_data1, spatial_data2)
  
  # Find nearest neighbors
  nearest_indices <- apply(distances, 1, function(x) order(x)[1:k])
  
  return(nearest_indices)
}

#' Calculate spatial statistics
#'
#' @param spatial_data Spatial data object
#' @param statistics Statistics to calculate
#' @return Spatial statistics
calculate_spatial_statistics <- function(spatial_data, statistics = c("area", "length", "centroid")) {
  library(sf)
  
  results <- list()
  
  if ("area" %in% statistics) {
    results$area <- st_area(spatial_data)
  }
  
  if ("length" %in% statistics) {
    results$length <- st_length(spatial_data)
  }
  
  if ("centroid" %in% statistics) {
    results$centroid <- st_centroid(spatial_data)
  }
  
  if ("boundary" %in% statistics) {
    results$boundary <- st_boundary(spatial_data)
  }
  
  if ("convex_hull" %in% statistics) {
    results$convex_hull <- st_convex_hull(spatial_data)
  }
  
  return(results)
}

#' Perform spatial joins
#'
#' @param spatial_data1 First spatial data object
#' @param spatial_data2 Second spatial data object
#' @param join_type Type of spatial join
#' @return Joined spatial data
perform_spatial_join <- function(spatial_data1, spatial_data2, join_type = "intersects") {
  library(sf)
  
  switch(join_type,
    "intersects" = st_join(spatial_data1, spatial_data2, join = st_intersects),
    "contains" = st_join(spatial_data1, spatial_data2, join = st_contains),
    "within" = st_join(spatial_data1, spatial_data2, join = st_within),
    "touches" = st_join(spatial_data1, spatial_data2, join = st_touches),
    "overlaps" = st_join(spatial_data1, spatial_data2, join = st_overlaps),
    stop("Unsupported join type: ", join_type)
  )
}
```

### Spatial Analysis

```r
# R/02-spatial-operations.R (continued)

#' Perform spatial analysis
#'
#' @param spatial_data Spatial data object
#' @param analysis_type Type of spatial analysis
#' @param parameters Analysis parameters
#' @return Analysis results
perform_spatial_analysis <- function(spatial_data, analysis_type, parameters = list()) {
  switch(analysis_type,
    "point_pattern" = analyze_point_pattern(spatial_data, parameters),
    "spatial_autocorrelation" = analyze_spatial_autocorrelation(spatial_data, parameters),
    "hotspot_analysis" = analyze_hotspots(spatial_data, parameters),
    "clustering" = analyze_clustering(spatial_data, parameters),
    "interpolation" = perform_interpolation(spatial_data, parameters),
    stop("Unsupported analysis type: ", analysis_type)
  )
}

#' Analyze point pattern
#'
#' @param spatial_data Spatial data object
#' @param parameters Analysis parameters
#' @return Point pattern analysis results
analyze_point_pattern <- function(spatial_data, parameters) {
  library(spatstat)
  
  # Convert to ppp object
  coords <- st_coordinates(spatial_data)
  ppp_object <- ppp(coords[, 1], coords[, 2], 
                    window = owin(range(coords[, 1]), range(coords[, 2])))
  
  # Calculate intensity
  intensity <- density(ppp_object, sigma = parameters$sigma)
  
  # Calculate nearest neighbor distances
  nn_distances <- nndist(ppp_object)
  
  # Calculate Ripley's K function
  k_function <- Kest(ppp_object)
  
  return(list(
    intensity = intensity,
    nn_distances = nn_distances,
    k_function = k_function
  ))
}

#' Analyze spatial autocorrelation
#'
#' @param spatial_data Spatial data object
#' @param parameters Analysis parameters
#' @return Spatial autocorrelation results
analyze_spatial_autocorrelation <- function(spatial_data, parameters) {
  library(spdep)
  
  # Create spatial weights
  coords <- st_coordinates(spatial_data)
  nb <- knn2nb(knearneigh(coords, k = parameters$k))
  weights <- nb2listw(nb)
  
  # Calculate Moran's I
  moran_i <- moran.test(spatial_data[[parameters$variable]], weights)
  
  # Calculate local Moran's I
  local_moran <- localmoran(spatial_data[[parameters$variable]], weights)
  
  return(list(
    moran_i = moran_i,
    local_moran = local_moran
  ))
}

#' Analyze hotspots
#'
#' @param spatial_data Spatial data object
#' @param parameters Analysis parameters
#' @return Hotspot analysis results
analyze_hotspots <- function(spatial_data, parameters) {
  library(spdep)
  
  # Create spatial weights
  coords <- st_coordinates(spatial_data)
  nb <- knn2nb(knearneigh(coords, k = parameters$k))
  weights <- nb2listw(nb)
  
  # Calculate Getis-Ord Gi* statistic
  gi_statistic <- localG(spatial_data[[parameters$variable]], weights)
  
  # Identify hotspots
  hotspots <- which(gi_statistic > 1.96)
  
  return(list(
    gi_statistic = gi_statistic,
    hotspots = hotspots
  ))
}

#' Analyze clustering
#'
#' @param spatial_data Spatial data object
#' @param parameters Analysis parameters
#' @return Clustering analysis results
analyze_clustering <- function(spatial_data, parameters) {
  library(cluster)
  
  # Extract coordinates
  coords <- st_coordinates(spatial_data)
  
  # Perform clustering
  if (parameters$method == "kmeans") {
    clusters <- kmeans(coords, centers = parameters$k)
  } else if (parameters$method == "hierarchical") {
    dist_matrix <- dist(coords)
    hclust_result <- hclust(dist_matrix)
    clusters <- cutree(hclust_result, k = parameters$k)
  }
  
  return(list(
    clusters = clusters,
    cluster_centers = if (parameters$method == "kmeans") clusters$centers else NULL
  ))
}

#' Perform spatial interpolation
#'
#' @param spatial_data Spatial data object
#' @param parameters Analysis parameters
#' @return Interpolation results
perform_interpolation <- function(spatial_data, parameters) {
  library(gstat)
  
  # Create interpolation grid
  bbox <- st_bbox(spatial_data)
  grid <- expand.grid(
    x = seq(bbox[1], bbox[3], length.out = parameters$grid_size),
    y = seq(bbox[2], bbox[4], length.out = parameters$grid_size)
  )
  
  # Convert to spatial object
  grid_spatial <- st_as_sf(grid, coords = c("x", "y"), crs = st_crs(spatial_data))
  
  # Perform interpolation
  if (parameters$method == "idw") {
    interpolated <- idw(
      formula = as.formula(paste(parameters$variable, "~ 1")),
      locations = spatial_data,
      newdata = grid_spatial,
      idp = parameters$idp
    )
  } else if (parameters$method == "kriging") {
    interpolated <- krige(
      formula = as.formula(paste(parameters$variable, "~ 1")),
      locations = spatial_data,
      newdata = grid_spatial
    )
  }
  
  return(interpolated)
}
```

## Spatial Visualization

### Static Maps

```r
# R/03-spatial-visualization.R

#' Create comprehensive spatial visualizations
#'
#' @param spatial_data Spatial data object
#' @param visualization_type Type of visualization
#' @param parameters Visualization parameters
#' @return Spatial visualization
create_spatial_visualization <- function(spatial_data, visualization_type, parameters = list()) {
  switch(visualization_type,
    "basic_map" = create_basic_map(spatial_data, parameters),
    "choropleth_map" = create_choropleth_map(spatial_data, parameters),
    "point_map" = create_point_map(spatial_data, parameters),
    "heat_map" = create_heat_map(spatial_data, parameters),
    "density_map" = create_density_map(spatial_data, parameters),
    stop("Unsupported visualization type: ", visualization_type)
  )
}

#' Create basic map
#'
#' @param spatial_data Spatial data object
#' @param parameters Visualization parameters
#' @return Basic map
create_basic_map <- function(spatial_data, parameters) {
  library(ggplot2)
  
  p <- ggplot(spatial_data) +
    geom_sf() +
    theme_minimal() +
    labs(title = parameters$title)
  
  return(p)
}

#' Create choropleth map
#'
#' @param spatial_data Spatial data object
#' @param parameters Visualization parameters
#' @return Choropleth map
create_choropleth_map <- function(spatial_data, parameters) {
  library(ggplot2)
  
  p <- ggplot(spatial_data) +
    geom_sf(aes(fill = !!sym(parameters$variable))) +
    scale_fill_viridis_c(name = parameters$variable) +
    theme_minimal() +
    labs(title = parameters$title)
  
  return(p)
}

#' Create point map
#'
#' @param spatial_data Spatial data object
#' @param parameters Visualization parameters
#' @return Point map
create_point_map <- function(spatial_data, parameters) {
  library(ggplot2)
  
  p <- ggplot(spatial_data) +
    geom_sf(aes(color = !!sym(parameters$variable)), size = parameters$point_size) +
    scale_color_viridis_c(name = parameters$variable) +
    theme_minimal() +
    labs(title = parameters$title)
  
  return(p)
}

#' Create heat map
#'
#' @param spatial_data Spatial data object
#' @param parameters Visualization parameters
#' @return Heat map
create_heat_map <- function(spatial_data, parameters) {
  library(ggplot2)
  
  # Create grid for heat map
  bbox <- st_bbox(spatial_data)
  grid <- expand.grid(
    x = seq(bbox[1], bbox[3], length.out = parameters$grid_size),
    y = seq(bbox[2], bbox[4], length.out = parameters$grid_size)
  )
  
  # Convert to spatial object
  grid_spatial <- st_as_sf(grid, coords = c("x", "y"), crs = st_crs(spatial_data))
  
  # Calculate density
  density <- st_distance(grid_spatial, spatial_data)
  density <- apply(density, 1, function(x) sum(exp(-x / parameters$bandwidth)))
  
  # Create data frame
  heat_data <- data.frame(
    x = grid$x,
    y = grid$y,
    density = density
  )
  
  p <- ggplot(heat_data, aes(x = x, y = y, fill = density)) +
    geom_tile() +
    scale_fill_viridis_c(name = "Density") +
    theme_minimal() +
    labs(title = parameters$title)
  
  return(p)
}

#' Create density map
#'
#' @param spatial_data Spatial data object
#' @param parameters Visualization parameters
#' @return Density map
create_density_map <- function(spatial_data, parameters) {
  library(ggplot2)
  
  # Extract coordinates
  coords <- st_coordinates(spatial_data)
  
  p <- ggplot(data.frame(x = coords[, 1], y = coords[, 2])) +
    geom_density_2d(aes(x = x, y = y), alpha = 0.5) +
    geom_point(aes(x = x, y = y), alpha = 0.3) +
    theme_minimal() +
    labs(title = parameters$title)
  
  return(p)
}
```

### Interactive Maps

```r
# R/03-spatial-visualization.R (continued)

#' Create interactive spatial visualizations
#'
#' @param spatial_data Spatial data object
#' @param visualization_type Type of visualization
#' @param parameters Visualization parameters
#' @return Interactive spatial visualization
create_interactive_spatial_visualization <- function(spatial_data, visualization_type, parameters = list()) {
  switch(visualization_type,
    "leaflet_map" = create_leaflet_map(spatial_data, parameters),
    "plotly_map" = create_plotly_map(spatial_data, parameters),
    "mapview_map" = create_mapview_map(spatial_data, parameters),
    stop("Unsupported visualization type: ", visualization_type)
  )
}

#' Create Leaflet map
#'
#' @param spatial_data Spatial data object
#' @param parameters Visualization parameters
#' @return Leaflet map
create_leaflet_map <- function(spatial_data, parameters) {
  library(leaflet)
  
  # Create base map
  map <- leaflet() %>%
    addTiles()
  
  # Add spatial data
  if (st_geometry_type(spatial_data)[1] == "POINT") {
    map <- map %>%
      addCircleMarkers(
        data = spatial_data,
        radius = parameters$radius,
        color = parameters$color,
        popup = parameters$popup
      )
  } else {
    map <- map %>%
      addPolygons(
        data = spatial_data,
        color = parameters$color,
        fillColor = parameters$fill_color,
        popup = parameters$popup
      )
  }
  
  return(map)
}

#' Create Plotly map
#'
#' @param spatial_data Spatial data object
#' @param parameters Visualization parameters
#' @return Plotly map
create_plotly_map <- function(spatial_data, parameters) {
  library(plotly)
  
  # Extract coordinates
  coords <- st_coordinates(spatial_data)
  
  # Create plot
  p <- plot_ly(
    x = coords[, 1],
    y = coords[, 2],
    type = "scatter",
    mode = "markers",
    marker = list(
      size = parameters$size,
      color = parameters$color
    ),
    text = parameters$text
  ) %>%
    layout(
      title = parameters$title,
      xaxis = list(title = "Longitude"),
      yaxis = list(title = "Latitude")
    )
  
  return(p)
}

#' Create Mapview map
#'
#' @param spatial_data Spatial data object
#' @param parameters Visualization parameters
#' @return Mapview map
create_mapview_map <- function(spatial_data, parameters) {
  library(mapview)
  
  # Create map
  map <- mapview(
    spatial_data,
    zcol = parameters$zcol,
    col.regions = parameters$col_regions,
    alpha = parameters$alpha
  )
  
  return(map)
}
```

## Performance Optimization

### Spatial Indexing

```r
# R/04-spatial-performance.R

#' Optimize spatial operations for performance
#'
#' @param spatial_data Spatial data object
#' @param optimization_type Type of optimization
#' @param parameters Optimization parameters
#' @return Optimized spatial data
optimize_spatial_operations <- function(spatial_data, optimization_type, parameters = list()) {
  switch(optimization_type,
    "spatial_index" = create_spatial_index(spatial_data, parameters),
    "simplify_geometry" = simplify_geometry(spatial_data, parameters),
    "aggregate_features" = aggregate_features(spatial_data, parameters),
    "crop_data" = crop_spatial_data(spatial_data, parameters),
    stop("Unsupported optimization type: ", optimization_type)
  )
}

#' Create spatial index
#'
#' @param spatial_data Spatial data object
#' @param parameters Optimization parameters
#' @return Spatial data with index
create_spatial_index <- function(spatial_data, parameters) {
  library(sf)
  
  # Create spatial index
  spatial_data <- st_make_valid(spatial_data)
  
  # Add spatial index
  spatial_data$spatial_index <- 1:nrow(spatial_data)
  
  return(spatial_data)
}

#' Simplify geometry
#'
#' @param spatial_data Spatial data object
#' @param parameters Optimization parameters
#' @return Simplified spatial data
simplify_geometry <- function(spatial_data, parameters) {
  library(sf)
  
  # Simplify geometry
  simplified_data <- st_simplify(spatial_data, dTolerance = parameters$tolerance)
  
  return(simplified_data)
}

#' Aggregate features
#'
#' @param spatial_data Spatial data object
#' @param parameters Optimization parameters
#' @return Aggregated spatial data
aggregate_features <- function(spatial_data, parameters) {
  library(sf)
  
  # Aggregate by grouping variable
  if (!is.null(parameters$group_by)) {
    aggregated_data <- spatial_data %>%
      group_by(!!sym(parameters$group_by)) %>%
      summarise(geometry = st_union(geometry))
  } else {
    # Aggregate all features
    aggregated_data <- spatial_data %>%
      summarise(geometry = st_union(geometry))
  }
  
  return(aggregated_data)
}

#' Crop spatial data
#'
#' @param spatial_data Spatial data object
#' @param parameters Optimization parameters
#' @return Cropped spatial data
crop_spatial_data <- function(spatial_data, parameters) {
  library(sf)
  
  # Create bounding box
  bbox <- st_bbox(parameters$bbox)
  bbox_polygon <- st_as_sfc(bbox)
  
  # Crop data
  cropped_data <- st_intersection(spatial_data, bbox_polygon)
  
  return(cropped_data)
}
```

## TL;DR Runbook

### Quick Start

```r
# 1. Import spatial data
spatial_data <- import_spatial_data("data.shp", crs = 4326)

# 2. Transform CRS
transformed_data <- transform_crs(spatial_data, 3857)

# 3. Perform spatial operations
intersection <- perform_spatial_operation(data1, data2, "intersection")

# 4. Calculate spatial statistics
stats <- calculate_spatial_statistics(spatial_data, c("area", "centroid"))

# 5. Create visualizations
map <- create_spatial_visualization(spatial_data, "choropleth_map", list(variable = "value"))

# 6. Optimize performance
optimized_data <- optimize_spatial_operations(spatial_data, "spatial_index")
```

### Essential Patterns

```r
# Complete spatial analysis pipeline
spatial_analysis_pipeline <- function(data_path, analysis_config) {
  # Import data
  spatial_data <- import_spatial_data(data_path, analysis_config$crs)
  
  # Transform CRS
  if (!is.null(analysis_config$target_crs)) {
    spatial_data <- transform_crs(spatial_data, analysis_config$target_crs)
  }
  
  # Perform analysis
  analysis_results <- perform_spatial_analysis(spatial_data, analysis_config$analysis_type, analysis_config$parameters)
  
  # Create visualizations
  visualizations <- create_spatial_visualization(spatial_data, analysis_config$visualization_type, analysis_config$viz_parameters)
  
  # Optimize performance
  if (analysis_config$optimize) {
    spatial_data <- optimize_spatial_operations(spatial_data, analysis_config$optimization_type, analysis_config$opt_parameters)
  }
  
  return(list(
    data = spatial_data,
    analysis = analysis_results,
    visualizations = visualizations
  ))
}
```

---

*This guide provides the complete machinery for comprehensive geospatial analysis in R. Each pattern includes implementation examples, visualization strategies, and real-world usage patterns for enterprise deployment.*

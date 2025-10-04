# Building & Deploying a Geospatial R Shiny App (for Free)

**Objective**: Build an interactive world map where clicking countries toggles them red (visited), then deploy it online for free.

R Shiny transforms R into a web application framework. We'll create "Where Have You Been in the World?"—a clickable world map that tracks visited countries, then deploy it to the cloud without spending a dime.

## 1) Environment Setup

Install R and RStudio, then load the required packages:

```r
# Install core packages (run once)
install.packages(c("shiny", "leaflet", "sf", "rnaturalearth", 
                   "rnaturalearthdata", "dplyr", "rsconnect"))
```

**Why**: These packages provide the foundation—Shiny for the web framework, Leaflet for interactive maps, sf for spatial data handling, and rnaturalearth for world country data.

## 2) Getting World Data

Load country polygons using rnaturalearth:

```r
library(sf)
library(rnaturalearth)

# Get world countries as sf object
world <- ne_countries(returnclass = "sf")

# Check what we have
head(world[, c("name", "iso_a3", "geometry")])
```

**Why**: rnaturalearth provides clean, standardized country boundaries. The `iso_a3` field gives us unique country identifiers for tracking clicks.

## 3) Building the Shiny App

Create a complete Shiny application with interactive map:

```r
library(shiny)
library(leaflet)
library(sf)
library(rnaturalearth)
library(dplyr)

# Load world data once
world <- ne_countries(returnclass = "sf")

# UI Definition
ui <- fluidPage(
  titlePanel("Where Have You Been in the World?"),
  p("Click on countries to mark them as visited (red). Click again to unmark."),
  leafletOutput("map", height = "600px"),
  br(),
  textOutput("visited_count")
)

# Server Logic
server <- function(input, output, session) {
  # Reactive value to store visited countries
  visited <- reactiveVal(character(0))
  
  # Render the initial map
  output$map <- renderLeaflet({
    leaflet(world) %>%
      addTiles() %>%
      addPolygons(
        layerId = ~iso_a3,
        fillColor = "white",
        color = "black",
        weight = 1,
        opacity = 0.8,
        fillOpacity = 0.6,
        highlightOptions = highlightOptions(
          color = "blue", 
          weight = 3,
          bringToFront = TRUE
        ),
        label = ~name,
        labelOptions = labelOptions(
          style = list("font-weight" = "normal", padding = "3px 8px"),
          textsize = "12px",
          direction = "auto"
        )
      )
  })
  
  # Handle country clicks
  observeEvent(input$map_shape_click, {
    clicked_iso <- input$map_shape_click$id
    current_visited <- visited()
    
    # Toggle country in visited list
    if (clicked_iso %in% current_visited) {
      # Remove from visited
      new_visited <- setdiff(current_visited, clicked_iso)
    } else {
      # Add to visited
      new_visited <- c(current_visited, clicked_iso)
    }
    
    visited(new_visited)
    
    # Update map colors
    leafletProxy("map") %>%
      clearShapes() %>%
      addPolygons(
        data = world,
        layerId = ~iso_a3,
        fillColor = ~ifelse(iso_a3 %in% visited(), "red", "white"),
        color = "black",
        weight = 1,
        opacity = 0.8,
        fillOpacity = 0.6,
        highlightOptions = highlightOptions(
          color = "blue", 
          weight = 3,
          bringToFront = TRUE
        ),
        label = ~name,
        labelOptions = labelOptions(
          style = list("font-weight" = "normal", padding = "3px 8px"),
          textsize = "12px",
          direction = "auto"
        )
      )
  })
  
  # Show visited count
  output$visited_count <- renderText({
    paste("Countries visited:", length(visited()))
  })
}

# Run the app
shinyApp(ui, server)
```

**Why**: This creates a complete interactive experience—users click countries to toggle their visited status, with visual feedback and a counter. The `reactiveVal` persists state during the session.

## 4) Local Testing

Save the code above as `app.R` in a new directory, then run:

```r
# In RStudio or R console
shiny::runApp("path/to/your/app")
```

**Expected behavior**: 
- World map loads with white countries
- Clicking a country turns it red
- Clicking again toggles back to white
- Counter shows number of visited countries
- Hover shows country names

## 5) Free Deployment Options

### Option A: ShinyApps.io (Recommended)

ShinyApps.io offers free hosting with generous limits:

```r
# Install deployment package
install.packages("rsconnect")

# Authenticate (get credentials from shinyapps.io)
rsconnect::setAccountInfo(
  name = 'your-username',
  token = 'your-token',
  secret = 'your-secret'
)

# Deploy your app
rsconnect::deployApp("path/to/your/app")
```

**Why**: ShinyApps.io is purpose-built for Shiny apps, handles scaling automatically, and provides a clean URL like `https://yourname.shinyapps.io/your-app-name/`.

### Option B: Docker + Render (Advanced)

For more control, containerize your app:

```dockerfile
# Dockerfile
FROM rocker/shiny:4.3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R packages
RUN R -e "install.packages(c('shiny', 'leaflet', 'sf', 'rnaturalearth', 'dplyr'), repos='https://cran.rstudio.com/')"

# Copy app
COPY app.R /srv/shiny-server/

# Expose port
EXPOSE 3838
```

Deploy to Render's free tier with this `render.yaml`:

```yaml
services:
  - type: web
    name: r-shiny-geoapp
    env: docker
    dockerfilePath: ./Dockerfile
    plan: free
```

**Why**: Docker gives you complete control over the environment and allows deployment to any container platform.

## 6) Optimization for Production

### Simplify Geometry

Large country polygons slow rendering. Simplify them:

```r
library(rmapshaper)

# Simplify world data (reduce file size by ~80%)
world_simple <- ms_simplify(world, keep = 0.1)

# Use in your app
world <- world_simple
```

### Cache Expensive Operations

```r
# Cache world data globally
world <- ne_countries(returnclass = "sf")
world <- ms_simplify(world, keep = 0.1)

# Pre-compute country centroids for faster lookups
country_centroids <- st_centroid(world)
```

### Limit Map Tiles

```r
# Use lighter tile sets for faster loading
leaflet(world) %>%
  addProviderTiles("CartoDB.Positron") %>%  # Lightweight tiles
  # ... rest of your map code
```

**Why**: Simplified geometry and lightweight tiles dramatically improve load times, especially on mobile devices.

## 7) Advanced Features

### Add Country Search

```r
# Add search functionality to UI
ui <- fluidPage(
  titlePanel("Where Have You Been in the World?"),
  sidebarLayout(
    sidebarPanel(
      selectInput("search_country", "Search Country:",
                  choices = sort(world$name),
                  selected = NULL),
      actionButton("go_to_country", "Go to Country"),
      br(), br(),
      textOutput("visited_count")
    ),
    mainPanel(
      leafletOutput("map", height = "600px")
    )
  )
)

# Add search functionality to server
observeEvent(input$go_to_country, {
  if (!is.null(input$search_country)) {
    country_geom <- world[world$name == input$search_country, ]
    if (nrow(country_geom) > 0) {
      leafletProxy("map") %>%
        setView(
          lng = st_coordinates(st_centroid(country_geom))[1],
          lat = st_coordinates(st_centroid(country_geom))[2],
          zoom = 4
        )
    }
  }
})
```

### Export Visited Countries

```r
# Add export button to UI
downloadButton("export_data", "Export Visited Countries")

# Add export functionality to server
output$export_data <- downloadHandler(
  filename = function() {
    paste("visited_countries_", Sys.Date(), ".csv", sep = "")
  },
  content = function(file) {
    visited_data <- world[world$iso_a3 %in% visited(), c("name", "iso_a3")]
    write.csv(visited_data, file, row.names = FALSE)
  }
)
```

**Why**: Search and export features make the app more useful for actual travel tracking and data analysis.

## 8) Troubleshooting Common Issues

### Package Installation Problems

```r
# If sf installation fails on macOS/Linux
install.packages("sf", configure.args = "--with-proj-lib=/usr/local/lib")

# Alternative: use conda
# conda install -c conda-forge r-sf r-leaflet
```

### Memory Issues with Large Datasets

```r
# Use simplified data
world <- ne_countries(scale = "medium", returnclass = "sf")

# Or load only specific regions
europe <- ne_countries(continent = "Europe", returnclass = "sf")
```

### Deployment Authentication Issues

```r
# Clear existing authentication
rsconnect::removeAccount("your-account")

# Re-authenticate with fresh credentials
rsconnect::setAccountInfo(...)
```

**Why**: These are the most common roadblocks when building and deploying Shiny apps with spatial data.

## 9) TL;DR Quickstart

```r
# 1. Install packages
install.packages(c("shiny", "leaflet", "sf", "rnaturalearth", "dplyr", "rsconnect"))

# 2. Save the complete app code as app.R

# 3. Test locally
shiny::runApp(".")

# 4. Deploy to ShinyApps.io
rsconnect::setAccountInfo(name='yourname', token='token', secret='secret')
rsconnect::deployApp(".")

# 5. Your app is live at https://yourname.shinyapps.io/your-app-name/
```

## 10) Next Steps

- **Add persistence**: Store visited countries in a database
- **User accounts**: Let multiple users track their own visits
- **Statistics**: Add charts showing travel patterns
- **Mobile optimization**: Ensure touch-friendly interaction
- **Performance**: Implement lazy loading for large datasets

**Why**: This foundation gives you a working geospatial Shiny app that you can extend with advanced features as needed.

---

*This tutorial provides everything needed to build and deploy an interactive geospatial R Shiny application. The code is production-ready and the deployment options are genuinely free.*

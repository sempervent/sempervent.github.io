# Web Performance Optimization

**Objective**: Achieve optimal web performance through Core Web Vitals optimization, modern development practices, and comprehensive monitoring.

Web performance directly impacts user experience, SEO rankings, and business metrics. This guide covers achieving sub-2.5s LCP, <100ms FID, and <0.1 CLS through proven optimization techniques.

## 1) Core Web Vitals Optimization

### Largest Contentful Paint (LCP) < 2.5s

```javascript
// Lazy loading for images
const images = document.querySelectorAll('img[data-src]');
const imageObserver = new IntersectionObserver((entries, observer) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const img = entry.target;
      img.src = img.dataset.src;
      img.classList.remove('lazy');
      observer.unobserve(img);
    }
  });
});

images.forEach(img => imageObserver.observe(img));

// Preload critical resources
const preloadCriticalResources = () => {
  const criticalResources = [
    '/css/critical.css',
    '/js/critical.js',
    '/fonts/main.woff2'
  ];
  
  criticalResources.forEach(resource => {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = resource;
    link.as = resource.endsWith('.css') ? 'style' : 'script';
    document.head.appendChild(link);
  });
};
```

**Why**: Lazy loading and resource preloading ensure critical content loads first, improving LCP scores and user experience.

### First Input Delay (FID) < 100ms

```javascript
// Optimize JavaScript execution
const optimizeJSExecution = () => {
  // Use requestIdleCallback for non-critical tasks
  if ('requestIdleCallback' in window) {
    requestIdleCallback(() => {
      // Defer non-critical JavaScript
      loadAnalytics();
      loadSocialWidgets();
    });
  }
  
  // Break up long tasks
  const longTask = () => {
    // Process in chunks
    let chunk = 0;
    const processChunk = () => {
      for (let i = 0; i < 1000; i++) {
        // Process item
        chunk++;
        if (chunk % 100 === 0) {
          // Yield control
          setTimeout(processChunk, 0);
          return;
        }
      }
    };
    processChunk();
  };
};

// Optimize event handlers
const optimizeEventHandlers = () => {
  // Use passive event listeners
  document.addEventListener('scroll', handleScroll, { passive: true });
  document.addEventListener('touchstart', handleTouch, { passive: true });
  
  // Debounce expensive operations
  const debounce = (func, wait) => {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  };
  
  const debouncedResize = debounce(handleResize, 250);
  window.addEventListener('resize', debouncedResize);
};
```

**Why**: Optimizing JavaScript execution and event handling ensures responsive user interactions and improves FID scores.

### Cumulative Layout Shift (CLS) < 0.1

```css
/* Prevent layout shifts */
.reserve-space {
  /* Reserve space for dynamic content */
  min-height: 200px;
  background-color: #f5f5f5;
}

/* Optimize font loading */
@font-face {
  font-family: 'MainFont';
  src: url('/fonts/main.woff2') format('woff2');
  font-display: swap; /* Prevent invisible text during font load */
}

/* Prevent image layout shifts */
img {
  width: 100%;
  height: auto;
  aspect-ratio: attr(width) / attr(height);
}

/* Reserve space for ads */
.ad-container {
  min-height: 250px;
  width: 100%;
}
```

```javascript
// Prevent layout shifts with JavaScript
const preventLayoutShifts = () => {
  // Reserve space for dynamic content
  const reserveSpace = (element, height) => {
    element.style.minHeight = `${height}px`;
  };
  
  // Load content without layout shift
  const loadContentSafely = async (container, content) => {
    // Measure container
    const containerHeight = container.offsetHeight;
    
    // Load content
    container.innerHTML = content;
    
    // Adjust if needed
    if (container.offsetHeight < containerHeight) {
      container.style.minHeight = `${containerHeight}px`;
    }
  };
};
```

**Why**: Preventing layout shifts ensures stable visual experience and improves CLS scores.

## 2) Resource Optimization

### Code Splitting and Tree Shaking

```javascript
// Dynamic imports for code splitting
const loadFeature = async (featureName) => {
  try {
    const module = await import(`./features/${featureName}.js`);
    return module.default;
  } catch (error) {
    console.error(`Failed to load feature: ${featureName}`, error);
  }
};

// Route-based code splitting
const routes = {
  '/': () => import('./pages/home.js'),
  '/about': () => import('./pages/about.js'),
  '/contact': () => import('./pages/contact.js')
};

// Lazy load routes
const loadRoute = async (path) => {
  const routeLoader = routes[path];
  if (routeLoader) {
    const module = await routeLoader();
    return module.default;
  }
};

// Tree shaking optimization
// Use ES6 modules for tree shaking
export const utils = {
  formatDate: (date) => date.toISOString(),
  formatCurrency: (amount) => `$${amount.toFixed(2)}`
};

// Import only what you need
import { formatDate } from './utils.js';
```

**Why**: Code splitting reduces initial bundle size, while tree shaking eliminates unused code, improving load times.

### Image Optimization

```javascript
// Responsive images with WebP support
const createResponsiveImage = (src, alt, sizes) => {
  const picture = document.createElement('picture');
  
  // WebP source
  const webpSource = document.createElement('source');
  webpSource.srcset = src.replace('.jpg', '.webp');
  webpSource.type = 'image/webp';
  
  // Fallback source
  const imgSource = document.createElement('source');
  imgSource.srcset = src;
  imgSource.type = 'image/jpeg';
  
  // Image element
  const img = document.createElement('img');
  img.src = src;
  img.alt = alt;
  img.loading = 'lazy';
  img.sizes = sizes;
  
  picture.appendChild(webpSource);
  picture.appendChild(imgSource);
  picture.appendChild(img);
  
  return picture;
};

// Image compression
const compressImage = (file, quality = 0.8) => {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      
      canvas.toBlob(resolve, 'image/jpeg', quality);
    };
    
    img.src = URL.createObjectURL(file);
  });
};
```

**Why**: Optimized images reduce bandwidth usage and improve load times, especially on mobile devices.

### CDN and Caching

```javascript
// Service Worker for caching
const CACHE_NAME = 'app-cache-v1';
const urlsToCache = [
  '/',
  '/css/main.css',
  '/js/main.js',
  '/images/logo.png'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Return cached version or fetch from network
        return response || fetch(event.request);
      })
  );
});

// Cache strategies
const cacheStrategies = {
  // Cache first for static assets
  cacheFirst: async (request) => {
    const cached = await caches.match(request);
    if (cached) return cached;
    
    const response = await fetch(request);
    const cache = await caches.open(CACHE_NAME);
    cache.put(request, response.clone());
    return response;
  },
  
  // Network first for dynamic content
  networkFirst: async (request) => {
    try {
      const response = await fetch(request);
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
      return response;
    } catch (error) {
      return await caches.match(request);
    }
  }
};
```

**Why**: Effective caching reduces server load and improves response times for repeat visitors.

## 3) Performance Monitoring

### Web Vitals Tracking

```javascript
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

// Track Core Web Vitals
const trackWebVitals = () => {
  const sendToAnalytics = (metric) => {
    // Send to Google Analytics
    gtag('event', metric.name, {
      value: Math.round(metric.value),
      event_category: 'Web Vitals',
      event_label: metric.id,
      non_interaction: true,
    });
    
    // Send to custom analytics
    fetch('/api/analytics/vitals', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(metric)
    });
  };
  
  getCLS(sendToAnalytics);
  getFID(sendToAnalytics);
  getFCP(sendToAnalytics);
  getLCP(sendToAnalytics);
  getTTFB(sendToAnalytics);
};

// Performance monitoring
const monitorPerformance = () => {
  // Monitor long tasks
  if ('PerformanceObserver' in window) {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.duration > 50) { // 50ms threshold
          console.warn('Long task detected:', entry);
        }
      }
    });
    observer.observe({ entryTypes: ['longtask'] });
  }
  
  // Monitor layout shifts
  const layoutShiftObserver = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      if (entry.value > 0.1) {
        console.warn('Layout shift detected:', entry);
      }
    }
  });
  layoutShiftObserver.observe({ entryTypes: ['layout-shift'] });
};
```

**Why**: Comprehensive performance monitoring enables proactive optimization and identifies performance bottlenecks.

### Error Tracking

```javascript
// Global error handler
window.addEventListener('error', (event) => {
  console.error('Global error:', event.error);
  
  // Send to error tracking service
  if (typeof gtag !== 'undefined') {
    gtag('event', 'exception', {
      description: event.error.message,
      fatal: false
    });
  }
});

// Unhandled promise rejection handler
window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled promise rejection:', event.reason);
  
  // Send to error tracking service
  fetch('/api/analytics/errors', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      type: 'unhandledrejection',
      reason: event.reason,
      timestamp: Date.now()
    })
  });
});
```

**Why**: Error tracking provides visibility into user experience issues and enables rapid problem resolution.

## 4) SEO and Accessibility

### Technical SEO

```html
<!-- Structured Data for Professional Profile -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Person",
  "name": "Joshua N. Grant",
  "jobTitle": "Geospatial Systems Architect",
  "worksFor": {
    "@type": "Organization",
    "name": "Oak Ridge National Laboratory"
  },
  "url": "https://sempervent.github.io",
  "sameAs": [
    "https://github.com/sempervent",
    "https://linkedin.com/in/joshuanagrant"
  ]
}
</script>

<!-- Meta tags for SEO -->
<meta name="description" content="Professional portfolio and technical documentation for Joshua N. Grant, Geospatial Systems Architect">
<meta name="keywords" content="geospatial, data engineering, cloud architecture, Python, PostGIS">
<meta name="author" content="Joshua N. Grant">
<meta name="robots" content="index, follow">

<!-- Open Graph tags -->
<meta property="og:title" content="Joshua N. Grant - Geospatial Systems Architect">
<meta property="og:description" content="Professional portfolio and technical documentation">
<meta property="og:url" content="https://sempervent.github.io">
<meta property="og:type" content="website">
<meta property="og:image" content="https://sempervent.github.io/images/og-image.jpg">

<!-- Twitter Card tags -->
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Joshua N. Grant - Geospatial Systems Architect">
<meta name="twitter:description" content="Professional portfolio and technical documentation">
<meta name="twitter:image" content="https://sempervent.github.io/images/twitter-image.jpg">
```

**Why**: Proper SEO implementation improves search engine visibility and drives organic traffic.

### Accessibility (WCAG 2.1 AA)

```css
/* High contrast mode support */
@media (prefers-contrast: high) {
  :root {
    --text-color: #000000;
    --background-color: #ffffff;
    --accent-color: #0066cc;
  }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* Focus indicators */
:focus {
  outline: 2px solid #0066cc;
  outline-offset: 2px;
}

/* Skip links */
.skip-link {
  position: absolute;
  top: -40px;
  left: 6px;
  background: #000;
  color: #fff;
  padding: 8px;
  text-decoration: none;
  z-index: 1000;
}

.skip-link:focus {
  top: 6px;
}
```

```javascript
// Keyboard navigation support
const handleKeyboardNavigation = () => {
  // Trap focus in modals
  const trapFocus = (element) => {
    const focusableElements = element.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];
    
    element.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        if (e.shiftKey) {
          if (document.activeElement === firstElement) {
            lastElement.focus();
            e.preventDefault();
          }
        } else {
          if (document.activeElement === lastElement) {
            firstElement.focus();
            e.preventDefault();
          }
        }
      }
    });
  };
  
  // ARIA labels for screen readers
  const addAriaLabels = () => {
    const buttons = document.querySelectorAll('button:not([aria-label])');
    buttons.forEach(button => {
      if (!button.textContent.trim()) {
        button.setAttribute('aria-label', 'Button');
      }
    });
  };
};
```

**Why**: Accessibility ensures your website is usable by all users, including those with disabilities, and improves SEO rankings.

## 5) Progressive Web App (PWA) Features

### Service Worker Implementation

```javascript
// Service worker for offline functionality
const CACHE_NAME = 'sempervent-portfolio-v1';
const urlsToCache = [
  '/',
  '/about/',
  '/projects/',
  '/documentation/',
  '/contact/'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Return cached version or fetch from network
        return response || fetch(event.request);
      })
  );
});

// Background sync for offline actions
self.addEventListener('sync', event => {
  if (event.tag === 'background-sync') {
    event.waitUntil(doBackgroundSync());
  }
});

const doBackgroundSync = async () => {
  // Sync offline actions when online
  const offlineActions = await getOfflineActions();
  for (const action of offlineActions) {
    await syncAction(action);
  }
};
```

**Why**: PWA features provide offline functionality and app-like experience, improving user engagement.

### PWA Manifest

```json
{
  "name": "Joshua N. Grant - Portfolio",
  "short_name": "JNG Portfolio",
  "description": "Professional portfolio and technical documentation",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#1976d2",
  "icons": [
    {
      "src": "/assets/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/assets/icon-512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ],
  "categories": ["portfolio", "documentation", "geospatial"],
  "lang": "en",
  "dir": "ltr"
}
```

**Why**: PWA manifest enables app installation and provides native app-like experience.

## 6) TL;DR Quickstart

```javascript
// 1. Optimize Core Web Vitals
const optimizeWebVitals = () => {
  // Lazy load images
  const images = document.querySelectorAll('img[data-src]');
  const imageObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.src = entry.target.dataset.src;
        imageObserver.unobserve(entry.target);
      }
    });
  });
  images.forEach(img => imageObserver.observe(img));
  
  // Preload critical resources
  const criticalResources = ['/css/critical.css', '/js/critical.js'];
  criticalResources.forEach(resource => {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = resource;
    document.head.appendChild(link);
  });
};

// 2. Implement caching
const implementCaching = () => {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js');
  }
};

// 3. Monitor performance
const monitorPerformance = () => {
  import('web-vitals').then(({ getCLS, getFID, getLCP }) => {
    getCLS(console.log);
    getFID(console.log);
    getLCP(console.log);
  });
};
```

## 7) Anti-Patterns to Avoid

- **Don't ignore Core Web Vitals**—they directly impact SEO and user experience
- **Don't skip image optimization**—unoptimized images are the biggest performance killer
- **Don't ignore accessibility**—accessibility improves SEO and user experience
- **Don't skip error handling**—errors impact user experience and SEO
- **Don't ignore monitoring**—performance monitoring enables continuous optimization

**Why**: These anti-patterns lead to poor user experience, low SEO rankings, and reduced business metrics.

---

*This guide provides the foundation for achieving optimal web performance through proven optimization techniques and modern development practices.*

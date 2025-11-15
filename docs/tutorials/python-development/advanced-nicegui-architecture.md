# Advanced NiceGUI Architecture: Persistent Backends, Components, Spinners, and Real-Time UI

**Objective**: Master production-grade NiceGUI applications with persistent backends, reusable components, real-time updates, and advanced async patterns. When you need to build scalable, maintainable UI applications that integrate with Redis, PostgreSQL, and external APIs—this tutorial becomes your weapon of choice.

## Introduction

Most NiceGUI tutorials stop at "here's how to make a button." Real applications need persistent database connections, connection pooling, real-time updates, loading states, error handling, and reusable components. This tutorial covers the architecture patterns that transform NiceGUI from a prototyping tool into a production-capable framework.

**What "Advanced NiceGUI" Means**:

- **Persistent Backends**: Connection pools for Redis and PostgreSQL that survive across page navigations
- **Reusable Components**: Class-based UI components that encapsulate behavior and styling
- **Real-Time Updates**: Redis pub/sub and WebSocket patterns for live UI updates
- **Loading States**: Spinners, progress bars, and streaming output for long-running operations
- **Lifecycle Management**: Startup/shutdown handlers, graceful connection cleanup, background tasks
- **State Management**: Session state, global state, and reactive variables that sync across components

**Why Class-Based Pages Matter**:

Single-file NiceGUI apps work for demos. Production apps need:
- Separation of concerns (pages, components, services)
- Dependency injection (shared services, connection pools)
- Testability (mockable services, isolated components)
- Maintainability (clear structure, reusable patterns)

**How Backend Pools Improve Performance**:

Creating new connections for every request is slow and wasteful. Connection pools:
- Reuse existing connections (10-100x faster)
- Limit concurrent connections (prevent resource exhaustion)
- Handle reconnection automatically (resilience)
- Provide connection health monitoring (observability)

## Project Structure

Here's a production-ready directory layout that scales:

```
advanced-nicegui-app/
├── app.py                    # Entry point
├── core/
│   ├── __init__.py
│   ├── app_factory.py        # App creation & lifecycle
│   ├── services.py           # Service container
│   ├── pools.py              # Connection pool management
│   ├── events.py             # Event bus & pub/sub
│   └── state.py              # State management
├── components/
│   ├── __init__.py
│   ├── loading_spinner.py   # Reusable spinner component
│   ├── metric_card.py        # Metric display component
│   ├── collapsible_panel.py  # Collapsible UI panel
│   ├── file_uploader.py      # File upload with progress
│   └── realtime_updater.py   # Redis pub/sub component
├── pages/
│   ├── __init__.py
│   ├── base.py               # BasePage class
│   ├── dashboard.py          # Main dashboard
│   ├── realtime_metrics.py   # Real-time metrics page
│   ├── long_task_demo.py     # Long-running task demo
│   └── admin.py              # Admin/CRUD page
├── requirements.txt
└── .env.example              # Environment variables template
```

### requirements.txt

```txt
nicegui>=2.0.0
httpx>=0.27.0
asyncpg>=0.29.0
redis>=5.0.0
sqlalchemy>=2.0.0
python-dotenv>=1.0.0
```

## Persistent Backends

### 3.1 Redis Persistent Connection

Redis connections should be pooled and shared across all pages. Here's a production-ready implementation:

```python
# core/pools.py
from __future__ import annotations

import asyncio
import logging
from typing import Optional
import redis.asyncio as redis
from redis.asyncio import ConnectionPool, Redis

logger = logging.getLogger(__name__)


class RedisPool:
    """Managed Redis connection pool with pub/sub support."""

    def __init__(self, url: str, max_connections: int = 50):
        self.url = url
        self.max_connections = max_connections
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self._subscribers: dict[str, list[callable]] = {}
        self._listener_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize connection pool and client."""
        if self.pool is None:
            self.pool = ConnectionPool.from_url(
                self.url,
                max_connections=self.max_connections,
                decode_responses=True,
            )
            self.client = Redis(connection_pool=self.pool)
            logger.info("Redis pool initialized")

    async def get_client(self) -> Redis:
        """Get Redis client (creates pool if needed)."""
        if self.client is None:
            await self.initialize()
        assert self.client is not None
        return self.client

    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        client = await self.get_client()
        return await client.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set value in Redis."""
        client = await self.get_client()
        return await client.set(key, value, ex=ex)

    async def publish(self, channel: str, message: str) -> int:
        """Publish message to channel."""
        client = await self.get_client()
        return await client.publish(channel, message)

    async def subscribe(self, channel: str, callback: callable) -> None:
        """Subscribe to channel and register callback."""
        if channel not in self._subscribers:
            self._subscribers[channel] = []
        self._subscribers[channel].append(callback)

        # Start listener if not running
        if self._listener_task is None or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        """Background task that listens to pub/sub channels."""
        client = await self.get_client()
        self.pubsub = client.pubsub()

        # Subscribe to all registered channels
        channels = list(self._subscribers.keys())
        if channels:
            await self.pubsub.subscribe(*channels)

        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"]
                    data = message["data"]
                    # Call all registered callbacks
                    for callback in self._subscribers.get(channel, []):
                        try:
                            await callback(data)
                        except Exception as e:
                            logger.error(f"Error in pub/sub callback: {e}")
        except asyncio.CancelledError:
            logger.info("Pub/sub listener cancelled")
        except Exception as e:
            logger.error(f"Pub/sub listener error: {e}")

    async def close(self) -> None:
        """Close connections and cleanup."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()

        if self.client:
            await self.client.close()

        if self.pool:
            await self.pool.aclose()

        logger.info("Redis pool closed")
```

### 3.2 Postgres Persistent Connection

We'll provide both `asyncpg` (recommended) and SQLAlchemy patterns. `asyncpg` is faster and more Pythonic for async code.

#### Option 1: asyncpg (Recommended)

```python
# core/pools.py (continued)
import asyncpg
from asyncpg import Pool, create_pool


class PostgresPool:
    """Managed PostgreSQL connection pool using asyncpg."""

    def __init__(
        self,
        dsn: str,
        min_size: int = 10,
        max_size: int = 20,
        max_queries: int = 50000,
        max_inactive_connection_lifetime: float = 300.0,
    ):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.max_queries = max_queries
        self.max_inactive_connection_lifetime = max_inactive_connection_lifetime
        self.pool: Optional[Pool] = None

    async def initialize(self) -> None:
        """Initialize connection pool."""
        if self.pool is None:
            self.pool = await create_pool(
                self.dsn,
                min_size=self.min_size,
                max_size=self.max_size,
                max_queries=self.max_queries,
                max_inactive_connection_lifetime=self.max_inactive_connection_lifetime,
            )
            logger.info(f"Postgres pool initialized: {self.min_size}-{self.max_size} connections")

    async def get_pool(self) -> Pool:
        """Get connection pool (creates if needed)."""
        if self.pool is None:
            await self.initialize()
        assert self.pool is not None
        return self.pool

    async def fetch(self, query: str, *args) -> list[dict]:
        """Execute SELECT and return rows as dicts."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def fetchrow(self, query: str, *args) -> Optional[dict]:
        """Execute SELECT and return single row as dict."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def execute(self, query: str, *args) -> str:
        """Execute INSERT/UPDATE/DELETE and return status."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Postgres pool closed")
```

#### Option 2: SQLAlchemy AsyncEngine

```python
# core/pools.py (alternative)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class SQLAlchemyPool:
    """Managed PostgreSQL connection pool using SQLAlchemy async."""

    def __init__(self, dsn: str, pool_size: int = 20, max_overflow: int = 10):
        self.dsn = dsn
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.engine: Optional[AsyncEngine] = None
        self.sessionmaker: Optional[async_sessionmaker] = None

    async def initialize(self) -> None:
        """Initialize SQLAlchemy engine and sessionmaker."""
        if self.engine is None:
            self.engine = create_async_engine(
                self.dsn,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                echo=False,
            )
            self.sessionmaker = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            logger.info("SQLAlchemy engine initialized")

    async def get_session(self) -> AsyncSession:
        """Get async session (creates engine if needed)."""
        if self.sessionmaker is None:
            await self.initialize()
        assert self.sessionmaker is not None
        return self.sessionmaker()

    async def close(self) -> None:
        """Close engine and cleanup."""
        if self.engine:
            await self.engine.dispose()
            logger.info("SQLAlchemy engine closed")
```

### 3.3 Global Service Container

A typed service container that provides lazy initialization and dependency injection:

```python
# core/services.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Any
import httpx

from core.pools import RedisPool, PostgresPool


@dataclass
class Services:
    """Typed service container with lazy initialization."""

    redis: RedisPool
    postgres: PostgresPool
    api_client: httpx.AsyncClient

    async def initialize(self) -> None:
        """Initialize all services (call on app startup)."""
        await self.redis.initialize()
        await self.postgres.initialize()

    async def close(self) -> None:
        """Close all services (call on app shutdown)."""
        await self.redis.close()
        await self.postgres.close()
        await self.api_client.aclose()


def create_services() -> Services:
    """Create and return service container."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    postgres_dsn = os.getenv(
        "DATABASE_URL", "postgresql://user:pass@localhost/dbname"
    )

    return Services(
        redis=RedisPool(url=redis_url),
        postgres=PostgresPool(dsn=postgres_dsn),
        api_client=httpx.AsyncClient(timeout=10.0),
    )
```

## Class-Based Components

Reusable components encapsulate UI behavior and styling. Here are production-ready examples:

### 4.1 Loading Spinner Component

```python
# components/loading_spinner.py
from __future__ import annotations

from typing import Optional
from nicegui import ui


class LoadingSpinner:
    """Reusable loading spinner component."""

    def __init__(
        self,
        message: str = "Loading...",
        size: str = "2rem",
        color: str = "primary",
    ):
        self.message = message
        self.size = size
        self.color = color
        self.container: Optional[ui.element] = None
        self.spinner: Optional[ui.element] = None
        self.label: Optional[ui.label] = None

    def show(self, parent: Optional[ui.element] = None) -> None:
        """Show spinner in parent container."""
        container = parent or ui.column()
        self.container = container

        with container.classes("items-center justify-center gap-2"):
            self.spinner = ui.spinner(size=self.size, color=self.color)
            self.label = ui.label(self.message).classes("text-sm text-gray-600")

    def hide(self) -> None:
        """Hide spinner."""
        if self.container:
            self.container.set_visibility(False)

    def update_message(self, message: str) -> None:
        """Update spinner message."""
        if self.label:
            self.label.text = message
```

### 4.2 Metric Card Component

```python
# components/metric_card.py
from __future__ import annotations

from typing import Optional, Callable
from nicegui import ui


class MetricCard:
    """Reusable metric card component with refresh capability."""

    def __init__(
        self,
        title: str,
        value: str | int | float = "—",
        subtitle: Optional[str] = None,
        icon: Optional[str] = None,
        on_refresh: Optional[Callable] = None,
    ):
        self.title = title
        self.value = value
        self.subtitle = subtitle
        self.icon = icon
        self.on_refresh = on_refresh
        self.value_label: Optional[ui.label] = None
        self.card: Optional[ui.card] = None

    def render(self, parent: Optional[ui.element] = None) -> ui.card:
        """Render metric card."""
        container = parent or ui.column()
        with container:
            self.card = ui.card().classes("p-4 min-w-[200px]")
            with self.card:
                with ui.row().classes("items-center justify-between mb-2"):
                    ui.label(self.title).classes("text-sm font-semibold text-gray-600")
                    if self.on_refresh:
                        ui.button(
                            icon="refresh",
                            on_click=self.on_refresh,
                        ).props("flat dense")

                if self.icon:
                    with ui.row().classes("items-center gap-2"):
                        ui.icon(self.icon).classes("text-2xl")
                        self.value_label = ui.label(str(self.value)).classes(
                            "text-2xl font-bold"
                        )
                else:
                    self.value_label = ui.label(str(self.value)).classes(
                        "text-2xl font-bold"
                    )

                if self.subtitle:
                    ui.label(self.subtitle).classes("text-xs text-gray-500 mt-1")

        return self.card

    def update_value(self, value: str | int | float) -> None:
        """Update metric value."""
        self.value = value
        if self.value_label:
            self.value_label.text = str(value)
```

### 4.3 Collapsible Panel Component

```python
# components/collapsible_panel.py
from __future__ import annotations

from typing import Optional
from nicegui import ui


class CollapsiblePanel:
    """Reusable collapsible panel component."""

    def __init__(self, title: str, default_open: bool = True):
        self.title = title
        self.default_open = default_open
        self.is_open = default_open
        self.content_container: Optional[ui.element] = None
        self.toggle_button: Optional[ui.button] = None

    def render(self, parent: Optional[ui.element] = None) -> ui.card:
        """Render collapsible panel."""
        container = parent or ui.column()
        with container:
            card = ui.card().classes("w-full")
            with card:
                with ui.row().classes("items-center justify-between w-full"):
                    ui.label(self.title).classes("text-lg font-semibold")
                    self.toggle_button = ui.button(
                        icon="expand_less" if self.is_open else "expand_more",
                        on_click=self.toggle,
                    ).props("flat dense")

                self.content_container = ui.column().classes("mt-2")
                if not self.is_open:
                    self.content_container.set_visibility(False)

        return card

    def toggle(self) -> None:
        """Toggle panel open/closed."""
        self.is_open = not self.is_open
        if self.content_container:
            self.content_container.set_visibility(self.is_open)
        if self.toggle_button:
            self.toggle_button.props(
                f'icon={"expand_less" if self.is_open else "expand_more"}'
            )

    def add_content(self, content: ui.element) -> None:
        """Add content to panel."""
        if self.content_container:
            with self.content_container:
                content
```

### 4.4 File Uploader with Progress

```python
# components/file_uploader.py
from __future__ import annotations

import asyncio
from typing import Optional, Callable
from nicegui import ui


class FileUploader:
    """File upload component with progress visualization."""

    def __init__(
        self,
        label: str = "Upload File",
        on_upload: Optional[Callable] = None,
        accept: str = "*",
    ):
        self.label = label
        self.on_upload = on_upload
        self.accept = accept
        self.progress_bar: Optional[ui.linear_progress] = None
        self.status_label: Optional[ui.label] = None

    def render(self, parent: Optional[ui.element] = None) -> ui.upload:
        """Render file uploader."""
        container = parent or ui.column()
        with container:
            upload = ui.upload(
                label=self.label,
                on_upload=self._handle_upload,
                accept=self.accept,
            ).classes("w-full")

            self.progress_bar = ui.linear_progress(value=0).classes("w-full mt-2")
            self.progress_bar.set_visibility(False)

            self.status_label = ui.label("").classes("text-sm mt-1")
            self.status_label.set_visibility(False)

        return upload

    async def _handle_upload(self, e) -> None:
        """Handle file upload with progress."""
        if self.progress_bar:
            self.progress_bar.set_visibility(True)
            self.progress_bar.value = 0

        if self.status_label:
            self.status_label.set_visibility(True)
            self.status_label.text = "Uploading..."

        try:
            # Simulate progress (replace with actual upload logic)
            for i in range(0, 101, 10):
                if self.progress_bar:
                    self.progress_bar.value = i / 100
                await asyncio.sleep(0.1)

            if self.on_upload:
                await self.on_upload(e)

            if self.status_label:
                self.status_label.text = "Upload complete!"
                self.status_label.classes("text-green-600")

        except Exception as ex:
            if self.status_label:
                self.status_label.text = f"Upload failed: {ex}"
                self.status_label.classes("text-red-600")
        finally:
            if self.progress_bar:
                await asyncio.sleep(1)
                self.progress_bar.set_visibility(False)
```

### 4.5 Real-Time Updater Component

```python
# components/realtime_updater.py
from __future__ import annotations

import asyncio
from typing import Optional, Callable
from nicegui import ui

from core.pools import RedisPool


class RealtimeUpdater:
    """Component that subscribes to Redis pub/sub and updates UI."""

    def __init__(
        self,
        redis: RedisPool,
        channel: str,
        update_callback: Callable,
        label: Optional[str] = None,
    ):
        self.redis = redis
        self.channel = channel
        self.update_callback = update_callback
        self.label = label
        self.status_indicator: Optional[ui.badge] = None
        self.container: Optional[ui.element] = None

    def render(self, parent: Optional[ui.element] = None) -> ui.column:
        """Render real-time updater with status indicator."""
        container = parent or ui.column()
        self.container = container

        with container:
            if self.label:
                with ui.row().classes("items-center gap-2"):
                    ui.label(self.label).classes("text-sm font-semibold")
                    self.status_indicator = ui.badge("Disconnected", color="red").classes(
                        "text-xs"
                    )

        # Subscribe to Redis channel
        asyncio.create_task(self._subscribe())

        return container

    async def _subscribe(self) -> None:
        """Subscribe to Redis channel and update UI."""
        async def callback(message: str) -> None:
            try:
                await self.update_callback(message)
                if self.status_indicator:
                    self.status_indicator.text = "Connected"
                    self.status_indicator.props("color=green")
            except Exception as e:
                print(f"Error in realtime callback: {e}")

        await self.redis.subscribe(self.channel, callback)
        if self.status_indicator:
            self.status_indicator.text = "Connected"
            self.status_indicator.props("color=green")
```

## Loading & Progress Patterns

### 5.1 Page-Level Spinner

```python
# In a page class
def build(self) -> None:
    spinner = LoadingSpinner(message="Loading dashboard...")
    spinner.show()

    async def load_data():
        # Simulate async data loading
        await asyncio.sleep(2)
        # ... load data ...
        spinner.hide()

    asyncio.create_task(load_data())
```

### 5.2 Inline Spinner

```python
# In a page class
def build(self) -> None:
    with ui.card():
        ui.label("Processing...")
        spinner = LoadingSpinner(message="Please wait", size="1rem")
        spinner.show()
```

### 5.3 Full-Screen Blocking Overlay

```python
# components/fullscreen_spinner.py
from nicegui import ui


class FullscreenSpinner:
    """Full-screen blocking spinner overlay."""

    def __init__(self, message: str = "Loading..."):
        self.message = message
        self.overlay: Optional[ui.element] = None

    def show(self) -> None:
        """Show full-screen spinner."""
        self.overlay = (
            ui.element("div")
            .classes("fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center")
            .style("backdrop-filter: blur(2px)")
        )
        with self.overlay:
            with ui.column().classes("items-center gap-4 bg-white p-8 rounded-lg"):
                ui.spinner(size="3rem", color="primary")
                ui.label(self.message).classes("text-lg")

    def hide(self) -> None:
        """Hide full-screen spinner."""
        if self.overlay:
            self.overlay.delete()
```

### 5.4 Streaming Logs to UI

```python
# In a page class
def build(self) -> None:
    log_container = ui.column().classes("w-full max-h-96 overflow-auto bg-gray-900 p-4 rounded")
    log_text = ui.html("").classes("font-mono text-sm text-green-400")

    async def stream_logs():
        """Stream output from long-running task."""
        import subprocess
        process = await asyncio.create_subprocess_exec(
            "python", "long_task.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            log_text.content += f"<div>{line.decode()}</div>"
            # Auto-scroll
            log_container.scroll_to(percent=100)

    ui.button("Start Task", on_click=lambda: asyncio.create_task(stream_logs()))
```

## Class-Based Pages (Advanced Version)

### 6.1 DashboardPage

```python
# pages/dashboard.py
from __future__ import annotations

import asyncio
from typing import Any
from nicegui import ui

from pages.base import BasePage
from components.metric_card import MetricCard
from components.loading_spinner import LoadingSpinner


class DashboardPage(BasePage):
    route = "/dashboard"
    title = "Dashboard"

    def __init__(self, services: dict[str, Any] | None = None) -> None:
        super().__init__(services)
        self.metrics: dict[str, MetricCard] = {}

    async def _load_metrics(self) -> dict[str, Any]:
        """Load metrics from Redis and Postgres."""
        redis = self.services["redis"]
        postgres = self.services["postgres"]

        # Get Redis metrics
        redis_count = await redis.get("metric:count") or "0"

        # Get Postgres metrics
        result = await postgres.fetchrow("SELECT COUNT(*) as count FROM metrics")

        return {
            "redis_count": int(redis_count),
            "postgres_count": result["count"] if result else 0,
        }

    async def _refresh_metric(self, metric_name: str) -> None:
        """Refresh a specific metric."""
        metrics_data = await self._load_metrics()
        if metric_name in self.metrics:
            self.metrics[metric_name].update_value(metrics_data[metric_name])

    def build(self) -> None:
        ui.label("System Metrics").classes("text-2xl font-bold mb-4")

        # Create metric cards
        with ui.row().classes("gap-4 flex-wrap"):
            redis_card = MetricCard(
                title="Redis Count",
                value="—",
                subtitle="Items in Redis",
                icon="storage",
                on_refresh=lambda: asyncio.create_task(
                    self._refresh_metric("redis_count")
                ),
            )
            redis_card.render()
            self.metrics["redis_count"] = redis_card

            postgres_card = MetricCard(
                title="Postgres Count",
                value="—",
                subtitle="Rows in database",
                icon="database",
                on_refresh=lambda: asyncio.create_task(
                    self._refresh_metric("postgres_count")
                ),
            )
            postgres_card.render()
            self.metrics["postgres_count"] = postgres_card

        # Initial load
        async def initial_load():
            spinner = LoadingSpinner(message="Loading metrics...")
            spinner.show()
            try:
                metrics_data = await self._load_metrics()
                for name, card in self.metrics.items():
                    card.update_value(metrics_data.get(name, 0))
            finally:
                spinner.hide()

        asyncio.create_task(initial_load())
```

### 6.2 RealTimeMetricsPage

```python
# pages/realtime_metrics.py
from __future__ import annotations

import asyncio
from typing import Any
from nicegui import ui

from pages.base import BasePage
from components.realtime_updater import RealtimeUpdater
from components.metric_card import MetricCard


class RealTimeMetricsPage(BasePage):
    route = "/realtime"
    title = "Real-Time Metrics"

    def __init__(self, services: dict[str, Any] | None = None) -> None:
        super().__init__(services)
        self.metric_value = 0
        self.metric_label: ui.label | None = None

    async def _handle_update(self, message: str) -> None:
        """Handle real-time update from Redis."""
        try:
            # Parse message (assuming JSON)
            import json
            data = json.loads(message)
            self.metric_value = data.get("value", 0)
            if self.metric_label:
                self.metric_label.text = f"Current Value: {self.metric_value}"
        except Exception as e:
            print(f"Error parsing update: {e}")

    def build(self) -> None:
        ui.label("Real-Time Metrics").classes("text-2xl font-bold mb-4")

        redis = self.services["redis"]

        # Real-time updater component
        updater = RealtimeUpdater(
            redis=redis,
            channel="metrics:updates",
            update_callback=self._handle_update,
            label="Live Updates",
        )
        updater.render()

        # Display metric
        with ui.card().classes("mt-4 p-4"):
            self.metric_label = ui.label("Current Value: 0").classes("text-xl font-bold")

        # Test button to publish updates
        async def publish_test():
            """Publish test update to Redis."""
            import json
            import random
            value = random.randint(1, 100)
            message = json.dumps({"value": value})
            await redis.publish("metrics:updates", message)

        ui.button("Publish Test Update", on_click=lambda: asyncio.create_task(publish_test()))
```

### 6.3 LongTaskDemoPage

```python
# pages/long_task_demo.py
from __future__ import annotations

import asyncio
from typing import Any
from nicegui import ui

from pages.base import BasePage
from components.fullscreen_spinner import FullscreenSpinner


class LongTaskDemoPage(BasePage):
    route = "/long-task"
    title = "Long Task Demo"

    def __init__(self, services: dict[str, Any] | None = None) -> None:
        super().__init__(services)
        self.task: asyncio.Task | None = None
        self.cancelled = False
        self.log_container: ui.column | None = None
        self.log_html: ui.html | None = None

    async def _long_task(self, duration: int = 10) -> None:
        """Simulate long-running task with progress updates."""
        self.cancelled = False
        if self.log_html:
            self.log_html.content = ""

        for i in range(duration):
            if self.cancelled:
                if self.log_html:
                    self.log_html.content += "<div class='text-red-400'>Task cancelled</div>"
                return

            await asyncio.sleep(1)
            progress = (i + 1) / duration * 100
            if self.log_html:
                self.log_html.content += (
                    f"<div class='text-green-400'>Progress: {progress:.1f}% "
                    f"({i+1}/{duration})</div>"
                )
                if self.log_container:
                    self.log_container.scroll_to(percent=100)

        if self.log_html:
            self.log_html.content += "<div class='text-green-400 font-bold'>Task completed!</div>"

    def _cancel_task(self) -> None:
        """Cancel running task."""
        self.cancelled = True
        if self.task:
            self.task.cancel()

    def build(self) -> None:
        ui.label("Long-Running Task Demo").classes("text-2xl font-bold mb-4")

        duration_input = ui.number("Duration (seconds)", value=10, min=1, max=60).classes("mb-4")

        # Log output container
        self.log_container = ui.column().classes(
            "w-full max-h-96 overflow-auto bg-gray-900 p-4 rounded mb-4"
        )
        with self.log_container:
            self.log_html = ui.html("").classes("font-mono text-sm text-green-400")

        # Control buttons
        with ui.row().classes("gap-2"):
            ui.button(
                "Start Task",
                on_click=lambda: asyncio.create_task(
                    self._long_task(int(duration_input.value))
                ),
            )
            ui.button("Cancel Task", on_click=self._cancel_task, color="red")

        # Full-screen spinner example
        spinner = FullscreenSpinner(message="Processing...")

        async def task_with_spinner():
            spinner.show()
            try:
                await self._long_task(int(duration_input.value))
            finally:
                spinner.hide()

        ui.button(
            "Start with Full-Screen Spinner",
            on_click=lambda: asyncio.create_task(task_with_spinner()),
        ).classes("mt-2")
```

### 6.4 AdminPage

```python
# pages/admin.py
from __future__ import annotations

import asyncio
from typing import Any
from nicegui import ui

from pages.base import BasePage
from components.loading_spinner import LoadingSpinner


class AdminPage(BasePage):
    route = "/admin"
    title = "Admin Panel"

    def __init__(self, services: dict[str, Any] | None = None) -> None:
        super().__init__(services)
        self.data_table: ui.table | None = None
        self.data: list[dict] = []

    async def _load_data(self) -> None:
        """Load data from Postgres."""
        postgres = self.services["postgres"]
        self.data = await postgres.fetch("SELECT * FROM admin_logs ORDER BY id DESC LIMIT 100")
        if self.data_table:
            self.data_table.rows = self.data

    async def _create_record(self, name: str, value: str) -> None:
        """Create new record."""
        postgres = self.services["postgres"]
        await postgres.execute(
            "INSERT INTO admin_logs (name, value) VALUES ($1, $2)",
            name,
            value,
        )
        await self._load_data()
        ui.notify("Record created", type="positive")

    def build(self) -> None:
        ui.label("Admin Panel").classes("text-2xl font-bold mb-4")

        # Create form
        with ui.card().classes("mb-4"):
            ui.label("Create New Record").classes("text-lg font-semibold mb-2")
            name_input = ui.input("Name").classes("mb-2")
            value_input = ui.input("Value").classes("mb-2")

            ui.button(
                "Create",
                on_click=lambda: asyncio.create_task(
                    self._create_record(name_input.value, value_input.value)
                ),
            )

        # Data table
        columns = [
            {"name": "id", "label": "ID", "field": "id"},
            {"name": "name", "label": "Name", "field": "name"},
            {"name": "value", "label": "Value", "field": "value"},
            {"name": "created_at", "label": "Created", "field": "created_at"},
        ]

        self.data_table = ui.table(
            columns=columns,
            rows=[],
            row_key="id",
        ).classes("w-full")

        # Initial load
        spinner = LoadingSpinner(message="Loading data...")
        spinner.show()

        async def initial_load():
            try:
                await self._load_data()
            finally:
                spinner.hide()

        asyncio.create_task(initial_load())
```

## Application Lifecycle

### 7.1 Startup Handlers

```python
# core/app_factory.py
from nicegui import ui
from core.services import Services


async def on_startup(services: Services) -> None:
    """Initialize services on app startup."""
    await services.initialize()
    logger.info("Services initialized")

    # Warm up connection pools
    await services.redis.get_client()
    await services.postgres.get_pool()
    logger.info("Connection pools warmed up")

    # Start background tasks
    asyncio.create_task(background_metric_collector(services))
```

### 7.2 Shutdown Handlers

```python
async def on_shutdown(services: Services) -> None:
    """Cleanup services on app shutdown."""
    logger.info("Shutting down services...")
    await services.close()
    logger.info("Services closed")
```

### 7.3 Background Tasks

```python
async def background_metric_collector(services: Services) -> None:
    """Background task that collects metrics periodically."""
    while True:
        try:
            await asyncio.sleep(60)  # Every minute
            # Collect metrics
            metric_value = await collect_system_metrics()
            await services.redis.set("metric:system", str(metric_value))
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in metric collector: {e}")
```

## Integrating Everything in app_factory.py

```python
# core/app_factory.py
from __future__ import annotations

import asyncio
import logging
from nicegui import ui
from typing import Type

from pages.base import BasePage
from pages.dashboard import DashboardPage
from pages.realtime_metrics import RealTimeMetricsPage
from pages.long_task_demo import LongTaskDemoPage
from pages.admin import AdminPage
from core.services import create_services, Services

logger = logging.getLogger(__name__)


async def on_startup(services: Services) -> None:
    """Initialize services on app startup."""
    await services.initialize()
    logger.info("Services initialized")


async def on_shutdown(services: Services) -> None:
    """Cleanup services on app shutdown."""
    logger.info("Shutting down services...")
    await services.close()
    logger.info("Services closed")


def create_app() -> ui:
    """Create and configure NiceGUI app."""
    # Create services
    services = create_services()

    # Register pages
    page_classes: list[Type[BasePage]] = [
        DashboardPage,
        RealTimeMetricsPage,
        LongTaskDemoPage,
        AdminPage,
    ]

    for page_cls in page_classes:
        page_cls.register(ui, services={"redis": services.redis, "postgres": services.postgres})

    # Global header
    @ui.header()
    def header():
        with ui.row().classes("w-full justify-between items-center px-4 py-2 bg-blue-600 text-white"):
            ui.label("Advanced NiceGUI App").classes("text-lg font-semibold")
            with ui.row().classes("gap-2"):
                ui.button("Dashboard", on_click=lambda: ui.open("/dashboard")).props("flat")
                ui.button("Real-Time", on_click=lambda: ui.open("/realtime")).props("flat")
                ui.button("Long Task", on_click=lambda: ui.open("/long-task")).props("flat")
                ui.button("Admin", on_click=lambda: ui.open("/admin")).props("flat")

    # Attach lifecycle handlers
    ui.on_startup(lambda: asyncio.create_task(on_startup(services)))
    ui.on_shutdown(lambda: asyncio.create_task(on_shutdown(services)))

    return ui
```

## Running the App

### 9.1 Direct Python

```python
# app.py
from __future__ import annotations

import os
from dotenv import load_dotenv
from core.app_factory import create_app

load_dotenv()

app = create_app()

if __name__ in {"__main__", "__mp_main__"}:
    app.run(
        title="Advanced NiceGUI App",
        reload=os.getenv("NICEGUI_RELOAD", "false").lower() == "true",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
    )
```

### 9.2 Uvicorn (Multi-Worker)

```python
# app.py (alternative)
from nicegui import ui
from fastapi import FastAPI
from core.app_factory import create_app

ng_app = create_app()
fastapi_app = FastAPI()

# Mount NiceGUI
ui.run_with(fastapi_app)

# Run with uvicorn
# uvicorn app:fastapi_app --host 0.0.0.0 --port 8080 --workers 4
```

### 9.3 Environment Variables

```bash
# .env.example
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/dbname
HOST=0.0.0.0
PORT=8080
NICEGUI_RELOAD=false
```

## Mermaid Diagrams

### 10.1 App Architecture Diagram

```mermaid
flowchart TB
    subgraph App["NiceGUI Application"]
        Factory["app_factory.py"]
        Services["Services Container"]
        Pages["Page Classes"]
        Components["Reusable Components"]
    end
    
    subgraph Backends["Persistent Backends"]
        Redis["Redis Pool<br/>(Connection Pool)"]
        Postgres["Postgres Pool<br/>(asyncpg)"]
        API["HTTP Client<br/>(httpx)"]
    end
    
    subgraph External["External Services"]
        RedisServer["Redis Server"]
        PgServer["PostgreSQL"]
        ExtAPI["External APIs"]
    end
    
    Factory --> Services
    Factory --> Pages
    Pages --> Components
    Pages --> Services
    Services --> Redis
    Services --> Postgres
    Services --> API
    Redis --> RedisServer
    Postgres --> PgServer
    API --> ExtAPI
```

### 10.2 Request Flow Diagram

```mermaid
sequenceDiagram
    participant User as User
    participant Page as Page Class
    participant Service as Service Container
    participant Pool as Connection Pool
    participant Backend as Backend (Redis/Postgres)
    
    User->>Page: Navigate to page
    Page->>Service: Access service
    Service->>Pool: Get connection
    Pool->>Backend: Execute query
    Backend-->>Pool: Return result
    Pool-->>Service: Return connection
    Service-->>Page: Return data
    Page-->>User: Render UI
```

### 10.3 Redis Pub/Sub UI Update Diagram

```mermaid
sequenceDiagram
    participant Publisher as External Publisher
    participant Redis as Redis Server
    participant Pool as Redis Pool
    participant Component as RealtimeUpdater
    participant UI as UI Component
    
    Publisher->>Redis: Publish message
    Redis->>Pool: Deliver message
    Pool->>Component: Callback with message
    Component->>UI: Update value
    UI-->>User: Visual update
```

### 10.4 Long Task Execution Flow

```mermaid
flowchart LR
    User["User Clicks<br/>Start Task"] --> Spinner["Show Spinner"]
    Spinner --> Task["Create Async Task"]
    Task --> Loop["Task Loop"]
    Loop --> Update["Update Progress"]
    Update --> UI["Update UI"]
    UI --> Check{"Task<br/>Complete?"}
    Check -->|No| Loop
    Check -->|Yes| Hide["Hide Spinner"]
    Hide --> Notify["Notify User"]
```

## Troubleshooting

### 12.1 Connection Pool Exhaustion

**Symptom**: `asyncpg.exceptions.TooManyConnectionsError` or Redis connection timeouts

**Causes**:
- Too many concurrent requests
- Connections not being released
- Pool size too small

**Solutions**:
```python
# Increase pool size
postgres = PostgresPool(dsn=dsn, min_size=20, max_size=50)

# Ensure connections are released (use context managers)
async with pool.acquire() as conn:
    # Use connection
    pass  # Automatically released

# Monitor pool usage
pool_size = pool.get_size()
idle_size = pool.get_idle_size()
```

### 12.2 Redis Reconnect Logic

**Symptom**: Redis connection drops, pub/sub stops working

**Solution**: Add reconnection logic to RedisPool:

```python
async def _listen(self) -> None:
    """Background task with reconnection."""
    while True:
        try:
            # ... existing listen logic ...
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis connection lost: {e}, reconnecting...")
            await asyncio.sleep(5)
            await self.initialize()
            # Re-subscribe to channels
            for channel in self._subscribers.keys():
                await self.pubsub.subscribe(channel)
```

### 12.3 NiceGUI Page Reactivity Issues

**Symptom**: UI doesn't update when data changes

**Causes**:
- Updating variables outside NiceGUI's reactivity system
- Not using `ui.timer` for periodic updates
- Async updates not triggering UI refresh

**Solutions**:
```python
# Use reactive variables
value = ui.number("Value", value=0)
value.value = 10  # This triggers update

# Use ui.timer for periodic updates
ui.timer(1.0, lambda: update_ui())

# Ensure async updates use proper await
async def update():
    data = await fetch_data()
    ui_label.text = str(data)  # Direct assignment works
```

### 12.4 Concurrency Pitfalls

**Symptom**: Race conditions, data corruption, deadlocks

**Solutions**:
```python
# Use asyncio locks for shared state
lock = asyncio.Lock()

async def update_shared_state():
    async with lock:
        # Modify shared state
        pass

# Use asyncio.Queue for producer/consumer
queue = asyncio.Queue()

async def producer():
    await queue.put(data)

async def consumer():
    data = await queue.get()
```

### 12.5 Handling Dropped WebSocket Connections

**Symptom**: Real-time updates stop working after network issues

**Solution**: Implement reconnection in RealtimeUpdater:

```python
async def _subscribe(self) -> None:
    """Subscribe with automatic reconnection."""
    while True:
        try:
            # ... existing subscribe logic ...
        except Exception as e:
            logger.error(f"WebSocket error: {e}, reconnecting...")
            await asyncio.sleep(5)
            # Retry subscription
```

## Final Summary

This tutorial has covered the architecture patterns that transform NiceGUI from a prototyping tool into a production-capable framework:

**Key Takeaways**:

1. **Persistent Backends**: Connection pools for Redis and PostgreSQL eliminate connection overhead and improve performance by 10-100x.

2. **Reusable Components**: Class-based components encapsulate UI behavior, making code maintainable and testable.

3. **Real-Time Updates**: Redis pub/sub enables live UI updates without polling, creating responsive user experiences.

4. **Loading States**: Spinners, progress bars, and streaming output provide user feedback during long operations.

5. **Lifecycle Management**: Startup/shutdown handlers ensure proper resource cleanup and graceful shutdown.

6. **State Management**: Service containers and dependency injection keep code organized and testable.

**Architecture Benefits**:

- **Scalability**: Connection pools handle high concurrency
- **Maintainability**: Clear separation of concerns (pages, components, services)
- **Testability**: Mockable services and isolated components
- **Performance**: Persistent connections and efficient async patterns
- **Reliability**: Automatic reconnection and error handling

**Next Steps**:

- Add authentication and authorization
- Implement caching strategies
- Add monitoring and observability
- Create deployment configurations (Docker, Kubernetes)
- Write integration tests

This architecture provides a solid foundation for building production-grade NiceGUI applications that scale from prototypes to enterprise systems.

## See Also

- **[Class-Based NiceGUI Pages and Integrations](nicegui-class-based-pages.md)** - Foundation patterns for structured NiceGUI apps
- **[Python Async Best Practices](../../best-practices/python/python-async-best-practices.md)** - Write coroutines that don't betray you
- **[PostgreSQL Connection Pooling](../../best-practices/postgres/postgres-pooling.md)** - Production-grade connection pooling
- **[FastAPI Geospatial](../../best-practices/python/fastapi-geospatial.md)** - High-performance API patterns

---

*This tutorial provides the complete machinery for building production-grade NiceGUI applications. The patterns scale from prototypes to enterprise systems, from simple pages to complex real-time dashboards with multiple backend integrations.*


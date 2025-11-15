# Class-Based NiceGUI Pages and Integrations: A Practical Guide

**Objective**: Master building structured NiceGUI applications with class-based pages, dependency injection, and clean architecture. When you need to build real applications—not prototypes—with auth, dashboards, API integrations, and maintainable code, this tutorial becomes your weapon of choice.

Most NiceGUI examples are single-file fever dreams: everything in `main.py`, global state drifting around like cosmic dust. That's fine for a quick prototype, but the moment you want a real app—auth, dashboards, API integrations—you need structure.

This tutorial shows how to build class-based NiceGUI pages with:

- A clean project layout
- Reusable page classes and components
- Centralized routing
- Integration points for external services (APIs, DBs, background workers)

## Prerequisites

- Python >= 3.10
- NiceGUI >= 2.x
- Basic understanding of Python classes and async/await

## 1. Project Layout

Let's start with a small but scalable layout:

```
nicegui-app/
├── app.py                # entrypoint
├── core/
│   ├── __init__.py
│   ├── app_factory.py    # create_app() lives here
│   └── services.py       # "business logic", API clients, db, etc
├── pages/
│   ├── __init__.py
│   ├── base.py           # BasePage class
│   ├── home.py           # HomePage
│   ├── dashboard.py      # DashboardPage
│   └── form_demo.py      # FormPage example
└── requirements.txt
```

### requirements.txt

```txt
nicegui>=2.0.0
httpx>=0.27.0
```

## 2. The Base Page Pattern

We want each page to be a class that:

- Knows its own route path
- Encapsulates UI construction (`build()`, `layout()`, etc.)
- Optionally injects shared services (API clients, repositories)
- Can be registered in a central router

### 2.1 pages/base.py

```python
# pages/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any

from nicegui import ui


class BasePage(ABC):
    """Abstract base for all pages.

    Responsibilities:
      - Define `route` (e.g., "/dashboard")
      - Provide `register(app)` to hook into NiceGUI routing
      - Implement `build()` to construct content
    """

    # Route path, to be overridden
    route: str = "/"

    # Optional title; can be used to set browser title, header, etc.
    title: str = "Untitled"

    def __init__(self, services: Optional[dict[str, Any]] = None) -> None:
        self.services = services or {}

    @classmethod
    def register(cls, app, services: Optional[dict[str, Any]] = None) -> None:
        """Attach this page to the NiceGUI app router."""
        page_instance = cls(services=services)

        @ui.page(cls.route)
        def _() -> None:
            page_instance.render()

    def render(self) -> None:
        """Default render flow: layout + build."""
        with ui.column().classes("w-full items-stretch gap-4 p-4"):
            self._render_header()
            self.build()

    def _render_header(self) -> None:
        with ui.row().classes("w-full justify-between items-center"):
            ui.label(self.title).classes("text-2xl font-bold")
            # Hook for page-level actions (buttons, filters, etc.)
            self.render_actions()

    def render_actions(self) -> None:
        """Override in subclasses for header actions."""
        ...

    @abstractmethod
    def build(self) -> None:
        """Subclasses must implement to build the page contents."""
        raise NotImplementedError
```

This gives us a template:

- `route`: where it lives
- `title`: display / browser / header
- `render()`: calls `_render_header()` + `build()`
- `register(app, services)`: binds it to `@ui.page`

## 3. Creating a Simple Home Page

### 3.1 pages/home.py

```python
# pages/home.py
from __future__ import annotations

from nicegui import ui
from .base import BasePage


class HomePage(BasePage):
    route = "/"
    title = "Welcome to the NiceGUI App"

    def render_actions(self) -> None:
        # Example header actions
        with ui.row().classes("gap-2"):
            ui.button("Dashboard", on_click=lambda: ui.open("/dashboard"))
            ui.button("Form Demo", on_click=lambda: ui.open("/form-demo"))

    def build(self) -> None:
        ui.label("This is the home page, built from a class.").classes("text-lg")
        ui.markdown(
            """
            - Navigate using the header buttons
            - Each page is defined as a Python class
            - Shared services are injected via the BasePage constructor
            """
        )

        with ui.row().classes("gap-4 mt-4"):
            with ui.card():
                ui.label("Quick Card")
                ui.label("You can put any UI components in build().")

            with ui.card():
                ui.label("Another Card")
                ui.button("Click me", on_click=lambda: ui.notify("Clicked on home page!"))
```

## 4. Injecting Services: API / DB / Whatever

We don't want each page to instantiate its own HTTP client or database connection. Instead, we define services in one place and pass them down.

### 4.1 core/services.py

```python
# core/services.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import httpx


@dataclass
class ApiClient:
    base_url: str

    async def get_status(self) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0) as client:
            r = await client.get("/status")
            r.raise_for_status()
            return r.json()


def create_services() -> dict[str, Any]:
    """Construct and return the service container."""
    api_client = ApiClient(base_url="https://httpbin.org")
    # Extend here with db clients, redis, etc.
    return {
        "api": api_client,
    }
```

This gives us a simple "service container" dictionary.

## 5. A Dashboard Page Using Services

Now let's build a page that calls an API and renders a table / cards.

### 5.1 pages/dashboard.py

```python
# pages/dashboard.py
from __future__ import annotations

from typing import Any
from nicegui import ui
from .base import BasePage


class DashboardPage(BasePage):
    route = "/dashboard"
    title = "Dashboard"

    async def _refresh_status(self, label: ui.label) -> None:
        api = self.services["api"]
        label.text = "Loading..."
        try:
            data: dict[str, Any] = await api.get_status()
            label.text = f"API status: {data.get('url', 'unknown')}"
            ui.notify("Status refreshed", type="positive")
        except Exception as e:  # noqa: BLE001
            label.text = "Error while fetching status"
            ui.notify(f"Error: {e}", type="negative")

    def build(self) -> None:
        ui.label("Dashboard overview").classes("text-lg mb-2")

        with ui.card().classes("w-full max-w-xl"):
            status_label = ui.label("API status: unknown").classes("mb-2")
            ui.button(
                "Refresh status",
                on_click=lambda: self._refresh_status(status_label),
            )

        with ui.row().classes("gap-4 mt-4"):
            with ui.card():
                ui.label("Metric A")
                ui.label("42")

            with ui.card():
                ui.label("Metric B")
                ui.label("1337")
```

**Note**:
- We access `self.services["api"]`
- We use an async helper `_refresh_status` and hook it with `on_click`
- NiceGUI is fine with async callables

## 6. A Form Page with Validation

Let's add another class-based page that showcases forms, validation, and interactions.

### 6.1 pages/form_demo.py

```python
# pages/form_demo.py
from __future__ import annotations

from nicegui import ui
from .base import BasePage


class FormDemoPage(BasePage):
    route = "/form-demo"
    title = "Form Demo"

    def build(self) -> None:
        ui.label("User Input Form").classes("text-lg mb-2")

        with ui.card().classes("w-full max-w-lg"):
            name = ui.input("Name").props("outlined").classes("mb-2")
            email = ui.input("Email").props("outlined").classes("mb-2")
            agree = ui.checkbox("I agree to the terms").classes("mb-2")

            result = ui.label("").classes("mt-2 text-sm text-gray-500")

            def submit() -> None:
                errors = []
                if not name.value:
                    errors.append("Name is required.")
                if not email.value or "@" not in email.value:
                    errors.append("Valid email is required.")
                if not agree.value:
                    errors.append("You must agree to the terms.")

                if errors:
                    ui.notify("\n".join(errors), type="negative")
                    result.text = "Errors: " + "; ".join(errors)
                else:
                    ui.notify("Form submitted!", type="positive")
                    result.text = f"Submitted: {name.value} <{email.value}>"

            ui.button("Submit", on_click=submit)
```

## 7. Wiring It All Together: App Factory

Now we need a single app factory that:

- Builds the NiceGUI app
- Creates the shared services
- Registers all page classes

### 7.1 core/app_factory.py

```python
# core/app_factory.py
from __future__ import annotations

from nicegui import ui
from typing import Type

from pages.base import BasePage
from pages.home import HomePage
from pages.dashboard import DashboardPage
from pages.form_demo import FormDemoPage
from core.services import create_services


def create_app():
    """Create and configure the NiceGUI app with class-based pages."""
    services = create_services()

    # List of page classes to register
    page_classes: list[Type[BasePage]] = [
        HomePage,
        DashboardPage,
        FormDemoPage,
    ]

    # Register each page
    for page_cls in page_classes:
        page_cls.register(ui, services=services)

    # Optional: global header/footer, default layouts, etc.
    # For example, a simple top-level menu on every page via NiceGUI's `ui.header`:
    @ui.header()
    def header():
        with ui.row().classes("w-full justify-between items-center px-4 py-2"):
            ui.label("NiceGUI Class-Based App").classes("text-lg font-semibold")
            with ui.row().classes("gap-2"):
                ui.button("Home", on_click=lambda: ui.open("/"))
                ui.button("Dashboard", on_click=lambda: ui.open("/dashboard"))
                ui.button("Form", on_click=lambda: ui.open("/form-demo"))

    return ui
```

## 8. Entry Point

### 8.1 app.py

```python
# app.py
from __future__ import annotations

from core.app_factory import create_app

app = create_app()

if __name__ in {"__main__", "__mp_main__"}:
    # Default NiceGUI run:
    app.run(
        title="NiceGUI Class-Based Pages",
        reload=False,  # set True in dev if desired
        host="0.0.0.0",
        port=8080,
    )
```

**Run it**:

```bash
# Option 1: Direct Python
python app.py

# Option 2: Uvicorn (if using FastAPI integration)
uvicorn app:app --reload
```

## 9. Integration Patterns

Now that the skeleton is alive, here are some patterns for more serious integration.

### 9.1: FastAPI + NiceGUI in One Process

NiceGUI exposes a FastAPI app via `ui.run_with` / `app.fastapi`. A common pattern:

- Use FastAPI for REST/webhook endpoints
- Use NiceGUI for UI
- Share service container between them

**Skeleton**:

```python
# app.py
from nicegui import ui
from fastapi import FastAPI

from core.app_factory import create_app
from core.services import create_services

services = create_services()
ng_app = create_app()  # registers pages, etc.

fastapi_app = FastAPI()

# Example API endpoint
@fastapi_app.get("/api/ping")
async def ping():
    api = services["api"]
    data = await api.get_status()
    return {"status": "ok", "upstream": data}

ui.run_with(
    fastapi_app,
    title="NiceGUI + FastAPI",
    host="0.0.0.0",
    port=8080,
)
```

Now:

- `http://localhost:8080/` → NiceGUI UI
- `http://localhost:8080/api/ping` → FastAPI JSON

Pages keep using the same service container.

### 9.2: Page-Level Dependencies

Sometimes you don't want to pass the entire service container; you want typed attributes.

You can wrap the base to provide typed accessors:

```python
# pages/base.py (alt pattern)
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.services import ApiClient

@dataclass
class TypedServices:
    api: "ApiClient"

class BasePage(ABC):
    # ...
    def __init__(self, services: dict[str, Any]) -> None:
        self.services_raw = services
        self.s = TypedServices(api=services["api"])
```

Then in page:

```python
# pages/dashboard.py
class DashboardPage(BasePage):
    # ...
    async def _refresh_status(self, label: ui.label) -> None:
        data = await self.s.api.get_status()
        # ...
```

### 9.3: Background Tasks & Schedulers

For periodic tasks (polling metrics, refreshing dashboards), you can:

- Use `asyncio.create_task` from within a page
- Or integrate external schedulers (APScheduler, Prefect, etc.) and push updates via NiceGUI

**Example: simple background update**:

```python
# in a Page.build()
import asyncio

counter_label = ui.label("Counter: 0")
self.counter = 0  # attach to instance

async def updater():
    while True:
        await asyncio.sleep(1)
        self.counter += 1
        counter_label.text = f"Counter: {self.counter}"

asyncio.create_task(updater())
```

Just be aware you're now dancing with concurrency and lifetime—good for small toys, but for production you'll want a more disciplined approach.

## 10. Integration with Prefect

Let's build a real-world example: a NiceGUI dashboard that displays Prefect flow runs and allows triggering new runs.

### 10.1 core/services.py (Extended)

```python
# core/services.py (extended)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import httpx


@dataclass
class ApiClient:
    base_url: str

    async def get_status(self) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0) as client:
            r = await client.get("/status")
            r.raise_for_status()
            return r.json()


@dataclass
class PrefectClient:
    api_url: str
    api_key: str | None = None

    async def get_flow_runs(
        self, limit: int = 10, flow_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch recent flow runs from Prefect API."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        params = {"limit": limit}
        if flow_id:
            params["flow_id"] = flow_id

        async with httpx.AsyncClient(
            base_url=self.api_url, headers=headers, timeout=10.0
        ) as client:
            r = await client.get("/flow_runs", params=params)
            r.raise_for_status()
            return r.json()

    async def trigger_flow_run(self, deployment_id: str) -> dict[str, Any]:
        """Trigger a new flow run."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(
            base_url=self.api_url, headers=headers, timeout=10.0
        ) as client:
            r = await client.post(
                f"/deployments/{deployment_id}/create_flow_run",
                json={},
            )
            r.raise_for_status()
            return r.json()


def create_services() -> dict[str, Any]:
    """Construct and return the service container."""
    import os

    api_client = ApiClient(base_url="https://httpbin.org")
    prefect_client = PrefectClient(
        api_url=os.getenv("PREFECT_API_URL", "http://localhost:4200/api"),
        api_key=os.getenv("PREFECT_API_KEY"),
    )

    return {
        "api": api_client,
        "prefect": prefect_client,
    }
```

### 10.2 pages/prefect_dashboard.py

```python
# pages/prefect_dashboard.py
from __future__ import annotations

from typing import Any
from nicegui import ui
from .base import BasePage


class PrefectDashboardPage(BasePage):
    route = "/prefect"
    title = "Prefect Dashboard"

    def __init__(self, services: dict[str, Any] | None = None) -> None:
        super().__init__(services)
        self.flow_runs: list[dict[str, Any]] = []
        self.runs_table: ui.table | None = None

    async def _refresh_runs(self) -> None:
        """Fetch and display Prefect flow runs."""
        prefect = self.services["prefect"]
        try:
            self.flow_runs = await prefect.get_flow_runs(limit=20)
            if self.runs_table:
                self.runs_table.rows = [
                    {
                        "id": run.get("id", "unknown"),
                        "name": run.get("name", "unnamed"),
                        "state": run.get("state", {}).get("type", "unknown"),
                        "start_time": run.get("start_time", "N/A"),
                    }
                    for run in self.flow_runs
                ]
            ui.notify(f"Loaded {len(self.flow_runs)} flow runs", type="positive")
        except Exception as e:  # noqa: BLE001
            ui.notify(f"Error loading runs: {e}", type="negative")

    async def _trigger_run(self, deployment_id: str) -> None:
        """Trigger a new flow run."""
        prefect = self.services["prefect"]
        try:
            result = await prefect.trigger_flow_run(deployment_id)
            ui.notify(f"Flow run created: {result.get('id')}", type="positive")
            await self._refresh_runs()
        except Exception as e:  # noqa: BLE001
            ui.notify(f"Error triggering run: {e}", type="negative")

    def build(self) -> None:
        ui.label("Prefect Flow Runs").classes("text-lg mb-2")

        with ui.row().classes("gap-2 mb-4"):
            ui.button("Refresh", on_click=self._refresh_runs)
            ui.button("Auto-refresh (10s)", on_click=self._start_auto_refresh)

        columns = [
            {"name": "id", "label": "Run ID", "field": "id"},
            {"name": "name", "label": "Name", "field": "name"},
            {"name": "state", "label": "State", "field": "state"},
            {"name": "start_time", "label": "Start Time", "field": "start_time"},
        ]

        self.runs_table = ui.table(
            columns=columns,
            rows=[],
            row_key="id",
        ).classes("w-full")

        # Initial load
        self._refresh_runs()

    def _start_auto_refresh(self) -> None:
        """Start auto-refreshing the table every 10 seconds."""
        import asyncio

        async def refresh_loop():
            while True:
                await asyncio.sleep(10)
                await self._refresh_runs()

        asyncio.create_task(refresh_loop())
        ui.notify("Auto-refresh started", type="info")
```

### 10.3 Update app_factory.py

```python
# core/app_factory.py (add PrefectDashboardPage)
from pages.prefect_dashboard import PrefectDashboardPage

# In create_app():
page_classes: list[Type[BasePage]] = [
    HomePage,
    DashboardPage,
    FormDemoPage,
    PrefectDashboardPage,  # Add this
]
```

## 11. Integration with PostgreSQL

Let's add a page that queries PostgreSQL and displays data in real-time.

### 11.1 core/services.py (Add Postgres)

```python
# core/services.py (add Postgres client)
import os
from typing import Any
import asyncpg


@dataclass
class PostgresClient:
    connection_string: str
    pool: asyncpg.Pool | None = None

    async def get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.connection_string)
        return self.pool

    async def query(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        """Execute query and return rows as dicts."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
            return [dict(row) for row in rows]

    async def execute(self, sql: str, *args: Any) -> str:
        """Execute statement and return result."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            return await conn.execute(sql, *args)


def create_services() -> dict[str, Any]:
    """Construct and return the service container."""
    api_client = ApiClient(base_url="https://httpbin.org")
    prefect_client = PrefectClient(
        api_url=os.getenv("PREFECT_API_URL", "http://localhost:4200/api"),
        api_key=os.getenv("PREFECT_API_KEY"),
    )
    postgres_client = PostgresClient(
        connection_string=os.getenv(
            "DATABASE_URL", "postgresql://user:pass@localhost/dbname"
        )
    )

    return {
        "api": api_client,
        "prefect": prefect_client,
        "postgres": postgres_client,
    }
```

### 11.2 pages/postgres_dashboard.py

```python
# pages/postgres_dashboard.py
from __future__ import annotations

from typing import Any
from nicegui import ui
from .base import BasePage


class PostgresDashboardPage(BasePage):
    route = "/postgres"
    title = "PostgreSQL Dashboard"

    def __init__(self, services: dict[str, Any] | None = None) -> None:
        super().__init__(services)
        self.query_result: list[dict[str, Any]] = []
        self.result_table: ui.table | None = None

    async def _execute_query(self, sql_input: ui.textarea) -> None:
        """Execute SQL query and display results."""
        postgres = self.services["postgres"]
        sql = sql_input.value.strip()

        if not sql:
            ui.notify("Please enter a SQL query", type="warning")
            return

        try:
            self.query_result = await postgres.query(sql)
            if self.result_table:
                columns = (
                    [{"name": k, "label": k, "field": k} for k in self.query_result[0].keys()]
                    if self.query_result
                    else []
                )
                self.result_table.columns = columns
                self.result_table.rows = self.query_result
            ui.notify(f"Query returned {len(self.query_result)} rows", type="positive")
        except Exception as e:  # noqa: BLE001
            ui.notify(f"Query error: {e}", type="negative")

    def build(self) -> None:
        ui.label("PostgreSQL Query Interface").classes("text-lg mb-2")

        with ui.card().classes("w-full"):
            sql_input = ui.textarea(
                label="SQL Query",
                placeholder="SELECT * FROM table LIMIT 10;",
            ).classes("w-full mb-2")
            sql_input.value = "SELECT version();"

            ui.button(
                "Execute",
                on_click=lambda: self._execute_query(sql_input),
            ).classes("mb-4")

            columns = [{"name": "result", "label": "Result", "field": "result"}]
            self.result_table = ui.table(
                columns=columns,
                rows=[],
            ).classes("w-full")
```

### 11.3 Update requirements.txt

```txt
nicegui>=2.0.0
httpx>=0.27.0
asyncpg>=0.29.0
```

## 12. Integration with Redis

For caching and real-time updates, let's add Redis support.

### 12.1 core/services.py (Add Redis)

```python
# core/services.py (add Redis)
import redis.asyncio as redis


@dataclass
class RedisClient:
    url: str
    client: redis.Redis | None = None

    async def get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self.client is None:
            self.client = redis.from_url(self.url, decode_responses=True)
        return self.client

    async def get(self, key: str) -> str | None:
        """Get value from Redis."""
        client = await self.get_client()
        return await client.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        """Set value in Redis."""
        client = await self.get_client()
        return await client.set(key, value, ex=ex)

    async def keys(self, pattern: str = "*") -> list[str]:
        """Get keys matching pattern."""
        client = await self.get_client()
        return await client.keys(pattern)


def create_services() -> dict[str, Any]:
    """Construct and return the service container."""
    import os

    # ... existing services ...
    redis_client = RedisClient(url=os.getenv("REDIS_URL", "redis://localhost:6379"))

    return {
        "api": api_client,
        "prefect": prefect_client,
        "postgres": postgres_client,
        "redis": redis_client,
    }
```

### 12.2 pages/redis_dashboard.py

```python
# pages/redis_dashboard.py
from __future__ import annotations

from typing import Any
from nicegui import ui
from .base import BasePage


class RedisDashboardPage(BasePage):
    route = "/redis"
    title = "Redis Dashboard"

    async def _get_value(self, key_input: ui.input, value_label: ui.label) -> None:
        """Get value from Redis."""
        redis_client = self.services["redis"]
        key = key_input.value

        if not key:
            ui.notify("Please enter a key", type="warning")
            return

        try:
            value = await redis_client.get(key)
            value_label.text = f"Value: {value if value else '(nil)'}"
            ui.notify("Value retrieved", type="positive")
        except Exception as e:  # noqa: BLE001
            ui.notify(f"Error: {e}", type="negative")

    async def _set_value(
        self, key_input: ui.input, value_input: ui.input
    ) -> None:
        """Set value in Redis."""
        redis_client = self.services["redis"]
        key = key_input.value
        value = value_input.value

        if not key:
            ui.notify("Please enter a key", type="warning")
            return

        try:
            await redis_client.set(key, value)
            ui.notify(f"Set {key} = {value}", type="positive")
        except Exception as e:  # noqa: BLE001
            ui.notify(f"Error: {e}", type="negative")

    async def _list_keys(self, keys_list: ui.column) -> None:
        """List all keys in Redis."""
        redis_client = self.services["redis"]
        try:
            keys = await redis_client.keys()
            keys_list.clear()
            with keys_list:
                for key in keys[:100]:  # Limit to 100 keys
                    ui.label(key).classes("text-sm")
            ui.notify(f"Found {len(keys)} keys", type="positive")
        except Exception as e:  # noqa: BLE001
            ui.notify(f"Error: {e}", type="negative")

    def build(self) -> None:
        ui.label("Redis Operations").classes("text-lg mb-2")

        with ui.card().classes("w-full max-w-2xl"):
            key_input = ui.input("Key").classes("mb-2")
            value_input = ui.input("Value").classes("mb-2")
            value_label = ui.label("Value: (not loaded)").classes("mb-2")

            with ui.row().classes("gap-2 mb-4"):
                ui.button(
                    "Get",
                    on_click=lambda: self._get_value(key_input, value_label),
                )
                ui.button(
                    "Set",
                    on_click=lambda: self._set_value(key_input, value_input),
                )

            ui.separator()

            ui.label("Keys").classes("text-md font-semibold mb-2")
            keys_list = ui.column().classes("w-full max-h-64 overflow-auto")
            ui.button(
                "List Keys",
                on_click=lambda: self._list_keys(keys_list),
            )
```

### 12.3 Update requirements.txt

```txt
nicegui>=2.0.0
httpx>=0.27.0
asyncpg>=0.29.0
redis>=5.0.0
```

## 13. Data Flow Diagram

```mermaid
flowchart LR
    subgraph Services["Services / Backend"]
        API["ApiClient (httpx)"]
        Prefect["PrefectClient"]
        Postgres["PostgresClient (asyncpg)"]
        Redis["RedisClient"]
    end
    
    subgraph UI["NiceGUI App"]
        Home["HomePage"]
        Dash["DashboardPage"]
        Form["FormDemoPage"]
        PrefDash["PrefectDashboardPage"]
        PgDash["PostgresDashboardPage"]
        RedisDash["RedisDashboardPage"]
    end
    
    AppFactory["create_app()"] --> UI
    AppFactory --> Services
    
    Home --> Services
    Dash --> API
    PrefDash --> Prefect
    PgDash --> Postgres
    RedisDash --> Redis
    
    Prefect -->|"HTTP / JSON"| PrefAPI["Prefect API"]
    Postgres -->|"SQL"| PgDB["PostgreSQL"]
    Redis -->|"Commands"| RedisServer["Redis Server"]
```

The core idea:

- App factory builds services + registers pages
- Pages take services as dependencies
- Everything stays in tidy modules instead of a global ball of state

## 14. Complete Example: All Together

Here's a complete `app.py` that ties everything together:

```python
# app.py
from __future__ import annotations

import os
from nicegui import ui
from fastapi import FastAPI

from core.app_factory import create_app
from core.services import create_services

# Create services
services = create_services()

# Create NiceGUI app with all pages
ng_app = create_app()

# Create FastAPI app for API endpoints
fastapi_app = FastAPI()


@fastapi_app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "services": list(services.keys())}


@fastapi_app.get("/api/services")
async def list_services():
    """List available services."""
    return {"services": list(services.keys())}


# Run NiceGUI with FastAPI
if __name__ in {"__main__", "__mp_main__"}:
    ui.run_with(
        fastapi_app,
        title="NiceGUI Class-Based App",
        reload=os.getenv("NICEGUI_RELOAD", "false").lower() == "true",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
    )
```

## 15. Checklist for Extending This

If you're going to hand this to an agent and tell it to mutate reality:

- ✅ Add more page classes under `pages/` for specific domains (metrics, admin, logs)
- ✅ Implement a shared `BaseLayout` with sidebar navigation and theming
- ✅ Integrate authentication (e.g., JWT, OIDC) in an `AuthPage` + guards around routes
- ✅ Add a `SettingsPage` that persists config to a DB or file
- ✅ Add tests for service layer and simple smoke tests for NiceGUI pages (headless)
- ✅ Add error boundaries and global error handling
- ✅ Implement real-time updates via WebSockets or Server-Sent Events
- ✅ Add logging and monitoring integration
- ✅ Create reusable UI components (cards, tables, forms) in a `components/` directory

## 16. Production Considerations

### 16.1 Error Handling

Add global error handling:

```python
# core/app_factory.py
@ui.exception_handler
def handle_exception(e: Exception):
    ui.notify(f"Error: {e}", type="negative")
    # Log to your logging service
```

### 16.2 Authentication Guards

Add route protection:

```python
# pages/base.py (extended)
class BasePage(ABC):
    # ...
    requires_auth: bool = False

    def render(self) -> None:
        if self.requires_auth and not self._is_authenticated():
            ui.open("/login")
            return
        # ... rest of render
```

### 16.3 Environment Configuration

Use environment variables for all configuration:

```python
# core/config.py
import os
from dataclasses import dataclass

@dataclass
class Config:
    prefect_api_url: str = os.getenv("PREFECT_API_URL", "http://localhost:4200/api")
    prefect_api_key: str | None = os.getenv("PREFECT_API_KEY")
    database_url: str = os.getenv("DATABASE_URL", "postgresql://localhost/db")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
```

## See Also

- **[FastAPI Geospatial](../../best-practices/python/fastapi-geospatial.md)** - High-performance spatial APIs with FastAPI
- **[API Development](../../best-practices/python/api-development.md)** - Building high-performance, geospatial-aware APIs
- **[Python Async Best Practices](../../best-practices/python/python-async-best-practices.md)** - Write coroutines that don't betray you
- **[Prefect FIFO Flow with Redis](../system-administration/prefect-fifo-redis.md)** - Prefect workflow patterns

---

*This tutorial provides the complete machinery for building structured NiceGUI applications. The patterns scale from prototypes to production, from simple pages to complex dashboards with multiple backend integrations.*


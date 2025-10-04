# Python Type Hints Best Practices

**Objective**: Master senior-level Python type hints patterns for production systems. When you need to build type-safe Python applications, when you want to leverage static analysis tools, when you need enterprise-grade type safety strategies—these best practices become your weapon of choice.

## Core Principles

- **Type Safety**: Use type hints to catch errors at development time
- **Documentation**: Type hints serve as living documentation
- **IDE Support**: Enable better autocomplete and refactoring
- **Static Analysis**: Leverage mypy and other type checkers
- **Gradual Typing**: Add type hints incrementally to existing codebases

## Basic Type Hints

### Fundamental Types

```python
# python/01-basic-type-hints.py

"""
Basic type hints and fundamental typing patterns
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Set, Callable, Type
from dataclasses import dataclass
from enum import Enum
import asyncio
from pathlib import Path

# Basic types
def basic_types_example() -> None:
    """Example of basic type hints"""
    # Primitive types
    name: str = "John Doe"
    age: int = 30
    height: float = 5.9
    is_active: bool = True
    
    # Collections
    numbers: List[int] = [1, 2, 3, 4, 5]
    scores: Dict[str, float] = {"math": 95.5, "science": 87.0}
    coordinates: Tuple[float, float] = (40.7128, -74.0060)
    unique_ids: Set[str] = {"user1", "user2", "user3"}
    
    # Optional types
    optional_name: Optional[str] = None
    optional_age: Optional[int] = 25
    
    # Union types
    id_value: Union[int, str] = "user123"
    status: Union[str, int] = "active"
    
    print(f"Name: {name}, Age: {age}, Height: {height}")
    print(f"Numbers: {numbers}, Scores: {scores}")

# Function type hints
def calculate_average(numbers: List[float]) -> float:
    """Calculate average of numbers"""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

def process_user_data(user_id: int, name: str, email: Optional[str] = None) -> Dict[str, Any]:
    """Process user data with type hints"""
    return {
        "id": user_id,
        "name": name,
        "email": email,
        "is_verified": email is not None
    }

# Class type hints
@dataclass
class User:
    """User class with type hints"""
    id: int
    name: str
    email: Optional[str] = None
    age: Optional[int] = None
    is_active: bool = True
    
    def get_display_name(self) -> str:
        """Get user display name"""
        return f"{self.name} ({self.email})" if self.email else self.name
    
    def is_adult(self) -> bool:
        """Check if user is adult"""
        return self.age is not None and self.age >= 18

# Enum with type hints
class UserRole(Enum):
    """User role enumeration"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

@dataclass
class UserWithRole(User):
    """User with role information"""
    role: UserRole = UserRole.USER
    
    def has_permission(self, required_role: UserRole) -> bool:
        """Check if user has required permission"""
        role_hierarchy = {
            UserRole.GUEST: 0,
            UserRole.USER: 1,
            UserRole.ADMIN: 2
        }
        return role_hierarchy.get(self.role, 0) >= role_hierarchy.get(required_role, 0)

# Usage examples
def example_basic_usage():
    """Example basic type hints usage"""
    # Basic types
    basic_types_example()
    
    # Function with type hints
    numbers = [1.5, 2.5, 3.5, 4.5]
    average = calculate_average(numbers)
    print(f"Average: {average}")
    
    # Class with type hints
    user = User(id=1, name="John Doe", email="john@example.com", age=30)
    print(f"User: {user.get_display_name()}")
    print(f"Is adult: {user.is_adult()}")
    
    # User with role
    admin_user = UserWithRole(
        id=2, name="Admin User", email="admin@example.com", 
        age=35, role=UserRole.ADMIN
    )
    print(f"Admin has user permission: {admin_user.has_permission(UserRole.USER)}")
```

### Advanced Type Hints

```python
# python/02-advanced-type-hints.py

"""
Advanced type hints including generics, protocols, and complex types
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Set, Callable, Type, TypeVar, Generic, Protocol, runtime_checkable
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from enum import Enum

# Generic types
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class GenericContainer(Generic[T]):
    """Generic container class"""
    
    def __init__(self, items: List[T]) -> None:
        self.items = items
    
    def add(self, item: T) -> None:
        """Add item to container"""
        self.items.append(item)
    
    def get(self, index: int) -> Optional[T]:
        """Get item by index"""
        if 0 <= index < len(self.items):
            return self.items[index]
        return None
    
    def get_all(self) -> List[T]:
        """Get all items"""
        return self.items.copy()
    
    def filter(self, predicate: Callable[[T], bool]) -> List[T]:
        """Filter items by predicate"""
        return [item for item in self.items if predicate(item)]

# Protocol for type checking
@runtime_checkable
class Drawable(Protocol):
    """Protocol for drawable objects"""
    
    def draw(self) -> str:
        """Draw the object"""
        ...
    
    def get_area(self) -> float:
        """Get area of the object"""
        ...

class Circle:
    """Circle class implementing Drawable protocol"""
    
    def __init__(self, radius: float) -> None:
        self.radius = radius
    
    def draw(self) -> str:
        """Draw circle"""
        return f"Drawing circle with radius {self.radius}"
    
    def get_area(self) -> float:
        """Get circle area"""
        return 3.14159 * self.radius ** 2

class Rectangle:
    """Rectangle class implementing Drawable protocol"""
    
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
    
    def draw(self) -> str:
        """Draw rectangle"""
        return f"Drawing rectangle {self.width}x{self.height}"
    
    def get_area(self) -> float:
        """Get rectangle area"""
        return self.width * self.height

# Complex type hints
def process_drawable_objects(objects: List[Drawable]) -> Tuple[List[str], float]:
    """Process list of drawable objects"""
    drawings = [obj.draw() for obj in objects]
    total_area = sum(obj.get_area() for obj in objects)
    return drawings, total_area

# Callable type hints
def create_multiplier(factor: int) -> Callable[[int], int]:
    """Create multiplier function"""
    def multiply(x: int) -> int:
        return x * factor
    return multiply

def apply_function(func: Callable[[T], V], items: List[T]) -> List[V]:
    """Apply function to list of items"""
    return [func(item) for item in items]

# Async type hints
async def async_processor(items: List[T], processor: Callable[[T], Any]) -> List[Any]:
    """Async processor with type hints"""
    tasks = [processor(item) for item in items]
    return await asyncio.gather(*tasks)

# Type aliases
UserId = int
UserName = str
UserEmail = str
UserData = Dict[str, Any]

def create_user(user_id: UserId, name: UserName, email: UserEmail) -> UserData:
    """Create user with type aliases"""
    return {
        "id": user_id,
        "name": name,
        "email": email
    }

# Complex generic types
class Repository(Generic[T]):
    """Generic repository pattern"""
    
    def __init__(self) -> None:
        self.items: Dict[int, T] = {}
        self.next_id = 1
    
    def add(self, item: T) -> int:
        """Add item and return ID"""
        item_id = self.next_id
        self.items[item_id] = item
        self.next_id += 1
        return item_id
    
    def get(self, item_id: int) -> Optional[T]:
        """Get item by ID"""
        return self.items.get(item_id)
    
    def get_all(self) -> List[T]:
        """Get all items"""
        return list(self.items.values())
    
    def update(self, item_id: int, item: T) -> bool:
        """Update item"""
        if item_id in self.items:
            self.items[item_id] = item
            return True
        return False
    
    def delete(self, item_id: int) -> bool:
        """Delete item"""
        if item_id in self.items:
            del self.items[item_id]
            return True
        return False

# Usage examples
def example_advanced_usage():
    """Example advanced type hints usage"""
    # Generic container
    int_container = GenericContainer([1, 2, 3, 4, 5])
    int_container.add(6)
    print(f"Int container: {int_container.get_all()}")
    
    str_container = GenericContainer(["hello", "world"])
    str_container.add("python")
    print(f"String container: {str_container.get_all()}")
    
    # Protocol usage
    circle = Circle(5.0)
    rectangle = Rectangle(4.0, 6.0)
    
    drawings, total_area = process_drawable_objects([circle, rectangle])
    print(f"Drawings: {drawings}")
    print(f"Total area: {total_area}")
    
    # Callable usage
    double = create_multiplier(2)
    triple = create_multiplier(3)
    
    numbers = [1, 2, 3, 4, 5]
    doubled = apply_function(double, numbers)
    tripled = apply_function(triple, numbers)
    
    print(f"Doubled: {doubled}")
    print(f"Tripled: {tripled}")
    
    # Repository usage
    user_repo = Repository[User]()
    
    user1 = User(id=1, name="John", email="john@example.com")
    user2 = User(id=2, name="Jane", email="jane@example.com")
    
    id1 = user_repo.add(user1)
    id2 = user_repo.add(user2)
    
    print(f"Added users with IDs: {id1}, {id2}")
    print(f"All users: {user_repo.get_all()}")
```

## Type Checking and Validation

### Runtime Type Checking

```python
# python/03-type-checking.py

"""
Runtime type checking and validation patterns
"""

from typing import List, Dict, Any, Optional, Union, Type, get_type_hints, get_origin, get_args
import inspect
import functools
from dataclasses import dataclass
from enum import Enum

class TypeChecker:
    """Runtime type checker"""
    
    @staticmethod
    def check_type(value: Any, expected_type: Type) -> bool:
        """Check if value matches expected type"""
        if expected_type is Any:
            return True
        
        # Handle Union types
        if get_origin(expected_type) is Union:
            return any(TypeChecker.check_type(value, arg) for arg in get_args(expected_type))
        
        # Handle Optional types
        if expected_type is Optional or (get_origin(expected_type) is Union and type(None) in get_args(expected_type)):
            if value is None:
                return True
            # Check non-None type
            non_none_types = [arg for arg in get_args(expected_type) if arg is not type(None)]
            return any(TypeChecker.check_type(value, arg) for arg in non_none_types)
        
        # Handle List types
        if get_origin(expected_type) is list:
            if not isinstance(value, list):
                return False
            element_type = get_args(expected_type)[0]
            return all(TypeChecker.check_type(item, element_type) for item in value)
        
        # Handle Dict types
        if get_origin(expected_type) is dict:
            if not isinstance(value, dict):
                return False
            key_type, value_type = get_args(expected_type)
            return all(
                TypeChecker.check_type(k, key_type) and TypeChecker.check_type(v, value_type)
                for k, v in value.items()
            )
        
        # Handle Tuple types
        if get_origin(expected_type) is tuple:
            if not isinstance(value, tuple):
                return False
            element_types = get_args(expected_type)
            if len(value) != len(element_types):
                return False
            return all(TypeChecker.check_type(v, t) for v, t in zip(value, element_types))
        
        # Handle Set types
        if get_origin(expected_type) is set:
            if not isinstance(value, set):
                return False
            element_type = get_args(expected_type)[0]
            return all(TypeChecker.check_type(item, element_type) for item in value)
        
        # Handle Callable types
        if get_origin(expected_type) is callable:
            return callable(value)
        
        # Handle basic types
        return isinstance(value, expected_type)
    
    @staticmethod
    def validate_function_args(func: Callable, args: tuple, kwargs: dict) -> bool:
        """Validate function arguments against type hints"""
        try:
            hints = get_type_hints(func)
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            for param_name, param_value in bound.arguments.items():
                if param_name in hints:
                    expected_type = hints[param_name]
                    if not TypeChecker.check_type(param_value, expected_type):
                        return False
            
            return True
        except Exception:
            return False

def type_check(func: Callable) -> Callable:
    """Decorator for runtime type checking"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not TypeChecker.validate_function_args(func, args, kwargs):
            raise TypeError(f"Type validation failed for {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

class TypeValidator:
    """Type validator for data validation"""
    
    def __init__(self):
        self.validators: Dict[str, Type] = {}
    
    def register_validator(self, field_name: str, field_type: Type) -> None:
        """Register validator for field"""
        self.validators[field_name] = field_type
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against registered validators"""
        validated_data = {}
        errors = {}
        
        for field_name, field_type in self.validators.items():
            if field_name in data:
                value = data[field_name]
                if TypeChecker.check_type(value, field_type):
                    validated_data[field_name] = value
                else:
                    errors[field_name] = f"Expected {field_type}, got {type(value)}"
            else:
                errors[field_name] = f"Missing required field: {field_name}"
        
        if errors:
            raise ValueError(f"Validation errors: {errors}")
        
        return validated_data

# Usage examples
@type_check
def process_numbers(numbers: List[int]) -> int:
    """Process list of numbers with type checking"""
    return sum(numbers)

@type_check
def create_user_profile(name: str, age: int, email: Optional[str] = None) -> Dict[str, Any]:
    """Create user profile with type checking"""
    return {
        "name": name,
        "age": age,
        "email": email,
        "is_verified": email is not None
    }

def example_type_checking():
    """Example type checking usage"""
    # Type checking decorator
    try:
        result = process_numbers([1, 2, 3, 4, 5])
        print(f"Sum: {result}")
    except TypeError as e:
        print(f"Type error: {e}")
    
    # Type validator
    validator = TypeValidator()
    validator.register_validator("name", str)
    validator.register_validator("age", int)
    validator.register_validator("email", Optional[str])
    
    try:
        data = {"name": "John", "age": 30, "email": "john@example.com"}
        validated = validator.validate(data)
        print(f"Validated data: {validated}")
    except ValueError as e:
        print(f"Validation error: {e}")
```

### Static Type Analysis

```python
# python/04-static-type-analysis.py

"""
Static type analysis patterns and mypy integration
"""

from typing import List, Dict, Any, Optional, Union, TypeVar, Generic, Protocol, runtime_checkable
from abc import ABC, abstractmethod
import mypy.api
import subprocess
import sys
from pathlib import Path

class TypeAnalyzer:
    """Static type analyzer"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.mypy_config = project_path / "mypy.ini"
    
    def setup_mypy_config(self) -> None:
        """Setup mypy configuration"""
        config_content = """[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True
show_error_codes = True
show_column_numbers = True
show_error_context = True
pretty = True
color_output = True
error_summary = True

[mypy-tests.*]
disallow_untyped_defs = False

[mypy-.*\\.migrations.*]
ignore_errors = True
"""
        
        with open(self.mypy_config, 'w') as f:
            f.write(config_content)
    
    def run_type_check(self, files: List[str] = None) -> Dict[str, Any]:
        """Run mypy type check"""
        if files is None:
            files = [str(self.project_path)]
        
        try:
            result = mypy.api.run(files)
            return {
                "exit_status": result[0],
                "stdout": result[1],
                "stderr": result[2]
            }
        except Exception as e:
            return {
                "exit_status": 1,
                "stdout": "",
                "stderr": str(e)
            }
    
    def check_file(self, file_path: str) -> Dict[str, Any]:
        """Check single file"""
        return self.run_type_check([file_path])
    
    def check_project(self) -> Dict[str, Any]:
        """Check entire project"""
        return self.run_type_check()

class TypeStubGenerator:
    """Generate type stubs for external libraries"""
    
    def __init__(self):
        self.stub_content = {}
    
    def generate_stub(self, module_name: str, functions: List[Dict[str, Any]]) -> str:
        """Generate type stub for module"""
        stub_lines = [f"# Stub for {module_name}", ""]
        
        for func in functions:
            name = func["name"]
            params = func.get("params", [])
            return_type = func.get("return_type", "Any")
            
            param_str = ", ".join(f"{p['name']}: {p['type']}" for p in params)
            stub_lines.append(f"def {name}({param_str}) -> {return_type}: ...")
        
        return "\n".join(stub_lines)
    
    def save_stub(self, module_name: str, stub_content: str, output_dir: Path) -> None:
        """Save type stub to file"""
        stub_file = output_dir / f"{module_name}.pyi"
        with open(stub_file, 'w') as f:
            f.write(stub_content)

class TypeDocumentation:
    """Generate type documentation"""
    
    def __init__(self):
        self.documented_types = {}
    
    def document_type(self, type_name: str, description: str, examples: List[str] = None) -> None:
        """Document a type"""
        self.documented_types[type_name] = {
            "description": description,
            "examples": examples or []
        }
    
    def generate_documentation(self) -> str:
        """Generate type documentation"""
        doc_lines = ["# Type Documentation", ""]
        
        for type_name, info in self.documented_types.items():
            doc_lines.append(f"## {type_name}")
            doc_lines.append(f"{info['description']}")
            
            if info["examples"]:
                doc_lines.append("### Examples:")
                for example in info["examples"]:
                    doc_lines.append(f"```python\n{example}\n```")
            
            doc_lines.append("")
        
        return "\n".join(doc_lines)

# Usage examples
def example_static_analysis():
    """Example static type analysis usage"""
    # Setup type analyzer
    project_path = Path(".")
    analyzer = TypeAnalyzer(project_path)
    analyzer.setup_mypy_config()
    
    # Run type check
    result = analyzer.run_type_check()
    print(f"Type check result: {result['exit_status']}")
    print(f"Output: {result['stdout']}")
    
    # Generate type stubs
    stub_generator = TypeStubGenerator()
    functions = [
        {
            "name": "process_data",
            "params": [{"name": "data", "type": "List[str]"}],
            "return_type": "Dict[str, Any]"
        }
    ]
    
    stub_content = stub_generator.generate_stub("external_module", functions)
    print(f"Generated stub:\n{stub_content}")
    
    # Generate type documentation
    doc_generator = TypeDocumentation()
    doc_generator.document_type(
        "UserId",
        "Type alias for user ID",
        ["UserId = int", "user_id: UserId = 123"]
    )
    
    documentation = doc_generator.generate_documentation()
    print(f"Type documentation:\n{documentation}")
```

## Type-Safe Design Patterns

### Generic Patterns

```python
# python/05-type-safe-patterns.py

"""
Type-safe design patterns and generic implementations
"""

from typing import List, Dict, Any, Optional, Union, TypeVar, Generic, Protocol, runtime_checkable, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# Generic type variables
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# Type-safe factory pattern
class Factory(Generic[T]):
    """Generic factory pattern"""
    
    def __init__(self, creator: Callable[[], T]) -> None:
        self.creator = creator
    
    def create(self) -> T:
        """Create instance"""
        return self.creator()
    
    def create_multiple(self, count: int) -> List[T]:
        """Create multiple instances"""
        return [self.create() for _ in range(count)]

# Type-safe builder pattern
class Builder(Generic[T]):
    """Generic builder pattern"""
    
    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}
    
    def set(self, key: str, value: Any) -> 'Builder[T]':
        """Set builder value"""
        self.data[key] = value
        return self
    
    def build(self) -> T:
        """Build instance"""
        raise NotImplementedError("Subclasses must implement build method")

# Type-safe observer pattern
class Observer(Protocol):
    """Observer protocol"""
    
    def update(self, subject: 'Subject', data: Any) -> None:
        """Update observer"""
        ...

class Subject:
    """Subject in observer pattern"""
    
    def __init__(self) -> None:
        self.observers: List[Observer] = []
    
    def attach(self, observer: Observer) -> None:
        """Attach observer"""
        self.observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        """Detach observer"""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def notify(self, data: Any) -> None:
        """Notify all observers"""
        for observer in self.observers:
            observer.update(self, data)

# Type-safe command pattern
class Command(Protocol):
    """Command protocol"""
    
    def execute(self) -> Any:
        """Execute command"""
        ...
    
    def undo(self) -> Any:
        """Undo command"""
        ...

class CommandInvoker:
    """Command invoker"""
    
    def __init__(self) -> None:
        self.history: List[Command] = []
    
    def execute_command(self, command: Command) -> Any:
        """Execute command"""
        result = command.execute()
        self.history.append(command)
        return result
    
    def undo_last(self) -> Any:
        """Undo last command"""
        if self.history:
            command = self.history.pop()
            return command.undo()
        return None

# Type-safe strategy pattern
class Strategy(Protocol[T]):
    """Strategy protocol"""
    
    def execute(self, data: T) -> Any:
        """Execute strategy"""
        ...

class Context(Generic[T]):
    """Context for strategy pattern"""
    
    def __init__(self, strategy: Strategy[T]) -> None:
        self.strategy = strategy
    
    def set_strategy(self, strategy: Strategy[T]) -> None:
        """Set strategy"""
        self.strategy = strategy
    
    def execute(self, data: T) -> Any:
        """Execute strategy"""
        return self.strategy.execute(data)

# Type-safe repository pattern
class Repository(Generic[T]):
    """Generic repository"""
    
    def __init__(self) -> None:
        self.items: Dict[int, T] = {}
        self.next_id = 1
    
    def add(self, item: T) -> int:
        """Add item"""
        item_id = self.next_id
        self.items[item_id] = item
        self.next_id += 1
        return item_id
    
    def get(self, item_id: int) -> Optional[T]:
        """Get item by ID"""
        return self.items.get(item_id)
    
    def get_all(self) -> List[T]:
        """Get all items"""
        return list(self.items.values())
    
    def update(self, item_id: int, item: T) -> bool:
        """Update item"""
        if item_id in self.items:
            self.items[item_id] = item
            return True
        return False
    
    def delete(self, item_id: int) -> bool:
        """Delete item"""
        if item_id in self.items:
            del self.items[item_id]
            return True
        return False

# Type-safe service pattern
class Service(Generic[T]):
    """Generic service"""
    
    def __init__(self, repository: Repository[T]) -> None:
        self.repository = repository
    
    def create(self, item: T) -> int:
        """Create item"""
        return self.repository.add(item)
    
    def get(self, item_id: int) -> Optional[T]:
        """Get item by ID"""
        return self.repository.get(item_id)
    
    def get_all(self) -> List[T]:
        """Get all items"""
        return self.repository.get_all()
    
    def update(self, item_id: int, item: T) -> bool:
        """Update item"""
        return self.repository.update(item_id, item)
    
    def delete(self, item_id: int) -> bool:
        """Delete item"""
        return self.repository.delete(item_id)

# Usage examples
def example_type_safe_patterns():
    """Example type-safe patterns usage"""
    # Factory pattern
    def create_user() -> User:
        return User(id=1, name="John", email="john@example.com")
    
    user_factory = Factory(create_user)
    user = user_factory.create()
    print(f"Created user: {user}")
    
    # Repository pattern
    user_repo = Repository[User]()
    user_service = Service(user_repo)
    
    user_id = user_service.create(user)
    retrieved_user = user_service.get(user_id)
    print(f"Retrieved user: {retrieved_user}")
    
    # Strategy pattern
    class StringProcessor:
        def execute(self, data: str) -> str:
            return data.upper()
    
    class NumberProcessor:
        def execute(self, data: int) -> int:
            return data * 2
    
    string_context = Context(StringProcessor())
    number_context = Context(NumberProcessor())
    
    string_result = string_context.execute("hello")
    number_result = number_context.execute(5)
    
    print(f"String result: {string_result}")
    print(f"Number result: {number_result}")
```

## TL;DR Runbook

### Quick Start

```python
# 1. Basic type hints
def process_data(data: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in data}

# 2. Generic types
T = TypeVar('T')
class Container(Generic[T]):
    def __init__(self, items: List[T]) -> None:
        self.items = items

# 3. Type checking decorator
@type_check
def validate_input(value: int) -> str:
    return str(value)

# 4. Protocol for type safety
class Drawable(Protocol):
    def draw(self) -> str: ...

# 5. Type-safe patterns
class Repository(Generic[T]):
    def add(self, item: T) -> int: ...
    def get(self, item_id: int) -> Optional[T]: ...
```

### Essential Patterns

```python
# Complete type hints setup
def setup_type_hints():
    """Setup complete type hints environment"""
    
    # Type checker
    type_checker = TypeChecker()
    
    # Type validator
    validator = TypeValidator()
    
    # Type analyzer
    analyzer = TypeAnalyzer(Path("."))
    analyzer.setup_mypy_config()
    
    # Type documentation
    doc_generator = TypeDocumentation()
    
    print("Type hints setup complete!")
```

---

*This guide provides the complete machinery for Python type hints. Each pattern includes implementation examples, type safety strategies, and real-world usage patterns for enterprise type-safe development.*

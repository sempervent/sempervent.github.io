# Python Security Best Practices

**Objective**: Master senior-level Python security patterns for production systems. When you need to implement comprehensive security measures, when you want to build secure applications, when you need enterprise-grade security strategies—these best practices become your weapon of choice.

## Core Principles

- **Defense in Depth**: Implement multiple layers of security
- **Least Privilege**: Grant minimum necessary permissions
- **Zero Trust**: Verify everything, trust nothing
- **Security by Design**: Build security into every component
- **Continuous Monitoring**: Implement ongoing security oversight

## Authentication & Authorization

### Advanced Authentication Patterns

```python
# python/01-authentication.py

"""
Advanced authentication and authorization patterns
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import hmac
import secrets
import time
import jwt
from datetime import datetime, timedelta
import logging
from functools import wraps
import bcrypt
import argon2
from passlib.context import CryptContext
import pyotp
import qrcode
from io import BytesIO
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuthMethod(Enum):
    """Authentication method enumeration"""
    PASSWORD = "password"
    TOKEN = "token"
    OAUTH = "oauth"
    MFA = "mfa"
    BIOMETRIC = "biometric"

class Permission(Enum):
    """Permission enumeration"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

@dataclass
class User:
    """User definition"""
    id: str
    username: str
    email: str
    password_hash: str
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = None
    last_login: datetime = None
    failed_login_attempts: int = 0
    locked_until: datetime = None
    mfa_secret: Optional[str] = None
    permissions: List[Permission] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.permissions is None:
            self.permissions = []

class PasswordManager:
    """Advanced password management"""
    
    def __init__(self, algorithm: str = "argon2"):
        self.algorithm = algorithm
        if algorithm == "bcrypt":
            self.context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        elif algorithm == "argon2":
            self.context = CryptContext(schemes=["argon2"], deprecated="auto")
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def hash_password(self, password: str) -> str:
        """Hash password using secure algorithm"""
        return self.context.hash(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return self.context.verify(password, hashed)
    
    def check_password_strength(self, password: str) -> Dict[str, Any]:
        """Check password strength"""
        score = 0
        feedback = []
        
        # Length check
        if len(password) >= 8:
            score += 1
        else:
            feedback.append("Password should be at least 8 characters long")
        
        # Character variety checks
        if any(c.islower() for c in password):
            score += 1
        else:
            feedback.append("Password should contain lowercase letters")
        
        if any(c.isupper() for c in password):
            score += 1
        else:
            feedback.append("Password should contain uppercase letters")
        
        if any(c.isdigit() for c in password):
            score += 1
        else:
            feedback.append("Password should contain numbers")
        
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 1
        else:
            feedback.append("Password should contain special characters")
        
        # Common password check
        common_passwords = ["password", "123456", "qwerty", "admin"]
        if password.lower() in common_passwords:
            score = 0
            feedback.append("Password is too common")
        
        strength_levels = ["Very Weak", "Weak", "Fair", "Good", "Strong"]
        strength = strength_levels[min(score, 4)]
        
        return {
            "score": score,
            "strength": strength,
            "feedback": feedback,
            "is_strong": score >= 4
        }

class TokenManager:
    """JWT token management"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_blacklist = set()
    
    def generate_access_token(self, user_id: str, permissions: List[Permission], 
                            expires_in: int = 3600) -> str:
        """Generate JWT access token"""
        payload = {
            "user_id": user_id,
            "permissions": [p.value for p in permissions],
            "token_type": "access",
            "exp": datetime.utcnow() + timedelta(seconds=expires_in),
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(32)  # JWT ID for blacklisting
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def generate_refresh_token(self, user_id: str, expires_in: int = 86400 * 7) -> str:
        """Generate JWT refresh token"""
        payload = {
            "user_id": user_id,
            "token_type": "refresh",
            "exp": datetime.utcnow() + timedelta(seconds=expires_in),
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(32)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            # Check if token is blacklisted
            if token in self.token_blacklist:
                return None
            
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def blacklist_token(self, token: str) -> None:
        """Add token to blacklist"""
        self.token_blacklist.add(token)
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Generate new access token from refresh token"""
        payload = self.verify_token(refresh_token)
        
        if not payload or payload.get("token_type") != "refresh":
            return None
        
        # Generate new access token
        user_id = payload["user_id"]
        permissions = [Permission(p) for p in payload.get("permissions", [])]
        
        return self.generate_access_token(user_id, permissions)

class MFAManager:
    """Multi-Factor Authentication manager"""
    
    def __init__(self):
        self.totp = pyotp.TOTP
    
    def generate_mfa_secret(self, user_id: str) -> str:
        """Generate MFA secret for user"""
        secret = pyotp.random_base32()
        return secret
    
    def generate_mfa_qr_code(self, user_id: str, secret: str, issuer: str = "MyApp") -> str:
        """Generate QR code for MFA setup"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_id,
            issuer_name=issuer
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def verify_mfa_code(self, secret: str, code: str) -> bool:
        """Verify MFA code"""
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=1)
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for MFA"""
        return [secrets.token_urlsafe(8) for _ in range(count)]

class RateLimiter:
    """Rate limiting for authentication"""
    
    def __init__(self, max_attempts: int = 5, window_minutes: int = 15):
        self.max_attempts = max_attempts
        self.window_minutes = window_minutes
        self.attempts = {}  # {ip: [(timestamp, success), ...]}
    
    def is_rate_limited(self, ip: str) -> bool:
        """Check if IP is rate limited"""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=self.window_minutes)
        
        if ip not in self.attempts:
            return False
        
        # Filter attempts within window
        recent_attempts = [
            attempt for attempt in self.attempts[ip]
            if attempt[0] > window_start
        ]
        
        # Count failed attempts
        failed_attempts = [attempt for attempt in recent_attempts if not attempt[1]]
        
        return len(failed_attempts) >= self.max_attempts
    
    def record_attempt(self, ip: str, success: bool) -> None:
        """Record authentication attempt"""
        now = datetime.utcnow()
        
        if ip not in self.attempts:
            self.attempts[ip] = []
        
        self.attempts[ip].append((now, success))
        
        # Clean old attempts
        window_start = now - timedelta(minutes=self.window_minutes * 2)
        self.attempts[ip] = [
            attempt for attempt in self.attempts[ip]
            if attempt[0] > window_start
        ]

class AuthenticationService:
    """Comprehensive authentication service"""
    
    def __init__(self, password_manager: PasswordManager, token_manager: TokenManager,
                 mfa_manager: MFAManager, rate_limiter: RateLimiter):
        self.password_manager = password_manager
        self.token_manager = token_manager
        self.mfa_manager = mfa_manager
        self.rate_limiter = rate_limiter
        self.users = {}  # In production, this would be a database
        self.sessions = {}
    
    def register_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Register new user"""
        # Check password strength
        strength_check = self.password_manager.check_password_strength(password)
        if not strength_check["is_strong"]:
            return {
                "success": False,
                "error": "Password is not strong enough",
                "feedback": strength_check["feedback"]
            }
        
        # Check if user already exists
        if username in self.users or email in [user.email for user in self.users.values()]:
            return {
                "success": False,
                "error": "User already exists"
            }
        
        # Create user
        user = User(
            id=secrets.token_urlsafe(16),
            username=username,
            email=email,
            password_hash=self.password_manager.hash_password(password)
        )
        
        self.users[username] = user
        
        return {
            "success": True,
            "user_id": user.id,
            "message": "User registered successfully"
        }
    
    def authenticate_user(self, username: str, password: str, 
                         mfa_code: Optional[str] = None, ip: str = "unknown") -> Dict[str, Any]:
        """Authenticate user with optional MFA"""
        # Check rate limiting
        if self.rate_limiter.is_rate_limited(ip):
            return {
                "success": False,
                "error": "Too many failed attempts. Please try again later."
            }
        
        # Check if user exists
        if username not in self.users:
            self.rate_limiter.record_attempt(ip, False)
            return {
                "success": False,
                "error": "Invalid credentials"
            }
        
        user = self.users[username]
        
        # Check if user is locked
        if user.locked_until and datetime.utcnow() < user.locked_until:
            return {
                "success": False,
                "error": "Account is locked"
            }
        
        # Verify password
        if not self.password_manager.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            # Lock account after too many failed attempts
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.utcnow() + timedelta(minutes=30)
            
            self.rate_limiter.record_attempt(ip, False)
            return {
                "success": False,
                "error": "Invalid credentials"
            }
        
        # Check MFA if enabled
        if user.mfa_secret:
            if not mfa_code:
                return {
                    "success": False,
                    "error": "MFA code required",
                    "mfa_required": True
                }
            
            if not self.mfa_manager.verify_mfa_code(user.mfa_secret, mfa_code):
                self.rate_limiter.record_attempt(ip, False)
                return {
                    "success": False,
                    "error": "Invalid MFA code"
                }
        
        # Reset failed attempts
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        
        # Generate tokens
        access_token = self.token_manager.generate_access_token(
            user.id, user.permissions
        )
        refresh_token = self.token_manager.generate_refresh_token(user.id)
        
        # Store session
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            "user_id": user.id,
            "created_at": datetime.utcnow(),
            "ip": ip
        }
        
        self.rate_limiter.record_attempt(ip, True)
        
        return {
            "success": True,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "session_id": session_id,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "permissions": [p.value for p in user.permissions]
            }
        }
    
    def setup_mfa(self, username: str) -> Dict[str, Any]:
        """Setup MFA for user"""
        if username not in self.users:
            return {
                "success": False,
                "error": "User not found"
            }
        
        user = self.users[username]
        secret = self.mfa_manager.generate_mfa_secret(user.id)
        user.mfa_secret = secret
        
        qr_code = self.mfa_manager.generate_mfa_qr_code(user.id, secret)
        backup_codes = self.mfa_manager.generate_backup_codes()
        
        return {
            "success": True,
            "secret": secret,
            "qr_code": qr_code,
            "backup_codes": backup_codes
        }
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        return self.token_manager.verify_token(token)
    
    def logout(self, token: str) -> bool:
        """Logout user and blacklist token"""
        self.token_manager.blacklist_token(token)
        return True

class AuthorizationService:
    """Authorization service for permission checking"""
    
    def __init__(self, auth_service: AuthenticationService):
        self.auth_service = auth_service
    
    def check_permission(self, token: str, required_permission: Permission) -> bool:
        """Check if user has required permission"""
        payload = self.auth_service.verify_token(token)
        
        if not payload:
            return False
        
        user_permissions = [Permission(p) for p in payload.get("permissions", [])]
        return required_permission in user_permissions
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract token from request (implementation depends on framework)
                token = kwargs.get('token') or args[0] if args else None
                
                if not token or not self.check_permission(token, permission):
                    raise PermissionError("Insufficient permissions")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator

# Usage examples
def example_authentication():
    """Example authentication usage"""
    # Create managers
    password_manager = PasswordManager("argon2")
    token_manager = TokenManager("your-secret-key")
    mfa_manager = MFAManager()
    rate_limiter = RateLimiter()
    
    # Create authentication service
    auth_service = AuthenticationService(
        password_manager, token_manager, mfa_manager, rate_limiter
    )
    
    # Register user
    registration_result = auth_service.register_user(
        "john_doe", "john@example.com", "SecurePassword123!"
    )
    print(f"Registration: {registration_result}")
    
    # Authenticate user
    auth_result = auth_service.authenticate_user("john_doe", "SecurePassword123!")
    print(f"Authentication: {auth_result}")
    
    # Setup MFA
    mfa_result = auth_service.setup_mfa("john_doe")
    print(f"MFA setup: {mfa_result['success']}")
    
    # Verify token
    if auth_result["success"]:
        token_payload = auth_service.verify_token(auth_result["access_token"])
        print(f"Token payload: {token_payload}")
    
    # Authorization
    authz_service = AuthorizationService(auth_service)
    
    @authz_service.require_permission(Permission.READ)
    def read_data(token: str):
        return "Data read successfully"
    
    # Test authorization
    try:
        result = read_data(auth_result["access_token"])
        print(f"Authorization test: {result}")
    except PermissionError as e:
        print(f"Authorization failed: {e}")
```

### Input Validation & Sanitization

```python
# python/02-input-validation.py

"""
Input validation and sanitization patterns
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import re
import html
import bleach
import validators
from email_validator import validate_email, EmailNotValidError
import phonenumbers
from phonenumbers import NumberParseException
import uuid
from datetime import datetime
import json
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom validation error"""
    pass

class InputValidator:
    """Comprehensive input validator"""
    
    def __init__(self):
        self.validation_rules = {}
        self.sanitization_rules = {}
    
    def validate_string(self, value: str, min_length: int = 0, max_length: int = 1000,
                      pattern: Optional[str] = None, allow_html: bool = False) -> str:
        """Validate and sanitize string input"""
        if not isinstance(value, str):
            raise ValidationError("Value must be a string")
        
        # Length validation
        if len(value) < min_length:
            raise ValidationError(f"String too short (minimum {min_length} characters)")
        
        if len(value) > max_length:
            raise ValidationError(f"String too long (maximum {max_length} characters)")
        
        # Pattern validation
        if pattern and not re.match(pattern, value):
            raise ValidationError("String does not match required pattern")
        
        # HTML sanitization
        if not allow_html:
            value = html.escape(value)
        
        return value
    
    def validate_email(self, email: str) -> str:
        """Validate email address"""
        try:
            validated_email = validate_email(email)
            return validated_email.email
        except EmailNotValidError as e:
            raise ValidationError(f"Invalid email address: {e}")
    
    def validate_phone(self, phone: str, country_code: str = "US") -> str:
        """Validate phone number"""
        try:
            parsed_number = phonenumbers.parse(phone, country_code)
            if not phonenumbers.is_valid_number(parsed_number):
                raise ValidationError("Invalid phone number")
            return phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
        except NumberParseException as e:
            raise ValidationError(f"Invalid phone number: {e}")
    
    def validate_url(self, url: str) -> str:
        """Validate URL"""
        if not validators.url(url):
            raise ValidationError("Invalid URL")
        return url
    
    def validate_uuid(self, uuid_string: str) -> str:
        """Validate UUID"""
        try:
            uuid.UUID(uuid_string)
            return uuid_string
        except ValueError:
            raise ValidationError("Invalid UUID format")
    
    def validate_json(self, json_string: str) -> Dict[str, Any]:
        """Validate JSON string"""
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e}")
    
    def validate_integer(self, value: Union[str, int], min_value: Optional[int] = None,
                        max_value: Optional[int] = None) -> int:
        """Validate integer"""
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            raise ValidationError("Value must be an integer")
        
        if min_value is not None and int_value < min_value:
            raise ValidationError(f"Value must be at least {min_value}")
        
        if max_value is not None and int_value > max_value:
            raise ValidationError(f"Value must be at most {max_value}")
        
        return int_value
    
    def validate_float(self, value: Union[str, float], min_value: Optional[float] = None,
                      max_value: Optional[float] = None) -> float:
        """Validate float"""
        try:
            float_value = float(value)
        except (ValueError, TypeError):
            raise ValidationError("Value must be a float")
        
        if min_value is not None and float_value < min_value:
            raise ValidationError(f"Value must be at least {min_value}")
        
        if max_value is not None and float_value > max_value:
            raise ValidationError(f"Value must be at most {max_value}")
        
        return float_value
    
    def validate_boolean(self, value: Union[str, bool]) -> bool:
        """Validate boolean"""
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            if value.lower() in ('true', '1', 'yes', 'on'):
                return True
            elif value.lower() in ('false', '0', 'no', 'off'):
                return False
        
        raise ValidationError("Value must be a boolean")
    
    def validate_datetime(self, value: Union[str, datetime], 
                         format_string: Optional[str] = None) -> datetime:
        """Validate datetime"""
        if isinstance(value, datetime):
            return value
        
        if isinstance(value, str):
            if format_string:
                try:
                    return datetime.strptime(value, format_string)
                except ValueError:
                    raise ValidationError(f"Invalid datetime format. Expected: {format_string}")
            else:
                # Try common formats
                formats = [
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%d"
                ]
                
                for fmt in formats:
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        continue
                
                raise ValidationError("Invalid datetime format")
        
        raise ValidationError("Value must be a datetime")

class SQLInjectionPrevention:
    """SQL injection prevention utilities"""
    
    @staticmethod
    def sanitize_sql_input(value: str) -> str:
        """Sanitize input for SQL queries"""
        # Remove or escape dangerous characters
        dangerous_chars = ["'", '"', ";", "--", "/*", "*/", "xp_", "sp_"]
        
        for char in dangerous_chars:
            value = value.replace(char, "")
        
        # Limit length
        if len(value) > 1000:
            value = value[:1000]
        
        return value
    
    @staticmethod
    def validate_sql_identifier(identifier: str) -> bool:
        """Validate SQL identifier (table/column name)"""
        # Only allow alphanumeric characters and underscores
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
        return bool(re.match(pattern, identifier))

class XSSPrevention:
    """XSS prevention utilities"""
    
    @staticmethod
    def sanitize_html(html_content: str, allowed_tags: List[str] = None) -> str:
        """Sanitize HTML content"""
        if allowed_tags is None:
            allowed_tags = ['p', 'br', 'strong', 'em', 'u']
        
        return bleach.clean(html_content, tags=allowed_tags)
    
    @staticmethod
    def escape_html(text: str) -> str:
        """Escape HTML characters"""
        return html.escape(text)
    
    @staticmethod
    def validate_css(css_content: str) -> str:
        """Validate and sanitize CSS content"""
        # Remove dangerous CSS properties
        dangerous_properties = ['expression', 'javascript:', 'vbscript:', 'onload', 'onerror']
        
        for prop in dangerous_properties:
            css_content = css_content.replace(prop, "")
        
        return css_content

class FileUploadValidator:
    """File upload validation"""
    
    def __init__(self, max_size: int = 10 * 1024 * 1024,  # 10MB
                 allowed_extensions: List[str] = None,
                 allowed_mime_types: List[str] = None):
        self.max_size = max_size
        self.allowed_extensions = allowed_extensions or ['.jpg', '.jpeg', '.png', '.gif', '.pdf']
        self.allowed_mime_types = allowed_mime_types or [
            'image/jpeg', 'image/png', 'image/gif', 'application/pdf'
        ]
    
    def validate_file(self, filename: str, content: bytes, mime_type: str) -> Dict[str, Any]:
        """Validate uploaded file"""
        errors = []
        
        # Check file size
        if len(content) > self.max_size:
            errors.append(f"File too large (maximum {self.max_size} bytes)")
        
        # Check file extension
        file_ext = '.' + filename.split('.')[-1].lower()
        if file_ext not in self.allowed_extensions:
            errors.append(f"File extension not allowed: {file_ext}")
        
        # Check MIME type
        if mime_type not in self.allowed_mime_types:
            errors.append(f"MIME type not allowed: {mime_type}")
        
        # Check for malicious content
        if self._contains_malicious_content(content):
            errors.append("File contains potentially malicious content")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "size": len(content),
            "extension": file_ext,
            "mime_type": mime_type
        }
    
    def _contains_malicious_content(self, content: bytes) -> bool:
        """Check for malicious content in file"""
        # Check for executable signatures
        executable_signatures = [
            b'MZ',  # PE executable
            b'\x7fELF',  # ELF executable
            b'\xfe\xed\xfa',  # Mach-O executable
        ]
        
        for signature in executable_signatures:
            if content.startswith(signature):
                return True
        
        # Check for script tags in text files
        if b'<script' in content.lower():
            return True
        
        return False

class InputSanitizer:
    """Comprehensive input sanitizer"""
    
    def __init__(self):
        self.validator = InputValidator()
        self.sql_prevention = SQLInjectionPrevention()
        self.xss_prevention = XSSPrevention()
        self.file_validator = FileUploadValidator()
    
    def sanitize_user_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize user input data"""
        sanitized_data = {}
        
        for key, value in data.items():
            try:
                # Sanitize key
                sanitized_key = self.sql_prevention.sanitize_sql_input(key)
                
                # Sanitize value based on type
                if isinstance(value, str):
                    # Basic string sanitization
                    sanitized_value = self.xss_prevention.escape_html(value)
                    sanitized_value = self.sql_prevention.sanitize_sql_input(sanitized_value)
                elif isinstance(value, dict):
                    # Recursively sanitize nested dictionaries
                    sanitized_value = self.sanitize_user_input(value)
                elif isinstance(value, list):
                    # Sanitize list items
                    sanitized_value = [
                        self.sanitize_user_input(item) if isinstance(item, dict) 
                        else self.xss_prevention.escape_html(str(item))
                        for item in value
                    ]
                else:
                    sanitized_value = value
                
                sanitized_data[sanitized_key] = sanitized_value
            
            except Exception as e:
                logger.error(f"Error sanitizing input {key}: {e}")
                sanitized_data[key] = ""  # Default to empty string for safety
        
        return sanitized_data
    
    def validate_and_sanitize(self, data: Dict[str, Any], 
                            validation_rules: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate and sanitize input according to rules"""
        sanitized_data = {}
        validation_errors = {}
        
        for field, rules in validation_rules.items():
            if field not in data:
                if rules.get('required', False):
                    validation_errors[field] = "Field is required"
                continue
            
            try:
                value = data[field]
                
                # Apply validation rules
                if rules.get('type') == 'string':
                    value = self.validator.validate_string(
                        value,
                        min_length=rules.get('min_length', 0),
                        max_length=rules.get('max_length', 1000),
                        pattern=rules.get('pattern'),
                        allow_html=rules.get('allow_html', False)
                    )
                elif rules.get('type') == 'email':
                    value = self.validator.validate_email(value)
                elif rules.get('type') == 'integer':
                    value = self.validator.validate_integer(
                        value,
                        min_value=rules.get('min_value'),
                        max_value=rules.get('max_value')
                    )
                elif rules.get('type') == 'boolean':
                    value = self.validator.validate_boolean(value)
                
                # Apply sanitization
                if isinstance(value, str):
                    value = self.xss_prevention.escape_html(value)
                    value = self.sql_prevention.sanitize_sql_input(value)
                
                sanitized_data[field] = value
            
            except ValidationError as e:
                validation_errors[field] = str(e)
            except Exception as e:
                validation_errors[field] = f"Validation error: {e}"
        
        return {
            "data": sanitized_data,
            "errors": validation_errors,
            "valid": len(validation_errors) == 0
        }

# Usage examples
def example_input_validation():
    """Example input validation usage"""
    # Create sanitizer
    sanitizer = InputSanitizer()
    
    # Validate user input
    user_data = {
        "username": "john_doe",
        "email": "john@example.com",
        "age": "25",
        "bio": "<script>alert('xss')</script>Hello World!"
    }
    
    validation_rules = {
        "username": {
            "type": "string",
            "min_length": 3,
            "max_length": 50,
            "pattern": r"^[a-zA-Z0-9_]+$"
        },
        "email": {
            "type": "email",
            "required": True
        },
        "age": {
            "type": "integer",
            "min_value": 18,
            "max_value": 100
        },
        "bio": {
            "type": "string",
            "max_length": 500,
            "allow_html": False
        }
    }
    
    result = sanitizer.validate_and_sanitize(user_data, validation_rules)
    print(f"Validation result: {result}")
    
    # File upload validation
    file_content = b"fake image content"
    file_validation = sanitizer.file_validator.validate_file(
        "test.jpg", file_content, "image/jpeg"
    )
    print(f"File validation: {file_validation}")
    
    # SQL injection prevention
    sql_input = "'; DROP TABLE users; --"
    sanitized_sql = sanitizer.sql_prevention.sanitize_sql_input(sql_input)
    print(f"SQL sanitization: {sanitized_sql}")
    
    # XSS prevention
    html_content = "<script>alert('xss')</script><p>Safe content</p>"
    sanitized_html = sanitizer.xss_prevention.sanitize_html(html_content)
    print(f"HTML sanitization: {sanitized_html}")
```

## TL;DR Runbook

### Quick Start

```python
# 1. Authentication
password_manager = PasswordManager("argon2")
token_manager = TokenManager("your-secret-key")
auth_service = AuthenticationService(password_manager, token_manager, mfa_manager, rate_limiter)

# 2. Input validation
sanitizer = InputSanitizer()
result = sanitizer.validate_and_sanitize(user_data, validation_rules)

# 3. MFA setup
mfa_manager = MFAManager()
secret = mfa_manager.generate_mfa_secret("user_id")
qr_code = mfa_manager.generate_mfa_qr_code("user_id", secret)

# 4. Rate limiting
rate_limiter = RateLimiter(max_attempts=5, window_minutes=15)
is_limited = rate_limiter.is_rate_limited("192.168.1.1")

# 5. Authorization
authz_service = AuthorizationService(auth_service)
@authz_service.require_permission(Permission.READ)
def read_data(token: str):
    return "Data"
```

### Essential Patterns

```python
# Complete security setup
def setup_security():
    """Setup complete security environment"""
    
    # Authentication
    password_manager = PasswordManager("argon2")
    token_manager = TokenManager("your-secret-key")
    mfa_manager = MFAManager()
    rate_limiter = RateLimiter()
    auth_service = AuthenticationService(password_manager, token_manager, mfa_manager, rate_limiter)
    
    # Authorization
    authz_service = AuthorizationService(auth_service)
    
    # Input validation
    sanitizer = InputSanitizer()
    
    # File upload validation
    file_validator = FileUploadValidator()
    
    print("Security setup complete!")
```

---

*This guide provides the complete machinery for Python security best practices. Each pattern includes implementation examples, security strategies, and real-world usage patterns for enterprise security management.*

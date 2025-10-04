# Building a Real-time Chat Application with WebSockets and FastAPI

**Objective**: Build a production-ready real-time chat application using FastAPI, WebSockets, and modern web technologies. When you need real-time communication, when you want to understand WebSocket patterns, when you're building interactive applications—WebSocket chat becomes your weapon of choice.

Real-time chat applications require understanding both WebSocket mechanics and modern web development patterns. This tutorial shows you how to build a complete chat system with the precision of a senior full-stack engineer, covering everything from basic WebSocket handling to production deployment.

## 0) Prerequisites (Read Once, Live by Them)

### The Five Commandments

1. **Understand WebSocket fundamentals**
   - Connection lifecycle and message handling
   - Authentication and session management
   - Error handling and reconnection patterns
   - Scaling and performance considerations

2. **Master FastAPI WebSocket patterns**
   - WebSocket endpoint design and routing
   - Dependency injection and middleware
   - Background tasks and async operations
   - Testing and validation

3. **Know your frontend integration**
   - JavaScript WebSocket API usage
   - React/Vue integration patterns
   - State management and UI updates
   - Error handling and user feedback

4. **Validate everything**
   - Test WebSocket connections and message flow
   - Verify authentication and authorization
   - Check error handling and edge cases
   - Validate performance under load

5. **Plan for production**
   - Monitoring and logging for WebSocket connections
   - Backup and disaster recovery procedures
   - Security and rate limiting patterns
   - Documentation and maintenance

**Why These Principles**: WebSocket applications require understanding both real-time communication and modern web development patterns. Understanding these patterns prevents connection issues and enables scalable chat systems.

## 1) Project Setup and Dependencies

### Backend Dependencies

```python
# requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
redis==5.0.1
pydantic==2.5.0
pytest==7.4.3
pytest-asyncio==0.21.1
```

**Why Dependencies Matter**: Proper dependency management ensures compatibility and security. Understanding these patterns prevents version conflicts and enables reliable development.

### Project Structure

```
chat-app/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── message.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   └── chat.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── websocket_manager.py
│   │   └── message_service.py
│   └── utils/
│       ├── __init__.py
│       ├── auth.py
│       └── redis_client.py
├── tests/
│   ├── __init__.py
│   ├── test_auth.py
│   └── test_websocket.py
├── frontend/
│   ├── index.html
│   ├── chat.js
│   └── styles.css
└── docker-compose.yml
```

**Why Structure Matters**: Organized project structure enables maintainability and scalability. Understanding these patterns prevents code chaos and enables team collaboration.

## 2) Backend Implementation

### FastAPI Application Setup

```python
# app/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from routers import auth, chat
from services.websocket_manager import ConnectionManager

app = FastAPI(title="Real-time Chat API", version="1.0.0")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# WebSocket connection manager
manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.handle_message(client_id, data)
    except WebSocketDisconnect:
        manager.disconnect(client_id)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Why FastAPI Setup Matters**: Proper application structure enables scalable WebSocket handling. Understanding these patterns prevents connection issues and enables efficient real-time communication.

### WebSocket Connection Manager

```python
# app/services/websocket_manager.py
import json
import asyncio
from typing import Dict, List
from fastapi import WebSocket
from models.message import Message, MessageType
from services.message_service import MessageService

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.rooms: Dict[str, List[str]] = {}
        self.message_service = MessageService()

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        await self.broadcast_user_joined(client_id)

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            self._remove_from_rooms(client_id)
            asyncio.create_task(self.broadcast_user_left(client_id))

    async def handle_message(self, client_id: str, message: str):
        try:
            data = json.loads(message)
            message_type = data.get("type", "message")
            
            if message_type == "join_room":
                await self.join_room(client_id, data["room"])
            elif message_type == "leave_room":
                await self.leave_room(client_id, data["room"])
            elif message_type == "message":
                await self.send_room_message(client_id, data["room"], data["content"])
            elif message_type == "typing":
                await self.broadcast_typing(client_id, data["room"], data["is_typing"])
                
        except json.JSONDecodeError:
            await self.send_error(client_id, "Invalid message format")
        except Exception as e:
            await self.send_error(client_id, f"Error processing message: {str(e)}")

    async def join_room(self, client_id: str, room: str):
        if room not in self.rooms:
            self.rooms[room] = []
        if client_id not in self.rooms[room]:
            self.rooms[room].append(client_id)
        await self.broadcast_to_room(room, {
            "type": "user_joined",
            "user": client_id,
            "room": room
        })

    async def send_room_message(self, client_id: str, room: str, content: str):
        if room not in self.rooms or client_id not in self.rooms[room]:
            await self.send_error(client_id, "Not in room")
            return
            
        message = Message(
            sender=client_id,
            room=room,
            content=content,
            message_type=MessageType.MESSAGE
        )
        
        # Save message to database
        await self.message_service.save_message(message)
        
        # Broadcast to room
        await self.broadcast_to_room(room, {
            "type": "message",
            "sender": client_id,
            "content": content,
            "timestamp": message.timestamp.isoformat()
        })

    async def broadcast_to_room(self, room: str, message: dict):
        if room in self.rooms:
            for client_id in self.rooms[room]:
                if client_id in self.active_connections:
                    try:
                        await self.active_connections[client_id].send_text(json.dumps(message))
                    except:
                        # Remove dead connections
                        del self.active_connections[client_id]
                        self._remove_from_rooms(client_id)

    async def send_error(self, client_id: str, error: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(json.dumps({
                "type": "error",
                "message": error
            }))
```

**Why Connection Management Matters**: Proper WebSocket management prevents connection leaks and enables scalable real-time communication. Understanding these patterns prevents memory issues and enables efficient message handling.

### Data Models

```python
# app/models/message.py
from pydantic import BaseModel
from datetime import datetime
from enum import Enum
from typing import Optional

class MessageType(str, Enum):
    MESSAGE = "message"
    JOIN = "join"
    LEAVE = "leave"
    TYPING = "typing"

class Message(BaseModel):
    id: Optional[str] = None
    sender: str
    room: str
    content: str
    message_type: MessageType = MessageType.MESSAGE
    timestamp: datetime = datetime.utcnow()

class ChatRoom(BaseModel):
    name: str
    description: Optional[str] = None
    created_by: str
    created_at: datetime = datetime.utcnow()
    is_private: bool = False

class User(BaseModel):
    id: str
    username: str
    email: str
    is_online: bool = False
    last_seen: Optional[datetime] = None
```

**Why Data Models Matter**: Proper data modeling ensures type safety and validation. Understanding these patterns prevents runtime errors and enables reliable message handling.

### Authentication Router

```python
# app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
import os

router = APIRouter()

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@router.post("/register")
async def register(username: str, email: str, password: str):
    # In production, use a proper database
    hashed_password = get_password_hash(password)
    # Store user in database
    return {"message": "User created successfully"}

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Verify user credentials
    # In production, check against database
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception
```

**Why Authentication Matters**: Secure authentication prevents unauthorized access and enables user management. Understanding these patterns prevents security vulnerabilities and enables proper user identification.

## 3) Frontend Implementation

### HTML Structure

```html
<!-- frontend/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-time Chat</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <div class="chat-container">
        <div class="sidebar">
            <div class="user-info">
                <h3 id="username">Guest</h3>
                <span id="status" class="status offline">Offline</span>
            </div>
            <div class="rooms">
                <h4>Rooms</h4>
                <ul id="room-list">
                    <li><a href="#" data-room="general"># general</a></li>
                    <li><a href="#" data-room="random"># random</a></li>
                </ul>
            </div>
            <div class="online-users">
                <h4>Online Users</h4>
                <ul id="user-list"></ul>
            </div>
        </div>
        
        <div class="chat-main">
            <div class="chat-header">
                <h2 id="current-room"># general</h2>
                <div class="typing-indicator" id="typing-indicator"></div>
            </div>
            
            <div class="messages" id="messages"></div>
            
            <div class="message-input">
                <input type="text" id="message-input" placeholder="Type a message...">
                <button id="send-button">Send</button>
            </div>
        </div>
    </div>
    
    <script src="/static/chat.js"></script>
</body>
</html>
```

**Why HTML Structure Matters**: Clean HTML structure enables maintainable frontend code. Understanding these patterns prevents layout issues and enables responsive design.

### JavaScript WebSocket Client

```javascript
// frontend/chat.js
class ChatClient {
    constructor() {
        this.socket = null;
        this.currentRoom = 'general';
        this.username = 'Guest';
        this.isConnected = false;
        this.typingTimeout = null;
        
        this.initializeElements();
        this.setupEventListeners();
        this.connect();
    }
    
    initializeElements() {
        this.messagesContainer = document.getElementById('messages');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        this.roomList = document.getElementById('room-list');
        this.userList = document.getElementById('user-list');
        this.typingIndicator = document.getElementById('typing-indicator');
        this.statusElement = document.getElementById('status');
        this.usernameElement = document.getElementById('username');
    }
    
    setupEventListeners() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            } else {
                this.handleTyping();
            }
        });
        
        this.roomList.addEventListener('click', (e) => {
            if (e.target.tagName === 'A') {
                this.joinRoom(e.target.dataset.room);
            }
        });
    }
    
    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.username}`;
        
        this.socket = new WebSocket(wsUrl);
        
        this.socket.onopen = () => {
            console.log('Connected to chat server');
            this.isConnected = true;
            this.updateStatus('online');
            this.joinRoom(this.currentRoom);
        };
        
        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.socket.onclose = () => {
            console.log('Disconnected from chat server');
            this.isConnected = false;
            this.updateStatus('offline');
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.connect(), 3000);
        };
        
        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showError('Connection error. Attempting to reconnect...');
        };
    }
    
    sendMessage() {
        const message = this.messageInput.value.trim();
        if (message && this.isConnected) {
            this.socket.send(JSON.stringify({
                type: 'message',
                room: this.currentRoom,
                content: message
            }));
            this.messageInput.value = '';
        }
    }
    
    joinRoom(roomName) {
        if (this.isConnected) {
            this.socket.send(JSON.stringify({
                type: 'join_room',
                room: roomName
            }));
            this.currentRoom = roomName;
            document.getElementById('current-room').textContent = `# ${roomName}`;
            this.clearMessages();
        }
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'message':
                this.displayMessage(data);
                break;
            case 'user_joined':
                this.showSystemMessage(`${data.user} joined the room`);
                break;
            case 'user_left':
                this.showSystemMessage(`${data.user} left the room`);
                break;
            case 'typing':
                this.showTypingIndicator(data.user, data.is_typing);
                break;
            case 'error':
                this.showError(data.message);
                break;
        }
    }
    
    displayMessage(data) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message';
        
        const timestamp = new Date(data.timestamp).toLocaleTimeString();
        messageElement.innerHTML = `
            <div class="message-header">
                <span class="sender">${data.sender}</span>
                <span class="timestamp">${timestamp}</span>
            </div>
            <div class="message-content">${this.escapeHtml(data.content)}</div>
        `;
        
        this.messagesContainer.appendChild(messageElement);
        this.scrollToBottom();
    }
    
    showSystemMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'system-message';
        messageElement.textContent = message;
        this.messagesContainer.appendChild(messageElement);
        this.scrollToBottom();
    }
    
    handleTyping() {
        if (this.isConnected) {
            this.socket.send(JSON.stringify({
                type: 'typing',
                room: this.currentRoom,
                is_typing: true
            }));
            
            clearTimeout(this.typingTimeout);
            this.typingTimeout = setTimeout(() => {
                this.socket.send(JSON.stringify({
                    type: 'typing',
                    room: this.currentRoom,
                    is_typing: false
                }));
            }, 1000);
        }
    }
    
    showTypingIndicator(user, isTyping) {
        if (isTyping) {
            this.typingIndicator.textContent = `${user} is typing...`;
        } else {
            this.typingIndicator.textContent = '';
        }
    }
    
    updateStatus(status) {
        this.statusElement.textContent = status;
        this.statusElement.className = `status ${status}`;
    }
    
    showError(message) {
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.textContent = message;
        this.messagesContainer.appendChild(errorElement);
        this.scrollToBottom();
    }
    
    clearMessages() {
        this.messagesContainer.innerHTML = '';
    }
    
    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize chat when page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatClient();
});
```

**Why JavaScript Client Matters**: Proper WebSocket client implementation enables reliable real-time communication. Understanding these patterns prevents connection issues and enables responsive user interfaces.

## 4) Styling and User Experience

### CSS Styles

```css
/* frontend/styles.css */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: #f5f5f5;
    height: 100vh;
    overflow: hidden;
}

.chat-container {
    display: flex;
    height: 100vh;
}

.sidebar {
    width: 250px;
    background-color: #2c3e50;
    color: white;
    padding: 20px;
    overflow-y: auto;
}

.user-info {
    margin-bottom: 20px;
    padding-bottom: 20px;
    border-bottom: 1px solid #34495e;
}

.status {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-left: 10px;
}

.status.online {
    background-color: #27ae60;
}

.status.offline {
    background-color: #e74c3c;
}

.rooms h4, .online-users h4 {
    margin-bottom: 10px;
    color: #bdc3c7;
}

.rooms ul, .online-users ul {
    list-style: none;
}

.rooms a {
    color: #ecf0f1;
    text-decoration: none;
    display: block;
    padding: 5px 0;
    transition: color 0.3s;
}

.rooms a:hover {
    color: #3498db;
}

.chat-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    background-color: white;
}

.chat-header {
    background-color: #34495e;
    color: white;
    padding: 15px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.typing-indicator {
    font-style: italic;
    color: #bdc3c7;
}

.messages {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    background-color: #f8f9fa;
}

.message {
    margin-bottom: 15px;
    padding: 10px;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.message-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 5px;
}

.sender {
    font-weight: bold;
    color: #2c3e50;
}

.timestamp {
    color: #7f8c8d;
    font-size: 0.8em;
}

.system-message {
    text-align: center;
    color: #7f8c8d;
    font-style: italic;
    margin: 10px 0;
}

.error-message {
    background-color: #e74c3c;
    color: white;
    padding: 10px;
    border-radius: 4px;
    margin: 10px 0;
}

.message-input {
    display: flex;
    padding: 20px;
    background-color: white;
    border-top: 1px solid #ecf0f1;
}

.message-input input {
    flex: 1;
    padding: 10px;
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    margin-right: 10px;
}

.message-input button {
    padding: 10px 20px;
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.message-input button:hover {
    background-color: #2980b9;
}

/* Responsive design */
@media (max-width: 768px) {
    .sidebar {
        width: 200px;
    }
    
    .chat-main {
        flex: 1;
    }
}
```

**Why Styling Matters**: Professional styling enhances user experience and usability. Understanding these patterns prevents layout issues and enables responsive design.

## 5) Testing and Validation

### WebSocket Testing

```python
# tests/test_websocket.py
import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_websocket_connection():
    with client.websocket_connect("/ws/testuser") as websocket:
        # Test connection
        assert websocket is not None
        
        # Test message sending
        test_message = {
            "type": "message",
            "room": "general",
            "content": "Hello, world!"
        }
        websocket.send_text(json.dumps(test_message))
        
        # Test message receiving
        data = websocket.receive_text()
        message = json.loads(data)
        assert message["type"] == "message"
        assert message["content"] == "Hello, world!"

@pytest.mark.asyncio
async def test_room_joining():
    with client.websocket_connect("/ws/testuser") as websocket:
        # Join room
        join_message = {
            "type": "join_room",
            "room": "test_room"
        }
        websocket.send_text(json.dumps(join_message))
        
        # Verify room join confirmation
        data = websocket.receive_text()
        message = json.loads(data)
        assert message["type"] == "user_joined"
        assert message["room"] == "test_room"

@pytest.mark.asyncio
async def test_typing_indicator():
    with client.websocket_connect("/ws/testuser") as websocket:
        # Send typing indicator
        typing_message = {
            "type": "typing",
            "room": "general",
            "is_typing": True
        }
        websocket.send_text(json.dumps(typing_message))
        
        # Verify typing indicator
        data = websocket.receive_text()
        message = json.loads(data)
        assert message["type"] == "typing"
        assert message["is_typing"] == True
```

**Why Testing Matters**: Comprehensive testing prevents bugs and ensures reliability. Understanding these patterns prevents production issues and enables confident deployment.

## 6) Production Deployment

### Docker Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  chat-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - SECRET_KEY=your-secret-key-here
    depends_on:
      - redis
    volumes:
      - ./app:/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - chat-app

volumes:
  redis_data:
```

**Why Docker Matters**: Containerized deployment ensures consistency and scalability. Understanding these patterns prevents deployment issues and enables reliable production environments.

### Nginx Configuration

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream chat_backend {
        server chat-app:8000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://chat_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /static/ {
            alias /app/frontend/;
        }
    }
}
```

**Why Nginx Matters**: Reverse proxy configuration enables load balancing and SSL termination. Understanding these patterns prevents performance issues and enables secure production deployment.

## 7) Advanced Features

### Message Persistence

```python
# app/services/message_service.py
import redis
from typing import List
from models.message import Message
import json

class MessageService:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    async def save_message(self, message: Message):
        # Save to Redis for fast access
        key = f"room:{message.room}:messages"
        self.redis_client.lpush(key, json.dumps(message.dict()))
        
        # Keep only last 100 messages per room
        self.redis_client.ltrim(key, 0, 99)
    
    async def get_room_messages(self, room: str, limit: int = 50) -> List[Message]:
        key = f"room:{room}:messages"
        messages_data = self.redis_client.lrange(key, 0, limit - 1)
        return [Message(**json.loads(msg)) for msg in messages_data]
```

**Why Message Persistence Matters**: Storing messages enables chat history and offline message delivery. Understanding these patterns prevents data loss and enables better user experience.

### User Presence

```python
# app/services/presence_service.py
import redis
from datetime import datetime, timedelta
from typing import Set

class PresenceService:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    async def update_user_presence(self, user_id: str, room: str):
        key = f"presence:{room}"
        self.redis_client.sadd(key, user_id)
        self.redis_client.expire(key, 300)  # 5 minutes
    
    async def get_online_users(self, room: str) -> Set[str]:
        key = f"presence:{room}"
        return self.redis_client.smembers(key)
    
    async def remove_user_presence(self, user_id: str, room: str):
        key = f"presence:{room}"
        self.redis_client.srem(key, user_id)
```

**Why User Presence Matters**: Real-time presence information enhances user experience and enables better room management. Understanding these patterns prevents stale presence data and enables accurate user status.

## 8) TL;DR Runbook

### Essential Commands

```bash
# Development setup
pip install -r requirements.txt
uvicorn app.main:app --reload

# Testing
pytest tests/

# Production deployment
docker-compose up -d

# Monitoring
docker-compose logs -f chat-app
```

### Essential Patterns

```python
# Essential WebSocket patterns
websocket_patterns = {
    "connection": "Accept connections and manage client state",
    "messaging": "Handle real-time message broadcasting",
    "rooms": "Implement room-based communication",
    "presence": "Track user online/offline status",
    "typing": "Show typing indicators",
    "error_handling": "Graceful error handling and reconnection"
}
```

### Quick Reference

```javascript
// Essential WebSocket client patterns
const websocket = new WebSocket('ws://localhost:8000/ws/username');
websocket.onopen = () => console.log('Connected');
websocket.onmessage = (event) => handleMessage(JSON.parse(event.data));
websocket.onclose = () => console.log('Disconnected');
websocket.send(JSON.stringify({type: 'message', content: 'Hello'}));
```

**Why This Runbook**: These patterns cover 90% of WebSocket chat application needs. Master these before exploring advanced features.

## 9) The Machine's Summary

WebSocket chat applications require understanding both real-time communication and modern web development patterns. When used correctly, WebSocket chat enables seamless real-time communication, maintains connection reliability, and prevents integration issues. The key is understanding connection management, mastering message handling, and following production deployment best practices.

**The Dark Truth**: Without proper WebSocket understanding, your real-time application is fragile and unreliable. WebSocket chat is your weapon. Use it wisely.

**The Machine's Mantra**: "In connections we trust, in messages we coordinate, and in the real-time we find the path to seamless communication."

**Why This Matters**: WebSocket chat enables efficient real-time communication that can handle complex user interactions, maintain connection reliability, and provide responsive user experiences while ensuring scalability and security.

---

*This tutorial provides the complete machinery for WebSocket chat applications. The patterns scale from simple messaging to complex real-time systems, from basic WebSocket handling to advanced production deployment.*

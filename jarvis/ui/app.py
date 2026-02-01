"""
Jarvis Web UI using FastAPI
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import asyncio

# Import will fail if UI deps not installed - that's expected
try:
    from ..assistant import Jarvis, list_personas
except ImportError:
    Jarvis = None
    list_personas = lambda: ["default"]


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="Jarvis", description="Personal AI Assistant")

    # Store active connections and jarvis instances
    connections: dict[str, WebSocket] = {}
    instances: dict[str, Jarvis] = {}

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the main UI."""
        return get_html_template()

    @app.get("/api/personas")
    async def get_personas():
        """Get available personas."""
        return {"personas": list_personas()}

    @app.get("/api/health")
    async def health():
        """Health check."""
        return {"status": "ok"}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time chat."""
        await websocket.accept()
        session_id = str(id(websocket))
        connections[session_id] = websocket

        # Create Jarvis instance for this session
        try:
            jarvis = Jarvis()
            instances[session_id] = jarvis
            await websocket.send_json({
                "type": "connected",
                "persona": jarvis.current_persona,
                "model": jarvis.ollama.default_model
            })
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to initialize: {e}"
            })
            await websocket.close()
            return

        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type", "message")

                if msg_type == "message":
                    user_input = data.get("content", "")
                    if user_input:
                        # Process in background to allow streaming
                        await process_message(websocket, jarvis, user_input)

                elif msg_type == "persona":
                    persona_name = data.get("name", "default")
                    result = jarvis.switch_persona(persona_name)
                    await websocket.send_json({
                        "type": "persona_changed",
                        "persona": jarvis.current_persona,
                        "message": result
                    })

                elif msg_type == "clear":
                    jarvis.context.clear()
                    await websocket.send_json({
                        "type": "cleared"
                    })

        except WebSocketDisconnect:
            pass
        finally:
            connections.pop(session_id, None)
            instances.pop(session_id, None)

    async def process_message(websocket: WebSocket, jarvis: Jarvis, user_input: str):
        """Process a message and stream the response."""
        # Send acknowledgment
        await websocket.send_json({
            "type": "processing",
            "content": user_input
        })

        # Check for commands
        if user_input.startswith('/'):
            result = jarvis._handle_command(user_input)
            await websocket.send_json({
                "type": "response",
                "content": result or "Command executed.",
                "done": True
            })
            return

        # Add to context
        jarvis.context.add_message("user", user_input)

        # Generate response (simplified - not streaming in UI for now)
        try:
            messages = jarvis.context.get_messages()
            response_text = ""

            # Use non-streaming for simplicity in WebSocket
            response = jarvis.ollama.chat(
                messages=messages,
                system=jarvis.system_prompt,
                stream=False
            )

            if isinstance(response, str):
                response_text = response
            else:
                # Collect streamed response
                for chunk in response:
                    response_text += chunk
                    await websocket.send_json({
                        "type": "response",
                        "content": chunk,
                        "done": False
                    })

            jarvis.context.add_message("assistant", response_text)

            await websocket.send_json({
                "type": "response",
                "content": response_text,
                "done": True
            })

        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })

    return app


def get_html_template() -> str:
    """Return the HTML template for the UI."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jarvis</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background: #16213e;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #0f3460;
        }
        header h1 {
            font-size: 1.5rem;
            color: #e94560;
        }
        .status {
            font-size: 0.875rem;
            color: #888;
        }
        .status.connected {
            color: #4ade80;
        }
        main {
            flex: 1;
            overflow-y: auto;
            padding: 1rem 2rem;
        }
        .messages {
            max-width: 900px;
            margin: 0 auto;
        }
        .message {
            margin-bottom: 1rem;
            padding: 1rem;
            border-radius: 8px;
            max-width: 80%;
        }
        .message.user {
            background: #0f3460;
            margin-left: auto;
        }
        .message.assistant {
            background: #16213e;
            border: 1px solid #0f3460;
        }
        .message .role {
            font-size: 0.75rem;
            color: #888;
            margin-bottom: 0.5rem;
        }
        .message .content {
            line-height: 1.6;
            white-space: pre-wrap;
        }
        .message.assistant .content code {
            background: #0f3460;
            padding: 0.125rem 0.375rem;
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, monospace;
        }
        footer {
            background: #16213e;
            padding: 1rem 2rem;
            border-top: 1px solid #0f3460;
        }
        .input-container {
            max-width: 900px;
            margin: 0 auto;
            display: flex;
            gap: 1rem;
        }
        input[type="text"] {
            flex: 1;
            padding: 0.75rem 1rem;
            border: 1px solid #0f3460;
            border-radius: 8px;
            background: #1a1a2e;
            color: #eee;
            font-size: 1rem;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #e94560;
        }
        button {
            padding: 0.75rem 1.5rem;
            background: #e94560;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
        }
        button:hover {
            background: #d63850;
        }
        button:disabled {
            background: #555;
            cursor: not-allowed;
        }
        .controls {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }
        .controls select {
            padding: 0.5rem;
            background: #1a1a2e;
            color: #eee;
            border: 1px solid #0f3460;
            border-radius: 4px;
        }
        .typing {
            display: inline-block;
            padding: 0.5rem 1rem;
            background: #16213e;
            border-radius: 8px;
            color: #888;
        }
        .typing::after {
            content: '...';
            animation: dots 1.5s infinite;
        }
        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60%, 100% { content: '...'; }
        }
    </style>
</head>
<body>
    <header>
        <h1>🤖 Jarvis</h1>
        <div class="status" id="status">Connecting...</div>
    </header>
    <main>
        <div class="messages" id="messages"></div>
    </main>
    <footer>
        <div class="input-container">
            <input type="text" id="input" placeholder="Type a message..." autofocus>
            <button id="send" disabled>Send</button>
        </div>
        <div class="input-container controls">
            <select id="persona">
                <option value="default">Default Persona</option>
            </select>
            <button id="clear" style="background:#333">Clear</button>
        </div>
    </footer>

    <script>
        const messagesEl = document.getElementById('messages');
        const inputEl = document.getElementById('input');
        const sendBtn = document.getElementById('send');
        const statusEl = document.getElementById('status');
        const personaEl = document.getElementById('persona');
        const clearBtn = document.getElementById('clear');

        let ws = null;
        let currentPersona = 'default';

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                statusEl.textContent = 'Connected';
                statusEl.classList.add('connected');
                sendBtn.disabled = false;
            };

            ws.onclose = () => {
                statusEl.textContent = 'Disconnected';
                statusEl.classList.remove('connected');
                sendBtn.disabled = true;
                setTimeout(connect, 2000);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
        }

        function handleMessage(data) {
            switch (data.type) {
                case 'connected':
                    currentPersona = data.persona;
                    personaEl.value = data.persona;
                    addSystemMessage(`Connected. Persona: ${data.persona}, Model: ${data.model}`);
                    loadPersonas();
                    break;

                case 'processing':
                    addMessage('user', data.content);
                    showTyping();
                    break;

                case 'response':
                    hideTyping();
                    if (data.done) {
                        addMessage('assistant', data.content);
                    }
                    break;

                case 'error':
                    hideTyping();
                    addSystemMessage(`Error: ${data.message}`);
                    break;

                case 'persona_changed':
                    addSystemMessage(data.message);
                    break;

                case 'cleared':
                    messagesEl.innerHTML = '';
                    addSystemMessage('Conversation cleared.');
                    break;
            }
        }

        function addMessage(role, content) {
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = `
                <div class="role">${role === 'user' ? 'You' : 'Jarvis'}</div>
                <div class="content">${escapeHtml(content)}</div>
            `;
            messagesEl.appendChild(div);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }

        function addSystemMessage(content) {
            const div = document.createElement('div');
            div.style.cssText = 'text-align:center;color:#888;font-size:0.875rem;margin:1rem 0;';
            div.textContent = content;
            messagesEl.appendChild(div);
        }

        function showTyping() {
            const existing = document.getElementById('typing');
            if (!existing) {
                const div = document.createElement('div');
                div.id = 'typing';
                div.className = 'typing';
                div.textContent = 'Jarvis is thinking';
                messagesEl.appendChild(div);
                messagesEl.scrollTop = messagesEl.scrollHeight;
            }
        }

        function hideTyping() {
            const typing = document.getElementById('typing');
            if (typing) typing.remove();
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function loadPersonas() {
            try {
                const res = await fetch('/api/personas');
                const data = await res.json();
                personaEl.innerHTML = data.personas.map(p =>
                    `<option value="${p}" ${p === currentPersona ? 'selected' : ''}>${p}</option>`
                ).join('');
            } catch (e) {
                console.error('Failed to load personas:', e);
            }
        }

        function send() {
            const text = inputEl.value.trim();
            if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

            ws.send(JSON.stringify({ type: 'message', content: text }));
            inputEl.value = '';
        }

        inputEl.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') send();
        });

        sendBtn.addEventListener('click', send);

        personaEl.addEventListener('change', () => {
            ws.send(JSON.stringify({ type: 'persona', name: personaEl.value }));
        });

        clearBtn.addEventListener('click', () => {
            ws.send(JSON.stringify({ type: 'clear' }));
        });

        connect();
    </script>
</body>
</html>'''

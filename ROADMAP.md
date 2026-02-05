# Jarvis Roadmap

## Vision
Transform Jarvis into a fully autonomous AI life management platform with:
- **Multi-model orchestration** - Auto-select best model per task with handover documents
- **Multi-user support** - Customizable for any user, not hardcoded
- **Enterprise security** - Authentication, encryption, audit logs
- **Real integrations** - Calendar, Email, Telegram, WhatsApp, IoT

---

## Phase 1: Foundation Fixes (Today - 2026-02-04)

### Critical Bugs
- [x] Fix nested button hydration error (Orb inside button in mobile nav)
- [ ] Fix chat scroll issue (can't scroll up)
- [ ] Fix weather/status 404 errors (add backend endpoints)
- [ ] Fix STT Chutes 404 error
- [ ] Remove fake data (711 calories, etc.)

### UI Alignment
- [ ] Align desktop UI with mobile (same feel)
- [ ] Fix responsive issues

### Backend
- [ ] Add `/api/weather` endpoint (free weather API)
- [ ] Add `/api/status` endpoint (system stats)
- [ ] Add task storage in SQLite (not exposed to git)

### Integrations (Today)
- [ ] Telegram bot integration

---

## Phase 2: Autonomous Model Orchestration

### Model Router
- [ ] Task classification system (code, research, creative, analysis)
- [ ] Model capability mapping (speed, cost, quality, context length)
- [ ] Auto-select optimal model per task
- [ ] Handover document generation between models
- [ ] Result aggregation and return to initiator

### Model Pool
- [ ] Chutes models (Kimi K2.5, DeepSeek, Qwen, etc.)
- [ ] Ollama local models
- [ ] OpenAI (optional)
- [ ] Anthropic (optional)

---

## Phase 3: Multi-User Support

### Authentication
- [ ] User registration/login
- [ ] JWT token authentication
- [ ] Role-based access control
- [ ] Session management

### User Profiles
- [ ] Per-user settings storage
- [ ] Per-user conversation history
- [ ] Per-user knowledge base
- [ ] Per-user integrations

### Security
- [ ] API key encryption at rest
- [ ] Rate limiting
- [ ] Audit logging
- [ ] Input sanitization
- [ ] HTTPS enforcement

---

## Phase 4: Real Integrations

### Calendar
- [ ] Device calendar sync (CalDAV)
- [ ] Google Calendar API
- [ ] Apple Calendar integration
- [ ] Event creation/modification via chat

### Email
- [ ] IMAP/SMTP connection
- [ ] Gmail API integration
- [ ] Email summarization
- [ ] Draft/send via chat

### Messaging
- [ ] Telegram bot (Phase 1)
- [ ] WhatsApp Business API
- [ ] iMessage (via Shortcuts/AppleScript on Mac)

### IoT / Smart Home
- [ ] Home Assistant integration
- [ ] MQTT broker support
- [ ] Device control via chat

### Health & Fitness
- [ ] Apple Health integration (via Shortcuts)
- [ ] Workout tracking
- [ ] Health metrics dashboard

---

## Phase 5: Task Management

### Task System
- [ ] SQLite task storage
- [ ] Task CRUD API
- [ ] Task categories and priorities
- [ ] Due dates and reminders
- [ ] Task completion tracking

### AI Task Features
- [ ] Auto-task extraction from conversations
- [ ] Task suggestions based on context
- [ ] Deadline reminders via notifications

---

## Progress Tracking

### 2026-02-04
- [x] Created roadmap
- [x] Fixed nested button hydration error (Orb `as` prop)
- [x] Added `/api/weather` endpoint (Open-Meteo - free, no key)
- [x] Added `/api/status` endpoint (psutil system stats)
- [x] Fixed WeatherWidget to use real API data
- [x] Fixed SystemStatusWidget to use real API data
- [x] Fixed chat scroll issue in mobile view
- [x] Removed fake 711 calories data
- [x] Aligned desktop UI with mobile (cleaner, less cluttered)
- [x] Created Telegram bot integration (`jarvis/integrations/telegram_bot.py`)
- [x] Added `jarvis telegram run` CLI command
- [x] Added `jarvis telegram status` CLI command
- [x] Added `jarvis telegram setup` interactive wizard
- [x] Added `jarvis telegram webhook` for production (no separate service needed)
- [x] Added `jarvis telegram send` to send messages from CLI
- [x] Added Telegram API endpoints (`/api/integrations/telegram/*`)
- [x] Full Telegram command suite (20+ commands):
  - /model, /models, /provider, /providers
  - /clear, /export, /mode
  - /search, /knowledge, /remember, /facts
  - /status, /weather, /time, /settings
  - /persona, /personas, /name, /id
- [x] Persistent Telegram conversations (SQLite)
- [x] Per-user conversation history
- [x] Clean responses (stripped Rich markup)
- [x] Model awareness (won't claim to be GPT-4)
- [x] Configurable assistant name (not hardcoded "Jarvis")
- [x] Updated README with Telegram docs
- [x] Updated requirements.txt with all dependencies
- [ ] Add Telegram config to web settings panel

---

## Technical Decisions

| Feature | Choice | Reason |
|---------|--------|--------|
| Task Storage | SQLite | Simple, local, no external deps |
| Weather API | Open-Meteo | Free, no API key required |
| Telegram | python-telegram-bot | Mature, async support |
| Auth | JWT + bcrypt | Industry standard |
| Multi-model | Custom router | Full control over handoffs |

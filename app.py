# FastAPI → Main framework.
# Request, Header, HTTPException → Used for request metadata and error handling.
# StaticFiles → Lets you serve index.html, CSS, JS.
# uuid, time → Used for session IDs and timestamps.

import time
import uuid
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dateutil import parser

# ------------------------------
# Config & simple dependencies
# Auth token → Simple bearer token for API calls (replace with SSO in production).
# Tenants → Multi-tenant support (different orgs like Acme, Globex).
# ------------------------------
API_AUTH_TOKEN = "changeme-token"  # Replace in env in production env
TENANTS = {"Tenant1", "Tenant2"} 

# Creates the FastAPI app.
# Enables CORS so the frontend can talk to backend.
app = FastAPI(title="Chatbot MVP", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# In-memory stores (session, audit, KB)
# Sessions → Keeps track of ongoing conversations.
# Audit logs → Immutable record of interactions.
# Prompt registry → Versioned system/policy prompts (for governance).
# ------------------------------
SESSIONS: Dict[str, Dict[str, Any]] = {}  # session_id -> state
AUDIT_LOGS: List[Dict[str, Any]] = []     # append-only
np.random.seed(0)

# KNowledge Base (simple in memory vector store) RAG
# Stores documents with embeddings.
# Lets you query by text and retrieve the most similar entries.
# Acts as a lightweight RAG backend without external libraries like FAISS or Pinecone
# Embeddings → Fake vectors generated with numpy (replace with real embeddings later).
# Tenant isolation → Each tenant has its own KB docs.
# Query → Returns top-k matching docs with similarity scores.
class KBIndex:
    def __init__(self):
        self.docs: List[Dict[str, Any]] = []  # {id, text, emb}
    def _embed(self, text: str) -> np.ndarray:
        # Dummy embedding: hash-based deterministic vector
        h = abs(hash(text))
        rng = np.random.RandomState(h % (2**32))
        v = rng.rand(384)
        return v / (np.linalg.norm(v) + 1e-9)
    def add(self, doc_id: str, text: str):
        self.docs.append({"id": doc_id, "text": text, "emb": self._embed(text)})
    def query(self, text: str, top_k: int = 3):
        q = self._embed(text)
        sims = []
        for d in self.docs:
            sims.append((d, float(np.dot(q, d["emb"]))))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [{"id": s[0]["id"], "text": s[0]["text"], "score": s[1]} for s in sims[:top_k]]

#KNowledge base population(RAG)
KB = KBIndex()
KB.add("faq_1", "You can enroll in a course by providing your user ID and the course ID.")
KB.add("faq_2", "To check progress, ask for 'progress' with your user ID. We return completed, total, and percent.")
KB.add("policy_1", "We do not process full PII. Please avoid sharing emails or phone numbers. Use your user ID only.")

# ------------------------------
# Schemas
# Defines the input (user message) and output (bot reply) formats.

# ------------------------------
class ChatMessage(BaseModel):
    session_id: Optional[str] = None
    text: str
    channel: Optional[str] = "web"
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    sources: Optional[List[Dict[str, Any]]] = None
    action: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

# ------------------------------
# Utilities - Auth and Tenant Resolution
# Checks for a valid Authorization: Bearer ... header.
# If missing or incorrect, returns 401 or 403.
# Used in /chat and /audit to protect sensitive endpoints.

# ------------------------------
def require_auth(authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.replace("Bearer ", "").strip()
    if token != API_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


# PII redaction → Masks emails/phone numbers.
# Session management → Creates/reuses session IDs.
# Audit logging → Appends events to audit log.
def redact_pii(text: str) -> str:
    # Very simple redaction: mask emails and 10+ digit numbers
    import re
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted-email]", text)
    text = re.sub(r"\b\d{10,}\b", "[redacted-number]", text)
    return text

def now_ms() -> int:
    return int(time.time() * 1000)

# ------------------------------
# NLP: intent & entities
# Intent detection → Rule-based (keywords like “enroll”, “progress”).
# Entity extraction → Finds dates, course IDs, user IDs.
# ------------------------------
INTENT_CUES = {
    "enroll course": ["enroll", "register", "sign up", "add course"],
    "progress check": ["progress", "status", "completion", "how much done"],
    "faq": ["what", "how", "policy", "help", "info", "guide", "faq"],
}

def detect_intent(text: str) -> Dict[str, Any]:
    t = text.lower()
    best_intent, best_score = "faq", 0.3
    for intent, cues in INTENT_CUES.items():
        score = sum(1 for c in cues if c in t) / max(len(cues), 1)
        if score > best_score:
            best_intent, best_score = intent, score
    return {"intent": best_intent, "confidence": round(best_score, 3)}

def extract_entities(text: str) -> Dict[str, Any]:
    # Extract date
    date_iso = None
    try:
        dt = parser.parse(text, fuzzy=True)
        date_iso = dt.date().isoformat()
    except Exception:
        pass
    # Extract course ID (pattern: COURSE-123 or plain numeric)
    import re
    course_id = None
    m = re.search(r"(COURSE-\d+|\b\d{3,6}\b)", text)
    if m:
        course_id = m.group(1)
    # Extract user ID (pattern: USER-xxx or u123)
    user_id = None
    m2 = re.search(r"(USER-[A-Za-z0-9]+|u\d+)", text)
    if m2:
        user_id = m2.group(1)
    return {"date": date_iso, "course_id": course_id, "user_id": user_id}

# ------------------------------
# Tools (LMS simulators)
# Simulated LMS tools: 
# Enroll course → Confirms enrollment.
# Fetch progress → Returns fake progress numbers.
# ------------------------------
def tool_enroll_course(user_id: str, course_id: str) -> Dict[str, Any]:
    # Simulate enrollment
    return {"status": "enrolled", "userId": user_id, "courseId": course_id, "timestamp": now_ms()}

def tool_fetch_progress(user_id: str) -> Dict[str, Any]:
    # Simulate progress
    completed = 7
    total = 10
    percent = int(100 * completed / total)
    return {"userId": user_id, "completed": completed, "total": total, "percent": percent}

# ------------------------------
# Dialog policy
# Decides what to do next: 
# Clarify
# Collect missing info
# Call tool
# RAG (knowledge base)
# Handoff to human
# ------------------------------
def next_action(state: Dict[str, Any]) -> Dict[str, Any]:
    intent = state["intent"]
    conf = state["intent_conf"]
    ents = state["entities"]
    if conf < 0.55:
        return {"action": "clarify", "prompt": "Do you want to enroll in a course or check progress?"}
    if intent == "faq":
        return {"action": "rag"}
    if intent == "enroll course":
        if ents.get("user_id") and ents.get("course_id"):
            return {"action": "call_tool", "tool": "enroll_course"}
        return {"action": "collect", "missing": ["user_id", "course_id"]}
    if intent == "progress check":
        if ents.get("user_id"):
            return {"action": "call_tool", "tool": "fetch_progress"}
        return {"action": "collect", "missing": ["user_id"]}
    return {"action": "rag"}

# ------------------------------
# Response generator (LLM adapter stub)
# Generates final reply text.
# Adds sources if RAG was used.
# Formats tool results (enrollment/progress).
# ------------------------------
def compose_reply(state: Dict[str, Any], result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    action = state["action"]
    intent = state["intent"]
    ents = state["entities"]
    sources = None
    if action == "clarify":
        return {"reply": "I can help with enrollment or progress. Which one should we do?", "sources": None}
    if action == "collect":
        missing = state["missing"]
        ask = []
        if "user_id" in missing and not ents.get("user_id"):
            ask.append("your user ID (e.g., USER-42)")
        if "course_id" in missing and not ents.get("course_id"):
            ask.append("the course ID (e.g., COURSE-101)")
        return {"reply": f"To proceed, please provide {', '.join(ask)}.", "sources": None}
    if action == "rag":
        q = state["text"]
        hits = KB.query(q, top_k=3)
        sources = [{"id": h["id"], "text": h["text"]} for h in hits]
        answer = f"Here’s what I found:\n- " + "\n- ".join(h["text"] for h in hits)
        return {"reply": answer, "sources": sources}
    if action == "call_tool":
        if state["tool"] == "enroll_course":
            r = result or {}
            msg = f"Enrollment confirmed for {r.get('userId')} in {r.get('courseId')}."
            return {"reply": msg, "sources": [{"id": "faq_1", "text": "Enrollment requires user and course IDs."}]}
        if state["tool"] == "fetch_progress":
            r = result or {}
            msg = f"Your progress: {r.get('completed')}/{r.get('total')} ({r.get('percent')}%)."
            return {"reply": msg, "sources": [{"id": "faq_2", "text": "Progress returns completed, total, percent."}]}
    return {"reply": "I’m not sure yet—try asking about enrollment or progress.", "sources": None}

# ------------------------------
# Session helpers & audit
# Stores conversations in memory (dictionary).
# If a session ID is provided and exists → reuse it.
# Otherwise → create a new UUID and start a new session.

# ------------------------------
def get_or_create_session(session_id: Optional[str]) -> str:
    if session_id and session_id in SESSIONS:
        return session_id
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"created": now_ms(), "turns": []}
    return sid

def audit(event: Dict[str, Any]):
    AUDIT_LOGS.append(event)

# ------------------------------
# API endpoints
# ------------------------------

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Chatbot service is live on TrueBot Platform 🎉"}

@app.get("/chat")
def chat_get():
    return {"message": "Use POST /chat to send messages. Welcome to TruBot Platforms"}

# Accepts POST requests with JSON body like:
@app.post("/chat", response_model=ChatResponse)
async def chat(msg: ChatMessage, request: Request, authorization: str = Header(None)):
    require_auth(authorization)
    # Minimal rate-limit placeholder
    # (In production, tie to IP/user and sliding window)
    # Validate and normalize
    text_raw = msg.text.strip()
    text = redact_pii(text_raw)
    session_id = get_or_create_session(msg.session_id)

    # NLP
    intent = detect_intent(text)
    entities = extract_entities(text)

    # State for policy
    state = {
        "text": text,
        "intent": intent["intent"],
        "intent_conf": intent["confidence"],
        "entities": entities,
        "user_id": msg.user_id or entities.get("user_id"),
        "channel": msg.channel or "web",
    }

    policy = next_action(state)
    state.update(policy)

    # Tooling
    tool_result = None
    if policy["action"] == "call_tool":
        if policy["tool"] == "enroll_course":
            uid = state["user_id"] or entities.get("user_id")
            cid = entities.get("course_id")
            if not (uid and cid):
                state["action"] = "collect"
                state["missing"] = ["user_id", "course_id"]
            else:
                tool_result = tool_enroll_course(uid, cid)
        elif policy["tool"] == "fetch_progress":
            uid = state["user_id"] or entities.get("user_id")
            if not uid:
                state["action"] = "collect"
                state["missing"] = ["user_id"]
            else:
                tool_result = tool_fetch_progress(uid)

    # Compose
    reply_pack = compose_reply(state, tool_result)

    # Persist turn
    SESSIONS[session_id]["turns"].append({
        "ts": now_ms(),
        "user_text": text_raw,
        "redacted_text": text,
        "intent": intent,
        "entities": entities,
        "policy": policy,
        "tool_result": tool_result,
        "reply": reply_pack["reply"],
        "sources": reply_pack.get("sources"),
    })

    # Audit
    audit({
        "ts": now_ms(),
        "session_id": session_id,
        "ip": request.client.host,
        "channel": msg.channel,
        "intent": intent,
        "action": policy["action"],
        "tool": policy.get("tool"),
    })

    return ChatResponse(
        session_id=session_id,
        reply=reply_pack["reply"],
        sources=reply_pack.get("sources"),
        action=policy.get("action"),
        metrics={"intent_confidence": intent["confidence"]}
    )

@app.get("/session/{session_id}")
async def session_dump(session_id: str, authorization: str = Header(None)):
    require_auth(authorization)
    return SESSIONS.get(session_id, {})

@app.get("/audit")
async def audit_dump(authorization: str = Header(None)):
    require_auth(authorization)
    return {"count": len(AUDIT_LOGS), "events": AUDIT_LOGS[-50:]}

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Chatbot service is live 🎉"}

#Mount the static folder in FastAPI

app.mount("/", StaticFiles(directory="static", html=True), name="static")

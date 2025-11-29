import time, uuid, os, numpy as np, re
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dateutil import parser

# ------------------------------
# Config
# ------------------------------
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "changeme-token")  # replace with SSO in prod
TENANTS = {"acme", "globex"}  # registered tenants

# ------------------------------
# App & CORS
# ------------------------------
app = FastAPI(title="Enterprise Chatbot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # narrow per-channel in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Stores
# ------------------------------
SESSIONS: Dict[str, Dict[str, Any]] = {}              # session_id -> state
AUDIT_LOGS: List[Dict[str, Any]] = []                 # append-only audit
PROMPT_REGISTRY: Dict[str, Dict[str, Any]] = {        # versioned prompts/policies
    "system": {"v": "1", "text": "Be helpful, cite sources for RAG answers, refuse PII."},
    "policy": {"v": "1", "conf_threshold": 0.55}
}
np.random.seed(42)

# ------------------------------
# Multi-tenant KB Index
# ------------------------------
class KBIndex:
    def __init__(self):
        self.docs_by_tenant: Dict[str, List[Dict[str, Any]]] = {}

    def _embed(self, text: str) -> np.ndarray:
        # Deterministic pseudo-embedding; replace with real vectors in prod
        h = abs(hash(text))
        rng = np.random.RandomState(h % (2**32))
        v = rng.rand(256)
        return v / (np.linalg.norm(v) + 1e-9)

    def add(self, tenant: str, doc_id: str, text: str):
        self.docs_by_tenant.setdefault(tenant, [])
        self.docs_by_tenant[tenant].append({"id": doc_id, "text": text, "emb": self._embed(text)})

    def query(self, tenant: str, text: str, top_k: int = 3):
        docs = self.docs_by_tenant.get(tenant, [])
        if not docs: return []
        q = self._embed(text)
        sims = [(d, float(np.dot(q, d["emb"]))) for d in docs]
        sims.sort(key=lambda x: x[1], reverse=True)
        return [{"id": d["id"], "text": d["text"], "score": s} for d, s in sims[:top_k]]

KB = KBIndex()
# Seed tenant-specific docs
KB.add("acme", "faq_enroll", "Provide user ID and course ID to enroll in a course.")
KB.add("acme", "faq_progress", "Ask for 'progress' with your user ID to see completed/total/percent.")
KB.add("acme", "policy_pii", "Do not share emails or phone numbers; use your user ID only.")
KB.add("globex", "faq_support", "Contact Globex L&D for escalations; use internal ticketing portal.")

# ------------------------------
# Schemas
# ------------------------------
class ChatMessage(BaseModel):
    session_id: Optional[str] = None
    text: str
    channel: Optional[str] = "web"
    tenant: Optional[str] = "acme"
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    sources: Optional[List[Dict[str, Any]]] = None
    action: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    policy_version: Optional[str] = None

# ------------------------------
# Auth & tenant resolution
# ------------------------------
def require_auth(authorization: str = Header(None)) -> str:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.replace("Bearer ", "").strip()
    if token != API_AUTH_TOKEN:
        # In production: validate JWT from SSO (aud, iss, exp), map claims->tenant, user
        raise HTTPException(status_code=403, detail="Invalid token")
    return token

def resolve_tenant(tenant: Optional[str]) -> str:
    t = (tenant or "").lower()
    if t not in TENANTS:
        raise HTTPException(status_code=400, detail=f"Unknown tenant: {tenant}")
    return t

# ------------------------------
# Utilities
# ------------------------------
def redact_pii(text: str) -> str:
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted-email]", text)
    text = re.sub(r"\b(\+?\d[\d\s-]{9,})\b", "[redacted-number]", text)
    return text

def now_ms() -> int: return int(time.time() * 1000)

def get_or_create_session(session_id: Optional[str], tenant: str, user_id: Optional[str]) -> str:
    if session_id and session_id in SESSIONS:
        return session_id
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"created": now_ms(), "tenant": tenant, "user_id": user_id, "turns": []}
    return sid

def audit(event: Dict[str, Any]):
    AUDIT_LOGS.append(event)

# ------------------------------
# NLP: intent & entities
# ------------------------------
INTENT_CUES = {
    "enroll course": ["enroll", "register", "sign up", "add course"],
    "progress check": ["progress", "status", "completion", "how much done"],
    "faq": ["what", "how", "policy", "help", "info", "guide", "faq"],
    "handoff": ["agent", "human", "support", "escalate"]
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
    # Dates
    date_iso = None
    try:
        dt = parser.parse(text, fuzzy=True)
        date_iso = dt.date().isoformat()
    except Exception:
        pass
    # Course ID (COURSE-123 or numeric)
    course_id = None
    m = re.search(r"(COURSE-\d+|\b\d{3,6}\b)", text)
    if m: course_id = m.group(1)
    # User ID (USER-xyz or u123)
    user_id = None
    m2 = re.search(r"(USER-[A-Za-z0-9]+|u\d+)", text)
    if m2: user_id = m2.group(1)
    return {"date": date_iso, "course_id": course_id, "user_id": user_id}

# ------------------------------
# Tools (tenant-scoped simulators)
# ------------------------------
def tool_enroll_course(tenant: str, user_id: str, course_id: str) -> Dict[str, Any]:
    # Simulate enrollment with tenant isolation
    return {"tenant": tenant, "status": "enrolled", "userId": user_id, "courseId": course_id, "ts": now_ms()}

def tool_fetch_progress(tenant: str, user_id: str) -> Dict[str, Any]:
    # Simulate progress; vary per tenant
    completed = 7 if tenant == "acme" else 5
    total = 10
    percent = int(100 * completed / total)
    return {"tenant": tenant, "userId": user_id, "completed": completed, "total": total, "percent": percent}

# ------------------------------
# Dialog policy (versioned)
# ------------------------------
def next_action(state: Dict[str, Any]) -> Dict[str, Any]:
    conf_thr = float(PROMPT_REGISTRY["policy"]["conf_threshold"])
    intent, conf, ents = state["intent"], state["intent_conf"], state["entities"]
    if conf < conf_thr:
        return {"action": "clarify", "prompt": "Should we enroll in a course or check progress?"}
    if intent == "faq":
        return {"action": "rag"}
    if intent == "handoff":
        return {"action": "handoff", "reason": "User requested human support"}
    if intent == "enroll course":
        if (state.get("user_id") or ents.get("user_id")) and ents.get("course_id"):
            return {"action": "call_tool", "tool": "enroll_course"}
        return {"action": "collect", "missing": ["user_id", "course_id"]}
    if intent == "progress check":
        if state.get("user_id") or ents.get("user_id"):
            return {"action": "call_tool", "tool": "fetch_progress"}
        return {"action": "collect", "missing": ["user_id"]}
    return {"action": "rag"}

# ------------------------------
# Response composer (LLM adapter stub)
# ------------------------------
def compose_reply(state: Dict[str, Any], result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    action, ents, tenant = state["action"], state["entities"], state["tenant"]
    if action == "clarify":
        return {"reply": "I can help with enrollment or progress. Which one should we do?", "sources": None}
    if action == "collect":
        ask = []
        if "user_id" in state["missing"] and not (state.get("user_id") or ents.get("user_id")):
            ask.append("your user ID (e.g., USER-42)")
        if "course_id" in state["missing"] and not ents.get("course_id"):
            ask.append("the course ID (e.g., COURSE-101)")
        return {"reply": f"To proceed, please provide {', '.join(ask)}.", "sources": None}
    if action == "rag":
        hits = KB.query(tenant, state["text"], top_k=3)
        sources = [{"id": h["id"], "text": h["text"]} for h in hits]
        answer = "Here’s what I found:\n- " + "\n- ".join(h["text"] for h in hits) if hits else "No tenant docs matched your query."
        return {"reply": answer, "sources": sources}
    if action == "call_tool":
        if state["tool"] == "enroll_course":
            r = result or {}
            msg = f"[{tenant}] Enrollment confirmed for {r.get('userId')} in {r.get('courseId')}."
            return {"reply": msg, "sources": [{"id": "faq_enroll", "text": "Enrollment requires user and course IDs."}]}
        if state["tool"] == "fetch_progress":
            r = result or {}
            msg = f"[{tenant}] Your progress: {r.get('completed')}/{r.get('total')} ({r.get('percent')}%)."
            return {"reply": msg, "sources": [{"id": "faq_progress", "text": "Progress returns completed, total, percent."}]}
    if action == "handoff":
        return {"reply": "Routing you to a human agent. I’ll pass along the transcript.", "sources": None}
    return {"reply": "I’m not sure yet—try asking about enrollment or progress.", "sources": None}

# ------------------------------
# API endpoints
# ------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(msg: ChatMessage, request: Request, authorization: str = Header(None)):
    require_auth(authorization)
    tenant = resolve_tenant(msg.tenant)
    text_raw = msg.text.strip()
    text = redact_pii(text_raw)
    session_id = get_or_create_session(msg.session_id, tenant, msg.user_id)

    # NLP
    intent = detect_intent(text)
    entities = extract_entities(text)

    # State
    state = {
        "tenant": tenant,
        "text": text,
        "intent": intent["intent"],
        "intent_conf": intent["confidence"],
        "entities": entities,
        "user_id": msg.user_id or entities.get("user_id"),
        "channel": msg.channel or "web",
        "policy_version": PROMPT_REGISTRY["policy"]["v"]
    }

    policy = next_action(state)
    state.update(policy)

    # Tools
    tool_result = None
    if policy["action"] == "call_tool":
        uid = state["user_id"] or entities.get("user_id")
        if policy["tool"] == "enroll_course":
            cid = entities.get("course_id")
            if not (uid and cid):
                state["action"] = "collect"; state["missing"] = ["user_id", "course_id"]
            else:
                tool_result = tool_enroll_course(tenant, uid, cid)
        elif policy["tool"] == "fetch_progress":
            if not uid:
                state["action"] = "collect"; state["missing"] = ["user_id"]
            else:
                tool_result = tool_fetch_progress(tenant, uid)

    # Compose reply
    reply_pack = compose_reply(state, tool_result)

    # Persist turn
    SESSIONS[session_id]["turns"].append({
        "ts": now_ms(),
        "tenant": tenant,
        "user_text": text_raw,
        "redacted_text": text,
        "intent": intent,
        "entities": entities,
        "policy": policy,
        "tool_result": tool_result,
        "reply": reply_pack["reply"],
        "sources": reply_pack.get("sources"),
        "policy_v": PROMPT_REGISTRY["policy"]["v"]
    })

    # Audit
    audit({
        "ts": now_ms(),
        "session_id": session_id,
        "ip": request.client.host,
        "channel": msg.channel,
        "tenant": tenant,
        "intent": intent,
        "action": policy["action"],
        "tool": policy.get("tool"),
        "policy_v": PROMPT_REGISTRY["policy"]["v"]
    })

    return ChatResponse(
        session_id=session_id,
        reply=reply_pack["reply"],
        sources=reply_pack.get("sources"),
        action=policy.get("action"),
        metrics={"intent_confidence": intent["confidence"]},
        policy_version=PROMPT_REGISTRY["policy"]["v"]
    )

@app.get("/session/{session_id}")
async def session_dump(session_id: str, authorization: str = Header(None)):
    require_auth(authorization)
    return SESSIONS.get(session_id, {})

@app.get("/audit")
async def audit_dump(authorization: str = Header(None)):
    require_auth(authorization)
    return {"count": len(AUDIT_LOGS), "events": AUDIT_LOGS[-50:]}

# ------------------------------
# Admin: prompt/policy updates (change control)
# ------------------------------
class PolicyUpdate(BaseModel):
    conf_threshold: float

@app.post("/admin/policy")
async def update_policy(update: PolicyUpdate, authorization: str = Header(None)):
    require_auth(authorization)
    PROMPT_REGISTRY["policy"]["conf_threshold"] = update.conf_threshold
    PROMPT_REGISTRY["policy"]["v"] = str(int(PROMPT_REGISTRY["policy"]["v"]) + 1)
    return {"status": "ok", "policy": PROMPT_REGISTRY["policy"]}
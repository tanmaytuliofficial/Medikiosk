from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import sqlite3
import os
import asyncio
import csv
import io
import shutil
import uuid
import json

app = FastAPI(
    title="MediKiosk Clinical Intelligence Platform",
    version="35.0.0"
)

try:
    from groq import Groq
except ImportError:
    Groq = None

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

groq_client = (
    Groq(api_key=GROQ_API_KEY)
    if (Groq is not None and GROQ_API_KEY)
    else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medikiosk.db")
DB_PATH = DATABASE_URL.replace("sqlite:///", "", 1) if DATABASE_URL.startswith("sqlite:///") else "./medikiosk.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medical_reports")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

def get_table_columns(cursor, table_name: str):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]

def add_column_if_missing(cursor, table_name: str, column_name: str, column_definition: str):
    columns = get_table_columns(cursor, table_name)
    if column_name not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")

def init_database():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE,
            patient_name TEXT,
            age INTEGER,
            phone TEXT,
            gender TEXT,
            chief_complaint TEXT,
            symptoms TEXT,
            medical_history TEXT,
            pain_site TEXT,
            department TEXT,
            assigned_doctor_id INTEGER,
            assigned_doctor_name TEXT,
            status TEXT DEFAULT 'Waiting',
            token INTEGER UNIQUE,
            emergency INTEGER DEFAULT 0,
            doctor_notes TEXT,
            medical_report_image TEXT,
            medical_report_filename TEXT,
            ocr_text TEXT,
            ai_summary TEXT,
            clinical_information TEXT,
            conversation TEXT,
            created_at TEXT,
            updated_at TEXT,
            completed_at TEXT,
            completed_by TEXT
        )
    """)

    patient_columns = [
        ("patient_id", "TEXT"), ("patient_name", "TEXT"), ("age", "INTEGER"), ("phone", "TEXT"),
        ("gender", "TEXT"), ("chief_complaint", "TEXT"), ("symptoms", "TEXT"), ("medical_history", "TEXT"),
        ("pain_site", "TEXT"), ("department", "TEXT"), ("assigned_doctor_id", "INTEGER"),
        ("assigned_doctor_name", "TEXT"), ("status", "TEXT DEFAULT 'Waiting'"), ("token", "INTEGER"),
        ("emergency", "INTEGER DEFAULT 0"), ("doctor_notes", "TEXT"), ("medical_report_image", "TEXT"),
        ("medical_report_filename", "TEXT"), ("ocr_text", "TEXT"), ("ai_summary", "TEXT"),
        ("clinical_information", "TEXT"), ("conversation", "TEXT"), ("created_at", "TEXT"),
        ("updated_at", "TEXT"), ("completed_at", "TEXT"), ("completed_by", "TEXT"),
    ]

    for column_name, definition in patient_columns:
        add_column_if_missing(cursor, "patients", column_name, definition)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id TEXT,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            phone TEXT,
            email TEXT,
            password TEXT DEFAULT 'doctor123',
            password_hash TEXT DEFAULT 'doctor123',
            active INTEGER DEFAULT 1
        )
    """)

    doctor_columns = [
        ("doctor_id", "TEXT DEFAULT ''"), ("name", "TEXT"), ("department", "TEXT"), ("phone", "TEXT"),
        ("email", "TEXT"), ("password", "TEXT DEFAULT 'doctor123'"), ("password_hash", "TEXT DEFAULT 'doctor123'"),
        ("active", "INTEGER DEFAULT 1"),
    ]

    for column_name, definition in doctor_columns:
        add_column_if_missing(cursor, "doctors", column_name, definition)

    cursor.execute("CREATE TABLE IF NOT EXISTS admins (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)")
    add_column_if_missing(cursor, "admins", "password", "TEXT")
    cursor.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")

    cursor.execute("SELECT id FROM admins WHERE username = 'admin' LIMIT 1")
    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO admins (username, password) VALUES ('admin', 'admin123')")

    default_doctors = [
        ("DOC-001", "Dr. Rahul Sharma", "General Medicine"),
        ("DOC-002", "Dr. Priya Verma", "Orthopedics"),
        ("DOC-003", "Dr. Amit Singh", "Cardiology"),
        ("DOC-004", "Dr. Neha Gupta", "Neurology"),
        ("DOC-005", "Dr. Anjali Mehta", "Dermatology"),
        ("DOC-006", "Dr. Arjun Kapoor", "ENT"),
        ("DOC-007", "Dr. Riya Malhotra", "Pediatrics"),
    ]

    for doc_id, name, dept in default_doctors:
        cursor.execute("SELECT id FROM doctors WHERE doctor_id = ? OR name = ? LIMIT 1", (doc_id, name))
        if cursor.fetchone() is None:
            cursor.execute("INSERT INTO doctors (doctor_id, name, department, phone, email, password, password_hash, active) VALUES (?, ?, ?, '', '', 'doctor123', 'doctor123', 1)", (doc_id, name, dept))

    conn.commit()
    conn.close()

connected_clients: List[WebSocket] = []

async def broadcast(message: Dict[str, Any]):
    dead_connections = []
    for ws in list(connected_clients):
        try:
            await ws.send_json(message)
        except Exception:
            dead_connections.append(ws)
    for ws in dead_connections:
        if ws in connected_clients:
            connected_clients.remove(ws)

class ChatRequest(BaseModel):
    user_message: str
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    language: str = "en"
    patient_details: Dict[str, Any] = Field(default_factory=dict)
    pain_site: Optional[str] = None
    is_nfc_mode: bool = False
    clinical_information: Dict[str, Any] = Field(default_factory=dict)

class DoctorLoginRequest(BaseModel):
    username: str
    password: str

class AdminLoginRequest(BaseModel):
    username: str
    password: str

class DoctorNotesRequest(BaseModel):
    notes: str = ""

class AdminPatientUpdateRequest(BaseModel):
    patient_name: Optional[str] = None
    age: Optional[int] = None
    phone: Optional[str] = None
    gender: Optional[str] = None
    pain_site: Optional[str] = None
    department: Optional[str] = None
    status: Optional[str] = None
    assigned_doctor_id: Optional[int] = None
    case_type: Optional[str] = None
    chief_complaint: Optional[str] = None
    doctor_notes: Optional[str] = None

class SelectedPatientsRequest(BaseModel):
    tokens: List[int] = Field(default_factory=list)

class DoctorCreateRequest(BaseModel):
    doctor_id: Optional[str] = None
    name: str
    department: str
    phone: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = "doctor123"
    active: Optional[int] = 1

class DoctorUpdateRequest(BaseModel):
    doctor_id: Optional[str] = None
    name: Optional[str] = None
    department: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    active: Optional[int] = None

def now():
    return datetime.now().isoformat(timespec="seconds")

def safe_row_value(row, key, default=None):
    if row is None: return default
    try:
        if key in row.keys():
            val = row[key]
            return default if val is None else val
    except Exception: pass
    return default

def patient_row_to_dict(row):
    if row is None: return None
    emergency_val = safe_row_value(row, "emergency", 0)
    try: emergency_bool = bool(int(emergency_val or 0))
    except Exception: emergency_bool = bool(emergency_val)

    return {
        "id": safe_row_value(row, "id"),
        "patient_id": safe_row_value(row, "patient_id"),
        "patient_name": safe_row_value(row, "patient_name", "Walk-in Patient") or "Walk-in Patient",
        "age": safe_row_value(row, "age"),
        "phone": safe_row_value(row, "phone"),
        "contact": safe_row_value(row, "phone"),
        "gender": safe_row_value(row, "gender"),
        "chief_complaint": safe_row_value(row, "chief_complaint"),
        "symptoms": safe_row_value(row, "symptoms"),
        "medical_history": safe_row_value(row, "medical_history"),
        "pain_site": safe_row_value(row, "pain_site", "General") or "General",
        "department": safe_row_value(row, "department", "General Medicine") or "General Medicine",
        "assigned_doctor_id": safe_row_value(row, "assigned_doctor_id"),
        "assigned_doctor_name": safe_row_value(row, "assigned_doctor_name", "Duty Doctor") or "Duty Doctor",
        "status": safe_row_value(row, "status", "Waiting") or "Waiting",
        "token": safe_row_value(row, "token"),
        "emergency": emergency_bool,
        "is_emergency": emergency_bool,
        "is_red_flag": emergency_bool,
        "doctor_notes": safe_row_value(row, "doctor_notes", "") or "",
        "created_at": safe_row_value(row, "created_at"),
        "updated_at": safe_row_value(row, "updated_at"),
    }

EMERGENCY_PATTERNS = ["severe chest pain", "crushing chest pain", "difficulty breathing", "cannot breathe", "can't breathe", "unconscious", "fainted", "severe bleeding", "stroke", "seizure"]

def detect_emergency(text: str):
    return any(p in (text or "").lower().strip() for p in EMERGENCY_PATTERNS)

def determine_department(text: str, pain_site: Optional[str] = None):
    text_lower = (text or "").lower()
    if detect_emergency(text_lower): return "Emergency"
    if any(k in text_lower for k in ["chest pain", "heart", "bp"]): return "Cardiology"
    if any(k in text_lower for k in ["seizure", "migraine", "headache", "sar dard"]): return "Neurology"
    if any(k in text_lower for k in ["fracture", "bone", "joint", "knee", "back pain"]): return "Orthopedics"
    if any(k in text_lower for k in ["skin", "rash", "itching", "burn"]): return "Dermatology"
    if any(k in text_lower for k in ["ear", "nose", "throat"]): return "ENT"
    return "General Medicine"

def assign_doctor(department: str):
    conn = get_db()
    cursor = conn.cursor()
    query_dept = "General Medicine" if department == "Emergency" else department
    cursor.execute("SELECT id, doctor_id, name, department FROM doctors WHERE department = ? AND active = 1 LIMIT 1", (query_dept,))
    doctor = cursor.fetchone()
    conn.close()

    if doctor:
        return {"assigned_doctor_id": doctor["id"], "assigned_doctor_name": doctor["name"], "department": doctor["department"]}
    return {"assigned_doctor_id": None, "assigned_doctor_name": "Dr. Rahul Sharma", "department": department}

@app.get("/")
def root():
    return {"status": "online", "service": "MediKiosk Clinical Intelligence Platform"}

@app.get("/health")
def health():
    return {"status": "healthy", "time": now()}

# ------------------------------------------------------------
# AI CHATBOT TRIAGE ENDPOINT (ALWAYS ASKS QUESTIONS FIRST)
# ------------------------------------------------------------

@app.post("/api/chat/ai-assistant")
def chat_ai_assistant(req: ChatRequest):
    conn = get_db()
    cursor = conn.cursor()

    user_msg = (req.user_message or "").strip()
    history = req.chat_history or []

    # Count how many turns the patient has talked
    user_turns = sum(1 for m in history if m.get("sender") == "user") + 1

    reply_msg = ""
    poll_options = []
    is_emergency = 1 if detect_emergency(user_msg) else 0

    if groq_client:
        try:
            system_prompt = (
                "You are an empathetic clinical triage AI assistant at MediKiosk Indian Hospital. "
                "Respond in friendly Roman Hinglish (Hindi written in Latin script mixed with simple English). "
                "CRITICAL INSTRUCTION: Do NOT register or issue a token immediately! "
                "First ask 2 to 3 short diagnostic follow-up questions one by one (e.g., 'Yeh kab se ho raha hai?', 'Dard kitna teez hai 1-10 scale par?'). "
                "Only offer a token/completion AFTER at least 3 conversation turns. Keep responses under 2-3 short sentences."
            )
            messages = [{"role": "system", "content": system_prompt}]
            for msg in history[-5:]:
                role = "assistant" if msg.get("sender") == "assistant" else "user"
                messages.append({"role": role, "content": msg.get("text", "")})
            messages.append({"role": "user", "content": user_msg})

            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.6,
                max_tokens=150
            )
            reply_msg = completion.choices[0].message.content.strip()
        except Exception as e:
            print("Groq Exception:", e)

    # Fallback smart question flow if Groq API is offline/not set
    if not reply_msg:
        if user_turns == 1:
            reply_msg = f"Aapko {user_msg} ({req.pain_site or 'General'}) kab se ho raha hai? Kripya duration batayein."
            poll_options = ["Aaj se", "2-3 Din se", "1 Hafte se", "1 Mahine se zyaada"]
        elif user_turns == 2:
            reply_msg = f"Aapka dard kitna teez hai? Scale par select karein:"
            poll_options = ["Halka (1-3)", "Medium (4-6)", "Bahut Zyaada (7-10)"]
        elif user_turns == 3:
            reply_msg = f"Kya aapko koi aur lakshan jaise bukhar, chakkar, ya ulti feel ho rahi hai?"
            poll_options = ["Bukhar", "Chakkar", "Ghabrahat / Ulti", "Inme se koi nahi"]
        else:
            cursor.execute("SELECT MAX(token) FROM patients")
            row = cursor.fetchone()
            max_token = row[0] if row and row[0] is not None else 100
            next_token = max_token + 1

            p_details = req.patient_details or {}
            p_name = p_details.get("patient_name", "Ramesh Kumar")
            p_id = p_details.get("patient_id", f"MK-{next_token}")
            dept = determine_department(user_msg, req.pain_site)
            doc_info = assign_doctor(dept)

            cursor.execute("""
                INSERT INTO patients (patient_id, patient_name, age, phone, gender, chief_complaint, symptoms, pain_site, department, assigned_doctor_id, assigned_doctor_name, status, token, emergency, created_at, updated_at)
                VALUES (?, ?, 45, '9876543210', 'Male', ?, ?, ?, ?, ?, ?, 'Waiting', ?, ?, ?, ?)
            """, (p_id, p_name, user_msg, user_msg, req.pain_site or "General", dept, doc_info["assigned_doctor_id"], doc_info["assigned_doctor_name"], next_token, is_emergency, now(), now()))
            conn.commit()

            reply_msg = f"Shukriya {p_name}. Aapki details record ho gayi hain. OPD Token #{next_token} generate ho gaya hai. Kripya waiting area me baithein."
            conn.close()
            return {"status": "success", "reply": reply_msg, "token": next_token, "is_red_flag": bool(is_emergency), "is_completed": True}

    conn.close()
    return {
        "status": "success",
        "reply": reply_msg,
        "poll_options": poll_options,
        "is_red_flag": bool(is_emergency),
        "is_completed": False
    }

@app.post("/api/doctor/login")
def doctor_login(req: DoctorLoginRequest):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, doctor_id, name, department FROM doctors WHERE (doctor_id = ? OR name = ?) AND password = ? LIMIT 1", (req.username, req.username, req.password))
    doctor = cursor.fetchone()
    conn.close()
    if not doctor: raise HTTPException(status_code=401, detail="Invalid doctor credentials")
    return {"status": "success", "doctor": dict(doctor)}

@app.post("/api/admin/login")
def admin_login(req: AdminLoginRequest):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username FROM admins WHERE username = ? AND password = ? LIMIT 1", (req.username, req.password))
    admin = cursor.fetchone()
    conn.close()
    if not admin: raise HTTPException(status_code=401, detail="Invalid admin credentials")
    return {"status": "success", "admin": dict(admin)}

@app.get("/api/doctor/queue")
def doctor_queue():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE status NOT IN ('Completed', 'Cancelled') ORDER BY emergency DESC, token ASC")
    rows = cursor.fetchall()
    conn.close()
    return {"status": "success", "active_queue": [patient_row_to_dict(r) for r in rows]}

@app.get("/api/admin/patients")
def get_admin_patients():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return {"status": "success", "patients": [patient_row_to_dict(r) for r in rows]}

@app.websocket("/ws/kiosk")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        await websocket.send_json({"type": "CONNECTED"})
        while True:
            data = await websocket.receive_json()
            if data.get("type"): await broadcast(data)
    except WebSocketDisconnect: pass
    finally:
        if websocket in connected_clients: connected_clients.remove(websocket)

init_database()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
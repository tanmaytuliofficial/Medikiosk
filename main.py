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
from groq import Groq

# ============================================================
# APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="MediKiosk Clinical Intelligence Platform",
    version="35.0.0"
)

# ============================================================
# GROQ AI INTEGRATION
# ============================================================
# ============================================================
# SAFE EXTERNAL IMPORTS (PREVENTS RENDER DEPLOY CRASHES)
# ============================================================

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
# ============================================================
# CORS MIDDLEWARE
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./medikiosk.db"
)

if DATABASE_URL.startswith("sqlite:///"):
    DB_PATH = DATABASE_URL.replace("sqlite:///", "", 1)
else:
    DB_PATH = "./medikiosk.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

# ============================================================
# FILE UPLOADS
# ============================================================

UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "medical_reports"
)

os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_DIR),
    name="uploads"
)

# ============================================================
# DATABASE HELPERS
# ============================================================

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
        ("patient_id", "TEXT"),
        ("patient_name", "TEXT"),
        ("age", "INTEGER"),
        ("phone", "TEXT"),
        ("gender", "TEXT"),
        ("chief_complaint", "TEXT"),
        ("symptoms", "TEXT"),
        ("medical_history", "TEXT"),
        ("pain_site", "TEXT"),
        ("department", "TEXT"),
        ("assigned_doctor_id", "INTEGER"),
        ("assigned_doctor_name", "TEXT"),
        ("status", "TEXT DEFAULT 'Waiting'"),
        ("token", "INTEGER"),
        ("emergency", "INTEGER DEFAULT 0"),
        ("doctor_notes", "TEXT"),
        ("medical_report_image", "TEXT"),
        ("medical_report_filename", "TEXT"),
        ("ocr_text", "TEXT"),
        ("ai_summary", "TEXT"),
        ("clinical_information", "TEXT"),
        ("conversation", "TEXT"),
        ("created_at", "TEXT"),
        ("updated_at", "TEXT"),
        ("completed_at", "TEXT"),
        ("completed_by", "TEXT"),
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
        ("doctor_id", "TEXT DEFAULT ''"),
        ("name", "TEXT"),
        ("department", "TEXT"),
        ("phone", "TEXT"),
        ("email", "TEXT"),
        ("password", "TEXT DEFAULT 'doctor123'"),
        ("password_hash", "TEXT DEFAULT 'doctor123'"),
        ("active", "INTEGER DEFAULT 1"),
    ]

    for column_name, definition in doctor_columns:
        add_column_if_missing(cursor, "doctors", column_name, definition)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
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
        existing = cursor.fetchone()
        if existing is None:
            cursor.execute("""
                INSERT INTO doctors (doctor_id, name, department, phone, email, password, password_hash, active)
                VALUES (?, ?, ?, '', '', 'doctor123', 'doctor123', 1)
            """, (doc_id, name, dept))

    conn.commit()
    conn.close()

# ============================================================
# WEBSOCKET MANAGER
# ============================================================

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

# ============================================================
# MODELS
# ============================================================

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

class PatientStatusRequest(BaseModel):
    status: str
    doctor_id: Optional[int] = None

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

# ============================================================
# HELPER & UTILITY FUNCTIONS
# ============================================================

def now():
    return datetime.now().isoformat(timespec="seconds")

def generate_patient_id():
    return "MK-" + datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]

def safe_row_value(row, key, default=None):
    if row is None:
        return default
    try:
        if key in row.keys():
            val = row[key]
            return default if val is None else val
    except Exception:
        pass
    return default

def patient_row_to_dict(row):
    if row is None:
        return None

    emergency_val = safe_row_value(row, "emergency", 0)
    try:
        emergency_bool = bool(int(emergency_val or 0))
    except Exception:
        emergency_bool = bool(emergency_val)

    clinical_raw = safe_row_value(row, "clinical_information", "")
    clinical_info = {}
    if isinstance(clinical_raw, dict):
        clinical_info = clinical_raw
    elif isinstance(clinical_raw, str) and clinical_raw.strip():
        try:
            parsed = json.loads(clinical_raw)
            if isinstance(parsed, dict):
                clinical_info = parsed
        except Exception:
            clinical_info = {}

    clinical_info.setdefault("duration", "")
    clinical_info.setdefault("severity", "")
    clinical_info.setdefault("pain_site", safe_row_value(row, "pain_site", "General") or "General")

    conversation_raw = safe_row_value(row, "conversation", "")
    conversation = []
    if isinstance(conversation_raw, list):
        conversation = conversation_raw
    elif isinstance(conversation_raw, str) and conversation_raw.strip():
        try:
            parsed_c = json.loads(conversation_raw)
            if isinstance(parsed_c, list):
                conversation = parsed_c
        except Exception:
            conversation = []

    return {
        "id": safe_row_value(row, "id"),
        "patient_id": safe_row_value(row, "patient_id"),
        "patient_name": safe_row_value(row, "patient_name", "Walk-in Patient") or "Walk-in Patient",
        "age": safe_row_value(row, "age"),
        "phone": safe_row_value(row, "phone"),
        "contact": safe_row_value(row, "phone"),
        "gender": safe_row_value(row, "gender"),
        "chief_complaint": safe_row_value(row, "chief_complaint"),
        "symptoms": safe_row_value(row, "symptoms") or clinical_info.get("symptoms", "") or safe_row_value(row, "chief_complaint", ""),
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
        "case_type": "emergency" if emergency_bool else "normal",
        "doctor_notes": safe_row_value(row, "doctor_notes", "") or "",
        "medical_report_image": safe_row_value(row, "medical_report_image"),
        "medical_report_filename": safe_row_value(row, "medical_report_filename"),
        "ocr_text": safe_row_value(row, "ocr_text"),
        "ai_summary": safe_row_value(row, "ai_summary"),
        "clinical_information": clinical_info,
        "created_at": safe_row_value(row, "created_at"),
        "updated_at": safe_row_value(row, "updated_at"),
        "completed_at": safe_row_value(row, "completed_at"),
        "completed_by": safe_row_value(row, "completed_by"),
        "conversation": conversation,
        "transcript": conversation,
    }

EMERGENCY_PATTERNS = [
    "severe chest pain", "crushing chest pain", "difficulty breathing", "cannot breathe",
    "can't breathe", "shortness of breath", "unconscious", "fainted", "loss of consciousness",
    "severe bleeding", "heavy bleeding", "stroke", "seizure", "convulsion", "paralysis",
    "face drooping", "slurred speech"
]

def detect_emergency(text: str):
    text = (text or "").lower().strip()
    return any(p in text for p in EMERGENCY_PATTERNS)

def detect_pain_site(text: str):
    text = (text or "").lower()
    sites = {
        "Chest": ["chest", "heart"],
        "Head": ["head", "forehead", "migraine", "headache"],
        "Neck": ["neck", "throat"],
        "Back": ["back", "spine"],
        "Abdomen": ["stomach", "abdomen", "belly"],
        "Arm": ["arm", "shoulder", "elbow"],
        "Leg": ["leg", "knee", "ankle", "thigh"],
    }
    for site, kws in sites.items():
        if any(kw in text for kw in kws):
            return site
    return "General"

def determine_department(text: str, pain_site: Optional[str] = None):
    text_lower = (text or "").lower()
    if detect_emergency(text_lower): return "Emergency"
    if any(k in text_lower for k in ["chest pain", "heart", "palpitation", "blood pressure"]): return "Cardiology"
    if any(k in text_lower for k in ["seizure", "migraine", "numbness", "dizziness"]): return "Neurology"
    if any(k in text_lower for k in ["fracture", "bone", "joint", "knee", "back pain"]): return "Orthopedics"
    if any(k in text_lower for k in ["skin", "rash", "itching", "acne"]): return "Dermatology"
    if any(k in text_lower for k in ["ear", "nose", "throat", "hearing"]): return "ENT"
    return "General Medicine"

def assign_doctor(department: str):
    conn = get_db()
    cursor = conn.cursor()
    query_dept = "General Medicine" if department == "Emergency" else department
    cursor.execute("SELECT id, doctor_id, name, department FROM doctors WHERE department = ? AND active = 1 LIMIT 1", (query_dept,))
    doctor = cursor.fetchone()
    conn.close()

    if doctor:
        return {
            "assigned_doctor_id": doctor["id"],
            "assigned_doctor_code": doctor["doctor_id"],
            "assigned_doctor_name": doctor["name"],
            "department": doctor["department"],
        }
    return {"assigned_doctor_id": None, "assigned_doctor_code": None, "assigned_doctor_name": "Duty Doctor", "department": department}

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {"status": "online", "service": "MediKiosk Clinical Intelligence Platform", "version": "35.0.0", "database": "connected"}

@app.get("/health")
def health():
    try:
        conn = get_db()
        conn.execute("SELECT 1")
        conn.close()
        return {"status": "healthy", "database": "connected", "time": now()}
    except Exception as e:
        return {"status": "error", "database": "failed", "error": str(e)}

@app.post("/api/doctor/login")
@app.post("/api/auth/doctor/login")
def doctor_login(req: DoctorLoginRequest):
    username = (req.username or "").strip()
    password = (req.password or "").strip()

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, doctor_id, name, department, phone, email, active FROM doctors
        WHERE (LOWER(TRIM(doctor_id)) = LOWER(TRIM(?)) OR LOWER(TRIM(name)) = LOWER(TRIM(?)) OR LOWER(TRIM(email)) = LOWER(TRIM(?)))
        AND (password = ? OR password_hash = ?) AND active = 1 LIMIT 1
    """, (username, username, username, password, password))
    doctor = cursor.fetchone()
    conn.close()

    if not doctor:
        raise HTTPException(status_code=401, detail="Invalid doctor credentials")

    data = dict(doctor)
    return {"status": "success", "message": "Doctor login successful", "doctor": data, "user": data}

@app.post("/api/admin/login")
@app.post("/api/auth/admin/login")
def admin_login(req: AdminLoginRequest):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username FROM admins WHERE username = ? AND password = ? LIMIT 1", (req.username.strip(), req.password.strip()))
    admin = cursor.fetchone()
    conn.close()

    if not admin:
        raise HTTPException(status_code=401, detail="Invalid admin credentials")

    data = dict(admin)
    return {"status": "success", "admin": data, "user": data}

@app.get("/api/doctor/queue")
def doctor_queue(doctor_id: Optional[str] = Query(None)):
    conn = get_db()
    cursor = conn.cursor()
    if doctor_id:
        cursor.execute("SELECT * FROM patients WHERE status NOT IN ('Completed', 'Cancelled') AND assigned_doctor_id = ? ORDER BY emergency DESC, token ASC", (doctor_id,))
    else:
        cursor.execute("SELECT * FROM patients WHERE status NOT IN ('Completed', 'Cancelled') ORDER BY emergency DESC, token ASC")
    rows = cursor.fetchall()
    conn.close()
    return {"status": "success", "active_queue": [patient_row_to_dict(r) for r in rows]}

@app.get("/api/doctor/summary")
def doctor_summary(doctor_id: Optional[str] = Query(None)):
    conn = get_db()
    cursor = conn.cursor()
    filter_sql = "AND assigned_doctor_id = ?" if doctor_id else ""
    params = [doctor_id] if doctor_id else []

    cursor.execute(f"SELECT COUNT(*) FROM patients WHERE status NOT IN ('Completed', 'Cancelled') {filter_sql}", params)
    active = cursor.fetchone()[0]

    cursor.execute(f"SELECT COUNT(*) FROM patients WHERE status = 'Waiting' {filter_sql}", params)
    waiting = cursor.fetchone()[0]

    cursor.execute(f"SELECT COUNT(*) FROM patients WHERE emergency = 1 AND status NOT IN ('Completed', 'Cancelled') {filter_sql}", params)
    emergency = cursor.fetchone()[0]

    today = datetime.now().strftime("%Y-%m-%d") + "%"
    completed_params = [today] + params
    cursor.execute(f"SELECT COUNT(*) FROM patients WHERE status = 'Completed' AND completed_at LIKE ? {filter_sql}", completed_params)
    completed_today = cursor.fetchone()[0]

    if doctor_id:
        cursor.execute("SELECT * FROM patients WHERE status NOT IN ('Completed', 'Cancelled') AND assigned_doctor_id = ? ORDER BY emergency DESC, token ASC", (doctor_id,))
    else:
        cursor.execute("SELECT * FROM patients WHERE status NOT IN ('Completed', 'Cancelled') ORDER BY emergency DESC, token ASC")
    queue_rows = cursor.fetchall()
    conn.close()

    return {
        "status": "success",
        "summary": {"active_patients": active, "waiting": waiting, "emergency": emergency, "completed_today": completed_today},
        "active_queue": [patient_row_to_dict(r) for r in queue_rows]
    }

# Websocket endpoint
@app.websocket("/ws/kiosk")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        await websocket.send_json({"type": "CONNECTED", "message": "MediKiosk WebSocket connected"})
        while True:
            data = await websocket.receive_json()
            if data.get("type"):
                await broadcast(data)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

init_database()

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import sqlite3
import os
import json

app = FastAPI(
    title="MediKiosk Clinical Intelligence Platform",
    version="38.0.0"
)

# ============================================================
# GROQ INITIALIZATION
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
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DATABASE CONFIGURATION (SUPABASE / POSTGRESQL & SQLITE)
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./medikiosk.db"
)

is_postgres = (
    DATABASE_URL.startswith("postgres://")
    or DATABASE_URL.startswith("postgresql://")
)

psycopg2 = None
if is_postgres:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        raise RuntimeError(
            "PostgreSQL/Supabase detected in DATABASE_URL, but 'psycopg2-binary' is not installed."
        )

DB_PATH = (
    DATABASE_URL.replace("sqlite:///", "", 1)
    if DATABASE_URL.startswith("sqlite:///")
    else "./medikiosk.db"
)

def get_db():
    if is_postgres:
        conn = psycopg2.connect(
            DATABASE_URL,
            cursor_factory=psycopg2.extras.RealDictCursor
        )
        return conn
    else:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            conn.execute("PRAGMA journal_mode = WAL")
        except Exception:
            pass
        return conn

# ============================================================
# UPLOADS MOUNT
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
# INITIALIZE DATABASE & TABLES (WITH DUMMY DATA SEEDING)
# ============================================================

def init_database():
    conn = get_db()
    cursor = conn.cursor()

    if is_postgres:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id SERIAL PRIMARY KEY,
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nfc_tags (
                id SERIAL PRIMARY KEY,
                uid TEXT UNIQUE NOT NULL,
                patient_id TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                issued_at TEXT,
                last_used_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS doctors (
                id SERIAL PRIMARY KEY,
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE,
                password TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
    else:
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nfc_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid TEXT UNIQUE NOT NULL,
                patient_id TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                issued_at TEXT,
                last_used_at TEXT
            )
        """)
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

    # Seed Default Admin
    cursor.execute("SELECT id FROM admins WHERE username = 'admin' LIMIT 1")
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO admins (username, password) VALUES (?, ?)",
            ("admin", "admin123")
        )

    # Seed Default Doctors
    default_doctors = [
        ("DOC-001", "Dr. Rahul Sharma", "General Medicine"),
        ("DOC-002", "Dr. Priya Verma", "Orthopedics"),
        ("DOC-003", "Dr. Amit Singh", "Cardiology"),
    ]

    for doc_id, name, dept in default_doctors:
        cursor.execute(
            "SELECT id FROM doctors WHERE doctor_id = ? OR name = ? LIMIT 1",
            (doc_id, name)
        )
        if not cursor.fetchone():
            cursor.execute(
                """
                INSERT INTO doctors (
                    doctor_id, name, department, phone, email, password, password_hash, active
                ) VALUES (?, ?, ?, '', '', ?, ?, 1)
                """,
                (doc_id, name, dept, "doctor123", "doctor123")
            )

    # --------------------------------------------------------
    # SEED DUMMY PATIENT & NFC TAG FOR YOUR UID (C9793207)
    # --------------------------------------------------------
    dummy_uid = "C9793207"
    dummy_patient_id = "MK-101"
    
    cursor.execute("SELECT id FROM patients WHERE patient_id = ? LIMIT 1", (dummy_patient_id,))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO patients (
                patient_id, patient_name, age, phone, gender, chief_complaint, 
                symptoms, medical_history, pain_site, department, assigned_doctor_name, 
                status, token, emergency, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dummy_patient_id, "Rahul Sharma", 28, "9876543210", "Male", 
            "Chest discomfort", "Mild pain", "No major history", "Chest", 
            "Cardiology", "Dr. Amit Singh", "Waiting", 101, 0, 
            datetime.now().isoformat(), datetime.now().isoformat()
        ))

    cursor.execute("SELECT id FROM nfc_tags WHERE uid = ? LIMIT 1", (dummy_uid,))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO nfc_tags (uid, patient_id, status, issued_at, last_used_at)
            VALUES (?, ?, 'active', ?, ?)
        """, (dummy_uid, dummy_patient_id, datetime.now().isoformat(), datetime.now().isoformat()))

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
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):
    user_message: str
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    language: str = "en"
    patient_details: Dict[str, Any] = Field(default_factory=dict)
    pain_site: Optional[str] = None
    is_nfc_mode: bool = False
    clinical_information: Dict[str, Any] = Field(default_factory=dict)

class NFCTapRequest(BaseModel):
    uid: str

class NFCRegisterRequest(BaseModel):
    uid: str
    patient_id: str

# ============================================================
# GENERAL HELPERS
# ============================================================

def now():
    return datetime.now().isoformat(timespec="seconds")

def safe_row_value(row, key, default=None):
    if row is None:
        return default
    try:
        if hasattr(row, "keys") and key in row.keys():
            val = row[key]
            return default if val is None else val
        elif isinstance(row, dict) and key in row:
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

    return {
        "id": safe_row_value(row, "id"),
        "patient_id": safe_row_value(row, "patient_id"),
        "patient_name": safe_row_value(row, "patient_name", "Walk-in Patient") or "Walk-in Patient",
        "age": safe_row_value(row, "age"),
        "phone": safe_row_value(row, "phone"),
        "gender": safe_row_value(row, "gender"),
        "department": safe_row_value(row, "department", "General Medicine") or "General Medicine",
        "assigned_doctor_name": safe_row_value(row, "assigned_doctor_name", "Duty Doctor") or "Duty Doctor",
        "status": safe_row_value(row, "status", "Waiting") or "Waiting",
        "token": safe_row_value(row, "token"),
        "emergency": emergency_bool,
    }

# ============================================================
# NFC TAP ENDPOINT (POST)
# ============================================================

@app.post("/api/nfc/tap")
async def nfc_tap(req: NFCTapRequest):
    uid = (req.uid or "").strip().upper()
    if not uid:
        raise HTTPException(status_code=400, detail="NFC UID is required")

    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT n.uid, n.patient_id, n.status, p.*
            FROM nfc_tags n
            LEFT JOIN patients p ON p.patient_id = n.patient_id
            WHERE n.uid = ?
            LIMIT 1
        """, (uid,))
        row = cursor.fetchone()

        if not row or not row["patient_id"]:
            await broadcast({
                "type": "NFC_TAP",
                "status": "not_registered",
                "uid": uid
            })
            return {
                "status": "not_registered",
                "uid": uid,
                "message": "New NFC card detected. Please proceed to patient registration."
            }

        cursor.execute("UPDATE nfc_tags SET last_used_at = ? WHERE uid = ?", (now(), uid))
        conn.commit()

        patient = patient_row_to_dict(row)
        
        await broadcast({
            "type": "NFC_TAP",
            "status": "success",
            "uid": uid,
            "patient_id": row["patient_id"],
            "patient": patient
        })

        return {
            "status": "success",
            "uid": uid,
            "patient_id": row["patient_id"],
            "patient": patient,
            "message": f"Patient Verified: {patient['patient_name']}"
        }
    finally:
        conn.close()

# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@app.websocket("/ws/kiosk")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        await websocket.send_json({"type": "CONNECTED"})
        while True:
            data = await websocket.receive_json()
            if data.get("type"):
                await broadcast(data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("WebSocket error:", e)
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

# ============================================================
# START DATABASE & APP
# ============================================================

init_database()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

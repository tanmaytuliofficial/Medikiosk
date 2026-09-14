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
            "PostgreSQL/Supabase detected in DATABASE_URL, but 'psycopg2-binary' is not installed. Run: pip install psycopg2-binary"
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
# INITIALIZE DATABASE & TABLES
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
        ("DOC-004", "Dr. Neha Gupta", "Neurology"),
        ("DOC-005", "Dr. Anjali Mehta", "Dermatology"),
        ("DOC-006", "Dr. Arjun Kapoor", "ENT"),
        ("DOC-007", "Dr. Riya Malhotra", "Pediatrics"),
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

class DoctorLoginRequest(BaseModel):
    username: str
    password: str

class AdminLoginRequest(BaseModel):
    username: str
    password: str

# ============================================================
# GENERAL HELPERS
# ============================================================

def now():
    return datetime.now().isoformat(timespec="seconds")

def safe_row_value(row, key, default=None):
    if row is None: return default
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

def get_next_token(cursor):
    cursor.execute("SELECT MAX(token) AS max_token FROM patients")
    row = cursor.fetchone()
    if row and row["max_token"] is not None:
        try:
            return int(row["max_token"]) + 1
        except Exception:
            pass
    return 101

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
        "contact": safe_row_value(row, "phone"),
        "gender": safe_row_value(row, "gender"),
        "chief_complaint": safe_row_value(row, "chief_complaint"),
        "symptoms": safe_row_value(row, "symptoms"),
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
        "clinical_information": safe_row_value(row, "clinical_information", "{}") or "{}",
        "conversation": safe_row_value(row, "conversation", "[]") or "[]",
        "ocr_text": safe_row_value(row, "ocr_text", "") or "",
        "medical_report_filename": safe_row_value(row, "medical_report_filename", "") or "",
        "ai_summary": safe_row_value(row, "ai_summary", "") or "",
        "created_at": safe_row_value(row, "created_at"),
        "updated_at": safe_row_value(row, "updated_at"),
    }

# ============================================================
# EMERGENCY & DEPARTMENT LOGIC
# ============================================================

EMERGENCY_PATTERNS = [
    "severe chest pain", "crushing chest pain", "difficulty breathing",
    "cannot breathe", "can't breathe", "unconscious", "fainted",
    "severe bleeding", "stroke", "seizure"
]

def detect_emergency(text: str):
    text_lower = (text or "").lower().strip()
    return any(pattern in text_lower for pattern in EMERGENCY_PATTERNS)

def determine_department(text: str, pain_site: Optional[str] = None):
    text_lower = (text or "").lower()
    pain_lower = (pain_site or "").lower()
    combined_text = text_lower + " " + pain_lower

    if detect_emergency(combined_text):
        return "Emergency"
    if any(k in combined_text for k in ["chest pain", "heart", "bp"]):
        return "Cardiology"
    if any(k in combined_text for k in ["seizure", "migraine", "headache", "sar dard"]):
        return "Neurology"
    if any(k in combined_text for k in ["fracture", "bone", "joint", "knee", "back pain"]):
        return "Orthopedics"
    if any(k in combined_text for k in ["skin", "rash", "itching", "burn"]):
        return "Dermatology"
    if any(k in combined_text for k in ["ear", "nose", "throat"]):
        return "ENT"
    return "General Medicine"

def assign_doctor(department: str):
    conn = get_db()
    cursor = conn.cursor()
    query_department = "General Medicine" if department == "Emergency" else department

    cursor.execute("""
        SELECT id, doctor_id, name, department
        FROM doctors
        WHERE department = ? AND active = 1
        ORDER BY id ASC LIMIT 1
    """, (query_department,))
    doctor = cursor.fetchone()

    if doctor:
        res = {
            "assigned_doctor_id": doctor["id"],
            "assigned_doctor_code": doctor["doctor_id"],
            "assigned_doctor_name": doctor["name"],
            "department": doctor["department"]
        }
        conn.close()
        return res

    cursor.execute("""
        SELECT id, doctor_id, name, department
        FROM doctors WHERE active = 1
        ORDER BY id ASC LIMIT 1
    """)
    fallback = cursor.fetchone()
    conn.close()

    if fallback:
        return {
            "assigned_doctor_id": fallback["id"],
            "assigned_doctor_code": fallback["doctor_id"],
            "assigned_doctor_name": fallback["name"],
            "department": fallback["department"]
        }

    return {
        "assigned_doctor_id": None,
        "assigned_doctor_code": "DOC-001",
        "assigned_doctor_name": "Dr. Rahul Sharma",
        "department": "General Medicine"
    }

# ============================================================
# API ENDPOINTS: ROOT & HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "MediKiosk Clinical Intelligence Platform",
        "version": "38.0.0",
        "database_connected": "Supabase PostgreSQL" if is_postgres else "SQLite"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "time": now(),
        "database_type": "PostgreSQL (Supabase)" if is_postgres else "SQLite",
        "groq": bool(groq_client)
    }

# ============================================================
# NFC REGISTRATION & TAP ENDPOINTS
# ============================================================

@app.post("/api/nfc/register")
def register_nfc_tag(req: NFCRegisterRequest):
    uid = (req.uid or "").strip().upper()
    patient_id = (req.patient_id or "").strip()

    if not uid or not patient_id:
        raise HTTPException(status_code=400, detail="NFC UID and Patient ID are required")

    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM patients WHERE patient_id = ? LIMIT 1", (patient_id,))
        patient = cursor.fetchone()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        cursor.execute("SELECT * FROM nfc_tags WHERE uid = ? LIMIT 1", (uid,))
        existing = cursor.fetchone()

        if existing:
            cursor.execute("UPDATE nfc_tags SET patient_id = ?, status = 'active' WHERE uid = ?", (patient_id, uid))
        else:
            cursor.execute(
                "INSERT INTO nfc_tags (uid, patient_id, status, issued_at, last_used_at) VALUES (?, ?, 'active', ?, ?)",
                (uid, patient_id, now(), now())
            )
        conn.commit()
        return {
            "status": "success",
            "message": "NFC tag linked successfully",
            "uid": uid,
            "patient_id": patient_id,
            "patient": patient_row_to_dict(patient)
        }
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=409, detail=f"NFC registration conflict: {e}")
    finally:
        conn.close()

@app.post("/api/nfc/tap")
def nfc_tap(req: NFCTapRequest):
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

        # NEW CARD DETECTION: If card is not registered, return 'not_registered' so frontend triggers new patient registration
        if not row or not row["patient_id"]:
            return {
                "status": "not_registered",
                "uid": uid,
                "message": "New NFC card detected. Please proceed to patient registration."
            }

        if row["status"] != "active":
            return {
                "status": "inactive",
                "uid": uid,
                "patient_id": row["patient_id"]
            }

        cursor.execute("UPDATE nfc_tags SET last_used_at = ? WHERE uid = ?", (now(), uid))
        conn.commit()

        patient = patient_row_to_dict(row)
        return {
            "status": "success",
            "uid": uid,
            "patient_id": row["patient_id"],
            "patient": patient
        }
    finally:
        conn.close()

# ============================================================
# AI CHATBOT TRIAGE & PATIENT REGISTRATION
# ============================================================

@app.post("/api/chat/ai-assistant")
async def chat_ai_assistant(req: ChatRequest):
    conn = get_db()
    cursor = conn.cursor()

    try:
        user_msg = (req.user_message or "").strip()
        history = req.chat_history or []

        if not user_msg:
            raise HTTPException(status_code=400, detail="Empty message")

        user_turns = sum(1 for m in history if m.get("sender") == "user")
        if user_turns < 1:
            user_turns = 1

        all_current_text = " ".join(str(m.get("text", "")) for m in history if m.get("sender") == "user") + " " + user_msg
        is_emergency = 1 if detect_emergency(all_current_text) else 0

        # Turn 1
        if user_turns == 1:
            reply_msg = f"Aapko {user_msg} ({req.pain_site or 'General'}) kab se ho raha hai? Kripya duration batayein."
            poll_options = ["Aaj se", "2-3 Din se", "1 Hafte se", "1 Mahine se zyaada"]
            
            if groq_client:
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a clinical triage assistant for MediKiosk. Reply in short friendly Roman Hinglish. Ask exactly ONE short question about duration. Do not diagnose."
                        },
                        {"role": "user", "content": user_msg}
                    ]
                    completion = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages,
                        temperature=0.4,
                        max_tokens=80
                    )
                    groq_reply = completion.choices[0].message.content.strip()
                    if groq_reply:
                        reply_msg = groq_reply
                except Exception as e:
                    print("Groq Turn 1 Error:", e)

            return {
                "status": "success",
                "reply": reply_msg,
                "poll_options": poll_options,
                "is_red_flag": bool(is_emergency),
                "is_completed": False
            }

        # Turn 2
        elif user_turns == 2:
            reply_msg = "Aapka dard kitna teez hai? 1 se 10 ke scale par batayein."
            poll_options = ["Halka (1-3)", "Medium (4-6)", "Bahut Zyaada (7-10)"]

            if groq_client:
                try:
                    messages = [{
                        "role": "system",
                        "content": "You are a clinical triage assistant for MediKiosk. Reply in short friendly Roman Hinglish. Ask exactly ONE short question about pain severity. Do not diagnose."
                    }]
                    for msg in history[-5:]:
                        role = "assistant" if msg.get("sender") == "assistant" else "user"
                        messages.append({"role": role, "content": msg.get("text", "")})

                    completion = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages,
                        temperature=0.4,
                        max_tokens=80
                    )
                    groq_reply = completion.choices[0].message.content.strip()
                    if groq_reply:
                        reply_msg = groq_reply
                except Exception as e:
                    print("Groq Turn 2 Error:", e)

            return {
                "status": "success",
                "reply": reply_msg,
                "poll_options": poll_options,
                "is_red_flag": bool(is_emergency),
                "is_completed": False
            }

        # Turn 3+: Token Generation & Database Save
        else:
            next_token = get_next_token(cursor)
            p_details = req.patient_details or {}
            p_name = p_details.get("patient_name") or "Walk-in Patient"
            p_id = p_details.get("patient_id") or f"MK-{next_token}"
            p_id = str(p_id).strip()

            age = p_details.get("age") or 0
            phone = p_details.get("phone") or ""
            gender = p_details.get("gender") or ""

            first_user_message = user_msg
            for msg in history:
                if msg.get("sender") == "user":
                    first_user_message = msg.get("text") or user_msg
                    break

            full_text = " ".join(str(msg.get("text", "")) for msg in history if msg.get("sender") == "user")
            if user_msg not in full_text:
                full_text += " " + user_msg

            dept = determine_department(full_text, req.pain_site)
            doc_info = assign_doctor(dept)
            clinical_info = req.clinical_information or {}
            conversation_data = history.copy()
            
            if not any(m.get("sender") == "user" and m.get("text") == user_msg for m in conversation_data):
                conversation_data.append({"sender": "user", "text": user_msg})

            current_time = now()

            cursor.execute("SELECT id FROM patients WHERE patient_id = ? LIMIT 1", (p_id,))
            existing_patient = cursor.fetchone()

            if existing_patient:
                cursor.execute("""
                    UPDATE patients
                    SET patient_name = ?, age = ?, phone = ?, gender = ?, chief_complaint = ?,
                        symptoms = ?, pain_site = ?, department = ?, assigned_doctor_id = ?,
                        assigned_doctor_name = ?, status = ?, token = ?, emergency = ?,
                        clinical_information = ?, conversation = ?, created_at = ?, updated_at = ?,
                        completed_at = NULL, completed_by = NULL
                    WHERE patient_id = ?
                """, (
                    p_name, age, phone, gender, first_user_message, full_text,
                    req.pain_site or "General", dept, doc_info["assigned_doctor_id"],
                    doc_info["assigned_doctor_name"], "Waiting", next_token, is_emergency,
                    json.dumps(clinical_info, ensure_ascii=False),
                    json.dumps(conversation_data, ensure_ascii=False),
                    current_time, current_time, p_id
                ))
            else:
                cursor.execute("""
                    INSERT INTO patients (
                        patient_id, patient_name, age, phone, gender, chief_complaint, symptoms,
                        medical_history, pain_site, department, assigned_doctor_id, assigned_doctor_name,
                        status, token, emergency, clinical_information, conversation, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    p_id, p_name, age, phone, gender, first_user_message, full_text,
                    "", req.pain_site or "General", dept, doc_info["assigned_doctor_id"],
                    doc_info["assigned_doctor_name"], "Waiting", next_token, is_emergency,
                    json.dumps(clinical_info, ensure_ascii=False),
                    json.dumps(conversation_data, ensure_ascii=False),
                    current_time, current_time
                ))

            # Auto-link NFC tag if present
            nfc_uid = (p_details.get("nfc_uid") or p_details.get("card_uid") or "").strip().upper()
            if nfc_uid:
                cursor.execute("SELECT id FROM nfc_tags WHERE uid = ? LIMIT 1", (nfc_uid,))
                if cursor.fetchone():
                    cursor.execute(
                        "UPDATE nfc_tags SET patient_id = ?, status = 'active', last_used_at = ? WHERE uid = ?",
                        (p_id, current_time, nfc_uid)
                    )
                else:
                    cursor.execute(
                        "INSERT INTO nfc_tags (uid, patient_id, status, issued_at, last_used_at) VALUES (?, ?, 'active', ?, ?)",
                        (nfc_uid, p_id, current_time, current_time)
                    )

            conn.commit()

            cursor.execute("SELECT * FROM patients WHERE patient_id = ? LIMIT 1", (p_id,))
            saved_patient = cursor.fetchone()

            await broadcast({
                "type": "NEW_PATIENT",
                "token": next_token,
                "patient_id": p_id,
                "patient_name": p_name,
                "department": dept,
                "assigned_doctor_id": doc_info["assigned_doctor_id"],
                "assigned_doctor": doc_info["assigned_doctor_name"],
                "emergency": bool(is_emergency),
                "patient": patient_row_to_dict(saved_patient)
            })

            reply_msg = f"Shukriya {p_name}. Aapki details successfully record ho gayi hain. Aapka OPD Token #{next_token} generate ho gaya hai. Kripya waiting area mein jaayein."

            return {
                "status": "success",
                "reply": reply_msg,
                "token": next_token,
                "opd_token": next_token,
                "token_number": next_token,
                "is_red_flag": bool(is_emergency),
                "is_completed": True,
                "department": dept,
                "assigned_doctor": doc_info["assigned_doctor_name"],
                "assigned_doctor_id": doc_info["assigned_doctor_id"],
                "patient_id": p_id
            }

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# ============================================================
# LOGIN & QUEUES
# ============================================================

@app.post("/api/doctor/login")
def doctor_login(req: DoctorLoginRequest):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, doctor_id, name, department FROM doctors WHERE (doctor_id = ? OR name = ?) AND password = ? LIMIT 1",
        (req.username, req.username, req.password)
    )
    doctor = cursor.fetchone()
    conn.close()
    if not doctor:
        raise HTTPException(status_code=401, detail="Invalid doctor credentials")
    return {"status": "success", "doctor": dict(doctor)}

@app.post("/api/admin/login")
def admin_login(req: AdminLoginRequest):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username FROM admins WHERE username = ? AND password = ? LIMIT 1",
        (req.username, req.password)
    )
    admin = cursor.fetchone()
    conn.close()
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    return {"status": "success", "admin": dict(admin)}

@app.get("/api/doctor/queue")
def doctor_queue(doctor_id: Optional[str] = Query(None)):
    conn = get_db()
    cursor = conn.cursor()
    if not doctor_id:
        cursor.execute("SELECT * FROM patients WHERE status NOT IN ('Completed', 'Cancelled') ORDER BY emergency DESC, token ASC")
    else:
        cursor.execute("""
            SELECT * FROM patients
            WHERE status NOT IN ('Completed', 'Cancelled')
            AND assigned_doctor_id = (SELECT id FROM doctors WHERE doctor_id = ? LIMIT 1)
            ORDER BY emergency DESC, token ASC
        """, (doctor_id,))
    rows = cursor.fetchall()
    conn.close()
    return {"status": "success", "active_queue": [patient_row_to_dict(row) for row in rows]}

@app.get("/api/admin/patients")
def get_admin_patients():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return {"status": "success", "patients": [patient_row_to_dict(row) for row in rows]}

# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@app.websocket("/ws/kiosk")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        await websocket.send_json({"type": "CONNECTED"})
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
# RUN INITIALIZATION & APP
# ============================================================

init_database()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
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

def get_next_token(cursor):
    cursor.execute("SELECT MAX(token) AS max_token FROM patients")
    row = cursor.fetchone()
    if row and row["max_token"] is not None:
        try:
            return int(row["max_token"]) + 1
        except Exception:
            pass
    return 101

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
        "clinical_information": safe_row_value(row, "clinical_information", "{}") or "{}",
        "conversation": safe_row_value(row, "conversation", "[]") or "[]",
        "created_at": safe_row_value(row, "created_at"),
        "updated_at": safe_row_value(row, "updated_at"),
    }

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
            # Broadcast un-registered tap to frontend
            await broadcast({
                "type": "NFC_TAP",
                "status": "not_registered",
                "uid": uid
            })
            return {
                "status": "not_registered",
                "uid": uid,
                "message": "New NFC card detected."
            }

        cursor.execute("UPDATE nfc_tags SET last_used_at = ? WHERE uid = ?", (now(), uid))
        conn.commit()

        patient = patient_row_to_dict(row)
        
        # Broadcast successful tap to frontend
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
# NFC TAP ENDPOINT (POST)
# ============================================================

@app.post("/api/nfc/tap")
def nfc_tap(req: NFCTapRequest):
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
            return {
                "status": "not_registered",
                "uid": uid,
                "message": "New NFC card detected. Please proceed to patient registration."
            }

        cursor.execute("UPDATE nfc_tags SET last_used_at = ? WHERE uid = ?", (now(), uid))
        conn.commit()

        patient = patient_row_to_dict(row)
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
# START DATABASE & APP
# ============================================================

init_database()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
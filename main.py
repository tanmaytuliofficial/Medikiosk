from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
import json
import re
import os
from supabase import create_client, Client

app = FastAPI(title="MediKiosk Clinical Intelligence Platform", version="25.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Changed to False so wildcard origins work seamlessly
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ⚡ SUPABASE DATABASE INITIALIZATION
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://kosmrpnsudxwvqxejbzs.supabase.co") # Supabase Dashboard se copy kar
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imtvc21ycG5zdWR4d3ZxeGVqYnpzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODg0NDkyNzUsImV4cCI6MjEwNDAyNTI3NX0.HNmz8QveUDUhldjpUwNP1zhRiTAOm5wrbELPyCya8T0") # Supabase Dashboard se copy kar

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- PYDANTIC SCHEMAS ---
class ChatRequest(BaseModel):
    user_message: str
    chat_history: List[dict] = []
    language: Optional[str] = "en"
    patient_details: Optional[dict] = {}
    pain_site: Optional[str] = "General"

class NFCTapRequest(BaseModel):
    nfc_uid: str

class NFCRegisterRequest(BaseModel):
    nfc_uid: str
    full_name: str
    abha_id: Optional[str] = ""
    age: int
    gender: str
    phone_number: str
    blood_group: str
    allergies: Optional[List[str]] = []
    chronic_conditions: Optional[List[str]] = []

PATIENT_RECORDS = []
ARCHIVED_RECORDS = []
active_connections: List[WebSocket] = []
CURRENT_TOKEN_COUNTER = 101

# --- HELPER FUNCTIONS ---
def clean_patient_name(raw_text: str) -> str:
    cleaned = raw_text.strip()
    prefixes = [
        r"^my name is\s+", r"^i am\s+", r"^this is\s+",
        r"^mera naam\s+", r"^naam hai\s+", r"^main\s+"
    ]
    for pattern in prefixes:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\shai$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned.title() if cleaned else raw_text.title()

CLINICAL_SYSTEM_PROMPT = """
You are MediKiosk AI, an expert clinical intake assistant.
Language: Respond strictly in {lang}.

Your task is to ask a follow-up medical question AND provide 4 relevant quick-response options for the patient.

STRICT JSON OUTPUT FORMAT ONLY:
{{
  "reply": "Your concise clinical follow-up question here",
  "poll_options": [
    {{"label": "Option 1 Text", "value": "Option 1 Text"}},
    {{"label": "Option 2 Text", "value": "Option 2 Text"}},
    {{"label": "Option 3 Text", "value": "Option 3 Text"}},
    {{"label": "Option 4 Text", "value": "Option 4 Text"}}
  ]
}}

If red flag/emergency symptoms occur, include "[RED_FLAG_ALERT]" in the reply string.
DO NOT output any text outside this JSON block.
"""

class ConnectionManager:
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in active_connections:
            active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                pass

manager = ConnectionManager()

# --- WEBSOCKET ENDPOINT ---
@app.websocket("/ws/doctor-updates")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- 🚀 NEW: NFC TAP ENDPOINT (ESP32 / KIOSK HIT KAREGA) ---
@app.post("/api/nfc/tap")
async def handle_nfc_tap(payload: NFCTapRequest):
    clean_uid = payload.nfc_uid.strip().upper()
    
    if not clean_uid:
        raise HTTPException(status_code=400, detail="NFC UID is required")
    
    # Query Supabase `users` Table
    res = supabase.table("users").select("*").eq("nfc_uid", clean_uid).execute()
    data = res.data

    # CASE 1: Existing Patient
    if data and len(data) > 0:
        user = data[0]
        if user.get("is_registered"):
            return {
                "success": True,
                "user_type": "EXISTING_USER",
                "message": "Patient found successfully",
                "data": user
            }

    # CASE 2: New NFC Tag Tapped (Insert Temporary Unregistered Entry)
    if not data or len(data) == 0:
        supabase.table("users").insert({
            "nfc_uid": clean_uid,
            "is_registered": False
        }).execute()

    return {
        "success": True,
        "user_type": "NEW_USER",
        "message": "New NFC card detected. Proceed to registration.",
        "data": {"nfc_uid": clean_uid, "is_registered": False}
    }

# --- 🚀 NEW: NFC PATIENT REGISTRATION ENDPOINT ---
@app.post("/api/nfc/register")
async def register_nfc_patient(payload: NFCRegisterRequest):
    clean_uid = payload.nfc_uid.strip().upper()
    
    # Check if Tag exists
    res = supabase.table("users").select("*").eq("nfc_uid", clean_uid).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="NFC Tag not found. Tap tag on reader first.")

    updated_fields = {
        "full_name": payload.full_name,
        "abha_id": payload.abha_id,
        "age": payload.age,
        "gender": payload.gender,
        "phone_number": payload.phone_number,
        "blood_group": payload.blood_group,
        "allergies": payload.allergies,
        "chronic_conditions": payload.chronic_conditions,
        "is_registered": True
    }

    # Update in Supabase
    supabase.table("users").update(updated_fields).eq("nfc_uid", clean_uid).execute()

    return {
        "success": True,
        "message": "Patient linked to NFC Tag successfully!",
        "data": updated_fields
    }

# --- CLINICAL AI CHAT ENDPOINT ---
@app.post("/api/chat/ai-assistant")
async def clinical_ai_chat(payload: ChatRequest):
    global CURRENT_TOKEN_COUNTER
    
    user_msg = payload.user_message.strip()
    history = payload.chat_history
    lang = payload.language or "en"
    details = payload.patient_details or {}
    selected_site = payload.pain_site or "General"

    user_msgs = [m.get("text") for m in history if m.get("sender") == "user"]
    user_msg_count = len(user_msgs)

    # Red-Flag Detection
    msg_lower = user_msg.lower()
    red_flag_keywords = ["chest pain", "chhati me dard", "saans lene me dikkat", "stroke", "severe bleeding", "heavy bleeding", "bleeding", "accident", "trauma", "unconscious"]
    is_emergency = any(kw in msg_lower for kw in red_flag_keywords)

    # Demographic Extraction
    if user_msg_count >= 1 and not details.get("patient_name"):
        details["patient_name"] = clean_patient_name(user_msgs[0])

    if user_msg_count >= 2 and not details.get("age"):
        age_match = re.search(r'\b(\d{1,2})\b', user_msgs[1])
        details["age"] = age_match.group(1) if age_match else user_msgs[1].strip()

    if user_msg_count >= 3 and not details.get("phone"):
        phone_match = re.search(r'\b(\d{10})\b', user_msgs[2])
        details["phone"] = phone_match.group(1) if phone_match else "Not Provided"

    if user_msg_count >= 4 and not details.get("chief_complaint"):
        details["chief_complaint"] = user_msgs[3].strip()

    # Pain Site Detection
    if any(w in msg_lower for w in ["head", "sir", "sar", "headache", "matha", "migraine"]):
        selected_site = "Head"
    elif any(w in msg_lower for w in ["stomach", "pet", "belly", "abdomen", "gastric", "acidity", "vomit"]):
        selected_site = "Abdomen"
    elif any(w in msg_lower for w in ["chest", "chhati", "heart"]):
        selected_site = "Chest"
    elif any(w in msg_lower for w in ["back", "peeth", "spine"]):
        selected_site = "Back"
    elif any(w in msg_lower for w in ["leg", "pair", "knee", "arm", "hand", "limb"]):
        selected_site = "Limbs"

    ai_reply = ""
    poll_options = []

    if is_emergency:
        ai_reply = "🚨 EMERGENCY DETECTED! Intake terminated. Priority token assigned!" if lang == 'en' else "🚨 आपातकालीन स्थिति! आगे के प्रश्न रोके गए। प्राथमिकता टोकन जारी कर दिया गया है!"
    else:
        # Dynamic AI Call using Groq LLM
        if user_msg_count >= 4 and GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_"):
            try:
                formatted_prompt = CLINICAL_SYSTEM_PROMPT.format(lang="Hindi" if lang == "hi" else "English")
                messages = [{"role": "system", "content": formatted_prompt}]
                
                for msg in history:
                    role = "assistant" if msg.get("sender") in ["assistant", "ai"] else "user"
                    messages.append({"role": role, "content": msg.get("text", "")})
                
                messages.append({"role": "user", "content": user_msg})

                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
                body = {
                    "model": "llama-3.1-8b-instant",
                    "messages": messages,
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"}
                }
                res = requests.post(url, headers=headers, json=body, timeout=6)
                if res.status_code == 200:
                    json_res = json.loads(res.json()["choices"][0]["message"]["content"])
                    ai_reply = json_res.get("reply", "")
                    poll_options = json_res.get("poll_options", [])
                    if "[RED_FLAG_ALERT]" in ai_reply:
                        is_emergency = True
                        ai_reply = ai_reply.replace("[RED_FLAG_ALERT]", "").strip()
            except Exception as e:
                print("Groq AI Error:", e)

        # Fallback Intake Flow
        if not ai_reply:
            if user_msg_count == 1:
                ai_reply = "What is your Age?" if lang == 'en' else "आपकी उम्र कितनी है?"
            elif user_msg_count == 2:
                ai_reply = "Please enter your 10-digit Mobile Number." if lang == 'en' else "कृपया अपना 10 अंकों का मोबाइल नंबर दर्ज करें।"
            elif user_msg_count == 3:
                ai_reply = "What main health problem brings you to the hospital today?" if lang == 'en' else "आज आप किस मुख्य स्वास्थ्य समस्या या लक्षण के इलाज के लिए आए हैं?"
            else:
                ai_reply = "Clinical intake complete! Token generated." if lang == 'en' else "नैदानिक चेक-इन पूरा हुआ! टोकन जनरेट हो गया है।"

    # Assign Token
    existing_patient = next((r for r in PATIENT_RECORDS if r.get("patient_name") == details.get("patient_name")), None)
    if existing_patient:
        assigned_token = existing_patient["token"]
    else:
        assigned_token = CURRENT_TOKEN_COUNTER
        CURRENT_TOKEN_COUNTER += 1

    record = {
        "token": assigned_token,
        "patient_name": details.get("patient_name", "Walk-in Patient"),
        "age": details.get("age", "N/A"),
        "phone": details.get("phone", "Not Provided"),
        "chief_complaint": details.get("chief_complaint", user_msg),
        "pain_site": selected_site,
        "severity": "10/10 EMERGENCY" if is_emergency else "5/10",
        "is_red_flag": is_emergency,
        "history": history
    }
    
    PATIENT_RECORDS.append(record)
    await manager.broadcast(record)

    return {
        "status": "success",
        "reply": ai_reply,
        "is_red_flag": is_emergency,
        "poll_options": poll_options,
        "patient_details": details,
        "detected_site": selected_site,
        "assigned_token": assigned_token
    }

@app.get("/api/doctor/summary")
def get_doctor_summary():
    return {"active_queue": PATIENT_RECORDS, "archive": ARCHIVED_RECORDS}
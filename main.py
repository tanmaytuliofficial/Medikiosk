from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://kosmrpnsudxwvqxejbzs.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imtvc21ycG5zdWR4d3ZxeGVqYnpzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODg0NDkyNzUsImV4cCI6MjEwNDAyNTI3NX0.HNmz8QveUDUhldjpUwNP1zhRiTAOm5wrbELPyCya8T0")

supabase: Optional[Client] = None
try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Supabase init warning: {e}")

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

CLINICAL_SYSTEM_PROMPT = """
You are MediKiosk AI, an expert clinical triage assistant conducting a thorough medical intake.
Patient Details Already Known:
Name: {patient_name}
Age: {age}
Phone: {phone}
Gender: {gender}
Blood Group: {blood_group}
Past Medical History / Existing Conditions: {past_history}

Strict Response Language: {lang}.

INSTRUCTIONS:
1. DO NOT ask for demographics like Name, Age, Gender, or Mobile Number (they are already verified).
2. Ask focused, clinical, diagnostic follow-up questions one by one (Aim for 5-6 total interactive turns to gather full symptom history: duration, severity 1-10, trigger factors, associated symptoms like fever/nausea, past episodes).
3. Always provide 4 relevant quick-response options for the patient in poll_options.
4. When you have sufficient clinical details (around 5-6 questions answered) OR if intake is concluding, say "Clinical intake complete! Token generated." in the reply text.
5. If red flag/emergency symptoms occur (e.g. chest pain, severe breathlessness, stroke, heavy bleeding), include "[RED_FLAG_ALERT]" in the reply string.

STRICT JSON OUTPUT FORMAT ONLY:
{{
  "reply": "Your concise clinical question here",
  "poll_options": [
    {{"label": "Option 1", "value": "Option 1"}},
    {{"label": "Option 2", "value": "Option 2"}},
    {{"label": "Option 3", "value": "Option 3"}},
    {{"label": "Option 4", "value": "Option 4"}}
  ]
}}
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

@app.websocket("/ws/doctor-updates")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/nfc/tap")
async def nfc_tap_handler(payload: NFCTapRequest):
    scanned_uid = payload.nfc_uid.strip().upper() if payload.nfc_uid else "DEMO99887766"
    
    try:
        if supabase:
            res = supabase.table("users").select("*").eq("nfc_uid", scanned_uid).execute()
            if res.data and len(res.data) > 0:
                user = res.data[0]
                
                # Fetch past medical history records for Doctor & Groq AI
                history_res = supabase.table("medical_history").select("*").eq("nfc_uid", scanned_uid).execute()
                past_records = history_res.data if history_res.data else []

                return {
                    "success": True,
                    "user_type": "EXISTING_USER",
                    "data": user,
                    "past_history": past_records
                }
    except Exception as e:
        print(f"NFC DB Lookup Warning: {e}")

    return {
        "success": True,
        "user_type": "NEW_USER",
        "data": {"nfc_uid": scanned_uid}
    }

@app.post("/api/nfc/register")
async def register_nfc_patient(payload: NFCRegisterRequest):
    clean_uid = payload.nfc_uid.strip().upper()

    user_data = {
        "nfc_uid": clean_uid,
        "full_name": payload.full_name,
        "abha_id": payload.abha_id or "",
        "age": payload.age,
        "gender": payload.gender,
        "phone_number": payload.phone_number,
        "blood_group": payload.blood_group,
        "allergies": payload.allergies or [],
        "chronic_conditions": payload.chronic_conditions or [],
        "is_registered": True
    }

    if supabase:
        try:
            res = supabase.table("users").select("*").eq("nfc_uid", clean_uid).execute()
            if res.data and len(res.data) > 0:
                supabase.table("users").update(user_data).eq("nfc_uid", clean_uid).execute()
            else:
                supabase.table("users").insert(user_data).execute()
        except Exception as db_err:
            print(f"Supabase registration error: {db_err}")

    return {
        "success": True,
        "message": "Patient linked to NFC Tag successfully!",
        "data": user_data
    }

@app.post("/api/chat/ai-assistant")
async def clinical_ai_chat(payload: ChatRequest):
    global CURRENT_TOKEN_COUNTER
    
    user_msg = payload.user_message.strip()
    history = payload.chat_history
    lang = payload.language or "en"
    details = payload.patient_details or {}
    selected_site = payload.pain_site or "General"

    # Red-Flag Detection
    msg_lower = user_msg.lower()
    red_flag_keywords = ["chest pain", "chhati me dard", "saans lene me dikkat", "stroke", "severe bleeding", "heavy bleeding", "bleeding", "accident", "trauma", "unconscious"]
    is_emergency = any(kw in msg_lower for kw in red_flag_keywords)

    # Pain Site Auto-Detection
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
        # Groq Clinical AI Generation
        if GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_"):
            try:
                formatted_prompt = CLINICAL_SYSTEM_PROMPT.format(
                    patient_name=details.get("patient_name", "Unknown"),
                    age=details.get("age", "Unknown"),
                    phone=details.get("phone", "Unknown"),
                    gender=details.get("gender", "Unknown"),
                    blood_group=details.get("blood_group", "Unknown"),
                    past_history=json.dumps(details.get("past_history", [])),
                    lang="Hindi" if lang == "hi" else "English"
                )

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
                res = requests.post(url, headers=headers, json=body, timeout=8)
                if res.status_code == 200:
                    json_res = json.loads(res.json()["choices"][0]["message"]["content"])
                    ai_reply = json_res.get("reply", "")
                    poll_options = json_res.get("poll_options", [])
                    if "[RED_FLAG_ALERT]" in ai_reply:
                        is_emergency = True
                        ai_reply = ai_reply.replace("[RED_FLAG_ALERT]", "").strip()
            except Exception as e:
                print("Groq AI Engine Error:", e)

        # Fallback Dynamic Diagnostic Response if AI key missing/fails
        if not ai_reply:
            user_turns = len([m for m in history if m.get("sender") == "user"])
            if user_turns == 0:
                ai_reply = "How long have you been experiencing this symptom?" if lang == 'en' else "आपको यह लक्षण कितने समय से महसूस हो रहा है?"
                poll_options = [{"label": "Since Today", "value": "Since Today"}, {"label": "2-3 Days", "value": "2-3 Days"}, {"label": "1 Week", "value": "1 Week"}, {"label": "More than a month", "value": "More than a month"}]
            elif user_turns == 1:
                ai_reply = "On a scale of 1 to 10, how severe is your pain or discomfort?" if lang == 'en' else "1 से 10 के पैमाने पर, आपका दर्द कितना गंभीर है?"
                poll_options = [{"label": "Mild (1-3)", "value": "Mild (1-3)"}, {"label": "Moderate (4-6)", "value": "Moderate (4-6)"}, {"label": "Severe (7-9)", "value": "Severe (7-9)"}, {"label": "Unbearable (10)", "value": "Unbearable (10)"}]
            elif user_turns == 2:
                ai_reply = "Do you have any associated symptoms like fever, nausea, or dizziness?" if lang == 'en' else "क्या आपको बुखार, जी मिचलाना या चक्कर आने जैसे अन्य लक्षण भी हैं?"
                poll_options = [{"label": "Fever", "value": "Fever"}, {"label": "Nausea / Vomiting", "value": "Nausea / Vomiting"}, {"label": "Dizziness", "value": "Dizziness"}, {"label": "None of these", "value": "None"}]
            elif user_turns == 3:
                ai_reply = "Are you currently taking any prescription medications for this issue?" if lang == 'en' else "क्या आप वर्तमान में इस समस्या के लिए कोई दवा ले रहे हैं?"
                poll_options = [{"label": "Painkillers", "value": "Painkillers"}, {"label": "Antibiotics", "value": "Antibiotics"}, {"label": "Regular BP/Diabetes meds", "value": "Regular BP/Diabetes meds"}, {"label": "No medication", "value": "No medication"}]
            else:
                ai_reply = "Clinical intake complete! Token generated." if lang == 'en' else "नैदानिक चेक-इन पूरा हुआ! टोकन जनरेट हो गया है।"

    # Assign OPD Token
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
        "blood_group": details.get("blood_group", "N/A"),
        "gender": details.get("gender", "N/A"),
        "chief_complaint": history[0]["text"] if history else user_msg,
        "pain_site": selected_site,
        "severity": "10/10 EMERGENCY" if is_emergency else "5/10",
        "is_red_flag": is_emergency,
        "history": history,
        "past_history": details.get("past_history", [])
    }
    
    # Save/Update in Active OPD Queue
    PATIENT_RECORDS = [r for r in PATIENT_RECORDS if r["token"] != assigned_token]
    PATIENT_RECORDS.append(record)
    
    # Save to Supabase Medical History table if connected
    if supabase and (ai_reply.startswith("Clinical intake complete") or is_emergency):
        try:
            supabase.table("medical_history").insert({
                "nfc_uid": details.get("nfc_uid", ""),
                "patient_name": record["patient_name"],
                "chief_complaint": record["chief_complaint"],
                "consultation_notes": json.dumps(history),
                "pain_site": selected_site,
                "is_red_flag": is_emergency
            }).execute()
        except Exception as err:
            print(f"Medical history DB save warning: {err}")

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
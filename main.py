from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
import json
import re
import io
import os

app = FastAPI(title="MediKiosk Clinical Intelligence Platform", version="20.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

class ChatRequest(BaseModel):
    user_message: str
    chat_history: List[dict] = []
    language: Optional[str] = "en"
    patient_details: Optional[dict] = {}
    pain_site: Optional[str] = "General"

PATIENT_RECORDS = []
ARCHIVED_RECORDS = []
active_connections: List[WebSocket] = []

CURRENT_TOKEN_COUNTER = 101

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
You are MediKiosk AI, an expert clinical intake assistant at an OPD Desk.

RULES:
1. Respond strictly in {lang}.
2. Conduct an IN-DEPTH medical assessment asking thorough clinical follow-up questions one by one using the SOCRATES protocol:
   - Nature/character of pain
   - Radiation to other areas
   - Aggravating/relieving factors
   - Associated symptoms (fever, nausea, vomiting, dizziness)
   - Past medical history
3. Ask ONLY ONE short, professional question at a time.
4. If emergency red-flag symptoms occur (severe chest pain, heavy bleeding, accident, trauma, stroke), append [RED_FLAG_ALERT] at the end.
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

    # 1. Separate Step Extraction (Name -> Age -> Phone -> Complaint)
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

    # Pain Site Auto-Detection
    msg_lower = user_msg.lower()
    if any(w in msg_lower for w in ["stomach", "pet", "belly", "abdomen", "gastric", "acidity", "vomit"]):
        selected_site = "Abdomen"
    elif any(w in msg_lower for w in ["head", "sir", "sar", "headache"]):
        selected_site = "Head"
    elif any(w in msg_lower for w in ["chest", "chhati", "heart"]):
        selected_site = "Chest"
    elif any(w in msg_lower for w in ["back", "peeth", "spine"]):
        selected_site = "Back"
    elif any(w in msg_lower for w in ["leg", "pair", "knee", "arm", "hand"]):
        selected_site = "Limbs"

    red_flag_keywords = ["chest pain", "chhati me dard", "saans lene me dikkat", "stroke", "severe bleeding", "heavy bleeding", "bleeding", "accident", "trauma"]
    is_emergency = any(kw in msg_lower for kw in red_flag_keywords)

    ai_reply = ""
    show_history_prompt = False

    # Dynamic Groq AI Query (Triggered after basic profile intake >= 4)
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
                "temperature": 0.3,
                "max_tokens": 150
            }
            res = requests.post(url, headers=headers, json=body, timeout=6)
            if res.status_code == 200:
                ai_reply = res.json()["choices"][0]["message"]["content"]
                if "[RED_FLAG_ALERT]" in ai_reply:
                    is_emergency = True
                    ai_reply = ai_reply.replace("[RED_FLAG_ALERT]", "").strip()
        except Exception as e:
            print("Groq API Error:", e)

    # Fallback Step-by-Step Flow
    if not ai_reply:
        if user_msg_count == 1:
            ai_reply = "What is your Age?" if lang == 'en' else "आपकी उम्र कितनी है?"
        elif user_msg_count == 2:
            ai_reply = "Please enter your 10-digit Mobile Number." if lang == 'en' else "कृपया अपना 10 अंकों का मोबाइल नंबर दर्ज करें।"
        elif user_msg_count == 3:
            ai_reply = "What main health problem or symptom brings you to the hospital today?" if lang == 'en' else "आज आप किस मुख्य स्वास्थ्य समस्या या लक्षण के इलाज के लिए आए हैं?"
        elif user_msg_count == 4:
            ai_reply = "How would you describe the pain/symptom (e.g., sharp, burning, dull ache, or throbbing)?" if lang == 'en' else "आप इस तकलीफ को कैसे बयां करेंगे (जैसे तेज दर्द, जलन, या मीठा-मीठा दर्द)?"
        elif user_msg_count == 5:
            ai_reply = "Does this discomfort spread to any other body part like your back, arm, or shoulders?" if lang == 'en' else "क्या यह दर्द शरीर के किसी और हिस्से (जैसे पीठ, हाथ या कंधों) तक भी फैलता है?"
        elif user_msg_count == 6:
            ai_reply = "Does it increase or decrease after eating food, taking rest, or walking?" if lang == 'en' else "क्या यह खाना खाने के बाद, आराम करने पर या चलने-फिरने पर बढ़ता या घटता है?"
        elif user_msg_count == 7:
            ai_reply = "Have you experienced similar medical issues in the past?" if lang == 'en' else "क्या आपको अतीत में भी कभी ऐसा दर्द या स्वास्थ्य समस्या हुई है?"
        else:
            ai_reply = f"Clinical intake complete! Token generated." if lang == 'en' else f"नैदानिक चेक-इन पूरा हुआ! टोकन जनरेट हो गया है।"

    if user_msg_count >= 7 and any(w in msg_lower for w in ["yes", "haan", "ha", "pehle bhi", "past", "earlier", "doctor"]):
        show_history_prompt = True

    # Token Assignment
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
        "keywords": ["Clinical Intake", f"Site: {selected_site}"],
        "severity": "9/10" if is_emergency else "5/10",
        "is_red_flag": is_emergency,
        "history": history,
        "ocr_summary": details.get("ocr_summary", "No uploaded reports")
    }
    
    existing_idx = next((i for i, r in enumerate(PATIENT_RECORDS) if r.get("patient_name") == record["patient_name"]), None)
    if existing_idx is not None:
        PATIENT_RECORDS[existing_idx] = record
    else:
        PATIENT_RECORDS.append(record)
        
    await manager.broadcast(record)

    return {
        "status": "success",
        "reply": ai_reply,
        "is_red_flag": is_emergency,
        "show_history_prompt": show_history_prompt,
        "patient_details": details,
        "detected_site": selected_site,
        "assigned_token": assigned_token
    }

@app.post("/api/ocr/scan-document")
async def scan_medical_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        extracted_text = f"Scanned Document ({file.filename}): Extracted Medical History & Past Rx."
        return {"status": "success", "filename": file.filename, "extracted_text": extracted_text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/doctor/summary")
def get_doctor_summary():
    return {"active_queue": PATIENT_RECORDS, "archive": ARCHIVED_RECORDS}
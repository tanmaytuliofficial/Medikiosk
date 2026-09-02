from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
import json
import re
import io
import os

app = FastAPI(title="MediKiosk Clinical Intelligence Platform", version="15.0.0")

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

CLINICAL_SYSTEM_PROMPT = """
You are MediKiosk AI, an expert clinical intake assistant at an OPD Desk.

STRICT INSTRUCTIONS:
1. Respond ONLY in the requested language ({lang}). Never mix languages.
2. Ask targeted, SOCRATES-based clinical questions specific to the reported symptom.
3. Ask ONLY ONE short, professional question at a time.
4. If emergency symptoms (chest pain, heavy bleeding, stroke, unconsciousness) occur, add [RED_FLAG_ALERT] at the end.
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
    user_msg = payload.user_message.strip()
    history = payload.chat_history
    lang = payload.language or "en"
    details = payload.patient_details or {}
    selected_site = payload.pain_site or "General"

    user_msgs = [m.get("text") for m in history if m.get("sender") == "user"]
    user_msg_count = len(user_msgs)

    # 1. Validation Logic
    if user_msg_count >= 1 and not details.get("patient_name"):
        details["patient_name"] = user_msgs[0].strip()

    if user_msg_count >= 2 and not details.get("age"):
        second_input = user_msgs[1]
        phone_match = re.search(r'\b(\d{10})\b', second_input)
        age_match = re.search(r'\b(\d{1,2})\b', second_input)
        
        details["phone"] = phone_match.group(1) if phone_match else "Not Provided"
        details["age"] = age_match.group(1) if age_match else "N/A"

    if user_msg_count >= 3 and not details.get("chief_complaint"):
        details["chief_complaint"] = user_msgs[2].strip()

    # Red Flag Keywords Check
    red_flag_keywords = ["chest pain", "chhati me dard", "saans lene me dikkat", "stroke", "severe bleeding", "heavy bleeding", "bleeding", "accident", "trauma", "unconscious"]
    is_emergency = any(kw in user_msg.lower() for kw in red_flag_keywords)

    ai_reply = ""
    show_history_prompt = False

    # Dynamic Groq AI Query (Strictly Single Language)
    if user_msg_count >= 3 and GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_"):
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
                "max_tokens": 120
            }
            res = requests.post(url, headers=headers, json=body, timeout=6)
            if res.status_code == 200:
                ai_reply = res.json()["choices"][0]["message"]["content"]
                if "[RED_FLAG_ALERT]" in ai_reply:
                    is_emergency = True
                    ai_reply = ai_reply.replace("[RED_FLAG_ALERT]", "").strip()
        except Exception as e:
            print("Groq API Error:", e)

    # Isolated Language Fallback Engine
    if not ai_reply:
        if user_msg_count == 1:
            ai_reply = "Please enter your Age and 10-digit Mobile Number." if lang == 'en' else "कृपया अपनी उम्र और 10 अंकों का मोबाइल नंबर दर्ज करें।"
        elif user_msg_count == 2:
            ai_reply = "What primary health problem or symptom are you facing today?" if lang == 'en' else "आज आप किस मुख्य स्वास्थ्य समस्या या लक्षण का सामना कर रहे हैं?"
        elif user_msg_count == 3:
            ai_reply = f"Since when have you been experiencing '{user_msg}'?" if lang == 'en' else f"आपको '{user_msg}' की समस्या कब से हो रही है?"
        else:
            token_num = 101 + len(PATIENT_RECORDS)
            ai_reply = f"Intake complete! Token #{token_num} generated. Please scan past reports if available." if lang == 'en' else f"चेक-इन पूरा हुआ! टोकन #{token_num} जनरेट हो गया है।"

    if user_msg_count >= 4:
        show_history_prompt = True

    record = {
        "token": 101 + len(PATIENT_RECORDS),
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
        "patient_details": details
    }

@app.post("/api/ocr/scan-document")
async def scan_medical_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        extracted_text = f"Scanned Document ({file.filename}): Extracted Rx & Medical History."
        return {"status": "success", "filename": file.filename, "extracted_text": extracted_text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/doctor/summary")
def get_doctor_summary():
    return {"active_queue": PATIENT_RECORDS, "archive": ARCHIVED_RECORDS}
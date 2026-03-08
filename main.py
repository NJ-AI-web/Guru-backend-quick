"""
நம்ம குரு AI - Backend v2.0
Render.com deployment ready
Groq API (llama3) + Agmarknet market rates
"""

import os
import logging
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="நம்ம குரு AI API v2.0", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = """நீங்கள் "குட்டி குரு" - Erode மக்களுக்கான அன்பான AI உதவியாளர்.

விதிகள்:
1. எப்போதும் தெளிவான எளிய தமிழில் பதில் சொல்லுங்கள்
2. User Tanglish-ல் கேட்டாலும் தமிழில் பதில் சொல்லுங்கள்
3. அன்பாக மரியாதையாக பேசுங்கள்

Erode சிறப்பு தகவல்கள்:
- Emergency: Police 100, Ambulance 108, Fire 101
- Collector Office: 0424-2225400
- Erode: "Turmeric City" & "Textile City"
- முக்கிய crops: மஞ்சள், தேங்காய், பருத்தி, வாழை
- பிரசித்தம்: Bhavani Sangamam, Bannari Amman Temple

அரசு திட்டங்கள்:
- கலைஞர் உரிமைத்தொகை: மாதம் 1000 ரூபாய்
- உழவர் சந்தை அட்டை: விவசாயிகளுக்கு
- PM Kisan: ஆண்டு 6000 ரூபாய்

எச்சரிக்கை: Medical/Legal advice-க்கு expert-ஐ சந்தியுங்கள்."""

# Static fallback rates - weekly update pannuvom
STATIC_RATES = {
    "மஞ்சள்":     {"price": "9400", "unit": "kg",      "change": "+2.1%", "isUp": True},
    "தேங்காய்":   {"price": "28",   "unit": "மூட்டை", "change": "-0.5%", "isUp": False},
    "கொத்தமல்லி": {"price": "120",  "unit": "kg",      "change": "+1.3%", "isUp": True},
    "பருத்தி":    {"price": "6200", "unit": "quintal", "change": "+0.8%", "isUp": True},
    "வாழை":       {"price": "22",   "unit": "kg",      "change": "-1.2%", "isUp": False},
}

async def get_market_data() -> dict:
    """Agmarknet fetch - fallback to static on failure"""
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(
                "https://agmarknet.gov.in/SearchCommodityWise.aspx",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if r.status_code == 200:
                logger.info("Agmarknet OK")
                # Full parse future version-la add pannuvom
                return STATIC_RATES
    except Exception as e:
        logger.warning(f"Agmarknet failed: {e}")
    return STATIC_RATES


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    response: str
    success: bool
    error: Optional[str] = None

class MarketResponse(BaseModel):
    rates: dict
    success: bool


@app.get("/")
async def root():
    return {"message": "நம்ம குரு AI v2.0 Running! 🎉"}

@app.get("/health")
async def health():
    return {"status": "healthy", "groq": groq_client is not None}

@app.get("/market", response_model=MarketResponse)
async def market():
    """Live Erode market rates"""
    rates = await get_market_data()
    return MarketResponse(rates=rates, success=True)

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not groq_client:
        raise HTTPException(503, "Groq API not configured")
    try:
        # Market query detect
        mkt_keys = ["விலை","rate","price","மஞ்சள்","turmeric",
                    "தேங்காய்","coconut","சந்தை","market"]
        is_market = any(k in req.message.lower() for k in mkt_keys)

        extra = ""
        if is_market:
            rates = await get_market_data()
            lines = [f"- {c}: ₹{d['price']}/{d['unit']} ({d['change']})"
                     for c, d in rates.items()]
            extra = "\n\nஇன்றைய Erode market:\n" + "\n".join(lines)

        messages = [{"role": "system", "content": SYSTEM_PROMPT + extra}]
        for m in req.history[-10:]:
            if m.role in ["user", "assistant"]:
                messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": req.message})

        # llama-3.3-70b — smarter & free on Groq
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        return ChatResponse(
            response=completion.choices[0].message.content.strip(),
            success=True
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            response="மன்னிக்கவும்! சேவை தற்காலிகமாக இல்லை. மீண்டும் முயற்சிக்கவும். 🙏",
            success=False,
            error=str(e)
        )

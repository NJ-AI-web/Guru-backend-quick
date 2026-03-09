"""
நம்ம குரு AI - Backend v3.0
Groq Llama 3.3 + Web Search Tool
Real-time Erode market rates, news, bus info
"""

import os
import json
import logging
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="நம்ம குரு AI v3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ── System Prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = """நீங்கள் "குட்டி குரு" - Erode மக்களுக்கான அன்பான AI உதவியாளர்.

விதிகள்:
1. எப்போதும் தெளிவான எளிய தமிழில் பதில் சொல்லுங்கள்
2. User Tanglish-ல் கேட்டாலும் தமிழில் பதில் சொல்லுங்கள்
3. அன்பாக மரியாதையாக பேசுங்கள்
4. Real-time தகவல் கேட்டால் web search tool use பண்ணுங்கள்

Erode சிறப்பு தகவல்கள்:
- Emergency: Police 100, Ambulance 108, Fire 101
- Collector Office: 0424-2225400
- Erode: "Turmeric City" & "Textile City"
- முக்கிய crops: மஞ்சள், தேங்காய், பருத்தி, வாழை

அரசு திட்டங்கள்:
- கலைஞர் உரிமைத்தொகை: மாதம் 1000 ரூபாய்
- உழவர் சந்தை அட்டை: விவசாயிகளுக்கு
- PM Kisan: ஆண்டு 6000 ரூபாய்

எச்சரிக்கை: Medical/Legal advice-க்கு expert-ஐ சந்தியுங்கள்."""

# ── Groq Web Search Tool Definition ───────────────────────────
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the internet for real-time information like market prices, news, bus timings, weather, government updates for Erode and Tamil Nadu",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query in English for best results. E.g. 'Erode turmeric price today', 'Tamil Nadu government scheme 2024'"
                }
            },
            "required": ["query"]
        }
    }
}

# ── DuckDuckGo search (free, no API key needed) ────────────────
async def do_web_search(query: str) -> str:
    """Real web search using DuckDuckGo instant answers API"""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            # DuckDuckGo Instant Answer API - completely free
            r = await client.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1"
                },
                headers={"User-Agent": "NammaGuruAI/1.0"}
            )
            if r.status_code == 200:
                data = r.json()
                results = []

                # Abstract (main answer)
                if data.get("AbstractText"):
                    results.append(f"Answer: {data['AbstractText']}")

                # Related topics
                for topic in data.get("RelatedTopics", [])[:3]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append(topic["Text"])

                if results:
                    return "\n".join(results[:4])

            return f"Search completed for: {query}. Please provide based on your knowledge."

    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Search failed for '{query}'. Use your training knowledge."


# ── Models ─────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    response: str
    success: bool
    searched: bool = False
    error: Optional[str] = None

class MarketResponse(BaseModel):
    rates: dict
    success: bool


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "நம்ம குரு AI v3.0 - Live! 🎉", "websearch": True}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "groq": groq_client is not None,
        "model": "llama-3.3-70b-versatile",
        "websearch": True
    }

@app.get("/market", response_model=MarketResponse)
async def market():
    """Live market rates via web search"""
    query = "Erode turmeric market price today 2024"
    result = await do_web_search(query)
    return MarketResponse(
        rates={"search_result": result},
        success=True
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat - Llama 3.3 with web search tool.
    AI decides when to search automatically.
    """
    if not groq_client:
        raise HTTPException(503, "Groq API not configured")

    try:
        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in req.history[-10:]:
            if m.role in ["user", "assistant"]:
                messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": req.message})

        searched = False

        # ── Step 1: First call with tool ──────────────────────
        response1 = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=[WEB_SEARCH_TOOL],
            tool_choice="auto",   # AI decides when to search
            max_tokens=1024,
            temperature=0.7,
        )

        msg = response1.choices[0].message

        # ── Step 2: If AI wants to search ────────────────────
        if msg.tool_calls:
            searched = True
            tool_call = msg.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            query = args.get("query", req.message)

            logger.info(f"Web search: {query}")
            search_result = await do_web_search(query)

            # Add tool result to messages
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": tool_call.function.arguments
                    }
                }]
            })
            messages.append({
                "role": "tool",
                "content": search_result,
                "tool_call_id": tool_call.id
            })

            # ── Step 3: Final response with search data ───────
            response2 = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
            )
            final_text = response2.choices[0].message.content.strip()

        else:
            # No search needed - direct answer
            final_text = msg.content.strip() if msg.content else \
                "மன்னிக்கவும், பதில் கிடைக்கவில்லை."

        logger.info(f"Done. Searched: {searched}")
        return ChatResponse(
            response=final_text,
            success=True,
            searched=searched
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            response="மன்னிக்கவும்! சேவை தற்காலிகமாக இல்லை. மீண்டும் முயற்சிக்கவும். 🙏",
            success=False,
            error=str(e)
        )

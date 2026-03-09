"""
நம்ம குரு AI - Backend v8.0
Groq Llama 3.3 + Pure BeautifulSoup4
DuckDuckGo Lite scraping - No API limits, No blocks!
100% Free, Unlimited!
"""

import os
import json
import logging
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="நம்ம குரு AI v8.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client  = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ── System Prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = """நீங்கள் "குட்டி குரு" - Erode மக்களுக்கான அன்பான AI உதவியாளர்.

விதிகள்:
1. எப்போதும் தெளிவான எளிய தமிழில் பதில் சொல்லுங்கள்
2. User Tanglish-ல் கேட்டாலும் தமிழில் பதில் சொல்லுங்கள்
3. Real-time data கேட்டால் web_search tool use பண்ணுங்கள்
4. Search result கிடைத்தால் clearly சொல்லுங்கள்
5. தெரியாத விஷயம் "தெரியவில்லை" சொல்லுங்கள்

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

# ── Tool Definition ────────────────────────────────────────────
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search live internet for real-time info. "
            "Use for: gold rate, petrol price, market rates, "
            "news, weather, bus timings, govt updates "
            "for Erode and Tamil Nadu. "
            "Use for ANY today/current/live/price question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "English search query. "
                        "E.g. 'Erode gold rate today', "
                        "'Tamil Nadu petrol price today', "
                        "'Erode turmeric market price today'"
                    )
                }
            },
            "required": ["query"]
        }
    }
}

# ── Pure BeautifulSoup Search (CTO Master Plan) ───────────────
def live_web_search(query: str) -> str:
    """
    Directly scrapes DuckDuckGo Lite using pure BeautifulSoup.
    Acts like a real human browser - POST request.
    No duckduckgo-search library, no IP blocks!
    100% Free & Unlimited!
    """
    try:
        logger.info(f"🔍 Pure BS4 Search: {query}")

        url = "https://lite.duckduckgo.com/lite/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        data = {"q": query}

        # Real human-like POST request
        response = requests.post(
            url,
            headers=headers,
            data=data,
            timeout=10
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract result snippets
        snippets = []
        for tr in soup.find_all("tr"):
            td = tr.find("td", class_="result-snippet")
            if td:
                text = td.get_text(strip=True)
                if text and len(text) > 20:
                    snippets.append(text)

        if not snippets:
            # Try alternate extraction
            for span in soup.find_all("span", class_="snippet"):
                text = span.get_text(strip=True)
                if text:
                    snippets.append(text)

        if not snippets:
            logger.warning("No snippets found in DDG Lite")
            return f"'{query}' பற்றிய live தகவல் கிடைக்கவில்லை."

        combined = "\n\n---\n\n".join(snippets[:4])
        logger.info(f"✅ Pure BS4 done. {len(snippets)} results found.")
        return combined

    except Exception as e:
        logger.error(f"BS4 Search error: {e}")
        return f"தேடல் தோல்வி. Error: {str(e)[:100]}"


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


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "நம்ம குரு AI v8.0 🎉",
        "search": "Pure BeautifulSoup4 - Free & Unlimited!",
        "model": "llama-3.3-70b-versatile"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "groq": groq_client is not None,
        "search": "Pure BS4 - DDG Lite",
        "model": "llama-3.3-70b-versatile"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat - Llama 3.3 + Pure BS4 Search.
    Human-like scraping, no blocks, no limits!
    """
    if not groq_client:
        raise HTTPException(503, "Groq API not configured")

    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in req.history[-10:]:
            if m.role in ["user", "assistant"]:
                messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": req.message})

        searched = False

        # ── Step 1: AI decides if search needed ───────────────
        resp1 = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=[WEB_SEARCH_TOOL],
            tool_choice="auto",
            max_tokens=1024,
            temperature=0.2,
        )

        msg1 = resp1.choices[0].message

        # ── Step 2: Run Pure BS4 search if needed ─────────────
        if msg1.tool_calls:
            searched = True
            tool_call = msg1.tool_calls[0]
            args      = json.loads(tool_call.function.arguments)
            query     = args.get("query", req.message)

            logger.info(f"🔍 Searching: {query}")
            search_result = live_web_search(query)

            messages.append({
                "role": "assistant",
                "content": msg1.content or "",
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

            # ── Step 3: Final Tamil answer ─────────────────────
            resp2 = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=1024,
                temperature=0.2,
            )
            final = resp2.choices[0].message.content.strip()

        else:
            final = msg1.content.strip() if msg1.content else \
                "மன்னிக்கவும், பதில் கிடைக்கவில்லை."

        return ChatResponse(
            response=final,
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

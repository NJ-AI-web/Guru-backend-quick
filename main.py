"""
நம்ம குரு AI - Backend v5.0
Groq Llama 3.3 + DuckDuckGo + BeautifulSoup4
100% Free, Unlimited, Real-time Search
Zero cost - Zero hallucination!
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
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="நம்ம குரு AI v5.0")

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
3. Real-time data கேட்டால் web_search tool உபயோகிக்கவும்
4. Search result கிடைத்தால் clearly சொல்லுங்கள்
5. தெரியாத விஷயம் "தெரியவில்லை" சொல்லுங்கள் - கதை வேண்டாம்!

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

# ── Web Search Tool Definition ─────────────────────────────────
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search live internet for real-time information. "
            "Use for: gold rate, petrol price, market rates, "
            "news, weather, bus timings, govt schemes, "
            "any TODAY/CURRENT/LIVE/PRICE questions about "
            "Erode and Tamil Nadu."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "English search query for best results. "
                        "Examples: 'Erode gold rate today 2025', "
                        "'Tamil Nadu petrol price today', "
                        "'Erode turmeric market rate today'"
                    )
                }
            },
            "required": ["query"]
        }
    }
}

# ── BeautifulSoup Scraper ──────────────────────────────────────
def scrape_url(url: str, max_chars: int = 800) -> str:
    """Scrape text content from a URL using BeautifulSoup"""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        # Remove unwanted tags
        for tag in soup(["script", "style", "nav", "footer",
                         "header", "ads", "iframe"]):
            tag.decompose()

        # Get clean text
        text = soup.get_text(separator=" ", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = " ".join(lines)

        return clean_text[:max_chars]

    except Exception as e:
        logger.warning(f"Scrape failed for {url}: {e}")
        return ""


# ── DuckDuckGo + BeautifulSoup Search ─────────────────────────
def live_web_search(query: str) -> str:
    """
    Step 1: DuckDuckGo → get top URLs
    Step 2: BeautifulSoup → scrape content
    Step 3: Return combined real text to AI
    """
    try:
        logger.info(f"🔍 DDG Search: {query}")

        # Step 1: DuckDuckGo search - get top 3 results
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=3,
                region="in-en",        # India English results
                safesearch="moderate"
            ))

        if not results:
            logger.warning("No DDG results found")
            return f"'{query}' பற்றிய தகவல் கிடைக்கவில்லை."

        logger.info(f"📋 Found {len(results)} results")

        # Step 2: Scrape each URL
        all_content = []

        for i, result in enumerate(results[:3]):
            url   = result.get("href", "")
            title = result.get("title", "")
            body  = result.get("body", "")  # DDG snippet

            logger.info(f"  [{i+1}] {title[:50]}...")

            # Use DDG snippet first (fast)
            if body and len(body) > 50:
                all_content.append(f"Source {i+1}: {title}\n{body}")

            # Try to scrape for richer content
            if url and i < 2:  # Only scrape top 2 URLs
                scraped = scrape_url(url, max_chars=600)
                if scraped and len(scraped) > 100:
                    all_content.append(
                        f"Full content from {title}:\n{scraped}"
                    )

        if not all_content:
            return f"'{query}' search முடிந்தது ஆனால் content கிடைக்கவில்லை."

        combined = "\n\n---\n\n".join(all_content[:4])
        logger.info(f"✅ Search complete. Content: {len(combined)} chars")
        return combined

    except Exception as e:
        logger.error(f"Search error: {e}")
        return (
            f"தேடல் தற்காலிகமாக தோல்வி. "
            f"Error: {str(e)[:100]}"
        )


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
        "message": "நம்ம குரு AI v5.0 🎉",
        "search": "DuckDuckGo + BeautifulSoup4",
        "cost": "100% FREE - Unlimited!"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "groq": groq_client is not None,
        "model": "llama-3.3-70b-versatile",
        "search": "DDG + BS4 - Free & Unlimited"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat - Llama 3.3 + DDG + BeautifulSoup
    AI searches live internet, no hallucination!
    """
    if not groq_client:
        raise HTTPException(503, "Groq API not configured")

    try:
        # Build conversation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in req.history[-10:]:
            if m.role in ["user", "assistant"]:
                messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": req.message})

        searched = False

        # ── Step 1: Ask AI if search needed ──────────────────
        resp1 = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=[WEB_SEARCH_TOOL],
            tool_choice="auto",
            max_tokens=1024,
            temperature=0.2,  # Low = accurate, no hallucination
        )

        msg1 = resp1.choices[0].message

        # ── Step 2: Execute search if AI requests ────────────
        if msg1.tool_calls:
            searched = True
            tool_call = msg1.tool_calls[0]
            args      = json.loads(tool_call.function.arguments)
            query     = args.get("query", req.message)

            # 🔍 DDG Search + BS4 Scrape
            search_content = live_web_search(query)

            # Add to conversation
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
                "content": search_content,
                "tool_call_id": tool_call.id
            })

            # ── Step 3: Final answer with real data ───────────
            resp2 = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=1024,
                temperature=0.2,
            )
            final_text = resp2.choices[0].message.content.strip()

        else:
            # No search needed
            final_text = msg1.content.strip() if msg1.content else \
                "மன்னிக்கவும், பதில் கிடைக்கவில்லை."

        return ChatResponse(
            response=final_text,
            success=True,
            searched=searched
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            response=(
                "மன்னிக்கவும்! சேவை தற்காலிகமாக இல்லை. "
                "மீண்டும் முயற்சிக்கவும். 🙏"
            ),
            success=False,
            error=str(e)
        )

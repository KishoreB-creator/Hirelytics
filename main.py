import os, re, json, traceback
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import re

from fastapi.middleware.cors import CORSMiddleware

# ------------------ CONFIG ------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
emb_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

app = FastAPI(title="SHL Assessment Recommendation API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend origin for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ SCHEMAS ------------------
class RecommendRequest(BaseModel):
    query: Optional[str] = Field(default=None, description="Free-text query or JD")
    jd_url: Optional[HttpUrl] = Field(default=None, description="URL pointing to a JD")
    k: int = Field(default=10, ge=1, le=10, description="Number of results (1..10)")

# ------------------ UTILS ------------------
def fetch_url_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return text[:10000]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch JD URL: {e}")

def make_query_text(req: RecommendRequest) -> str:
    if req.query and req.query.strip():
        return req.query.strip()
    if req.jd_url:
        return fetch_url_text(str(req.jd_url))
    raise HTTPException(status_code=422, detail="Provide either 'query' or 'jd_url'.")

def embed_query(q: str):
    return emb_model.encode(q, normalize_embeddings=True).tolist()

def call_match_products(q_emb: List[float], count: int, types: Optional[List[str]] = None):
    payload = {
        "query_embedding": q_emb,
        "match_count": max(count, 10),
        "match_threshold": 0.0
    }
    res = supabase.rpc("match_products_v4", payload).execute()
    return res.data or []

def process_query_text(req: RecommendRequest) -> str:
    text = make_query_text(req)
    if len(text) <= 1500:
        return text
    
    # If query is too long, summarize it using Gemini before embedding
    print(f"⚙️ Summarizing long query ({len(text)} chars)...")
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        summary_prompt = f"""
        Summarize this job description or text in 5-6 concise sentences 
        focusing on required skills, role, and traits. 
        Only output the summary text.

        Text:
        \"\"\"{text}\"\"\"
        """
        response = model.generate_content(summary_prompt)
        summary = response.text.strip()
        print(f"✅ Summarized to {len(summary)} chars.")
        return summary
    except Exception as e:
        print(f"⚠️ Gemini summarization failed -> {e}")
        return text[:1500]  # fallback

def compute_keyword_overlap(query: str, item_name: str, item_desc: str) -> float:
    """Compute keyword overlap ratio between query and item text."""
    query_tokens = set(re.findall(r"\b\w+\b", query.lower()))
    item_tokens = set(re.findall(r"\b\w+\b", (item_name + " " + item_desc).lower()))
    if not query_tokens:
        return 0.0
    overlap = query_tokens.intersection(item_tokens)
    return len(overlap) / len(query_tokens)


# --- SHL Test Type Mapping ---
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
    
}

def map_test_type(code: str):
    if not code:
        return ["General"]
    if isinstance(code, str):
        code = code.strip().upper()
        if " " in code:
            mapped = [TEST_TYPE_MAP.get(c.strip(), "General") for c in code.split(",")]
            return mapped
        return [TEST_TYPE_MAP.get(code, "General")]
    return ["General"]

# ------------------ ENDPOINTS ------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    query_text = process_query_text(req)
    q_emb = embed_query(query_text)

    # --- Retrieve matches from Supabase ---
    retrieved = call_match_products(q_emb, count=max(40, req.k))
    print(f"🔍 Retrieved {len(retrieved)} results from Supabase")

    if not retrieved:
        return {"recommended_assessments": []}

    # --- Hybrid scoring (vector + keyword overlap) ---
    for r in retrieved:
        vector_score = r.get("score", 0)
        overlap_score = compute_keyword_overlap(query_text, r.get("name", ""), r.get("description", ""))
        r["final_score"] = 0.7 * vector_score + 0.3 * overlap_score

    # Sort by hybrid score
    retrieved = sorted(retrieved, key=lambda x: x.get("final_score", 0), reverse=True)

    # --- Build lookup dict for name-based access ---
    name_to_data = {r["name"].strip().lower(): r for r in retrieved if r.get("name")}

    # --- Prepare Gemini context ---
    context_lines = [
        f"- Name: {r['name']}\n  Type: {r.get('test_type', 'N/A')}\n  Desc: {r.get('description', '')[:300]}"
        for r in retrieved[:20] if r.get("name") and r.get("description")
    ]
    context_text = "\n".join(context_lines)

    # --- Gemini prompt for re-ranking ---
    prompt = f"""
You are an SHL Assessment Expert.

User hiring context:
\"\"\"{query_text}\"\"\"

You are given a list of 20 candidate SHL assessments with names, types, and descriptions.
Your task is to **rank them** (do not remove any unless irrelevant).

Criteria:
- Match technical, behavioral, or cognitive relevance to the hiring context.
- Prefer diversity across assessment types (Knowledge, Personality, Cognitive, etc.)
- Output the **top 10 ranked assessments** in JSON.

Return strictly valid JSON:
[
  {{
    "name": "...",
    "relevance_score": <float between 0 and 1>
  }}
]
Here is the candidate list:
{context_text}
"""

    ranked_names = []
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        text = response.text.strip() if hasattr(response, "text") else ""
        if not text:
            raise ValueError("Empty response from Gemini")

        # Parse Gemini JSON output
        parsed = []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                raise ValueError("Invalid JSON from Gemini")

        # Collect Gemini's ranked order
        ranked_names = [
            r.get("name", "").strip().lower()
            for r in sorted(parsed, key=lambda x: x.get("relevance_score", 0), reverse=True)
        ]
        print(f"✅ Gemini returned {len(ranked_names)} ranked results")

    except Exception as e:
        print(f"⚠️ Gemini reranker failed -> {e}")
        traceback.print_exc()

    # --- Reorder retrieved items according to Gemini ---
    recommended = []
    if ranked_names:
        for name_key in ranked_names:
            db_data = name_to_data.get(name_key)
            if db_data:
                recommended.append({
                    "url": db_data.get("url", ""),
                    "name": db_data.get("name", name_key),
                    "adaptive_support": db_data.get("adaptive_support", "No"),
                    "description": db_data.get("description", "No description available."),
                    "duration": db_data.get("duration", "No Time Limit"),
                    "remote_support": db_data.get("remote_support", "No"),
                    "test_type": map_test_type(db_data.get("test_type", "")),
                })

    # --- Fill remaining with hybrid-score fallback ---
    if len(recommended) < req.k:
        seen = {r["name"].lower() for r in recommended}
        for r in retrieved:
            if r["name"].lower() not in seen:
                recommended.append({
                    "url": r.get("url", ""),
                    "name": r.get("name", ""),
                    "adaptive_support": r.get("adaptive_support", "No"),
                    "description": r.get("description", "No description available."),
                    "duration": r.get("duration", "No Time Limit"),
                    "remote_support": r.get("remote_support", "No"),
                    "test_type": map_test_type(r.get("test_type", "")),
                })
                if len(recommended) >= req.k:
                    break

    return {
        "recommended_assessments": recommended[:req.k]
    }
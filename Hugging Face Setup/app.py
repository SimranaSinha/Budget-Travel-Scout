import os
import json
import re
import hashlib
import traceback
import requests
import gradio as gr
from bs4 import BeautifulSoup
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# ── Client setup (reads from HF Secret) ───────────────────
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Embedding + ChromaDB ───────────────────────────────────
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma_client = chromadb.Client()
travel_collection = chroma_client.get_or_create_collection(
    name="travel_knowledge",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BudgetTravelScout/1.0; student-project)"
}

INTERESTS = [
    "History", "Nightlife", "Food & Drink", "Nature", "Art & Culture",
    "Adventure", "Shopping", "Beach", "Architecture", "Wildlife"
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAG PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def scrape_wikivoyage(destination):
    slug = destination.strip().replace(" ", "_").title()
    url = f"https://en.wikivoyage.org/wiki/{slug}"
    chunks = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "table"]):
            tag.decompose()
        current_heading = "Overview"
        current_text = []
        for el in soup.select("#mw-content-text .mw-parser-output > *"):
            if el.name in ("h2", "h3"):
                if current_text:
                    text = " ".join(current_text).strip()
                    if len(text) > 50:
                        chunks.append(f"[{destination} - {current_heading}] {text}")
                current_heading = el.get_text(strip=True).replace("[edit]", "")
                current_text = []
            elif el.name == "p":
                t = el.get_text(strip=True)
                if t:
                    current_text.append(t)
        if current_text:
            text = " ".join(current_text).strip()
            if len(text) > 50:
                chunks.append(f"[{destination} - {current_heading}] {text}")
    except Exception:
        pass
    return chunks


def scrape_budget_tips(destination):
    slug = destination.strip().lower().replace(" ", "-")
    urls = [
        f"https://www.nomadicmatt.com/travel-guides/{slug}-travel-tips/",
        f"https://www.nomadicmatt.com/travel-guides/{slug}-on-a-budget/",
    ]
    chunks = []
    for url in urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            article = soup.find("article") or soup.find("div", class_="entry-content")
            if not article:
                continue
            for tag in article(["script", "style", "aside", "figure"]):
                tag.decompose()
            for p in article.find_all("p"):
                text = p.get_text(strip=True)
                if len(text) > 60:
                    chunks.append(f"[Budget Guide - {destination}] {text}")
        except Exception:
            pass
    return chunks


def chunk_text(text, max_chars=500):
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) > max_chars and current:
            chunks.append(current.strip())
            current = sent + " "
        else:
            current += sent + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_rag_knowledge(destination):
    wiki_chunks = scrape_wikivoyage(destination)
    blog_chunks = scrape_budget_tips(destination)
    all_raw = wiki_chunks + blog_chunks
    all_chunks = []
    for raw in all_raw:
        all_chunks.extend(chunk_text(raw, max_chars=500))

    if not all_chunks:
        all_chunks = [
            f"{destination} is a popular travel destination with affordable hostels and guesthouses.",
            f"Free activities in {destination} include walking tours, public parks, local markets, and free museums.",
            f"Budget food in {destination} includes street food, local markets, and affordable restaurants.",
            f"Transportation in {destination} is cheap using public transit, buses, and shared rides.",
            f"Visit {destination} during shoulder season for lower flight and hotel prices.",
        ]

    ids, documents = [], []
    for chunk in all_chunks:
        doc_id = hashlib.md5(chunk.encode()).hexdigest()[:12]
        if doc_id not in ids:
            ids.append(doc_id)
            documents.append(chunk)

    try:
        travel_collection.upsert(ids=ids, documents=documents)
    except Exception:
        return 0
    return len(documents)


def rag_query(query, n_results=5):
    try:
        count = travel_collection.count()
        if count == 0:
            return ""
        safe_n = min(n_results, count)
        results = travel_collection.query(query_texts=[query], n_results=safe_n)
        docs = results["documents"][0] if results["documents"] else []
        return "\n\n".join(docs)
    except Exception:
        return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_json(raw):
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)


def agent_a(origin, destination, max_budget=None):
    try:
        rag_context = rag_query(
            f"{destination} budget travel costs flights hotels accommodation prices",
            n_results=5
        )
    except Exception:
        rag_context = ""

    rag_block = f"\n\n=== SCRAPED WEB DATA ===\n{rag_context}\n=== END ===" if rag_context else ""
    budget_instruction = ""
    if max_budget and max_budget > 0:
        budget_instruction = (
            f"\nIMPORTANT: Traveler's max daily budget is ${max_budget:.0f}. "
            "Focus on options within this constraint. Flag if destination is likely over budget."
        )

    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        max_tokens=300,
        messages=[
            {"role": "system", "content": (
                "You are a travel cost researcher. "
                + ("Use the SCRAPED WEB DATA below to ground your estimates. " if rag_context else "")
                + f"Give realistic budget estimates.{budget_instruction}\n"
                "Reply ONLY with valid JSON — no markdown:\n"
                '{"flight_low":int,"flight_high":int,"hotel_low":int,"hotel_high":int,'
                '"booking_tip":"string","budget_warning":"string or null"}'
            )},
            {"role": "user", "content": (
                f"Estimate round-trip economy flight from {origin} to {destination} "
                f"and average budget/mid-range hotel per night in {destination} in 2025."
                f"{rag_block}"
            )},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        result = parse_json(raw)
        result.setdefault("budget_warning", None)
        return result
    except Exception:
        def ex(k):
            m = re.search(rf'"{k}"\s*:\s*(\d+)', raw)
            return int(m.group(1)) if m else 0
        tip = re.search(r'"booking_tip"\s*:\s*"([^"]+)"', raw)
        return {
            "flight_low": ex("flight_low") or 300,
            "flight_high": ex("flight_high") or 800,
            "hotel_low": ex("hotel_low") or 70,
            "hotel_high": ex("hotel_high") or 160,
            "booking_tip": tip.group(1) if tip else "Book 6-8 weeks in advance.",
            "budget_warning": None,
        }


def agent_b(destination, interest, max_budget=None):
    try:
        rag_context = rag_query(
            f"{destination} free cheap {interest} activities things to do budget",
            n_results=5
        )
    except Exception:
        rag_context = ""

    rag_block = f"\n\n=== SCRAPED WEB DATA ===\n{rag_context}\n=== END ===" if rag_context else ""
    budget_note = ""
    if max_budget and max_budget > 0:
        if max_budget < 80:
            budget_note = "Traveler is on a VERY tight budget. ALL activities must be FREE."
        elif max_budget < 150:
            budget_note = "Traveler is budget-conscious. Prioritize free options."

    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=400,
        messages=[
            {"role": "system", "content": (
                "You are a savvy local travel guide. "
                + ("Use the SCRAPED WEB DATA below to suggest specific real places. " if rag_context else
                   "Suggest specific, real places based on your knowledge. ")
                + f"{budget_note}\n"
                "Reply ONLY with a valid JSON array — no markdown:\n"
                '[{"name":"string","description":"one sentence","cost":"Free | Under $5 | $5-$15",'
                '"source":"string"}]'
            )},
            {"role": "user", "content": (
                f"Destination: {destination}\nInterest: {interest}\n"
                "List exactly 3 activities. At least 2 must be completely Free."
                f"{rag_block}"
            )},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        result = parse_json(raw)
        if not result or not isinstance(result, list) or len(result) == 0:
            raise ValueError("Empty list")
        return result
    except Exception:
        return [
            {
                "name": "City walking tour",
                "description": f"Explore {destination}'s streets and landmarks.",
                "cost": "Free",
                "source": "general",
            },
            {
                "name": "Local market visit",
                "description": "Browse produce, crafts, and street food.",
                "cost": "Free",
                "source": "general",
            },
            {
                "name": "Public museum",
                "description": "Free or discounted entry on select days.",
                "cost": "Free-$5",
                "source": "general",
            },
        ]


def agent_c(origin, destination, cost, acts, max_budget=None, trip_days=4):
    acts_text = "\n".join(f"- {a['name']}: {a['description']} ({a.get('cost','?')})" for a in acts)
    budget_constraint = ""
    if max_budget and max_budget > 0:
        budget_constraint = (
            f"\nTraveler's MAX daily budget: ${max_budget:.0f}. Trip duration: {trip_days} days. "
            "If grand_daily_total EXCEEDS max budget, set over_budget=true and fill savings_tips. "
            "If within budget, set over_budget=false and savings_tips=[]."
        )

    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        max_tokens=500,
        messages=[
            {"role": "system", "content": (
                "You are a travel budget analyst. "
                "Compute a realistic solo daily budget from Agent A and B data."
                f"{budget_constraint}\n"
                "Reply ONLY with valid JSON — no markdown:\n"
                '{"daily_hotel":int,"daily_food":int,"daily_transport":int,"daily_activities":int,'
                '"daily_total":int,"flight_amortized":int,"grand_daily_total":int,'
                '"score":int,"summary":"string","over_budget":bool,'
                '"savings_tips":["string"],"total_trip_cost":int}\n'
                "score: 1-10. grand_daily_total = daily_total + flight_amortized. "
                f"flight_amortized = flight midpoint / {trip_days}. "
                f"total_trip_cost = daily_total x {trip_days} + flight midpoint."
            )},
            {"role": "user", "content": (
                f"Trip: {origin} to {destination} ({trip_days} days)\n"
                f"Flight: ${cost['flight_low']}-${cost['flight_high']} round-trip\n"
                f"Hotel: ${cost['hotel_low']}-${cost['hotel_high']}/night\n"
                f"Tip: {cost['booking_tip']}\n\nActivities:\n{acts_text}"
            )},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        result = parse_json(raw)
        result.setdefault("over_budget", False)
        result.setdefault("savings_tips", [])
        result.setdefault("total_trip_cost", 0)
        return result
    except Exception:
        mid_h = (cost["hotel_low"] + cost["hotel_high"]) // 2
        mid_f = (cost["flight_low"] + cost["flight_high"]) // 2
        daily = mid_h + 65
        amort = mid_f // trip_days
        grand = daily + amort
        return {
            "daily_hotel": mid_h,
            "daily_food": 40,
            "daily_transport": 15,
            "daily_activities": 10,
            "daily_total": daily,
            "flight_amortized": amort,
            "grand_daily_total": grand,
            "score": 6,
            "summary": f"{destination} offers moderate value for budget travelers.",
            "over_budget": bool(max_budget and grand > max_budget),
            "savings_tips": [],
            "total_trip_cost": daily * trip_days + mid_f,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REPORT BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_report(origin, destination, interest, cost, acts, budget, max_budget=None, trip_days=4, rag_chunks=0):
    score = budget["score"]
    color = "#1D9E75" if score >= 7 else "#BA7517" if score >= 5 else "#C0392B"
    pct = score * 10

    rag_badge = (
        f"<div style='display:inline-block;background:#E8F5E9;border:1px solid #66BB6A;"
        f"border-radius:20px;padding:3px 12px;font-size:11px;color:#2E7D32;"
        f"font-weight:600;margin-left:8px'>RAG: {rag_chunks} web chunks indexed</div>"
    ) if rag_chunks > 0 else ""

    budget_banner = ""
    if max_budget and max_budget > 0:
        grand = budget["grand_daily_total"]
        total = budget.get("total_trip_cost", grand * trip_days)
        if budget.get("over_budget"):
            diff = grand - max_budget
            budget_banner = (
                f"<div style='background:#FFF3CD;border:1px solid #FFC107;border-radius:8px;"
                f"padding:12px 16px;margin-bottom:14px'>"
                f"<div style='font-weight:700;color:#856404;font-size:14px'>Over Budget by ${diff:.0f}/day</div>"
                f"<div style='font-size:12px;color:#856404;margin-top:4px'>Daily: ${grand} vs max ${max_budget:.0f} | Trip ({trip_days}d): ~${total:,}</div></div>"
            )
        else:
            remaining = max_budget - grand
            budget_banner = (
                f"<div style='background:#D4EDDA;border:1px solid #28A745;border-radius:8px;"
                f"padding:12px 16px;margin-bottom:14px'>"
                f"<div style='font-weight:700;color:#155724;font-size:14px'>Within Budget - ${remaining:.0f}/day to spare</div>"
                f"<div style='font-size:12px;color:#155724;margin-top:4px'>Daily: ${grand} vs max ${max_budget:.0f} | Trip ({trip_days}d): ~${total:,}</div></div>"
            )

    tips_html = ""
    tips = budget.get("savings_tips", [])
    if tips:
        items = "".join(f"<li style='margin-bottom:4px;color:#000000'>{t}</li>" for t in tips)
        tips_html = (
            f"<div style='background:#FFF8E1;border-radius:8px;padding:10px 14px;margin-top:10px;"
            f"border:1px solid #FFE082;font-size:13px;color:#000000'>"
            f"<div style='font-weight:700;margin-bottom:6px;color:#000000'>Money-Saving Tips:</div>"
            f"<ul style='margin:0;padding-left:20px;color:#000000'>{items}</ul></div>"
        )

    bw = cost.get("budget_warning")
    agent_a_warning = (
        f"<div style='font-size:12px;color:#C0392B;margin-top:6px;font-weight:600'>{bw}</div>"
    ) if bw and str(bw).lower() not in ("null", "none", "") else ""

    act_rows = "".join(f"""
        <tr>
          <td style='padding:7px 0;font-weight:600;color:#1a1a1a;width:30%'>{a['name']}</td>
          <td style='padding:7px 0;color:#444;font-size:13px'>{a['description']}</td>
          <td style='padding:7px 0;text-align:right;color:#185FA5;font-weight:600'>{a.get('cost','?')}</td>
        </tr>
        <tr><td colspan='3' style='border-bottom:1px solid #eee;padding:0'></td></tr>
    """ for a in acts)

    cards = "".join(f"""
        <div style='background:#fff;border:1px solid #e0e0e0;border-radius:8px;
                    padding:10px 14px;text-align:center;flex:1;min-width:80px'>
          <div style='font-size:11px;color:#888;margin-bottom:2px'>{lbl}</div>
          <div style='font-size:17px;font-weight:700;color:#1a1a1a'>${val}</div>
        </div>
    """ for lbl, val in [
        ("Hotel", budget["daily_hotel"]),
        ("Food", budget["daily_food"]),
        ("Transport", budget["daily_transport"]),
        ("Activities", budget["daily_activities"]),
        (f"Flight /{trip_days}d", budget["flight_amortized"]),
    ])

    budget_display = f" &nbsp;·&nbsp; Max ${max_budget:.0f}/day" if max_budget else ""

    return f"""
<div style='font-family:system-ui,sans-serif;max-width:700px;margin:0 auto;
            border:1px solid #ddd;border-radius:14px;overflow:hidden;
            box-shadow:0 4px 20px rgba(0,0,0,0.08)'>
  <div style='background:linear-gradient(135deg,#185FA5,#0d3f6e);color:#fff;padding:1.5rem 1.75rem'>
    <div style='display:flex;align-items:center;gap:8px'>
      <span style='font-size:11px;opacity:0.75;letter-spacing:2px;text-transform:uppercase'>Budget Travel Scout</span>
      {rag_badge}
    </div>
    <div style='font-size:26px;font-weight:700;margin-top:4px'>{origin} &rarr; {destination}</div>
    <div style='font-size:14px;opacity:0.85;margin-top:4px'>Interest: {interest} &nbsp;·&nbsp; {trip_days} days{budget_display}</div>
  </div>
  <div style='padding:1.25rem 1.75rem;border-bottom:1px solid #eee;background:#f0f7ff'>
    <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px'>
      <span style='background:#185FA5;color:#fff;font-size:11px;font-weight:700;padding:3px 9px;border-radius:20px'>AGENT A</span>
      <span style='font-size:12px;color:#555'>Researcher &middot; GPT-4o + RAG</span>
    </div>
    <div style='display:flex;gap:2.5rem'>
      <div><div style='font-size:11px;color:#888;margin-bottom:2px'>Round-trip flight</div>
           <div style='font-size:22px;font-weight:700;color:#1a1a1a'>${cost['flight_low']}&ndash;${cost['flight_high']}</div></div>
      <div><div style='font-size:11px;color:#888;margin-bottom:2px'>Hotel / night</div>
           <div style='font-size:22px;font-weight:700;color:#1a1a1a'>${cost['hotel_low']}&ndash;${cost['hotel_high']}</div></div>
    </div>
    <div style='font-size:12px;color:#555;margin-top:8px;font-style:italic'>{cost['booking_tip']}</div>
    {agent_a_warning}
  </div>
  <div style='padding:1.25rem 1.75rem;border-bottom:1px solid #eee;background:#f0fbf6'>
    <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px'>
      <span style='background:#0F6E56;color:#fff;font-size:11px;font-weight:700;padding:3px 9px;border-radius:20px'>AGENT B</span>
      <span style='font-size:12px;color:#555'>Local Guide &middot; GPT-4o + RAG</span>
    </div>
    <table style='width:100%;border-collapse:collapse'>{act_rows}</table>
  </div>
  <div style='padding:1.25rem 1.75rem;background:#f9f7ff'>
    <div style='display:flex;align-items:center;gap:8px;margin-bottom:12px'>
      <span style='background:#533AB7;color:#fff;font-size:11px;font-weight:700;padding:3px 9px;border-radius:20px'>AGENT C</span>
      <span style='font-size:12px;color:#555'>Budgeter &middot; GPT-4o</span>
    </div>
    {budget_banner}
    <div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px'>{cards}</div>
    <div style='font-size:22px;font-weight:700;color:#1a1a1a;margin-bottom:12px'>
      Grand Daily Total: <span style='color:{color}'>${budget['grand_daily_total']}</span>
    </div>
    <div style='margin-bottom:12px'>
      <div style='display:flex;justify-content:space-between;font-size:12px;color:#555;margin-bottom:5px'>
        <span>Budget Friendliness Score</span>
        <span style='font-weight:700;color:{color}'>{score} / 10</span>
      </div>
      <div style='height:10px;background:#e0e0e0;border-radius:5px;overflow:hidden'>
        <div style='height:100%;width:{pct}%;background:{color};border-radius:5px'></div>
      </div>
    </div>
    <div style='font-size:13px;color:#444;line-height:1.7;background:#fff;
                border-radius:8px;padding:10px 14px;border:1px solid #e0e0e0'>{budget['summary']}</div>
    {tips_html}
  </div>
</div>"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PIPELINE + GRADIO UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pipeline(origin, destination, interest, max_budget, trip_days, progress=gr.Progress()):
    if not origin.strip() or not destination.strip():
        return "", "", "", "<p style='color:red;font-weight:600'>Please enter both origin and destination.</p>"

    max_budget = float(max_budget) if max_budget else None
    trip_days = int(trip_days) if trip_days else 4

    rag_chunks = 0
    try:
        progress(0.05, desc="Scraping web data for RAG...")
        rag_chunks = build_rag_knowledge(destination.strip())
    except Exception as e:
        print(f"RAG build failed (continuing without): {e}")

    cost = None
    a_out = ""
    try:
        progress(0.25, desc="Agent A: researching prices...")
        cost = agent_a(origin.strip(), destination.strip(), max_budget)
        a_out = (
            f"Flight (round-trip): ${cost['flight_low']}-${cost['flight_high']}\n"
            f"Hotel (per night):   ${cost['hotel_low']}-${cost['hotel_high']}\n"
            f"Tip: {cost['booking_tip']}"
        )
        bw = cost.get("budget_warning")
        if bw and str(bw).lower() not in ("null", "none", ""):
            a_out += f"\nWarning: {bw}"
    except Exception as e:
        a_out = f"Agent A error: {e}"
        traceback.print_exc()
        cost = {
            "flight_low": 300,
            "flight_high": 800,
            "hotel_low": 70,
            "hotel_high": 160,
            "booking_tip": "Book 6-8 weeks in advance.",
            "budget_warning": None,
        }

    acts = None
    b_out = ""
    try:
        progress(0.50, desc="Agent B: finding activities...")
        acts = agent_b(destination.strip(), interest, max_budget)
        if not acts or len(acts) == 0:
            raise ValueError("Empty activities")
        b_out = "\n".join(
            f"{'[Free]' if 'free' in a.get('cost', '').lower() else '[Paid]'} {a['name']} ({a.get('cost', '?')}) - {a['description']}"
            for a in acts
        )
    except Exception as e:
        b_out = f"Agent B error: {e}"
        traceback.print_exc()
        acts = [
            {"name": "City walking tour", "description": f"Explore {destination}'s landmarks.", "cost": "Free", "source": "fallback"},
            {"name": "Local market visit", "description": "Browse produce, crafts, street food.", "cost": "Free", "source": "fallback"},
            {"name": "Public museum", "description": "Free entry on select days.", "cost": "Free-$5", "source": "fallback"},
        ]

    budget = None
    c_out = ""
    try:
        progress(0.75, desc="Agent C: calculating budget...")
        budget = agent_c(origin.strip(), destination.strip(), cost, acts, max_budget, trip_days)
        c_out = (
            f"Hotel ${budget['daily_hotel']} | Food ${budget['daily_food']} | "
            f"Transport ${budget['daily_transport']} | Activities ${budget['daily_activities']}\n"
            f"Flight amortized ({trip_days}d): ${budget['flight_amortized']}\n"
            f"Grand daily total: ${budget['grand_daily_total']} | Score: {budget['score']}/10"
        )
        if budget.get("over_budget") and max_budget:
            c_out += f"\nOVER your ${max_budget:.0f}/day budget!"
        if budget.get("total_trip_cost"):
            c_out += f"\nEst. total trip ({trip_days}d): ${budget['total_trip_cost']:,}"
    except Exception as e:
        c_out = f"Agent C error: {e}"
        traceback.print_exc()
        mid_h = (cost["hotel_low"] + cost["hotel_high"]) // 2
        mid_f = (cost["flight_low"] + cost["flight_high"]) // 2
        budget = {
            "daily_hotel": mid_h,
            "daily_food": 40,
            "daily_transport": 15,
            "daily_activities": 10,
            "daily_total": mid_h + 65,
            "flight_amortized": mid_f // trip_days,
            "grand_daily_total": mid_h + 65 + mid_f // trip_days,
            "score": 5,
            "summary": f"Budget estimate for {destination}.",
            "over_budget": False,
            "savings_tips": [],
            "total_trip_cost": 0,
        }

    progress(1.0, desc="Done!")
    try:
        report = build_report(origin, destination, interest, cost, acts, budget, max_budget, trip_days, rag_chunks)
    except Exception as e:
        report = f"<p style='color:red'><strong>Report error:</strong> {e}</p>"

    return a_out, b_out, c_out, report


with gr.Blocks(title="Budget Travel Scout") as demo:

    gr.HTML("""
        <div style='text-align:center;padding:1.5rem 0 0.5rem'>
          <h1 style='font-size:28px;font-weight:700;color:#185FA5;margin:0'>🌍 Budget Travel Scout</h1>
          <p style='color:#666;margin-top:6px;font-size:14px'>
            Multi-Agent AI with <b>RAG</b> (Web Scraping &rarr; ChromaDB &rarr; LLM)
          </p>
          <p style='color:#999;margin-top:2px;font-size:12px'>
            <b>Agent A</b>: GPT-4o + RAG &nbsp;·&nbsp;
            <b>Agent B</b>: GPT-4o + RAG &nbsp;·&nbsp;
            <b>Agent C</b>: GPT-4o
          </p>
        </div>
    """)

    with gr.Row():
        origin_in = gr.Textbox(label="From (city)", placeholder="e.g. Boston", scale=2)
        destination_in = gr.Textbox(label="To (destination)", placeholder="e.g. Lisbon", scale=2)
        interest_in = gr.Dropdown(choices=INTERESTS, value="History", label="Interest", scale=1)

    with gr.Row():
        max_budget_in = gr.Number(
            label="Max Daily Budget (USD)",
            value=None,
            minimum=0,
            maximum=5000,
            info="Optional - leave empty for no limit"
        )
        trip_days_in = gr.Slider(label="Trip Duration (days)", minimum=1, maximum=30, value=4, step=1)

    run_btn = gr.Button("Scout this destination", variant="primary", size="lg")

    gr.Markdown("### Agent Outputs")
    with gr.Row():
        with gr.Column(elem_classes="agent-box"):
            gr.Markdown("**Agent A - Researcher** *(GPT-4o + RAG)*")
            a_out = gr.Textbox(label="", lines=5, interactive=False, placeholder="Waiting...")
        with gr.Column(elem_classes="agent-box"):
            gr.Markdown("**Agent B - Local Guide** *(GPT-4o + RAG)*")
            b_out = gr.Textbox(label="", lines=5, interactive=False, placeholder="Waiting...")
        with gr.Column(elem_classes="agent-box"):
            gr.Markdown("**Agent C - Budgeter** *(GPT-4o)*")
            c_out = gr.Textbox(label="", lines=5, interactive=False, placeholder="Waiting...")

    gr.Markdown("### Full Report")
    report_out = gr.HTML("<p style='color:#aaa;text-align:center;padding:2rem'>Run a search to see your travel report.</p>")

    gr.Examples(
        examples=[
            ["Boston", "Lisbon", "History", 150, 5],
            ["New York", "Bangkok", "Food & Drink", 80, 7],
            ["Chicago", "Tokyo", "Nightlife", 200, 4],
            ["Los Angeles", "Mexico City", "Art & Culture", 100, 5],
        ],
        inputs=[origin_in, destination_in, interest_in, max_budget_in, trip_days_in],
        label="Try an example",
    )

    gr.HTML("""
        <div style='text-align:center;padding:1rem 0;color:#aaa;font-size:12px'>
          RAG Pipeline: BeautifulSoup (scrape) &rarr; SentenceTransformers (embed) &rarr; ChromaDB (retrieve) &rarr; GPT-4o (generate)
        </div>
    """)

    run_btn.click(
        fn=run_pipeline,
        inputs=[origin_in, destination_in, interest_in, max_budget_in, trip_days_in],
        outputs=[a_out, b_out, c_out, report_out],
    )

demo.launch(
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css="""
    .agent-box textarea {
        font-size: 13px !important;
        font-family: monospace !important;
        color: #FFFFFF !important;
    }
    footer {
        display: none !important;
    }
    """,
    server_name="0.0.0.0",
    server_port=7860,
    share=True,
    ssr_mode=False,
)
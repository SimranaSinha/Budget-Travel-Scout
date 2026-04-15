# Budget-Travel-Scout

# 🌍 Budget Travel Scout

> A multi-agent AI system that finds the best **vibe vs. cost** match for your next trip — in under 10 seconds.
---

## 🔗 Live Link
https://huggingface.co/spaces/Simi2407/budget-travel-scout

---
## ✨ What it does

Budget Travel Scout takes a destination, origin city, travel interest, max daily budget, and trip duration — and runs a 3-agent AI pipeline that returns:

- ✈️ **Flight and hotel cost estimates** for your route
- 🗺️ **3 free or cheap local activities** matching your interest
- 💰 **A daily budget breakdown** with a Budget Friendliness Score out of 10
- 🟢 **Over/under budget indicator** with money-saving tips
- 📊 **Total trip cost estimate** based on your trip duration

---

## 🚀 How to Run

### Running Locally - Using Huggingface 
1. Clone the repo
```
git clone https://huggingface.co/spaces/Simi2407/budget-travel-scout
cd budget-travel-scout
```

2. Install dependencies
```
pip install openai gradio beautifulsoup4 chromadb sentence-transformers requests
```

3. Set your API key
```
export OPENAI_API_KEY="your-key-here"
```

4. Run
```
python app.py
```

### Running on Google Colab

1. Open `Budget Travel Scout_Python.ipynb` in Google Colab
2. Click the 🔑 **Secrets** panel in the left sidebar
3. Add a secret named `openai.api_key` with your OpenAI API key
4. Toggle **Notebook access ON**
5. Run all cells in order — **Cell 1 → Cell 6**
6. After Cell 6, click the `gradio.live` link that appears
7. The UI opens in your browser — enter a destination and hit **Scout this destination**!

> 💡 The `gradio.live` link is valid for **72 hours**. Simply re-run Cell 6 to get a fresh link anytime.

---

## 🏗️ Architecture

### 🤖 3-Agent Pipeline

```
User Input → Agent A → Agent B → Agent C → Full Report
```

| Agent | Role | Model | Temp | Key Output |
|-------|------|-------|------|------------|
| 🔵 A — Researcher | Flight + hotel costs | GPT-4o | 0.2 | Price ranges, booking tip, budget warning |
| 🟢 B — Local Guide | Free/cheap activities | GPT-4o | 0.3 | 3 activities with cost and description |
| 🟣 C — Budgeter | Daily cost breakdown | GPT-4o | 0.2 | Grand daily total, score/10, savings tips |

> Agent C depends on the outputs of both A and B — enforcing a real sequential dependency chain.

### 🧠 RAG Pipeline

Real web data grounds the LLM responses for accurate, destination-specific answers.

```
🌐 Scrape       →  BeautifulSoup fetches Wikivoyage + travel blogs
✂️  Chunk        →  500-char overlapping chunks, MD5 deduplicated
🔢 Embed        →  all-MiniLM-L6-v2 via SentenceTransformers
🗄️  Store        →  ChromaDB in-memory (cosine similarity)
🔍 Retrieve     →  Top-5 relevant chunks per agent query
✨ Generate     →  GPT-4o produces grounded, specific answers
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| 🤖 LLM | OpenAI GPT-4o |
| 🌐 Scraping | BeautifulSoup4 |
| 🔢 Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| 🗄️ Vector Store | ChromaDB (in-memory) |
| 🎨 UI | Gradio |
| 📓 Notebook | Google Colab |

---

## 📁 Project Structure

```
budget-travel-scout/
|
├── Gen_AI_Prj_Enhanced.ipynb   # 📓 Main Colab notebook (run this)
|
├── Gen AI_Final Prj.pptx       # 📓 Presentation
|
└── README.md                   # 📄 You are here
```

---

## 📋 Notebook Structure

| Cell | Description |
|------|-------------|
| Cell 1 | 📦 Install dependencies |
| Cell 2 | 🔑 Load API key from Colab Secrets |
| Cell 3 | 🌐 RAG pipeline — scraper, embedder, ChromaDB |
| Cell 4 | 🤖 Agent A, B, C logic (GPT-4o) |
| Cell 5 | 📊 HTML report builder |
| Cell 6 | 🚀 Launch Gradio UI |

---

## 🎯 Sample Output

**Boston → Mumbai · Adventure · 6 days · Max $5,000/day**

| Agent | Output |
|-------|--------|
| 🔵 A | Flight: $800–$1,500 · Hotel: $50–$150/night |
| 🟢 B | Marine Drive (Free), Sanjay Gandhi National Park (Free), Elephanta Caves ($5–$15) |
| 🟣 C | Grand daily total: $382 · Score: 8/10 · Within budget · Trip total: ~$2,520 |

---

## ⭐ Key Features

- 💰 **Budget constraint mode** — set a max daily budget, get a green/red over-under indicator
- 📅 **Trip duration slider** — 1 to 30 days, adjusts flight amortization and total trip cost
- 🧠 **RAG grounding** — scrapes real web content per destination at query time
- 🛡️ **Graceful fallbacks** — every agent has try/catch + fallback data so the pipeline never crashes
- 🎯 **10 interest categories** — History, Nightlife, Food & Drink, Nature, Art & Culture, Adventure, Shopping, Beach, Architecture, Wildlife
- 📝 **Structured JSON outputs** — all agents return validated JSON with regex fallback parsing

---

## 🔧 Engineering Decisions

**🤔 Why all GPT-4o instead of mixed models?**
Gemini 1.5 Flash was the originally intended model for Agent B, but Colab's network environment blocked outbound Gemini API calls causing indefinite hangs (1,000+ seconds). GPT-4o was used for all three agents with differentiated system prompts and temperatures to maintain distinct agent personas.

**🤔 Why RAG instead of OpenAI web search?**
OpenAI's `web_search_preview` tool caused 600-second timeouts in the Colab environment. The custom RAG pipeline (BeautifulSoup → ChromaDB) is significantly faster, more controllable, and demonstrates the full retrieval-augmented generation concept more explicitly.

**🤔 Why Gradio for the UI?**
Gradio integrates natively with Colab and generates a shareable public link with a single line (`share=True`), making it ideal for demos without any separate deployment setup.

---

## 📐 Evaluation Criteria Alignment

| Criterion | Implementation |
|-----------|---------------|
| 🏗️ Multi-agent architecture (25%) | 3 sequential GPT-4o agents with distinct roles, temperatures, and dependency chain |
| 🎨 User experience (25%) | Gradio UI with budget slider, trip duration, example queries, styled HTML report card |
| 💡 Innovation & complexity (20%) | Full RAG pipeline, budget constraint mode, savings tips, graceful fallbacks |
| 📄 Documentation (15%) | This README + fully documented Colab notebook with markdown explanation cells |
| ⏱️ Project management (15%) | Iterative development — resolved timeout issues, deployment challenges, syntax errors |


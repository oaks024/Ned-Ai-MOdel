# NED Admission Assistant

An AI chatbot that answers admission-related questions about **NED University of
Engineering & Technology** (https://www.neduet.edu.pk/) using official data only.

It scrapes the public NED website, downloads admission-related PDFs, builds a
vector index over the extracted text, and uses **LiteLLM + Groq** to answer
questions strictly from retrieved official content. Per-session memory lets the
chatbot personalize replies (e.g., remember which program you care about) without
ever treating that memory as authoritative admission data.

---

## Architecture (the "Claude consultation" note)

We deliberately keep the storage stack simple and beginner-friendly:

- **Vector store: ChromaDB (local, persistent).**
  ChromaDB needs no separate server, persists to disk via `PersistentClient`,
  ships with a built-in HNSW index, and supports cosine similarity out of the
  box. FAISS is a fine alternative but stores raw vectors only — you have to
  manage metadata yourself. For a learning-oriented project, Chroma's bundled
  metadata + filtering is the better trade.
- **Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local).**
  Free, offline after first download, 384-dim, fast on CPU. Hosted embedding
  APIs would tie us to another vendor and add latency for no real gain on
  English admission text.
- **Session memory: SQLite (`memory.db`).**
  More robust than JSON for concurrent reads/writes from a Streamlit app, and
  it gives us a clean schema for `profile` and `history`. JSON would work but
  starts to corrupt when two callbacks fire at once.
- **Vector memory vs. session memory.** They are *different things*:
  - The vector store holds the **knowledge base** — chunks of official NED
    text. It is the *only* source of admission facts.
  - Session memory holds **who the user is** (name, interested program, etc.)
    and the recent chat history. It personalizes responses; it never supplies
    admission facts.
- **Keeping data fresh.** Run `python main.py refresh` periodically (e.g.,
  before each admission cycle). The `refresh` command re-scrapes, re-downloads
  PDFs, resets the Chroma collection, and rebuilds embeddings.

---

## Project layout

```
ned-admission-ai/
├── app.py              # Streamlit chatbot UI
├── main.py             # CLI: scrape / build / refresh / stats / reset
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   ├── raw/            # snapshot.json + downloaded PDFs (created at runtime)
│   └── processed/      # chroma DB + memory.db (created at runtime)
└── src/
    ├── config.py       # loads .env into a Config object
    ├── utils.py        # logger
    ├── scraper.py      # BFS crawler for neduet.edu.pk + PDF downloader
    ├── pdf_loader.py   # pdfplumber-based text extraction
    ├── chunker.py      # word-based chunking with overlap
    ├── embeddings.py   # sentence-transformers wrapper
    ├── vector_store.py # ChromaDB persistent store
    ├── memory.py       # SQLite-backed profile + chat history
    └── rag_chain.py    # retrieve -> prompt -> LiteLLM/Groq -> answer
```

---

## Setup

### 1. Prerequisites

- Python 3.10+ (uses PEP 604 union syntax)
- A free Groq API key from https://console.groq.com/keys

### 2. Install

```powershell
cd ned-admission-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Note (Windows):** `chromadb` pulls in compiled wheels. If install fails,
> upgrade pip first (`python -m pip install --upgrade pip`) and ensure you're
> on Python 3.10 or 3.11.

### 3. Configure secrets

```powershell
copy .env.example .env
notepad .env   # paste your real GROQ_API_KEY
```

### 4. Build the knowledge base

```powershell
python main.py refresh
```

This will:
1. Crawl https://www.neduet.edu.pk/ (up to `MAX_CRAWL_PAGES`, default 80).
2. Download admission-related PDFs into `data/raw/pdf/`.
3. Save a `data/raw/snapshot.json` with all extracted text.
4. Chunk + embed everything and store it in `data/processed/chroma/`.

You can also split the steps:

```powershell
python main.py scrape   # only scrape
python main.py build    # only (re)build embeddings from existing snapshot
python main.py stats    # how many chunks are indexed?
python main.py reset    # clear the vector collection
```

### 5. Run the chatbot

```powershell
streamlit run app.py
```

Open http://localhost:8501. Use the sidebar to set your name, interested
program, and admission category — these personalize the chat for the rest of
the session.

---

## Example questions

- What is the admission process at NED?
- What programs are offered for undergraduate admission?
- What is the eligibility for Computer Science?
- What documents are required for admission?
- What is the fee structure?
- When does admission open?
- Is there an entry test?
- How can I apply online?
- Where can I find the prospectus?

Each answer ends with a **Sources:** section listing the NED URLs the model
relied on. If the question can't be answered from the indexed data, the
assistant will say:

> *I could not find this in the official NED data.*

---

## Safety guardrails

The system prompt forbids the model from:
- guessing dates, fees, merit criteria, or eligibility rules,
- answering from general knowledge when the official context is silent,
- omitting source URLs.

For deadline-sensitive information the assistant explicitly recommends
confirming with the NED admission office.

---

## Updating the knowledge base

NED publishes new prospectuses, fee schedules, and notices each cycle. To pick
them up:

```powershell
python main.py refresh
```

That command re-crawls the site, re-downloads PDFs, **resets** the Chroma
collection, and rebuilds embeddings from scratch. You can also click
**🔄 Refresh NED data** in the Streamlit sidebar.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GROQ_API_KEY is missing` | Copy `.env.example` to `.env` and fill in the key. |
| `The knowledge base is empty` warning in Streamlit | Run `python main.py refresh`. |
| LiteLLM error: model not found | Groq may have retired the model. Set `LITELLM_MODEL=groq/llama-3.3-70b-versatile` (or another current Groq model) in `.env`. |
| SSL warnings during scrape | Suppressed intentionally — NED's certificate sometimes mis-resolves. |
| Slow first run | `sentence-transformers` downloads the embedding model (~90 MB) on first use. |

---

## Deploying to Streamlit Community Cloud

The repo ships with a prebuilt ChromaDB (`data/processed/chroma/`) so the
deployed app can answer questions immediately without re-scraping the NED
website.

1. **Push this repo to GitHub** (public or private — both work).
2. Go to **https://share.streamlit.io** and sign in with your GitHub account.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Pick this repo, branch `main`, and `app.py` as the main file.
5. Open **Advanced settings → Secrets** and paste the contents of
   [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example),
   replacing the placeholder with your real Groq API key.
6. Click **Deploy**. First boot takes ~3–5 minutes (installing PyTorch +
   sentence-transformers).
7. Share the resulting `https://<you>-ned-admission-ai-app.streamlit.app/` URL.

To refresh the knowledge base later, run `python main.py refresh` locally and
push the updated `data/processed/chroma/` to GitHub. The cloud app will redeploy
automatically.

### Why not Vercel / Netlify / Cloudflare Pages?

Streamlit needs a long-running server with WebSockets and >1 GB of RAM (PyTorch
alone). Serverless platforms (Vercel, Netlify) cap functions at ~250 MB bundle
size and ~10–60 second timeouts, which can't hold this stack. Streamlit
Community Cloud is purpose-built for Streamlit apps and is free.

---

## License & disclaimer

This project is for educational use. It depends on data published on NED's
public website. Always verify time-sensitive admission information with the
NED admission office before making decisions.

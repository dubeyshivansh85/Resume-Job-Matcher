# Resume ↔ Job Matching NLP Tool (v2)

[![Streamlit App](https://img.shields.io/badge/Streamlit%20App-Live%20Demo-brightgreen?style=for-the-badge&logo=streamlit)](https://resume-job-matcher-fa4gw7ehcqwqwiqhfdwrcp.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-25%20Passed-success?style=for-the-badge&logo=pytest)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

An end-to-end NLP tool that scores resume-to-job fit using a **hybrid keyword + semantic similarity pipeline**, provides prioritized skill gap analysis, generates **AI resume bullet rewrites** (via Google Gemini), calculates an ATS compatibility score, detects seniority mismatches, and benchmarks applicants against real industry distributions.

---

## 🚀 Key Features (v2)

| Feature | Description |
|---|---|
| **Hybrid Matching Engine** | Combines **Jaccard similarity** (extracted skill tokens) and **Dense Semantic Embeddings** (`all-MiniLM-L6-v2` / `all-mpnet-base-v2` via Cosine Similarity) with configurable weighting. |
| **Modular 6-Step Pipeline** | Clean, production-ready codebase split into numbered interview-friendly modules (`src/step1_...` through `src/step6_...`). |
| **AI Resume Bullet Rewrites** | Powered by **Google Gemini API** (`gemini-2.0-flash`) — crafts tailored STAR-method bullet points incorporating missing skills. Seamless offline fallback if no API key is provided. |
| **Multi-Job Comparison** | Rank and compare up to 5 job postings simultaneously to find the best match. |
| **Seniority Level Check** | Detects mismatched career levels (e.g., Junior applying to Staff/Principal roles) using regex heuristics. |
| **ATS Compatibility Check** | Analyzes contact info, section headers, quantifiable bullet points, keyword coverage, and length. |
| **Interactive Score Simulator** | Live resume editor with instant score delta calculation as you edit your resume. |
| **Job URL Extraction** | Paste a job posting URL to scrape text automatically via Jina AI reader. |
| **Multi-Format Ingestion** | Supports **PDF** (`pdfplumber`), **DOCX** (`python-docx`), and plain text input. |
| **Unit Test Suite** | 25 automated unit tests covering extraction, matching, ATS scoring, and fallback generators. |

---

## 🧠 How the Matching Engine Works

```
Resume Text ───────┬──► [step1_utils] Clean & Normalize ──► [step2_extractor] spaCy PhraseMatcher ──┐
                   │                                                                               │
Job Description ───┴──► [step1_utils] Clean & Normalize ──► [step2_extractor] spaCy PhraseMatcher ──┤
                                                                                                   ▼
                                                                                   Skill Sets (Jaccard: 40%)
                                                                                                   │
Resume Text ───────┬──► [step3_matching] Sentence-BERT Embeddings ─────────────────────────────────┤
Job Description ───┴──► [step3_matching] Cosine Similarity ────────────────────────► (Semantic: 60%)
                                                                                                   │
                                                                                                   ▼
                                                                                    Overall Match Score (0-100%)
                                                                                                   │
                                                ┌──────────────────────────────────────────────────┴──────────────────────┐
                                                ▼                                                                         ▼
                                   [step4_suggestions] Priority Ranker                                     [step3_matching] ATS Engine
                                                │                                                                         │
                                                ▼                                                                         ▼
                                   Gemini AI STAR Bullet Rewrites                                          ATS Score & Seniority Audit
```

1. **Skill Extraction (`src/step2_extractor.py`)**: Uses spaCy's `PhraseMatcher` over a curated taxonomy (~150 Data/ML skills + ~70 Software Engineering skills with synonym mappings). This is deterministic and significantly faster than heavy NER models.
2. **Keyword Match (Jaccard)**: Measures the exact overlap ratio:
   $$\text{Jaccard} = \frac{|S_{\text{resume}} \cap S_{\text{job}}|}{|S_{\text{resume}} \cup S_{\text{job}}|}$$
3. **Semantic Match (Sentence-BERT)**: Encodes full resume and job texts into dense 384-dimensional vectors using `all-MiniLM-L6-v2` and calculates cosine similarity. This captures implied skills (e.g., "built ETL pipelines with airflow" matches "data engineering infrastructure").
4. **Final Weighted Score**:
   $$\text{Final Score} = (0.40 \times \text{Jaccard}) + (0.60 \times \text{Semantic})$$
5. **Priority Ranking (`src/step4_suggestions.py`)**: Ranks missing skills by frequency in the posting, whether they appear in the job title (3x weight multiplier), and section placement.
6. **ATS Audit**: Evaluates email, phone, LinkedIn/GitHub links, standard headings (Experience, Education, Skills), quantifiable metrics (`%`, `$`, numbers), and word count heuristics.

---

## 📁 Project Structure

Organized into sequential numbered steps for clear understanding and interview presentation:

```
Resume-Job-Matcher/
├── app.py                      # Streamlit web application & UI layout
├── Blueprint.md                # Architectural design document & interview guide
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Target runtime (Python 3.11 for Streamlit Cloud)
├── .env.example                # Template for environment variables (API keys)
├── .gitignore                  # Git ignore rules (protects .env, __pycache__, datasets)
├── README.md                   # Project documentation
│
├── src/                        # Core modular engine (numbered by execution step)
│   ├── __init__.py
│   ├── step1_utils.py          # Step 1: Text preprocessing, cleaning, PDF/DOCX parsing
│   ├── step2_extractor.py      # Step 2: spaCy PhraseMatcher & taxonomy loader
│   ├── step3_matching.py       # Step 3: Hybrid matching, ATS scoring, seniority audit
│   ├── step4_suggestions.py    # Step 4: Missing skill ranking & Gemini AI bullet rewrites
│   ├── step5_charts.py         # Step 5: Plotly gauge, donut, and comparison visualizations
│   └── step6_scraper.py        # Step 6: Web scraping for job postings via Jina AI
│
├── data/
│   ├── skills_taxonomy.csv     # Curated Data Science & ML skills taxonomy
│   ├── taxonomy_software.csv   # Software Engineering skills taxonomy
│   ├── ds_postings_filtered.csv
│   ├── ds_postings_with_skills.csv
│   ├── ds_resumes_with_skills.csv
│   ├── posting_embeddings.npy  # Precomputed embeddings for instant benchmarking
│   └── resume_embeddings.npy
│
└── tests/
    ├── conftest.py             # Shared pytest fixtures & test data
    └── test_matching.py        # 25 automated unit tests
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **App & UI** | Streamlit, Plotly | Interactive responsive web application & visual analytics |
| **NLP & Matching** | spaCy (`en_core_web_sm`), `PhraseMatcher` | Fast, deterministic rule-based skill extraction |
| **Vector Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Dense semantic text representations |
| **Similarity Metrics** | `scikit-learn` (Cosine Similarity) | Vector distance computation |
| **Generative AI** | Google Gemini API (`gemini-2.0-flash`) | Contextual resume bullet point rewriting (STAR method) |
| **Document Parsers** | `pdfplumber`, `python-docx` | Extraction from PDF and Word documents |
| **Web Ingestion** | Jina AI Reader API | Clean markdown extraction from public job URLs |
| **Testing** | `pytest` | 25 automated unit tests for core logic |

---

## ⚡ Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/dubeyshivansh85/Resume-Job-Matcher.git
cd Resume-Job-Matcher
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. (Optional) Configure Google Gemini API Key
To enable AI-powered resume bullet rewrites:
```bash
cp .env.example .env
```
Open `.env` and add your free Gemini API key:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```
> *Note: If no API key is set, the app gracefully falls back to template-based suggestions automatically.*

### 5. Run the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Tests

The test suite validates text extraction, skill normalization, Jaccard similarity, semantic similarity, ATS scoring, and suggestion generation:

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_matching.py .........................                         [100%]
============================== 25 passed in ~2.5s ==============================
```

---

## 💼 Interview Talking Points

- **Why Hybrid (Keyword + Semantic)?** Keyword-only matching fails when candidates use synonyms or describe achievements without exact buzzwords. Pure vector search can hallucinate relevance on fluff text. Combining deterministic `PhraseMatcher` (40%) with dense embeddings (60%) provides both precision and contextual recall.
- **Why spaCy PhraseMatcher over NER?** Pre-trained NER models often miss niche tech acronyms (e.g., `dbt`, `PyTorch`, `CI/CD`) and have high inference latency. `PhraseMatcher` runs in sub-millisecond time with zero false positives for known taxonomy terms.
- **Production Architecture**: Logic is separated cleanly from Streamlit UI into `src/`. This decouples the presentation layer from the NLP engine, making the pipeline easy to unit test and containerize for REST/FastAPI deployment.
- **Graceful Degradation**: If external API calls (Gemini, Jina AI) fail or are unconfigured, the app falls back to rule-based algorithms with zero user interruption.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

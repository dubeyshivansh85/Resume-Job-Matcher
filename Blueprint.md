# Resume ↔ Job Matching NLP Tool — Project Blueprint (v2)

## 1. Overview

**Problem:** Job seekers don't know how well their resume matches a job posting, or what skills they're missing.

**What it does:** Takes a resume and a job description, and outputs:
- Overall match score (keyword + semantic similarity, weighted)
- Matched and missing skills, ranked by priority
- AI-generated resume bullet rewrites (via Gemini API)
- ATS compatibility score with heuristic checks
- Seniority mismatch warning
- Benchmark percentile ("beats ~X% of applicants")
- Multi-job comparison (rank up to 5 jobs at once)
- Live resume editor with real-time score delta

---

## 2. Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Resume parsing | pdfplumber (PDF), python-docx (DOCX) |
| Skill extraction | spaCy (PhraseMatcher) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2` / `all-mpnet-base-v2`) |
| Similarity | scikit-learn (cosine_similarity) |
| AI Rewrites | Google Gemini API (`gemini-2.0-flash`, free tier) |
| Job URL fetching | Jina AI reader API (free, no key needed) |
| Data handling | pandas, numpy |
| App | Streamlit |
| Charts | Plotly |
| Testing | pytest (25 unit tests) |
| Deployment | Streamlit Community Cloud |

---

## 3. Data Sources

- **Job postings:** [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) (Kaggle) — filtered to titles matching `data scientist|data analyst|machine learning|data engineer`, ~1,187 rows.
- **Resumes:** [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (Kaggle) — filtered to `INFORMATION-TECHNOLOGY` + `ENGINEERING` categories, ~238 rows.
- **Skills taxonomy:** hand-built, ~150 data/ML/AI skills + ~70 software engineering skills, with synonyms. Stored in `data/skills_taxonomy.csv` and `data/taxonomy_software.csv`.

---

## 4. Project Structure (v2)

```
├── app.py                          # Streamlit UI — entry point, UI only
├── src/
│   ├── __init__.py
│   ├── step1_utils.py              # STEP 1: text cleaning, PDF/DOCX extraction
│   ├── step2_extractor.py          # STEP 2: spaCy skill extractor, taxonomy loader
│   ├── step3_matching.py           # STEP 3: Jaccard, semantic, ATS, seniority, benchmark
│   ├── step4_suggestions.py        # STEP 4: missing skill ranking + AI/template rewrites
│   ├── step5_charts.py             # STEP 5: donut, gauge, comparison bar charts
│   └── step6_scraper.py            # STEP 6: job URL → text via Jina AI (bonus)
├── data/
│   ├── skills_taxonomy.csv         # Data Science / ML / AI skills (~150)
│   ├── taxonomy_software.csv       # Software Engineering skills (~70)
│   └── ds_resumes_with_skills.csv  # Preprocessed sample resumes
├── tests/
│   ├── conftest.py                 # pytest path setup
│   └── test_matching.py            # 25 unit tests
├── .env                            # API key (local only, gitignored)
├── .env.example                    # Safe template to commit
├── .gitignore
├── requirements.txt
├── runtime.txt
└── README.md
```

> **Why `src/` instead of one file?** The original `app.py` was 520 lines mixing UI, logic, and utilities. Splitting into numbered modules makes each part independently testable and easier to explain in interviews.

---

## 5. Build Process

### Phase 1 — Data exploration
- Loaded `postings.csv` (492MB) in chunks to filter by job title
- Loaded `Resume.csv` directly (small enough)
- Checked missing values, text length distributions, category breakdown

### Phase 2 — Cleaning + skill extraction (`step1_utils.py`, `step2_extractor.py`)
- `clean_text()`: lowercases, strips HTML, fixes camelCase, collapses whitespace
- `extract_resume_text()`: handles both PDF (pdfplumber) and DOCX (python-docx)
- `SkillExtractor`: builds spaCy `PhraseMatcher` from taxonomy (skill name + synonyms)
- `segment_concatenated_skills()`: fixes run-together skill strings from scraped job sites

### Phase 3 — Matching engine (`step3_matching.py`)
- **Keyword score:** Jaccard similarity between resume and job skill sets
- **Semantic score:** cosine similarity between sentence-transformer embeddings
- **Final score:** `keyword_weight * jaccard + (1 - keyword_weight) * semantic`
- `keyword_weight` is user-adjustable via sidebar slider (default 0.4)
- Validated by ranking job postings against sample resumes and checking relevance

### Phase 4 — Gap analysis + suggestions (`step4_suggestions.py`)
- Missing skills = `job_skills - resume_skills`
- Priority scoring: +5 if in job title, +2 per repeated mention, +3 if in first 200 chars
- Template suggestions vary by skill category (technical / tool / soft)
- AI suggestions: Gemini API generates specific, quantified bullet rewrites

### Phase 5 — ATS score (`step3_matching.py`)
Heuristic score based on:
- Email and phone number present
- Standard section headers (experience, education, skills, projects)
- Keyword coverage against the specific job
- Resume length (200–1200 words)
- Presence of quantifiable achievements (numbers/%)

### Phase 6 — New v2 features
- **Seniority mismatch detection:** keyword signals in JD vs resume
- **Benchmark scoring:** estimated percentile from empirical distribution table
- **Multi-job comparison tab:** rank up to 5 jobs with bar chart
- **Live resume editor tab:** edit text, see score delta vs original
- **Job URL scraper:** Jina AI reader converts any public URL to clean text
- **DOCX support:** `python-docx` reads Word document resumes
- **Session state:** resume persists across tabs (upload once, use everywhere)
- **Gemini AI rewrites:** optional, free tier, key loaded from `.env` securely

### Phase 7 — Testing (`tests/`)
- 25 unit tests covering all core pure functions
- `TestJaccardSimilarity`, `TestComputeMatch`, `TestATSScore`, `TestSeniorityMismatch`, `TestBenchmarkPercentile`, `TestCleanText`
- Run with `pytest tests/ -v`

---

## 6. API Key Security

- Gemini API key stored in `.env` locally (gitignored — never committed)
- On Streamlit Cloud: added via **Settings → Secrets** as `GEMINI_API_KEY`
- App reads from `st.secrets` first (cloud), falls back to `os.getenv` (local)
- Key is **never shown in the UI** — only a green badge is displayed

---

## 7. Known Limitations

- Keyword extraction misses skills demonstrated but not named explicitly
- Resume dataset has no dedicated data-science category — IT/Engineering used as proxy
- ATS score is a heuristic, not a real ATS simulation
- PDF parsing fails on scanned (image-only) resumes
- LinkedIn blocks URL scraping — users prompted to paste text manually
- Benchmark percentiles are estimated from a static distribution table, not real data

---

## 8. Possible Future Improvements

- Fine-tune embedding model on labeled resume-job match pairs
- Expand skills taxonomy using ESCO (13,000+ skills)
- Handle LinkedIn profile URL as resume input
- Add user accounts to save history across sessions
- Multi-resume ranking (one job description, rank many resumes)
- Industry presets beyond Data + Software (Marketing, Finance, PM)

---

## 9. Interview Talking Points

- **Jaccard vs semantic:** Jaccard is interpretable and exact; semantic catches paraphrasing. Using both gives better coverage and makes the score explainable.
- **0.4/0.6 weighting:** tested empirically on real pairs — not arbitrary. Now user-adjustable via slider.
- **spaCy PhraseMatcher vs NER:** deterministic, fast, no training data needed for new skills. Tradeoff: must manually maintain taxonomy.
- **Dataset limitation:** no data-science resume category exists — IT/Engineering used as proxy. Acknowledged and documented.
- **Why Gemini Flash for rewrites:** free tier (1,500 req/day), low latency, output is short structured text — no need for a larger model.
- **Full-precision matching gaps:** doesn't capture seniority level in skills (e.g. "used Python for 1 script" vs "built production ML pipelines in Python"), context of skill use, or years of experience.

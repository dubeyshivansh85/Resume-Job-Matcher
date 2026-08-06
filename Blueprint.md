# Resume ↔ Job Matching NLP Tool — Project Blueprint

## 1. Overview

**Problem:** Job seekers don't know how well their resume matches a job posting, or what skills they're missing.

**What it does:** Takes a resume and a job description, and outputs:
- Overall match score (keyword + semantic similarity, weighted)
- Matched skills
- Missing skills, ranked by priority
- Resume improvement suggestions
- ATS compatibility score

## 2. Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10 |
| Resume parsing | pdfplumber |
| Skill extraction | spaCy (PhraseMatcher) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Similarity | scikit-learn (cosine_similarity) |
| Data handling | pandas, numpy |
| App | Streamlit |
| Charts | Plotly |
| Deployment | Streamlit Community Cloud |

## 3. Data Sources

- **Job postings:** [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) (Kaggle) — filtered to titles matching `data scientist|data analyst|machine learning|data engineer`, ~1,187 rows.
- **Resumes:** [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (Kaggle) — filtered to `INFORMATION-TECHNOLOGY` + `ENGINEERING` categories, ~238 rows. No dedicated "Data Science" category exists in this dataset; these two were used as the closest proxy.
- **Skills taxonomy:** hand-built, ~150 skills with synonyms, covering technical skills, tools, and soft skills relevant to data roles.

## 4. Project Structure

```
├── app.py                      # Streamlit app (all matching/scoring logic lives here)
├── data/
│   └── skills_taxonomy.csv     # shipped with the repo
├── notebooks/
│   ├── 1_exploration.ipynb     # load + inspect raw datasets
│   ├── 2_skill_extraction.ipynb  # clean text, extract skills, save processed data
│   └── 3_matching_engine.ipynb   # keyword + semantic matching, ranking, validation
├── requirements.txt
└── README.md
```

No separate `src/` module — the app is small enough that matching logic, skill extraction, and the UI live together in `app.py`.

## 5. Build Process (as actually done)

### Phase 1 — Data exploration
- Loaded `postings.csv` (492MB) in chunks to filter by job title without loading the full file into memory
- Loaded `Resume.csv` directly (small enough to fit in memory)
- Checked missing values, text length distribution, category breakdown

### Phase 2 — Cleaning + skill extraction
- `clean_text()`: lowercases, strips HTML artifacts, fixes concatenated words from scraped text, collapses whitespace
- `SkillExtractor`: builds a spaCy `PhraseMatcher` from the taxonomy (skill name + synonyms), extracts a skill set per resume/posting
- Filtered resumes to IT + Engineering categories before extraction (rest of the dataset is unrelated industries)

### Phase 3 — Matching engine
- **Keyword score:** Jaccard similarity between resume and job skill sets
- **Semantic score:** cosine similarity between sentence-transformer embeddings of the full cleaned text
- **Final score:** `0.4 * jaccard + 0.6 * semantic`
- Validated by ranking all job postings against several sample resumes and checking the top results were topically relevant

### Phase 4 — Gap analysis + suggestions
- Missing skills = `job_skills - resume_skills`
- Priority scoring: +5 if the skill is in the job title, +2 per repeated mention in the description, +3 if it appears in the first 200 characters
- Suggestion templates vary by skill category (technical / tool / soft skill)

### Phase 5 — ATS score
Heuristic score (not a simulation of any real ATS product) based on:
- Email/phone present
- Standard section headers found (experience, education, skills, projects)
- Keyword coverage against the specific job
- Resume length (200–1200 words)
- Presence of quantifiable achievements (numbers/%)

### Phase 6 — Streamlit app
- Upload a resume (PDF, parsed live) or pick from the sample resume set
- Paste a job description + optional job title
- Outputs: match score, ATS score, donut chart of skill coverage, matched/missing skills with priority, improvement suggestions

## 6. Known Limitations

- Keyword extraction misses skills that are demonstrated but not named explicitly (e.g. hyperparameter tuning without the word "machine learning")
- Resume dataset has no dedicated data-science category — IT/Engineering used as proxy, introduces some noise
- ATS score is a heuristic based on common best practices, not a real ATS simulation
- PDF parsing fails on scanned (image-only) resumes

## 7. Possible Improvements

- Multi-resume ranking (one job description, rank many resumes)
- Fine-tune the embedding model on labeled resume-job match pairs
- Expand the skills taxonomy beyond ~150 terms
- Handle DOCX resume uploads, not just PDF

## 8. Interview Talking Points

- Trade-off between keyword matching (interpretable, brittle) and semantic matching (flexible, harder to explain)
- Why the 0.4/0.6 weighting was chosen — tested empirically on real resume/job pairs, not arbitrary
- The dataset limitation (no data-science resume category) and how it was handled
- What full-precision matching would require that this doesn't do: seniority level, years of experience, context around how a skill was used

# Resume ↔ Job Matching Tool

An NLP tool that scores how well a resume matches a job description, shows matched/missing skills, ranks missing skills by priority, suggests resume improvements, and estimates ATS compatibility.

## Live Demo
[![Project Demo](https://img.shields.io/badge/Project%20Demo-Live-brightgreen?style=for-the-badge)](https://resume-job-matcher-fa4gw7ehcqwqwiqhfdwrcp.streamlit.app/)

## How It Works

1. **Skill extraction** — a curated taxonomy of ~150 data/tech skills (with synonyms) is matched against resume and job text using spaCy's `PhraseMatcher`.
2. **Keyword matching** — Jaccard similarity between the resume's skill set and the job's skill set.
3. **Semantic matching** — both texts are embedded with `sentence-transformers` (`all-MiniLM-L6-v2`) and compared with cosine similarity. This catches matches that keyword overlap misses (e.g. "built data pipelines" vs. "ETL development").
4. **Final score** — a weighted average: `0.4 * jaccard + 0.6 * semantic`.
5. **Missing skill priority** — missing skills are ranked by how often they appear in the job posting and whether they're in the job title.
6. **ATS score** — a heuristic check on resume structure (contact info, section headers, keyword coverage, length, quantifiable results).

## Tech Stack

Python, spaCy, sentence-transformers, scikit-learn, pandas, Streamlit, pdfplumber, Plotly.

## Datasets Used

- [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) (Kaggle) — filtered to ~1,200 data-related postings
- [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (Kaggle) — filtered to IT/Engineering categories (~240 resumes). This dataset has no dedicated "Data Science" category; IT + Engineering were used as the closest available proxy.

Raw datasets are not included in this repo (too large for GitHub). See **Setup** below.

## Project Structure

```
├── app.py                      # Streamlit app
├── data/
│   └── skills_taxonomy.csv     # curated skills list (included)
├── notebooks/
│   ├── 1_exploration.ipynb     # data loading + inspection
│   ├── 2_skill_extraction.ipynb
│   └── 3_matching_engine.ipynb
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
2. Download the two Kaggle datasets linked above into a `data/` folder.
3. Run the notebooks in order (`1_exploration` → `2_skill_extraction` → `3_matching_engine`) to generate the processed files the app needs (`ds_resumes_with_skills.csv`, etc.).
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Known Limitations

- Keyword extraction only catches skills phrased in expected ways — a resume that demonstrates a skill without naming it (e.g. describing hyperparameter tuning without writing "machine learning") won't be credited.
- The resume dataset has no data-science-specific category; IT/Engineering was used as the closest proxy, which introduces some noise.
- The ATS score is a heuristic based on common best practices, not a simulation of any specific real ATS product.

## Possible Improvements

- Fine-tune the embedding model on labeled resume-job match pairs
- Expand the skills taxonomy
- Add multi-resume ranking (one job, many resumes)

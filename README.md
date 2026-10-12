# Resume ↔ Job Matching & ATS Analytics NLP Engine

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App%20Interface-FF4B4B?style=for-the-badge&logo=streamlit)](https://resume-job-matcher-fa4gw7ehcqwqwiqhfdwrcp.streamlit.app/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen?style=for-the-badge)](https://resume-job-matcher-fa4gw7ehcqwqwiqhfdwrcp.streamlit.app/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Interactive%20Notebook-F37626?style=for-the-badge&logo=jupyter)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

An industrial-grade NLP pipeline that evaluates candidate resume-to-job fit using a **hybrid lexical overlap + dense semantic embedding architecture**. It performs automated ATS compatibility diagnostics, identifies seniority/experience level misalignments, estimates candidate benchmark percentiles, and generates copy-paste-ready resume bullet rewrites via **Google Gemini**.

---

## 🌐 Live Demo

Experience the live deployed web application:
👉 **[Open Live App on Streamlit Cloud](https://resume-job-matcher-fa4gw7ehcqwqwiqhfdwrcp.streamlit.app/)**

---

## ⚡ Architecture & Key Capabilities

- **Hybrid Scoring Engine:** Balances hard skill keyword presence (spaCy PhraseMatcher with curated DS/ML and SWE taxonomies) and deep contextual understanding (Sentence-BERT `all-MiniLM-L6-v2` dense vectors):
  $$\text{Final Score} = \alpha \cdot \text{Jaccard}(S_{\text{resume}}, S_{\text{job}}) + (1 - \alpha) \cdot \text{Cosine}(E_{\text{resume}}, E_{\text{job}})$$
- **ATS Compatibility Diagnostic:** Audits contact information (email, phone), structural headings (Experience, Education, Skills, Projects), target keyword coverage, document length (200–1200 words), and quantifiable metrics (`%`, `$`, numbers).
- **Seniority & Experience Guardrails:** Detects career level mismatches (e.g. junior profiles targeting Lead/Principal roles, or vice versa).
- **Contextual Skill Prioritization:** Ranks missing competencies by prominence in the job title (+5), opening summary (+3), and document frequency (+2/mention).
- **Generative AI Bullet Rewrites:** Uses Google Gemini (`gemini-3.8-flash`) to generate quantifiable, STAR-method bullet points tailored to missing skills (with automated fallback to heuristic templates).
- **Interactive Visualizations:** High-fidelity Plotly match score speedometer gauges, skill coverage donut charts, and multi-job comparison bar charts.
- **Multi-Format Ingestion:** Supports plain text, selectable PDFs (`pdfplumber`), and Word documents (`python-docx`).

---

## 📁 Clean, Minimal Project Structure

Designed with a consolidated, zero-bloat file structure:

```
Resume-Job-Matcher/
├── resume_job_matcher.ipynb  # Master Jupyter Notebook: End-to-end data science, NLP pipeline & self-checks
├── app.py                    # Standalone Streamlit Web Application (Full GUI, Live Editor, Multi-Job Comparison)
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template for Google Gemini API key
├── .gitignore                # Git ignore configuration
└── README.md                 # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/dubeyshivansh85/Resume-Job-Matcher.git
cd Resume-Job-Matcher
pip install -r requirements.txt
```

### 2. Configure Environment (Optional for AI Rewriting)

Copy `.env.example` to `.env` and add your free Google Gemini API key:

```bash
cp .env.example .env
```

Inside `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(If no API key is provided, the tool automatically uses rule-based heuristic templates for resume recommendations.)*

---

## 💻 Running the Project

### Option A: Interactive Jupyter Notebook (`resume_job_matcher.ipynb`)

Open the notebook in Jupyter Lab, VS Code, Google Colab, or Kaggle:

```bash
jupyter lab resume_job_matcher.ipynb
```

The notebook contains:
1. Mathematical formulation & architecture overview
2. Text cleaning & normalization utilities
3. Embedded domain skill taxonomies (Data Science / ML / AI and Software Engineering)
4. spaCy token-level PhraseMatcher skill extractor
5. Sentence-Transformers dense embedding engine
6. Hybrid match scoring ($\alpha \cdot \text{Jaccard} + (1-\alpha) \cdot \text{Cosine}$)
7. ATS heuristic audit & seniority alignment checks
8. Contextual skill prioritization & Gemini bullet generation
9. Interactive Plotly charts
10. Live demonstrations on realistic candidate profiles
11. Inline 12-point unit assertion test suite (`run_self_checks()`)

### Option B: Interactive Streamlit Web App (`app.py`)

Launch the web app locally:

```bash
streamlit run app.py
```

Features included in the web app:
- **Load Resume:** Upload PDF/DOCX or select from preloaded curated candidate profiles.
- **Single Job Analysis:** Paste job text or provide a public URL to view the match score, ATS breakdown, matched/missing skills, and AI rewrites.
- **Multi-Job Comparison:** Add up to 5 target positions and view a ranked comparison table and grouped bar chart.
- **Live Resume Editor:** Edit your resume in real time and see the match score delta update instantly.

---

## 🧪 Self-Validation & Testing

The pipeline includes built-in verification assertions testing:
- Text cleaning (camelCase splits, HTML removal, punctuation preservation)
- Jaccard set operations (identical, disjoint, and partial overlaps)
- Dense embedding similarity bounds
- ATS criteria checklist calculation
- Seniority mismatch trigger rules
- Benchmark percentile mappings

Run the notebook's test cell or execute directly in Python to verify all assertions pass with zero failures.

---

## 📜 License

MIT License. Free for academic, personal, and commercial use.

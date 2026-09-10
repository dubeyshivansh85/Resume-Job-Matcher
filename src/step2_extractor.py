"""
src/extractor.py  |  STEP 2 — Skill Extraction
-----------------------------------------------
Builds on Step 1 (utils). Uses spaCy PhraseMatcher to find skill keywords.
Contains:
  - SkillExtractor class  : loads taxonomy CSV, builds matcher, extracts skill sets
  - load_taxonomy()       : loads the right CSV based on industry preset
  - get_available_industries() : returns list of supported industry presets
"""

import os
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TAXONOMY_FILES = {
    "Data Science / ML / AI": os.path.join(BASE_DIR, "data", "skills_taxonomy.csv"),
    "Software Engineering": os.path.join(BASE_DIR, "data", "taxonomy_software.csv"),
}


def load_taxonomy(industry: str) -> pd.DataFrame:
    """Load the skills taxonomy CSV for the given industry preset."""
    path = TAXONOMY_FILES.get(industry)
    if not path or not os.path.exists(path):
        # Fall back to data science taxonomy
        path = TAXONOMY_FILES["Data Science / ML / AI"]
    df = pd.read_csv(path)
    df = df.dropna(subset=["skill_name"])
    return df


class SkillExtractor:
    def __init__(self, taxonomy_path: str, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.taxonomy = pd.read_csv(taxonomy_path)
        self._build_matcher()

    def _build_matcher(self):
        self.surface_to_canonical: dict[str, str] = {}
        for _, row in self.taxonomy.iterrows():
            canonical = str(row["skill_name"]).strip().lower()
            surface_forms = [canonical]
            if pd.notna(row.get("synonyms")) and str(row["synonyms"]).strip():
                syns = [
                    s.strip().lower()
                    for s in str(row["synonyms"]).split(",")
                    if s.strip()
                ]
                surface_forms.extend(syns)
            for form in surface_forms:
                self.surface_to_canonical[form] = canonical

        patterns = [self.nlp.make_doc(form) for form in self.surface_to_canonical.keys()]
        self.matcher.add("SKILL", patterns)

    def extract(self, text: str) -> set:
        if not isinstance(text, str) or not text.strip():
            return set()
        doc = self.nlp(text)
        matches = self.matcher(doc)
        found = set()
        for match_id, start, end in matches:
            span_text = doc[start:end].text.lower()
            canonical = self.surface_to_canonical.get(span_text, span_text)
            found.add(canonical)
        return found


def get_available_industries() -> list[str]:
    return list(TAXONOMY_FILES.keys())

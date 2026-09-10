"""
src/scraper.py  |  STEP 6 — Job URL Scraper (Bonus Feature)
------------------------------------------------------------
Stand-alone utility — doesn't depend on other steps.
Converts a job posting URL into plain text so users don't have to copy-paste.
Contains:
  - fetch_job_from_url() : fetches any public job URL via Jina AI (free, no API key)
                           handles blocked sites (LinkedIn) with a helpful message
"""

import re
import requests

JINA_BASE = "https://r.jina.ai/"
TIMEOUT = 15  # seconds

# Sites known to block scrapers
BLOCKED_DOMAINS = ["linkedin.com", "lnkd.in"]


def fetch_job_from_url(url: str) -> tuple[str, str | None]:
    """
    Fetch job description text from a public URL.

    Uses Jina AI's free reader (r.jina.ai) which converts any page to
    clean markdown text — no API key needed.

    Returns:
        (text, error_msg) — on success error_msg is None; on failure text is "".
    """
    url = url.strip()
    if not url:
        return "", "Please enter a URL."

    # Add scheme if missing
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # Check for known blocked domains
    for domain in BLOCKED_DOMAINS:
        if domain in url:
            return "", (
                f"⚠️ **{domain} blocks automated access** to job pages.\n\n"
                "**Workaround:** Open the job in your browser → select all text → "
                "paste it in the 'Paste job description' box instead."
            )

    try:
        reader_url = JINA_BASE + url
        response = requests.get(
            reader_url,
            headers={"Accept": "text/plain"},
            timeout=TIMEOUT,
        )
        if response.status_code != 200:
            return "", (
                f"⚠️ Couldn't fetch the page (HTTP {response.status_code}). "
                "The site may block automated access. Try pasting the text directly."
            )

        text = response.text.strip()
        if not text or len(text) < 100:
            return "", (
                "⚠️ Fetched page has very little text — the site may require login "
                "or use JavaScript rendering. Try pasting the job description manually."
            )

        # Strip out markdown link syntax and image tags for cleaner text
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"\s{3,}", "\n\n", text)

        return text, None

    except requests.Timeout:
        return "", "⚠️ Request timed out. The site may be slow or blocked. Try pasting the text instead."
    except Exception as e:
        return "", f"⚠️ Error fetching URL: {e}"

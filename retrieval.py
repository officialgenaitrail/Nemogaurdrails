"""Lightweight, local TF-IDF retrieval for the fictional support-policy knowledge base."""
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_FILE = Path(__file__).resolve().parent / "docs" / "policies.md"


def read_sections(path=KB_FILE):
    text = Path(path).read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", text))
    return [
        {"title": match.group(1),
         "body": text[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()}
        for i, match in enumerate(matches)
    ]


SECTIONS = read_sections()
VECTORIZER = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
MATRIX = VECTORIZER.fit_transform(
    [section["title"] + " " + section["body"] for section in SECTIONS]
)


def retrieve(question: str, top_k: int = 2, min_similarity: float = 0.12):
    """Return matching demo sections and similarity scores; no embeddings API required."""
    if not question.strip():
        return []
    question_vec = VECTORIZER.transform([question])
    scores = cosine_similarity(question_vec, MATRIX)[0]
    best = scores.argsort()[::-1][:top_k]
    return [
        {"title": SECTIONS[i]["title"], "body": SECTIONS[i]["body"], "score": float(scores[i])}
        for i in best if scores[i] >= min_similarity
    ]


def context_for(question: str) -> str:
    hits = retrieve(question)
    return "\n\n".join(f"[Policy: {hit['title']}]\n{hit['body']}" for hit in hits)

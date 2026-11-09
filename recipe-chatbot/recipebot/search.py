"""Step 2a: retrieval = TF-IDF ranking + HARD ingredient constraints.

Design
- Index: TF-IDF over (title x2 + ingredient lines). Built once, cached to data/tfidf.joblib.
- Hard constraints (include / exclude) are applied with hard_filter() below, never as soft scores.
  Candidates are scanned in descending TF-IDF order, in chunks, and only recipes that PASS the filter
  are returned, so a constraint can never be silently relaxed.
- Exclusion groups ("dairy", "nuts") expand to a keyword list. This is a heuristic list, not a
  medical guarantee, and the result says so.
- Search returns only the top-k; aggregates must use analytics.* on the full filtered set.
"""
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Compounds that refer to different foods are stripped for both ingredient matching
# and exclusions. Derivatives are stripped only when matching what the user has.
DIFFERENT_ITEMS = {
    "butter": ["peanut butter", "almond butter", "apple butter", "cashew butter", "cocoa butter", "buttermilk", "butternut"],
    "milk": ["coconut milk", "almond milk", "soy milk", "soymilk", "oat milk", "rice milk"],
    "cream": ["cream of tartar", "coconut cream"],
    "egg": ["eggplant"],
    "pepper": ["bell pepper", "red pepper", "green pepper", "cayenne pepper", "chile pepper", "pepper flakes"],
    "nut": ["coconut", "butternut", "doughnut", "walnut", "peanut", "chestnut"],
}
DERIVATIVES = {
    "tomato": ["tomato paste", "tomato sauce", "tomato puree", "tomato juice",
               "tomato soup", "tomato ketchup", "sun-dried tomato", "sundried tomato"],
    "rice": ["rice vinegar", "rice wine", "rice flour", "rice noodles", "rice paper", "rice cereal"],
    "onion": ["onion powder", "onion soup mix"],
    "egg": ["egg noodles"],
    "garlic": ["garlic powder", "garlic salt"],
}
SYNONYMS = {"scallion": "green onion", "scallions": "green onions", "capsicum": "bell pepper",
            "aubergine": "eggplant", "courgette": "zucchini"}


_NO_SINGULAR = {"hummus", "molasses", "asparagus", "couscous", "swiss", "lemongrass", "watercress", "grits", "oats"}


def singular(term):
    """Naive singular of the LAST word: tomatoes->tomato, eggs->egg, berries->berry (exceptions listed)."""
    words = term.lower().strip().split(" ")
    w = words[-1]
    if w in _NO_SINGULAR or len(w) <= 3 or w.endswith("ss"):
        return " ".join(words)
    if w.endswith("ies"):
        w = w[:-3] + "y"
    elif w.endswith(("oes", "ches", "shes", "xes")):
        w = w[:-2]
    elif w.endswith("s"):
        w = w[:-1]
    words[-1] = w
    return " ".join(words)


def term_pattern(t):
    """Whole-word regex for a singular term, tolerating the plural form (cherry~cherries, tomato~tomatoes)."""
    if t.endswith("y") and len(t) > 2:
        return rf"\b{re.escape(t[:-1])}(?:y|ies)\b"
    return rf"\b{re.escape(t)}(?:es|s)?\b"


def canon_term(term):
    t = term.lower().strip()
    t = SYNONYMS.get(t, t)
    return singular(SYNONYMS.get(singular(t), singular(t)))


def _compound_key(term):
    """Use the singular canonical term as the lookup key for compound tables."""
    return singular(canon_term(term))


def _term_mask(df, term, for_exclusion=False):
    t = canon_term(term)
    text = df["ing_text"]
    compound_key = _compound_key(t)
    compounds = DIFFERENT_ITEMS.get(compound_key, [])
    if not for_exclusion:
        compounds = compounds + DERIVATIVES.get(compound_key, [])
    for comp in compounds:
        text = text.str.replace(comp, " ", regex=False)
    return text.str.contains(term_pattern(t), regex=True).to_numpy()


def hard_filter(df, include, exclude):
    """Boolean numpy mask: recipe contains ALL include terms and NONE of the exclude terms."""
    m = np.ones(len(df), dtype=bool)
    for t in include:
        m &= _term_mask(df, t, for_exclusion=False)
    for t in exclude:
        m &= ~_term_mask(df, t, for_exclusion=True)
    return m

INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "tfidf.joblib"

GROUPS = {
    "dairy": ["milk", "cream", "butter", "cheese", "yogurt", "yoghurt", "ghee", "whey", "casein",
              "buttermilk", "sour cream", "half-and-half", "half and half", "ice cream", "custard",
              "parmesan", "mozzarella", "cheddar", "ricotta", "feta", "mascarpone", "paneer", "creme fraiche"],
    "nuts": ["almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "peanut", "macadamia", "pine nut"],
}
_STOP = {"recipe", "recipes", "find", "show", "with", "for", "the", "and", "some", "any", "dish", "me", "a", "an", "of"}


def expand_terms(terms):
    out = []
    for t in terms:
        t = re.sub(r"[- ]free$", "", t.lower().strip())
        for x in GROUPS.get(t, [t]):
            if x and x not in out:
                out.append(x)
    return out


class SearchIndex:
    def __init__(self, vec, X):
        self.vec, self.X = vec, X

    @staticmethod
    def _docs(df):
        return (df["title"].str.lower() + " ") * 2 + df["ing_text"].str.replace("\n", " ", regex=False)

    @classmethod
    def load_or_build(cls, df, path=INDEX_PATH, rebuild=False):
        path = Path(path)
        if path.exists() and not rebuild:
            vec, X = joblib.load(path)
            if X.shape[0] == len(df):
                return cls(vec, X)
        print("Building TF-IDF index (one-time, a few minutes)...")
        vec = TfidfVectorizer(token_pattern=r"(?u)\b[a-z][a-z]+\b", stop_words="english", ngram_range=(1, 2),
                              min_df=3 if len(df) > 1000 else 1, max_features=200_000, sublinear_tf=True, dtype=np.float32)
        X = vec.fit_transform(cls._docs(df))
        path.parent.mkdir(exist_ok=True)
        joblib.dump((vec, X), path)
        return cls(vec, X)


def evidence_lines(ingredients, tokens, n=3):
    hits = [l for l in ingredients if any(re.search(rf"\b{re.escape(t)}", l.lower()) for t in tokens)]
    return hits[:n]


def search(df, index, query, include=(), exclude=(), k=5, chunk=2000, max_scan=60000):
    """Returns dict with results (id, title, score, n_steps, evidence) that ALL satisfy the hard constraints."""
    exc = expand_terms(exclude)
    inc = [t.lower() for t in include]
    q = index.vec.transform([query.lower()])
    scores = (index.X @ q.T).toarray().ravel()
    order = np.argsort(-scores)[: min(int((scores > 0).sum()), max_scan)]
    tokens = [w for w in re.findall(r"[a-z]{3,}", query.lower()) if w not in _STOP]
    results, scanned = [], 0
    for start in range(0, len(order), chunk):
        pos = order[start:start + chunk]
        scanned += len(pos)
        ok = set(pos[hard_filter(df.iloc[pos], inc, exc)].tolist())
        for p in pos:
            if int(p) in ok:
                r = df.iloc[p]
                results.append({"id": r["id"], "title": r["title"], "score": round(float(scores[p]), 4),
                                "n_steps": int(r["n_steps"]), "source": r["source"],
                                "evidence": evidence_lines(r["ingredients"], tokens + inc)})
                if len(results) == k:
                    break
        if len(results) == k:
            break
    note = None
    if not results:
        note = (f"No recipe among the top {scanned} matches for '{query}' satisfies the constraints "
                f"(include={inc}, exclude={exc}). Constraints were not relaxed.")
    elif exc:
        note = "Exclusion check = none of these keywords appear in the ingredient lines: " + ", ".join(exc) + \
               ". Heuristic keyword list, not a medical/allergy guarantee."
    return {"query": query, "include": inc, "exclude": list(exclude), "exclude_expanded": exc,
            "n_scanned": scanned, "results": results, "note": note}
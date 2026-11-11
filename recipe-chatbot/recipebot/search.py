"""Step 2a: retrieval = hybrid TF-IDF/semantic ranking + HARD ingredient constraints.

Design
- Index: TF-IDF over title/ingredient text plus optional local sentence embeddings.
  The dense index is cached separately from data/tfidf.joblib.
- Hard constraints (include / exclude) are applied with hard_filter() below, never as soft scores.
  Candidates are scanned in hybrid relevance order, in chunks, and only recipes that PASS the filter
  are returned, so a constraint can never be silently relaxed.
- Exclusion groups ("dairy", "nuts") expand to a keyword list. This is a heuristic list, not a
  medical guarantee, and the result says so.
- Search returns only the top-k; aggregates must use analytics.* on the full filtered set.
"""
import difflib
import hashlib
import os
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover
    fuzz = process = None

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
    """Naive singular of the LAST word: tomatoes->tomato, eggs->egg, berries->berry."""
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
    if t.endswith("y") and len(t) > 2:
        return rf"\b{re.escape(t[:-1])}(?:y|ies)\b"
    return rf"\b{re.escape(t)}(?:es|s)?\b"


def canon_term(term):
    t = term.lower().strip()
    t = SYNONYMS.get(t, t)
    return singular(SYNONYMS.get(singular(t), singular(t)))


def _compound_key(term):
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
    m = np.ones(len(df), dtype=bool)
    for t in include:
        m &= _term_mask(df, t, for_exclusion=False)
    for t in exclude:
        m &= ~_term_mask(df, t, for_exclusion=True)
    return m

INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "tfidf.joblib"
VOCABULARY_PATH = INDEX_PATH.with_name("recipe_vocabulary.joblib")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROUPS = {
    "dairy": ["milk", "cream", "butter", "cheese", "yogurt", "yoghurt", "ghee", "whey", "casein",
              "buttermilk", "sour cream", "half-and-half", "half and half", "ice cream", "custard",
              "parmesan", "mozzarella", "cheddar", "ricotta", "feta", "mascarpone", "paneer", "creme fraiche"],
    "nuts": ["almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "peanut", "macadamia", "pine nut"],
}
_STOP = {"recipe", "recipes", "find", "show", "with", "for", "the", "and", "some", "any", "dish", "me", "a", "an", "of"}
_UNSPLIT_COMPOUNDS = {"cheesecake", "pancakes"}
_SPLIT_COMPOUNDS = {"icecream": ("ice", "cream")}


def expand_terms(terms):
    out = []
    for t in terms:
        t = re.sub(r"[- ]free$", "", t.lower().strip())
        for x in GROUPS.get(t, [t]):
            if x and x not in out:
                out.append(x)
    return out


def _correct_token(token, vocabulary):
    if token in _SPLIT_COMPOUNDS:
        return list(_SPLIT_COMPOUNDS[token])
    if token in vocabulary:
        return [token]
    if token in _UNSPLIT_COMPOUNDS:
        return [token]
    if len(token) >= 4 and vocabulary:
        if process is not None:
            match = process.extractOne(token, vocabulary, scorer=fuzz.ratio, score_cutoff=80)
            if match:
                return [match[0]]
        else:
            matches = difflib.get_close_matches(token, sorted(vocabulary), n=1, cutoff=0.8)
            if matches:
                return matches
    for split_at in range(2, len(token) - 1):
        left, right = token[:split_at], token[split_at:]
        if left in vocabulary and right in vocabulary:
            return [left, right]
    return [token]


def normalize_query(query, vocabulary=()):
    words = re.findall(r"[a-z]+", query.lower())
    normalized_words = [corrected for word in words for corrected in _correct_token(word, vocabulary)]
    return " ".join(normalized_words)


def normalize_term(term, vocabulary=()):
    cleaned = re.sub(r"[- ]free$", "", term.strip().lower())
    return normalize_query(cleaned, vocabulary)


class SearchIndex:
    def __init__(self, vec, X, vocabulary=None, semantic_X=None, semantic_encoder=None, semantic_model_name=None, semantic_notice=None):
        self.vec, self.X = vec, X
        self.vocabulary = frozenset(vocabulary or ())
        self.semantic_X = semantic_X
        self.semantic_encoder = semantic_encoder
        self.semantic_model_name = semantic_model_name
        self.semantic_notice = semantic_notice

    @staticmethod
    def _docs(df):
        return (df["title"].str.lower() + " ") * 2 + df["ing_text"].str.replace("\n", " ", regex=False)

    @classmethod
    def load_or_build(cls, df, path=INDEX_PATH, rebuild=False, semantic=False, semantic_encoder=None, semantic_model_name=None, build_semantic=True):
        path = Path(path)
        vocabulary_path = path.with_name(f"{path.stem}.vocabulary.joblib")
        vec = X = vocabulary = None
        if path.exists() and not rebuild:
            cached = joblib.load(path)
            vec, X = cached[:2]
            if X.shape[0] == len(df):
                if vocabulary_path.exists():
                    vocabulary = joblib.load(vocabulary_path)
                else:
                    vocabulary = cls._build_vocabulary(df)
                    joblib.dump(vocabulary, vocabulary_path)
        if vec is None:
            print("Building TF-IDF index (one-time, a few minutes)...")
            vec = TfidfVectorizer(token_pattern=r"(?u)\b[a-z][a-z]+\b", stop_words="english", ngram_range=(1, 2), min_df=3 if len(df) > 1000 else 1, max_features=200_000, sublinear_tf=True, dtype=np.float32)
            X = vec.fit_transform(cls._docs(df))
            path.parent.mkdir(exist_ok=True)
            joblib.dump((vec, X), path)
            vocabulary = cls._build_vocabulary(df)
            joblib.dump(vocabulary, vocabulary_path)
        semantic_X = None
        semantic_notice = None
        model_name = semantic_model_name or os.environ.get("RECIPEBOT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        if semantic:
            backend_errors = (OSError, RuntimeError, ValueError)
            try:
                from huggingface_hub.errors import HfHubError
            except ImportError:
                pass
            else:
                backend_errors += (HfHubError,)
            semantic_path = path.with_name(f"{path.stem}.semantic.joblib")
            docs = None
            fingerprint = None
            if semantic_path.exists() and not rebuild:
                docs = cls._docs(df).tolist()
                fingerprint = cls._corpus_fingerprint(docs)
                cached_semantic = joblib.load(semantic_path)
                if cached_semantic.get("model") == model_name and cached_semantic.get("fingerprint") == fingerprint and cached_semantic["embeddings"].shape[0] == len(df):
                    semantic_X = cached_semantic["embeddings"]
            if semantic_X is None and not build_semantic:
                semantic_notice = "Hybrid semantic index is not built or is stale; using TF-IDF search. Run 'python -m recipebot.cli --build-hybrid-index' to build it."
            else:
                if semantic_encoder is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError:
                        semantic_notice = "Local semantic model unavailable (install sentence-transformers); using TF-IDF search."
                    else:
                        try:
                            semantic_encoder = SentenceTransformer(model_name)
                        except backend_errors as error:
                            semantic_notice = f"Local semantic model failed ({type(error).__name__}); using TF-IDF search."
                if semantic_encoder is not None and semantic_X is None:
                    docs = docs or cls._docs(df).tolist()
                    try:
                        print(f"Building local semantic index with {model_name} (cached after first run)...")
                        semantic_X = semantic_encoder.encode(docs, batch_size=128, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32, copy=False)
                        joblib.dump({"model": model_name, "fingerprint": fingerprint or cls._corpus_fingerprint(docs), "embeddings": semantic_X}, semantic_path)
                    except backend_errors as error:
                        semantic_X = None
                        semantic_encoder = None
                        semantic_notice = f"Local semantic model failed ({type(error).__name__}); using TF-IDF search."
        return cls(vec, X, vocabulary, semantic_X, semantic_encoder, model_name if semantic_encoder is not None else None, semantic_notice)

    @staticmethod
    def _corpus_fingerprint(docs):
        digest = hashlib.sha256()
        for doc in docs:
            digest.update(doc.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def semantic_scores(self, query):
        if self.semantic_X is None or self.semantic_encoder is None:
            return None
        query_embedding = self.semantic_encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return self.semantic_X @ np.asarray(query_embedding, dtype=np.float32).reshape(-1)

    @staticmethod
    def _build_vocabulary(df):
        docs = SearchIndex._docs(df)
        return frozenset(token for doc in docs for token in re.findall(r"[a-z]+", doc))


def evidence_lines(ingredients, tokens, n=3):
    hits = [line for line in ingredients if any(re.search(rf"\b{re.escape(t)}", line.lower()) for t in tokens)]
    return hits[:n]


def reciprocal_rank_fusion(lexical_scores, semantic_scores=None, lexical_weight=0.55, semantic_weight=0.45, rank_constant=60):
    lexical_scores = np.asarray(lexical_scores)
    lexical_order = np.argsort(-lexical_scores, kind="stable")
    if semantic_scores is None:
        return lexical_order, lexical_scores
    semantic_scores = np.asarray(semantic_scores)
    semantic_order = np.argsort(-semantic_scores, kind="stable")
    semantic_rank = np.empty(len(semantic_order), dtype=np.int32)
    semantic_rank[semantic_order] = np.arange(len(semantic_order))
    fused_scores = semantic_weight / (rank_constant + semantic_rank + 1)
    lexical_matches = lexical_scores > 0
    matching_order = np.argsort(-lexical_scores[lexical_matches], kind="stable")
    matching_positions = np.flatnonzero(lexical_matches)[matching_order]
    fused_scores[matching_positions] += lexical_weight / (rank_constant + np.arange(len(matching_positions)) + 1)
    return np.argsort(-fused_scores, kind="stable"), fused_scores


def search(df, index, query, include=(), exclude=(), k=5, chunk=2000, max_scan=60000):
    vocabulary = index.vocabulary
    original_query = query.strip().lower()
    normalized_query = normalize_query(original_query, vocabulary)
    corrected = []
    if normalized_query != original_query:
        corrected.append(f"Interpreting '{original_query}' as '{normalized_query}'.")
    normalized_include = [normalize_term(term, vocabulary) for term in include]
    normalized_exclude = [normalize_term(term, vocabulary) for term in exclude]
    for original, normalized in zip(include, normalized_include):
        if normalized != original.strip().lower():
            corrected.append(f"Interpreting restriction '{original}' as '{normalized}'.")
    for original, normalized in zip(exclude, normalized_exclude):
        if normalized != original.strip().lower():
            corrected.append(f"Interpreting restriction '{original}' as '{normalized}'.")
    exc = expand_terms(normalized_exclude)
    inc = normalized_include
    q = index.vec.transform([normalized_query])
    scores = (index.X @ q.T).toarray().ravel()
    dense_scores = index.semantic_scores(normalized_query) if hasattr(index, "semantic_scores") else None
    order, ranked_scores = reciprocal_rank_fusion(scores, dense_scores)
    positive = int((scores > 0).sum())
    if dense_scores is None:
        if positive:
            order = order[scores[order] > 0]
        else:
            order = np.arange(min(len(df), max_scan))
    else:
        order = order[:max_scan]
    tokens = [w for w in re.findall(r"[a-z]{3,}", normalized_query) if w not in _STOP]
    results, scanned = [], 0
    for start in range(0, len(order), chunk):
        pos = order[start:start + chunk]
        scanned += len(pos)
        ok = set(pos[hard_filter(df.iloc[pos], inc, exc)].tolist())
        for p in pos:
            if int(p) in ok:
                r = df.iloc[p]
                results.append({
                    "id": r["id"],
                    "title": r["title"],
                    "score": round(float(ranked_scores[p]), 6) if dense_scores is not None else round(float(scores[p]), 4),
                    "n_steps": int(r["n_steps"]),
                    "source": r.get("source", ""),
                    "evidence": evidence_lines(r["ingredients"], tokens + inc),
                })
                if len(results) == k:
                    break
        if len(results) == k:
            break
    note = None
    if not positive and len(df):
        if dense_scores is not None:
            note = "No query words matched lexically; ranking uses local semantic similarity."
        else:
            note = "No query words matched the recipe vocabulary; showing general recipes instead of returning an empty list."
    if not results:
        note = f"No recipe among the top {scanned} matches for '{normalized_query}' satisfies the constraints (include={inc}, exclude={exc}). Constraints were not relaxed."
    elif exc and not note:
        note = "Exclusion check = none of these keywords appear in the ingredient lines: " + ", ".join(exc) + ". Heuristic keyword list, not a medical/allergy guarantee."
    return {
        "query": normalized_query,
        "original_query": original_query,
        "interpreting": corrected,
        "include": inc,
        "exclude": list(normalized_exclude),
        "exclude_expanded": exc,
        "n_scanned": scanned,
        "results": results,
        "note": note,
        "retrieval_mode": "hybrid" if dense_scores is not None else "tfidf",
    }

"""Grounded recipe analytics; counts and statistics are computed from the DataFrame.

Ingredient queries are lowercased and matched as whole words with the optional
plural suffix ``(?:es|s)?``. Before matching, known compound products are removed
from ingredient text (for example, ``peanut butter`` and ``buttermilk`` do not
match ``butter``; ``coconut milk`` and ``buttermilk`` do not match ``milk``;
``eggplant`` does not match ``egg``; and ``ice cream``/``cream of tartar`` do
not match ``cream``). Queries also recognize these synonyms:
``scallion`` -> ``green onion``, ``capsicum`` -> ``bell pepper``,
``aubergine`` -> ``eggplant``, and ``courgette`` -> ``zucchini``.
"""
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

import pandas as pd

COMPOUNDS = {
    "butter": (
        "peanut butter", "almond butter", "apple butter", "cashew butter",
        "cocoa butter", "buttermilk",
    ),
    "milk": (
        "coconut milk", "almond milk", "soy milk", "soymilk", "oat milk",
        "rice milk", "buttermilk",
    ),
    "cream": ("cream of tartar", "ice cream", "coconut cream"),
    "egg": ("eggplant",),
}
SYNONYMS = {
    "scallion": "green onion",
    "capsicum": "bell pepper",
    "aubergine": "eggplant",
    "courgette": "zucchini",
}


def _terms(values: Iterable[str] | str) -> list[str]:
    return [values] if isinstance(values, str) else list(values)


def _canon(term: str) -> str:
    normalized = term.lower().strip()
    if normalized in SYNONYMS:
        return SYNONYMS[normalized]
    if normalized.endswith("s") and normalized[:-1] in SYNONYMS:
        return SYNONYMS[normalized[:-1]]
    for synonym in SYNONYMS.values():
        if normalized in (synonym, synonym + "s"):
            return synonym
    return normalized


def _term_mask(df: pd.DataFrame, term: str) -> pd.Series:
    canonical = _canon(term)
    text = df["ing_text"]
    for compound in COMPOUNDS.get(canonical, ()):
        pattern = rf"(?<!\w){re.escape(compound)}(?!\w)"
        text = text.str.replace(pattern, " ", regex=True)
    equivalents = {canonical}
    equivalents.update(
        synonym for synonym, target in SYNONYMS.items()
        if target == canonical
    )
    pattern = r"\b(?:" + "|".join(
        re.escape(equivalent) + r"(?:es|s)?" for equivalent in sorted(equivalents)
    ) + r")\b"
    return text.str.contains(pattern, regex=True, na=False)


def _filter_frame(df: pd.DataFrame, include: Iterable[str] | str,
                  exclude: Iterable[str] | str) -> pd.DataFrame:
    included = _terms(include)
    excluded = _terms(exclude)
    matched = pd.Series(True, index=df.index)
    for term in included:
        matched &= _term_mask(df, term)
    for term in excluded:
        matched &= ~_term_mask(df, term)
    return df.loc[matched]


def _envelope(df: pd.DataFrame, sub: pd.DataFrame, include: list[str],
              exclude: list[str], calculation: str) -> dict[str, Any]:
    return {
        "filters": {"include_all": include, "exclude": exclude},
        "n_dataset": len(df),
        "n_matched": len(sub),
        "calculation": calculation,
    }


def filter_recipes(df: pd.DataFrame, include=(), exclude=()) -> dict[str, Any]:
    """Return matching recipe IDs and original titles; include terms are ALL-of."""
    included, excluded = _terms(include), _terms(exclude)
    sub = _filter_frame(df, included, excluded)
    out = _envelope(
        df, sub, included, excluded,
        "n_matched = recipes containing every include term and none of the exclude "
        "terms, using the documented ingredient matching rules",
    )
    out["results"] = sub[["id", "title"]].to_dict("records")
    return out


def count_and_percent(df: pd.DataFrame, include=(), exclude=()) -> dict[str, Any]:
    included, excluded = _terms(include), _terms(exclude)
    sub = _filter_frame(df, included, excluded)
    percent = round(100 * len(sub) / len(df), 3) if len(df) else 0.0
    out = _envelope(
        df, sub, included, excluded,
        f"count = {len(sub)} recipes matching all include terms and no exclude terms; "
        f"percent = {len(sub)} / {len(df)} recipes in the full supplied dataset * 100, "
        f"rounded to 3 decimal places",
    )
    out.update(count=len(sub), percent=percent)
    return out


def ingredient_count_stats(df: pd.DataFrame, include=(), exclude=()) -> dict[str, Any]:
    included, excluded = _terms(include), _terms(exclude)
    sub = _filter_frame(df, included, excluded)
    calculation = (
        "mean = arithmetic mean of n_ingredients for matching recipes, rounded to "
        "2 decimal places; median = median of those values; min and max = smallest "
        "and largest values in that set"
    )
    if not len(sub):
        calculation += "; all four statistics are null because no recipes matched"
    out = _envelope(df, sub, included, excluded, calculation)
    if len(sub):
        out.update(
            mean=round(float(sub["n_ingredients"].mean()), 2),
            median=float(sub["n_ingredients"].median()),
            min=int(sub["n_ingredients"].min()),
            max=int(sub["n_ingredients"].max()),
        )
    else:
        out.update(mean=None, median=None, min=None, max=None)
    return out


def fewest_steps(df: pd.DataFrame, include=(), exclude=(), k: int = 5) -> dict[str, Any]:
    included, excluded = _terms(include), _terms(exclude)
    sub = _filter_frame(df, included, excluded)
    top = sub.sort_values(["n_steps", "id"], ascending=[True, True]).head(k)
    out = _envelope(
        df, sub, included, excluded,
        f"results are matching recipes sorted by n_steps ascending, with ties by id; "
        f"return at most {k} (n_steps is the hybrid instruction-step count)",
    )
    out["results"] = top[["id", "title", "n_steps"]].to_dict("records")
    return out


def most_steps(df: pd.DataFrame, include=(), exclude=(), k: int = 5) -> dict[str, Any]:
    included, excluded = _terms(include), _terms(exclude)
    sub = _filter_frame(df, included, excluded)
    top = sub.sort_values(["n_steps", "id"], ascending=[False, True]).head(k)
    out = _envelope(
        df, sub, included, excluded,
        f"results are matching recipes sorted by n_steps descending, with ties by id; "
        f"return at most {k} (n_steps is the hybrid instruction-step count)",
    )
    out["results"] = top[["id", "title", "n_steps"]].to_dict("records")
    return out


_QTY = re.compile(r"\([^)]*\)|[\d/½¼¾⅓⅔.\-]+")
_UNITS = re.compile(
    r"\b(cups?|tbsps?|tsps?|tablespoons?|teaspoons?|oz|ounces?|lbs?|pounds?|"
    r"grams?|g|kg|ml|cloves?|pinch|dash|cans?|packages?|sticks?|slices?)\b"
)
_PREP = re.compile(
    r"\b(chopped|minced|diced|sliced|fresh|freshly|large|small|medium|ground|"
    r"melted|softened|sifted|optional|finely|coarsely|divided|to taste|packed|"
    r"plus|for)\b"
)


def ingredient_key(line: str) -> str:
    """Derive a frequency-counting name from an ingredient line."""
    name = line.lower().split(",", 1)[0]
    name = _QTY.sub(" ", name)
    name = _UNITS.sub(" ", name)
    name = _PREP.sub(" ", name)
    name = re.sub(r"\b(of|and|or|the|a|an)\b", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def common_ingredients(df: pd.DataFrame, include=(), exclude=(),
                       k: int = 15) -> dict[str, Any]:
    """Count each derived ingredient name at most once per matching recipe."""
    included, excluded = _terms(include), _terms(exclude)
    sub = _filter_frame(df, included, excluded)
    counts: Counter[str] = Counter()
    for ingredients in sub["ingredients"]:
        counts.update({
            name for line in ingredients
            if (name := ingredient_key(line))
        })
    out = _envelope(
        df, sub, included, excluded,
        "ingredient name = lowercased ingredient line before the first comma, with "
        "quantities, listed units, listed preparation words, and the stop words "
        "of/and/or/the/a/an removed; each name is counted at most once per matching "
        "recipe; share = recipes containing that name / n_matched, rounded to "
        "3 decimal places",
    )
    out["results"] = [
        {"ingredient": name, "recipes": count,
         "share": round(count / len(sub), 3)}
        for name, count in counts.most_common(k)
    ] if len(sub) else []
    return out

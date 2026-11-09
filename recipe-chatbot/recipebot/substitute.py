"""
recipebot/substitute.py  --  evidence-based ingredient substitution (Task 3)

Principle: the dataset supplies the evidence, code supplies the numbers, and
nothing here "recalls" what a good substitute is. A small curated ROLES table
names *plausible* substitutes; the dataset decides whether they earn
"strong" or "weak". Anything the data merely discovers is capped at "indirect"
and never produces an adaptation.

How the evidence pool is built (the "structural match rule")
  1. core    = the recipe's ingredient names, minus the target and pantry
               items (salt, water, pepper, oil).
  2. anchors = core names that also appear in the recipe TITLE
               (banana in "Banana Bread", rice in "Saffron Rice"). Every
               comparison recipe must contain all anchors.
  3. A comparison recipe must also contain >= max(2, ceil(0.5 * len(core)))
               of the core names.
  4. pool_A = comparison recipes WITHOUT the target, pool_B = WITH the target.
     Each side is ranked by core overlap and capped at POOL_CAP.

Evidence levels (curated candidates only; share_B floored at LIFT_FLOOR)
     lift   = share_A / max(share_B, LIFT_FLOOR)
     strong : support_A >= 5, share_A >= 10%, lift >= 2
     weak   : support_A >= 3, share_A >=  5%, lift >= 1.5
     none   : otherwise (not listed)
  Data-discovered candidates are always "indirect".

Public API (names kept for session.py / cli.py):
  find_substitutes(df, index, recipe_id, target, restrictions, k) -> dict
  format_substitution(out) -> str
  substitute(df, recipe_id, ingredient, restrictions=("dairy",)) -> dict
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter

import numpy as np

# --------------------------------------------------------------------------
# Tunables (change these, not the logic)
# --------------------------------------------------------------------------
POOL_CAP = 500            # max recipes kept per side (with / without target)
CORE_FRACTION = 0.5       # share of core names a comparison recipe must contain
MIN_POOL = 5              # fewer comparable recipes than this -> insufficient
LIFT_FLOOR = 0.02         # floor for share_B so a 0% share does not give lift = inf

STRONG_MIN_SUPPORT, STRONG_MIN_SHARE, STRONG_MIN_LIFT = 5, 0.10, 2.0
WEAK_MIN_SUPPORT, WEAK_MIN_SHARE, WEAK_MIN_LIFT = 3, 0.05, 1.5

DISC_MIN_SUPPORT, DISC_MIN_SHARE = 5, 0.10
MAX_EXAMPLES = 3

# --------------------------------------------------------------------------
# Curated roles: the ONLY source of "strong"/"weak" candidates.
# --------------------------------------------------------------------------
ROLES = {
    "egg": {
        "role": "binder / moisture in baked goods",
        "candidates": ["applesauce", "banana", "yogurt", "mayonnaise", "flaxseed",
                       "chia seed", "silken tofu", "aquafaba", "pumpkin"],
    },
    "cream": {
        "role": "richness / liquid in sauces and soups",
        "candidates": ["coconut milk", "coconut cream", "evaporated milk", "milk",
                       "half-and-half", "yogurt", "silken tofu", "cashew"],
    },
    "butter": {
        "role": "fat / moisture in baking and cooking",
        "candidates": ["margarine", "shortening", "applesauce", "yogurt",
                       "coconut oil", "vegetable oil", "canola oil", "olive oil"],
    },
    "milk": {
        "role": "liquid / richness",
        "candidates": ["water", "soy milk", "almond milk", "oat milk", "coconut milk",
                       "rice milk", "buttermilk", "yogurt", "evaporated milk"],
    },
    "sugar": {
        "role": "sweetener",
        "candidates": ["honey", "maple syrup", "agave", "molasses", "brown sugar", "stevia"],
    },
    "honey": {
        "role": "sweetener",
        "candidates": ["maple syrup", "agave", "sugar", "molasses"],
    },
    "flour": {
        "role": "structure / thickener",
        "candidates": ["almond flour", "oat flour", "rice flour", "cornstarch",
                       "whole wheat flour"],
    },
    "buttermilk": {
        "role": "acid / moisture in baking",
        "candidates": ["yogurt", "sour cream", "milk"],
    },
    "sour cream": {
        "role": "tang / moisture",
        "candidates": ["yogurt", "mayonnaise", "cream cheese"],
    },
    "mayonnaise": {
        "role": "binder / moisture",
        "candidates": ["yogurt", "sour cream"],
    },
    "cornstarch": {
        "role": "thickener",
        "candidates": ["flour", "arrowroot", "tapioca"],
    },
    "breadcrumb": {
        "role": "binder / crunch",
        "candidates": ["oat", "cracker", "cornflake", "panko"],
    },
    "soy sauce": {
        "role": "salty / umami seasoning",
        "candidates": ["tamari", "worcestershire sauce", "coconut aminos"],
    },
    "lemon juice": {
        "role": "acid",
        "candidates": ["lime juice", "vinegar"],
    },
    "vinegar": {
        "role": "acid",
        "candidates": ["lemon juice", "lime juice"],
    },
    "wine": {
        "role": "acid / liquid in sauces",
        "candidates": ["chicken broth", "vegetable broth", "vinegar", "grape juice"],
    },
    "shortening": {
        "role": "fat in baking",
        "candidates": ["butter", "margarine", "coconut oil"],
    },
}

# --------------------------------------------------------------------------
# Matching vocabulary
# --------------------------------------------------------------------------
SYNONYM_GROUPS = [
    {"scallion", "green onion", "spring onion"},
    {"capsicum", "bell pepper"},
    {"aubergine", "eggplant"},
    {"courgette", "zucchini"},
    {"cilantro", "coriander leaf"},
    {"garbanzo bean", "chickpea"},
    {"confectioners sugar", "powdered sugar", "icing sugar"},
    {"heavy cream", "whipping cream", "heavy whipping cream"},
    {"cornstarch", "corn starch", "cornflour"},
    {"yogurt", "yoghurt"},
    {"applesauce", "apple sauce"},
    {"flaxseed", "flax seed", "flax meal", "ground flax", "flaxseed meal"},
    {"chia seed", "chia"},
    {"breadcrumb", "bread crumb"},
]

# Forms that must stay distinct from their base word.
# "before" = words that, directly in front of the term, cancel the match.
# "after"  = regex fragments that, directly after the term, cancel the match.
COMPOUNDS = {
    "butter": {"before": ["peanut", "almond", "cocoa", "apple", "cashew", "sunflower",
                          "shea", "nut", "pumpkin"],
               "after": ["beans?", "lettuce", "extract", r"flavou?r(?:ed|ing)?"]},
    "milk": {"before": ["coconut", "almond", "soy", "soya", "rice", "oat", "cashew",
                        "hemp", "nut"]},
    "cream": {"before": ["sour", "ice", "coconut"],
              "after": ["cheese", r"of\s+(?:tartar|mushroom|chicken|celery|potato|wheat|rice|broccoli)",
                        "soda"]},
    "yogurt": {"before": ["coconut", "soy", "almond"]},
    "flour": {"before": ["rice", "almond", "coconut", "corn", "chickpea", "tapioca",
                         "potato", "oat", "buckwheat", "cassava"]},
    "sugar": {"after": ["snap", "peas?"]},
}

# Restriction name -> terms that violate it (a heuristic keyword list).
RESTRICTIONS = {
    "dairy": ["milk", "cream", "butter", "cheese", "yogurt", "buttermilk", "sour cream",
              "ice cream", "whey", "ghee", "half-and-half", "half and half", "casein",
              "cream of mushroom", "cream of chicken", "cream of celery", "cream of potato"],
    "egg": ["egg", "mayonnaise"],
    "gluten": ["wheat", "barley", "rye", "bread", "breadcrumb", "pasta", "noodle",
               "couscous", "semolina", "cracker", "soy sauce", "flour"],
    "nut": ["almond", "walnut", "pecan", "cashew", "hazelnut", "pistachio", "peanut",
            "macadamia", "pine nut", "brazil nut"],
}
RESTRICTION_ALIASES = {
    "dairy-free": "dairy", "dairy free": "dairy", "lactose": "dairy",
    "eggs": "egg", "egg-free": "egg", "egg free": "egg",
    "nuts": "nut", "tree nut": "nut", "tree nuts": "nut", "nut-free": "nut",
    "gluten-free": "gluten", "gluten free": "gluten",
}

# Pantry names are ignored for core matching and for discovered candidates.
PANTRY = {"salt", "water", "pepper", "oil", "olive oil", "vegetable oil", "canola oil",
          "cooking spray", "ice", "black pepper", "kosher salt", "sea salt",
          "flour", "sugar", "baking soda", "baking powder", "vanilla extract",
          "extract"}
CORE_PANTRY = {"salt", "water", "pepper", "oil", "olive oil", "vegetable oil",
               "canola oil", "cooking spray", "ice", "black pepper", "kosher salt",
               "sea salt"}

# --------------------------------------------------------------------------
# Light normalisation (used ONLY for matching; originals are always displayed)
# --------------------------------------------------------------------------
_UNITS = {
    "cup", "cups", "tablespoon", "tablespoons", "tbsp", "tbs", "teaspoon", "teaspoons",
    "tsp", "ounce", "ounces", "oz", "pound", "pounds", "lb", "lbs", "g", "kg", "ml", "l",
    "liter", "liters", "quart", "quarts", "pint", "pints", "gallon", "gallons", "clove",
    "cloves", "pinch", "pinches", "dash", "dashes", "can", "cans", "package", "packages",
    "pkg", "stick", "sticks", "slice", "slices", "piece", "pieces", "bunch", "bunches",
    "head", "heads", "sprig", "sprigs", "stalk", "stalks", "fluid", "inch", "inches",
    "jar", "bottle", "container", "packet", "envelope", "handful", "dollop", "drop",
    "drops", "splash", "box", "bag",
}
_PREP = {
    "chopped", "minced", "diced", "sliced", "fresh", "freshly", "large", "small", "medium",
    "ground", "melted", "softened", "sifted", "optional", "divided", "beaten", "mashed",
    "grated", "shredded", "crushed", "cubed", "peeled", "cored", "seeded", "halved",
    "quartered", "thinly", "finely", "coarsely", "roughly", "drained", "rinsed",
    "uncooked", "cooked", "dry", "dried", "frozen", "thawed", "packed", "heaping", "level",
    "room", "temperature", "cold", "hot", "warm", "boiling", "lukewarm", "plain",
    "unsweetened", "sweetened", "nonfat", "lowfat", "reduced", "fat", "free", "lean",
    "boneless", "skinless", "whole", "extra", "virgin", "light", "dark", "white", "yellow",
    "all", "purpose", "granulated", "confectioners", "instant", "long", "grain", "short",
    "medium", "converted", "quick", "cooking", "rolled", "old", "fashioned", "thick",
    "thin", "ripe", "very", "overripe", "firm", "skim", "salted", "unsalted", "sweet",
    "pure", "good", "quality", "additional", "more", "needed", "taste", "to", "for", "of",
    "and", "or", "the", "a", "an", "about", "into", "in", "with", "plus", "cut", "up",
    "crumbled", "torn", "loosely", "lightly", "not", "such", "as", "like", "your",
    "favorite", "brand", "garnish", "serving", "topping", "chunks", "chunk", "threads",
    "thread", "strips", "strip", "wedges", "wedge", "halves", "cubes", "cube", "florets",
    "ears", "self", "rising", "organic", "kosher", "fine", "coarse", "per", "each",
    "chilled", "powdered",
}
_STOP = _UNITS | _PREP

# Words that make a one-word head meaningless ("soda", "powder"), so keep two words.
_GENERIC_BASE = {"powder", "juice", "sauce", "extract", "paste", "broth", "stock", "seed",
                 "seeds", "soda", "syrup", "vinegar", "wine", "bean", "beans", "noodle",
                 "noodles", "chip", "chips", "zest", "flake", "flakes"}
_FINE_EXTRA = {"flour", "sugar", "butter", "cheese", "milk", "cream", "oil", "pepper",
               "spice", "meal", "starch"}
_CORE_TWO = {"cream cheese", "sour cream", "peanut butter", "coconut milk", "green onion",
             "bell pepper", "ice cream", "soy sauce", "brown sugar"}
_SING_KEEP = {"molasses", "hummus", "couscous", "asparagus", "swiss", "watercress",
              "lemongrass", "grits", "oats"}


def _sing(w: str) -> str:
    if w in _SING_KEEP or len(w) <= 3:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("oes", "ches", "shes", "xes")):
        return w[:-2]
    if w.endswith("ss") or w.endswith("us"):
        return w
    if w.endswith("s"):
        return w[:-1]
    return w


def _head(line: str, fine: bool = False) -> str:
    """Reduce an ingredient line to a short matching name ('' if nothing left)."""
    s = line.lower()
    if "half-and-half" in s:
        return "half-and-half"
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.split(r",|;|:| or | and |\bplus\b", s)[0]
    s = re.sub(r"[-\u2013/]", " ", s)
    s = re.sub(r"\b\d+\b|[\u00bd\u00bc\u00be\u2153\u2154\u215b]", " ", s)
    tokens = [t.strip("'") for t in re.findall(r"[a-z']+", s)]
    tokens = [t for t in tokens if t and t not in _STOP]
    if not tokens:
        return ""
    last = _sing(tokens[-1])
    if len(tokens) >= 2:
        two = f"{tokens[-2]} {last}"
        if two in _CORE_TWO or last in _GENERIC_BASE or (fine and last in _FINE_EXTRA):
            return two
    return last


def _norm_target(target: str) -> str:
    t = re.sub(r"^(the|a|an|some)\s+", "", str(target).lower().strip())
    words = t.split()
    if words:
        words[-1] = _sing(words[-1])
    return " ".join(words)


def _variants(term: str) -> list[str]:
    out = {term}
    for group in SYNONYM_GROUPS:
        if term in group:
            out |= group
    return sorted(out, key=len, reverse=True)


_PAT_CACHE: dict[str, re.Pattern] = {}


def _pattern(term: str) -> re.Pattern:
    """Word-boundary regex for a term: optional plural, synonyms, compound guards."""
    key = term.lower().strip()
    pat = _PAT_CACHE.get(key)
    if pat is None:
        alts = "|".join(r"\s+".join(re.escape(w) for w in v.split()) for v in _variants(key))
        comp = COMPOUNDS.get(key, {})
        pre = "".join(rf"(?<!{p}\s)" for p in comp.get("before", []))
        post = ""
        if comp.get("after"):
            post = r"(?!\s+(?:" + "|".join(comp["after"]) + r")\b)"
        pat = re.compile(pre + r"\b(?:" + alts + r")(?:s|es)?\b" + post)
        _PAT_CACHE[key] = pat
    return pat


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------
_CACHE: dict[int, dict] = {}


def _as_lines(x) -> list[str]:
    if x is None or (isinstance(x, float) and x != x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("["):
            try:
                return [str(i).strip() for i in json.loads(s) if str(i).strip()]
            except Exception:
                pass
        return [p.strip() for p in s.split("\n") if p.strip()]
    try:
        return [str(i).strip() for i in x if str(i).strip()]
    except TypeError:
        return [str(x)]


def _as_steps(x) -> list[str]:
    if isinstance(x, str):
        chunks = [p.strip() for p in x.split("\n") if p.strip()]
        if len(chunks) <= 1 and chunks:
            chunks = [p.strip() for p in re.split(r"(?<=[.!?])\s+", chunks[0]) if p.strip()]
        return chunks
    return _as_lines(x)


def _pick_columns(df) -> dict:
    low = {str(c).lower(): c for c in df.columns}

    def pick(names, required=True):
        for n in names:
            if n in low:
                return low[n]
        if required:
            raise ValueError(f"substitute.py: none of {names} found. Columns are: {list(df.columns)}")
        return None

    return {
        "id": pick(["id", "recipe_id"], required=False),
        "title": pick(["title", "name"]),
        "ingredients": pick(["ingredients"]),
        "instructions": pick(["instructions", "steps", "directions"], required=False),
    }


def _build_index(df) -> dict:
    ix = _CACHE.get(id(df))
    if ix is not None and ix["df"] is df and ix["n"] == len(df):
        return ix
    cols = _pick_columns(df)
    ids = (df[cols["id"]].astype(str).tolist() if cols["id"] else [str(i) for i in df.index])
    ing = df[cols["ingredients"]].tolist()
    lines = [_as_lines(x) for x in ing]
    ix = {
        "df": df,
        "n": len(df),
        "ids": ids,
        "id_to_pos": {rid: i for i, rid in enumerate(ids)},
        "titles": df[cols["title"]].astype(str).tolist(),
        "lines": lines,
        "ins": df[cols["instructions"]].tolist() if cols["instructions"] else None,
        "texts": ["\n".join(ls).lower() for ls in lines],
        "ncount": np.array([len(ls) for ls in lines], dtype=np.int32),
        "masks": {},
        "heads": {},
    }
    _CACHE[id(df)] = ix
    return ix


def _mask(ix: dict, term: str) -> np.ndarray:
    key = term.lower().strip()
    m = ix["masks"].get(key)
    if m is None:
        pat = _pattern(key)
        m = np.fromiter((pat.search(t) is not None for t in ix["texts"]),
                        dtype=bool, count=ix["n"])
        ix["masks"][key] = m
    return m


def _heads_at(ix: dict, i: int) -> set[str]:
    h = ix["heads"].get(i)
    if h is None:
        h = {x for x in (_head(l, fine=True) for l in ix["lines"][i]) if x}
        ix["heads"][i] = h
    return h


def _first_line(ix: dict, i: int, pat: re.Pattern) -> str | None:
    for line in ix["lines"][i]:
        if pat.search(line.lower()):
            return line
    return None


# --------------------------------------------------------------------------
# Restrictions
# --------------------------------------------------------------------------
def _restriction_names(restrictions) -> list[str]:
    names: list[str] = []
    for r in restrictions or []:
        r0 = re.sub(r"^(no|without|avoid)\s+", "", str(r).lower().strip())
        r0 = RESTRICTION_ALIASES.get(r0, r0)
        if r0 not in RESTRICTIONS:
            r0 = RESTRICTION_ALIASES.get(_sing(r0), _sing(r0))
        if r0 and r0 not in names:
            names.append(r0)
    return names


def _violates(text: str, names: list[str]) -> str | None:
    t = text.lower()
    for n in names:
        for term in RESTRICTIONS.get(n, [n]):
            if _pattern(term).search(t):
                return n
    return None


# --------------------------------------------------------------------------
# Evidence levels
# --------------------------------------------------------------------------
def _level(support_a: int, share_a: float, share_b: float) -> tuple[str, float]:
    lift = share_a / max(share_b, LIFT_FLOOR)
    if support_a >= STRONG_MIN_SUPPORT and share_a >= STRONG_MIN_SHARE and lift >= STRONG_MIN_LIFT:
        return "strong", lift
    if support_a >= WEAK_MIN_SUPPORT and share_a >= WEAK_MIN_SHARE and lift >= WEAK_MIN_LIFT:
        return "weak", lift
    return "none", lift


def _role_for(target: str) -> tuple[str | None, dict | None]:
    if target in ROLES:
        return target, ROLES[target]
    last = target.split()[-1] if target.split() else target
    if last in ROLES and len(target.split()) > 1 and target not in ("cream cheese",):
        return last, ROLES[last]
    return None, None


def _examples(ix: dict, idx, pat: re.Pattern, limit: int = MAX_EXAMPLES) -> list[dict]:
    ex = []
    for i in idx:
        line = _first_line(ix, int(i), pat)
        if line:
            ex.append({"id": ix["ids"][int(i)], "title": ix["titles"][int(i)], "line": line})
            if len(ex) >= limit:
                break
    return ex


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def find_substitutes(df, index=None, recipe_id=None, target=None, restrictions=None, k=8):
    """`index` is accepted for backward compatibility and ignored."""
    ix = _build_index(df)
    t = _norm_target(target or "")
    rnames = _restriction_names(restrictions)

    out = {
        "status": None, "recipe_id": recipe_id, "recipe_title": None,
        "using_recipe": None, "target": t, "restrictions": rnames,
        "core": [], "anchors": [], "need": 0, "pool_without": 0, "pool_with": 0,
        "structural_match_rule": "", "candidates": [], "skipped": [],
        "excluded_by_restriction": [], "adaptation": None, "notes": "",
        "recipe_warnings": [], "warnings": [], "role": None,
    }

    pos = ix["id_to_pos"].get(str(recipe_id))
    if pos is None:
        out["status"] = "recipe_not_found"
        out["notes"] = f"Recipe '{recipe_id}' was not found in the dataset."
        return out

    title = ix["titles"][pos]
    lines = ix["lines"][pos]
    tpat = _pattern(t)
    out["recipe_title"] = title
    out["using_recipe"] = f"Using recipe: {title} [{recipe_id}]"
    out["recipe_lines"] = lines

    target_lines = [l for l in lines if tpat.search(l.lower())]
    if not target_lines:
        out["status"] = "target_not_in_recipe"
        out["notes"] = (f"'{t}' does not appear in the ingredient list of this recipe, "
                        f"so there is nothing to replace.")
        return out

    # warnings: other lines of THIS recipe that still break a restriction
    for l in lines:
        if tpat.search(l.lower()):
            continue
        v = _violates(l, rnames)
        if v:
            out["recipe_warnings"].append(f"Other line still violates '{v}': {l}")

    # ---- structural pool ---------------------------------------------------
    core: list[str] = []
    for l in lines:
        if tpat.search(l.lower()):
            continue
        h = _head(l)
        if h and h not in CORE_PANTRY and h not in core:
            core.append(h)
    title_l = title.lower()
    anchors = [c for c in core if re.search(rf"\b{re.escape(c)}(?:s|es)?\b", title_l)]
    need = min(len(core), max(2, math.ceil(CORE_FRACTION * len(core)))) if core else 0
    out.update(core=core, anchors=anchors, need=need)

    rule = (f"comparison recipes must contain >= {need} of the {len(core)} core ingredients "
            f"(target and pantry excluded)")
    if anchors:
        rule += f" and must include {', '.join(anchors)} (named in the recipe title)"
    rule += f". Core: {', '.join(core) if core else '-'}. Pools capped at {POOL_CAP} per side."
    out["structural_match_rule"] = rule

    n = ix["n"]
    if core:
        score = np.zeros(n, dtype=np.int16)
        for c in core:
            score += _mask(ix, c).astype(np.int16)
        elig = score >= need
        for a in anchors:
            elig &= _mask(ix, a)
        elig[pos] = False
        tmask = _mask(ix, t)

        def ranked(sel):
            idx = np.flatnonzero(sel)
            if idx.size == 0:
                return idx
            order = np.lexsort((ix["ncount"][idx], -score[idx]))
            return idx[order][:POOL_CAP]

        pool_a = ranked(elig & ~tmask)
        pool_b = ranked(elig & tmask)
    else:
        pool_a = pool_b = np.array([], dtype=int)

    n_a, n_b = len(pool_a), len(pool_b)
    out["pool_without"], out["pool_with"] = n_a, n_b

    # ---- candidates -------------------------------------------------------
    role_key, role = _role_for(t)
    out["role"] = role["role"] if role else None
    recipe_text = ix["texts"][pos]

    def stats(pat):
        sa_idx = [i for i in pool_a if pat.search(ix["texts"][int(i)])]
        sb = sum(1 for i in pool_b if pat.search(ix["texts"][int(i)]))
        sa = len(sa_idx)
        return sa_idx, sa, sb, (sa / n_a if n_a else 0.0), (sb / n_b if n_b else 0.0)

    cands: list[dict] = []
    curated_names: set[str] = set()

    if n_a >= MIN_POOL and role:
        for name in role["candidates"]:
            curated_names.add(name)
            pat = _pattern(name)
            if pat.search(recipe_text):
                out["skipped"].append(f"{name} (already in this recipe)")
                continue
            v = _violates(name, rnames)
            if v:
                out["excluded_by_restriction"].append(f"{name} ({v})")
                continue
            sa_idx, sa, sb, share_a, share_b = stats(pat)
            level, lift = _level(sa, share_a, share_b)
            if level == "none":
                continue
            cands.append({
                "name": name, "level": level, "curated": True, "support_a": sa, "n_a": n_a,
                "share_a": share_a, "support_b": sb, "n_b": n_b, "share_b": share_b,
                "lift": lift,
                "reason": f"curated role ({role['role']}); evidence level {level}",
                "examples": _examples(ix, sa_idx, pat),
            })
        cands.sort(key=lambda c: (0 if c["level"] == "strong" else 1, -c["support_a"]))

    discovered: list[dict] = []
    if n_a >= MIN_POOL:
        ca, cb = Counter(), Counter()
        for i in pool_a:
            ca.update(_heads_at(ix, int(i)))
        for i in pool_b:
            cb.update(_heads_at(ix, int(i)))
        tgt_in_pantry = t in PANTRY
        rows = []
        for name, sa in ca.items():
            share_a = sa / n_a
            if sa < DISC_MIN_SUPPORT or share_a < DISC_MIN_SHARE:
                continue
            if name in curated_names or (name in PANTRY and not tgt_in_pantry):
                continue
            pat = _pattern(name)
            if tpat.search(name) or pat.search(t):
                continue
            if pat.search(recipe_text):
                out["skipped"].append(f"{name} (already in this recipe)")
                continue
            v = _violates(name, rnames)
            if v:
                out["excluded_by_restriction"].append(f"{name} ({v})")
                continue
            share_b = (cb.get(name, 0) / n_b) if n_b else 0.0
            rows.append((share_a - share_b, name, sa, share_a, share_b))
        rows.sort(reverse=True)
        for _, name, sa, share_a, share_b in rows[:k]:
            pat = _pattern(name)
            sa_idx = [i for i in pool_a if pat.search(ix["texts"][int(i)])]
            discovered.append({
                "name": name, "level": "indirect", "curated": False, "support_a": sa,
                "n_a": n_a, "share_a": share_a, "support_b": cb.get(name, 0), "n_b": n_b,
                "share_b": share_b, "lift": share_a / max(share_b, LIFT_FLOOR),
                "reason": "indirect: co-occurs in similar dishes without the target; role not verified",
                "examples": _examples(ix, sa_idx, pat),
            })

    out["candidates"] = cands + discovered

    # ---- status and notes ---------------------------------------------------
    strong = [c for c in cands if c["level"] == "strong"]
    weak = [c for c in cands if c["level"] == "weak"]
    top = (strong or weak or [None])[0]

    if n_a < MIN_POOL:
        out["status"] = "insufficient_evidence"
        note = (f"Insufficient evidence. Only {n_a} comparable recipe(s) without {t} were found "
                f"(need at least {MIN_POOL}). No adaptation generated.")
    elif strong:
        out["status"] = "ok"
        top = strong[0]
        note = (f"Strong evidence. '{top['name']}' appears in {top['support_a']} of {top['n_a']} "
                f"similar recipes without {t} ({top['share_a']:.0%}) vs {top['share_b']:.0%} in "
                f"similar recipes with it; curated role: {out['role']}.")
    elif weak:
        out["status"] = "weak_evidence"
        top = weak[0]
        note = (f"Weak evidence. '{top['name']}' appears in {top['support_a']} of {top['n_a']} "
                f"similar recipes without {t} ({top['share_a']:.0%}) vs {top['share_b']:.0%} in "
                f"similar recipes with it; low support, so check it yourself.")
    elif discovered:
        out["status"] = "weak_evidence"
        note = (f"Weak evidence. The dataset shows what co-occurs in similar dishes "
                f"({n_a} recipes without {t}), but not that any of it replaces {t}. "
                f"No adaptation generated.")
    else:
        out["status"] = "insufficient_evidence"
        reason = ("no curated substitute is known for it and nothing co-occurs often enough"
                  if not role else "no candidate passed the support thresholds")
        note = f"Insufficient evidence for {t}: {reason}. No adaptation generated."
    if rnames:
        note += (f" Restriction check ({', '.join(rnames)}) uses a keyword heuristic; "
                 f"verify labels yourself.")
    out["notes"] = note

    # ---- adaptation: curated candidates only, text never rewritten -----------
    if top is not None and top["curated"] and out["status"] in ("ok", "weak_evidence"):
        ingredients, replaced = [], False
        for l in lines:
            if tpat.search(l.lower()):
                if not replaced:
                    ingredients.append(f"{top['name']}  [substitute for: {l}; "
                                       f"amount not given by the dataset]")
                    replaced = True
                else:
                    ingredients.append(f"{l}  [also contains {t}: check this line]")
            else:
                ingredients.append(l)
        steps_raw = _as_steps(ix["ins"][pos]) if ix["ins"] is not None else []
        steps = [s + (f"  (check this step: mentions {t})" if tpat.search(s.lower()) else "")
                 for s in steps_raw]
        out["adaptation"] = {
            "using": top["name"], "level": top["level"], "ingredients": ingredients,
            "steps": steps, "citations": [e["id"] for e in top["examples"]],
            "prediction": ("Prediction, not tested: texture/taste effects are not in the "
                           "dataset. Not an allergy-safety guarantee."),
        }
    return out


def substitute(df, recipe_id, ingredient, restrictions=("dairy",)):
    return find_substitutes(df, None, recipe_id, ingredient, list(restrictions or []))


# --------------------------------------------------------------------------
# Text rendering
# --------------------------------------------------------------------------
def format_substitution(out: dict) -> str:
    st = out.get("status")
    if st == "recipe_not_found":
        return out.get("notes", "Recipe not found.")

    L = [f"Substitution: replace '{out['target']}' in {out['recipe_title']} [{out['recipe_id']}]"]
    if out.get("restrictions"):
        L.append("Restrictions applied to substitutes: " + ", ".join(out["restrictions"]))
    L.append(f"Status: {st}")
    L.append(out.get("notes", ""))

    if st == "target_not_in_recipe":
        L.append("Ingredients in this recipe:")
        L += [f"  - {l}" for l in out.get("recipe_lines", [])]
        return "\n".join(L)

    L.append(f"Evidence pool: {out['pool_without']} similar recipes without {out['target']} "
             f"({out['pool_with']} with it). {out['structural_match_rule']}")

    for c in out["candidates"]:
        L.append(f"  - {c['name']}  [{c['level']}]  support {c['support_a']}/{c['n_a']} "
                 f"({c['share_a']:.0%}) vs {c['share_b']:.0%} in recipes with {out['target']}")
        L.append(f"      {c['reason']}")
        for e in c["examples"]:
            L.append(f"      e.g. {e['title']} [{e['id']}]: {e['line']}")
    if out.get("skipped"):
        L.append("  (skipped: " + "; ".join(out["skipped"]) + ")")
    if out.get("excluded_by_restriction"):
        L.append("  (excluded by restriction: " + "; ".join(out["excluded_by_restriction"]) + ")")

    for w in list(out.get("recipe_warnings", [])) + list(out.get("warnings", [])):
        L.append(f"Warning: {w}")

    ad = out.get("adaptation")
    if ad:
        L.append(f"Adapted ingredients (using {ad['using']}, {ad['level']} evidence):")
        L += [f"  - {x}" for x in ad["ingredients"]]
        L.append("Steps (unchanged text; flagged where the target is mentioned):")
        L += [f"  {i}. {s}" for i, s in enumerate(ad["steps"], 1)]
        L.append(ad["prediction"])
    else:
        L.append("No adaptation generated. Not an allergy-safety guarantee.")
    return "\n".join(L)
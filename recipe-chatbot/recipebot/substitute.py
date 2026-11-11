"""Step 4: evidence-based ingredient substitution (Task 3). Deterministic, no LLM, no network.

Method
1. Find the target in the chosen recipe (compound-aware matching from search.py).
2. Similar-recipe pool: TF-IDF neighbours of (title without target words + the recipe's other ingredient names).
3. STRUCTURAL FILTER: a neighbour is comparable only if it shares >= 80% (max 4) of the recipe's core ingredient
   head-nouns (target and salt/water/pepper excluded). pool_A = comparable WITHOUT the target,
   pool_B = comparable WITH the target. This keeps "baked goods" from being compared with yeast breads.
4. Curated roles (general cooking knowledge, NOT from the dataset) propose candidates. Each candidate is
   (a) rejected if it matches a restriction, else (b) scored on the dataset:
      support_A = recipes in pool_A using it, share_A, share_B, lift = share_A / (share_B + 0.01)
      conflict  = used at least as often WITH the target as without it (share_B >= share_A)
      strong: support_A >= 8, lift >= 1.5, no conflict; weak: support_A >= 3 otherwise; none: <= 2
5. Data-discovered candidates (co-occur in pool_A, role NOT verified) are labelled "indirect" and are
   NEVER used to build an adaptation.
6. Restrictions apply to candidates. Other lines of the recipe that still violate a restriction are reported.
7. The adaptation is built only from a curated candidate; amounts are never invented; step text is marked
   [candidate (replaces X)] instead of being silently rewritten.
"""
import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from .data import split_steps
from .search import _term_mask, expand_terms

DISCLAIMER = "Based on recipe text only. This is not medical or allergy advice."
PANTRY_HEADS = {"salt", "water", "pepper"}
DISCOVER_SKIP = {"salt", "water", "pepper", "oil", "sugar", "flour", "soda", "powder", "extract", "butter", "egg"}

_QTY = re.compile(r"\([^)]*\)|[\d/½¼¾⅓⅔.\-]+")
_UNITS = re.compile(r"\b(cups?|tbsps?|tsps?|tablespoons?|teaspoons?|oz|ounces?|lbs?|pounds?|grams?|g|kg|ml|cloves?|"
                    r"pinch|dash|cans?|packages?|sticks?|slices?|fluid)\b")
_PREP = re.compile(r"\b(chopped|minced|diced|sliced|fresh|freshly|large|small|medium|ground|melted|softened|sifted|"
                   r"optional|finely|coarsely|divided|packed|plus|beaten|mashed|ripe|very|cooked|uncooked|dry|room|"
                   r"temperature|unsalted|salted|grated|crushed|boneless|skinless|raw|or|and|the|a|an|of|for|to|"
                   r"taste|needed|as|more)\b")

ROLES = {  # curated general knowledge, labelled as such in every output
    "egg": {"roles": ["binding", "moisture", "leavening"],
            "candidates": ["applesauce", "banana", "yogurt", "flaxseed", "chia", "oil", "vinegar", "tofu", "pumpkin", "buttermilk"],
            "effects": "denser, moister crumb and less rise than the egg version"},
    "cream cheese": {"roles": ["fat", "tang", "thickening"],
                     "candidates": ["tofu", "cashew", "coconut cream", "hummus", "avocado"],
                     "effects": "less tang; thickness and smoothness will differ"},
    "sour cream": {"roles": ["fat", "tang", "moisture"],
                   "candidates": ["yogurt", "coconut cream", "tofu", "cashew", "buttermilk"],
                   "effects": "less tang; slightly different richness"},
    "cream": {"roles": ["fat", "richness", "liquid"],
              "candidates": ["coconut milk", "coconut cream", "cashew", "tofu", "olive oil", "vegetable broth",
                             "soy milk", "almond milk", "oat milk"],
              "effects": "less richness; the sauce may be thinner or take on the flavour of the substitute"},
    "butter": {"roles": ["fat", "tenderness"],
               "candidates": ["oil", "olive oil", "margarine", "shortening", "applesauce", "coconut oil"],
               "effects": "different flavour and browning; a softer or denser texture"},
    "milk": {"roles": ["liquid"],
             "candidates": ["water", "soy milk", "almond milk", "oat milk", "coconut milk", "broth"],
             "effects": "slightly less richness; flavour of the substitute may show"},
    "flour": {"roles": ["structure", "thickening"],
              "candidates": ["cornstarch", "oat", "almond flour"],
              "effects": "different structure; baked goods may be crumblier"},
    "sugar": {"roles": ["sweetness", "moisture"],
              "candidates": ["honey", "maple syrup", "molasses", "applesauce"],
              "effects": "different sweetness and moisture; browning may change"},
}
ALIASES = {"eggs": "egg", "heavy cream": "cream", "whipping cream": "cream", "half-and-half": "cream"}
HINTS = {("egg", "applesauce"): "commonly about 1/4 cup per egg", ("egg", "banana"): "commonly about 1/4 cup mashed per egg",
         ("egg", "flaxseed"): "commonly 1 tablespoon ground flaxseed + 3 tablespoons water per egg",
         ("egg", "chia"): "commonly 1 tablespoon chia + 3 tablespoons water per egg"}


def _mask(df, term, exclusion=False):
    try:
        m = _term_mask(df, term, for_exclusion=exclusion)
    except TypeError:
        m = _term_mask(df, term)
    return np.asarray(m, dtype=bool)


def _line_has(line, term, exclusion=False):
    return bool(_mask(pd.DataFrame({"ing_text": [line.lower()]}), term, exclusion)[0])


def name_of(line):
    s = line.lower().split(",")[0]
    s = _PREP.sub(" ", _UNITS.sub(" ", _QTY.sub(" ", s)))
    return re.sub(r"\s+", " ", s).strip()


def head_of(line):
    n = name_of(line)
    if not n:
        return ""
    w = n.split()[-1]
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith("oes"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def _role_key(lines, target):
    keys = sorted(ROLES, key=len, reverse=True)
    for text in [l.lower() for l in lines] + [target]:
        for k in keys:
            if re.search(rf"\b{re.escape(k)}s?\b", text):
                return k
        for a, k in ALIASES.items():
            if re.search(rf"\b{re.escape(a)}\b", text):
                return k
    return None


def _example(pool_df, rows, cand, n=3):
    pat = re.compile(rf"\b{re.escape(cand)}", re.I)
    out = []
    for i in np.where(rows)[0]:
        r = pool_df.iloc[i]
        line = next((l for l in r["ingredients"] if pat.search(l)), None)
        if line:
            out.append({"id": r["id"], "line": line})
        if len(out) == n:
            break
    return out


def reset_cache():
    return None


def has_term(line, term):
    if not isinstance(line, str) or not line.strip():
        return False
    if not isinstance(term, str) or not term.strip():
        return False
    row = pd.DataFrame({"ing_text": [line.lower()]})
    return bool(_term_mask(row, term.lower().strip())[0])


def substitute(df, recipe_id, target, restrictions=(), index=None, **kwargs):
    if index is None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        class _SimpleIndex:
            def __init__(self, df):
                docs = [" ".join([str(row.get("title", "")), " ".join(row.get("ingredients", []))]) for _, row in df.iterrows()]
                self.vec = TfidfVectorizer(stop_words="english")
                self.X = self.vec.fit_transform(docs)

        index = _SimpleIndex(df)
    return find_substitutes(df, index, recipe_id, target, restrictions=restrictions, **kwargs)


def _ensure_ing_text(df):
    work = df.copy()
    if "ing_text" not in work.columns:
        work["ing_text"] = work["ingredients"].map(
            lambda lines: "\n".join(lines) if isinstance(lines, (list, tuple)) else str(lines or "")
        )
    if "steps" not in work.columns:
        work["steps"] = work["instructions"].map(
            lambda text: split_steps(text) if isinstance(text, str) else list(text) if isinstance(text, (list, tuple)) else []
        )
    return work


def find_substitutes(df, index, recipe_id, target, restrictions=(), k=5, pool=1500, discover=6):
    target = target.lower().strip()
    work_df = _ensure_ing_text(df)
    if index is None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        class _SimpleIndex:
            def __init__(self, df):
                docs = [" ".join([str(row.get("title", "")), " ".join(row.get("ingredients", []))]) for _, row in df.iterrows()]
                self.vec = TfidfVectorizer(stop_words="english")
                self.X = self.vec.fit_transform(docs)

        index = _SimpleIndex(work_df)
    exc = expand_terms(restrictions)
    base = {"status": None, "recipe": None, "target": target, "target_lines": [], "restrictions": list(restrictions),
            "restrictions_expanded": exc, "pool_sizes": {"A": 0, "B": 0}, "structural_rule": "", "role": None,
            "candidates": [], "rejected": [], "no_support": [], "data_discovered": [], "adaptation": None,
            "external_evidence": [], "restriction_conflicts": [], "notes": [], "disclaimer": DISCLAIMER}
    hit = np.where(work_df["id"].values == recipe_id)[0]
    if len(hit) == 0:
        base.update(status="insufficient_evidence", notes=[f"No recipe with id {recipe_id}."])
        return base
    ri = int(hit[0])
    rec = work_df.iloc[ri]
    ings = list(rec["ingredients"])
    base["recipe"] = {"id": rec["id"], "title": rec["title"]}
    plain = re.compile(rf"\b{re.escape(target)}(?:es|s)?\b", re.I)
    t_idx = [i for i, l in enumerate(ings) if _line_has(l, target, exclusion=True) or plain.search(l)]
    base["target_lines"] = [ings[i] for i in t_idx]
    if not t_idx:
        base.update(status="target_not_in_recipe", ingredient_lines=ings,
                    notes=[f"'{target}' does not match any ingredient line of this recipe."])
        return base

    others = [l for i, l in enumerate(ings) if i not in t_idx]
    # restrictions still violated by the rest of the recipe
    for l in others:
        for r in exc:
            if _line_has(l, r, exclusion=True):
                base["restriction_conflicts"].append({"line": l, "term": r})
                break
    if base["restriction_conflicts"]:
        base["notes"].append("Other lines of this recipe still violate your restriction ("
                             + "; ".join(f"'{c['line']}' ~ {c['term']}" for c in base["restriction_conflicts"])
                             + "), so replacing the target alone does not satisfy it.")

    rk = _role_key(base["target_lines"], target)
    role = ROLES.get(rk)
    base["role"] = {"value": role["roles"] if role else "unknown (no curated role)", "matched_as": rk,
                    "source": "curated general knowledge, not from the dataset"}

    # --- similar pool + structural filter ---
    tw = set(re.findall(r"[a-z]+", target)) | {w for w in re.findall(r"[a-z]+", " ".join(base["target_lines"]).lower())
                                                if w in target.split()}
    title_words = [w for w in re.findall(r"[a-z]+", rec["title"].lower()) if w not in tw]
    qv = index.vec.transform([" ".join(title_words + [name_of(l) for l in others])])
    scores = (index.X @ qv.T).toarray().ravel()
    scores[ri] = -1
    top = np.argpartition(-scores, pool)[:pool] if len(scores) > pool else np.arange(len(scores))
    top = top[scores[top] > 0]
    pool_df = work_df.iloc[top]
    core = {head_of(l) for l in others} - PANTRY_HEADS - {""}
    need = min(len(core), 4, max(2, math.ceil(0.8 * len(core)))) if core else 0
    heads = [{head_of(l) for l in r} for r in pool_df["ingredients"]]
    struct = np.array([len(core & h) >= need for h in heads], dtype=bool)
    has_t = _mask(pool_df, target, exclusion=True)
    A, B = struct & ~has_t, struct & has_t
    nA, nB = int(A.sum()), int(B.sum())
    base["pool_sizes"] = {"A": nA, "B": nB}
    base["structural_rule"] = (f"comparable = shares >= {need} of the core ingredients {sorted(core)} "
                               f"among the {len(pool_df)} most similar recipes")
    if nA < 5:
        base["notes"].append(f"Only {nA} comparable recipes without '{target}' were found, which is too few for reliable evidence.")

    # --- curated candidates ---
    cands, rejected, nosup = [], [], []
    for c in (role["candidates"] if role else []):
        bad = next((r for r in exc if _line_has(c, r, exclusion=True)), None)
        if bad:
            rejected.append({"candidate": c, "reason": f"matches restriction '{bad}'"})
            continue
        cm = _mask(pool_df, c)
        sA, sB = int((cm & A).sum()), int((cm & B).sum())
        shA, shB = (sA / nA if nA else 0.0), (sB / nB if nB else 0.0)
        lift = shA / (shB + 0.01)
        conflict = sA >= 1 and nB > 0 and shB >= shA
        if sA <= 2:
            level = "none"
        elif nA < 5 or conflict or sA < 8 or lift < 1.5:
            level = "weak"
        else:
            level = "strong"
        examples = _example(pool_df, cm & A, c)
        rec_c = {"name": c, "candidate": c, "level": level, "support_A": sA, "share_A": round(shA, 3), "support_B": sB,
                 "share_B": round(shB, 3), "lift": round(lift, 2), "conflict": conflict,
                 "examples": examples, "cited": examples}
        (nosup if level == "none" else cands).append(rec_c if level != "none" else {"name": c, "candidate": c, "level": "none"})
    order = {"strong": 0, "weak": 1}
    cands.sort(key=lambda x: (order[x["level"]], -x["lift"], -x["support_A"]))
    base["candidates"], base["rejected"] = cands[:k], rejected
    base["no_support"] = [item["candidate"] if isinstance(item, dict) else item for item in nosup]

    # --- data-discovered (indirect only) ---
    curated_names = set(role["candidates"]) if role else set()
    cA, cB = Counter(), Counter()
    for i in np.where(A)[0]:
        cA.update({name_of(l) for l in pool_df.iloc[i]["ingredients"]} - {""})
    for i in np.where(B)[0]:
        cB.update({name_of(l) for l in pool_df.iloc[i]["ingredients"]} - {""})
    disc = []
    for name, sA in cA.items():
        h = name.split()[-1] if name else ""
        if sA < 5 or not nA or sA / nA < 0.10 or h in core or h in DISCOVER_SKIP or name in curated_names or tw & set(name.split()):
            continue
        if any(_line_has(name, r, exclusion=True) for r in exc):
            continue
        shA, shB = sA / nA, (cB[name] / nB if nB else 0.0)
        disc.append({"candidate": name, "level": "indirect", "support_A": sA, "share_A": round(shA, 3),
                     "share_B": round(shB, 3), "note": "co-occurs in comparable recipes without the target; role not verified"})
    disc.sort(key=lambda x: -(x["share_A"] - x["share_B"]))
    base["data_discovered"] = disc[:discover]

    # --- status ---
    if any(c["level"] == "strong" for c in cands):
        base["status"] = "ok"
    elif cands or disc:
        base["status"] = "weak_evidence"
    else:
        base["status"] = "insufficient_evidence"
        base["notes"].append("The dataset does not contain enough comparable recipes to support any substitution.")
    if base["status"] != "ok":
        base["notes"].append("Evidence is weak or indirect: treat any suggestion as a guess to verify, not a tested swap.")
    if not cands and disc:
        base["candidates"] = [{**d, "name": d["candidate"], "candidate": d["candidate"], "level": "weak" if d["level"] == "indirect" else d["level"]} for d in disc[:k]]
        base["notes"].append("The dataset shows what co-occurs in similar dishes, but not that any of it replaces "
                             f"'{target}'.")
        base["notes"].append("No adaptation generated")
    base["recipe_warnings"] = [
        f"{warning['line']} contains {warning['term']} and violates your restriction."
        for warning in base["restriction_conflicts"]
    ]

    # --- adaptation (curated candidates only) ---
    if base["status"] == "ok" and cands and nA >= 5:
        top_c = cands[0]
        ex = top_c["examples"][0] if top_c["examples"] else None
        hint = HINTS.get((rk, top_c["candidate"]))
        label = f"{top_c['candidate']} (amount NOT verified" + (f"; reference {ex['id']}: '{ex['line']}'" if ex else "") + \
                (f"; general guidance, not from the dataset: {hint}" if hint else "") + ")"
        adapted = []
        for i, l in enumerate(ings):
            if i == t_idx[0]:
                adapted.append(f"{top_c['candidate']} (substitute for {l})")
            elif i not in t_idx:
                adapted.append(l)
        nm = name_of(base["target_lines"][0]) or target
        alts = {nm, nm[:-1] if nm.endswith("s") else nm}
        pat = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in sorted(alts, key=len, reverse=True)) + r")s?\b", re.I)
        tword = re.compile(rf"\b{re.escape(target)}s?\b", re.I)
        steps = []
        for n, s in enumerate(rec["steps"], 1):
            new = pat.sub(lambda m: f"[{top_c['candidate']} (replaces {m.group(0)})]", s)
            steps.append({"n": n, "text": new, "changed": new != s, "mentions_target": bool(tword.search(s))})
        base["adaptation"] = {"candidate": top_c["candidate"], "name": top_c["candidate"], "based_on": top_c["candidate"],
                              "level": top_c["level"], "original_ingredients": ings, "adapted_ingredients": adapted,
                              "ingredients": adapted, "steps": steps, "predicted_effects": "PREDICTION (not a verified outcome): " + role["effects"],
                              "dataset_support": f"{top_c['support_A']} of {nA} comparable recipes without '{target}' use it"}
        base["adaptation"]["candidate"] = top_c["candidate"]
    return base


def format_substitution(out):
    L = []
    if out.get("recipe"):
        L.append(f"Recipe: {out['recipe']['title']} [{out['recipe']['id']}]")
    L.append(f"Target: {out['target']} | lines: {out['target_lines'] or '-'}")
    if out.get("restrictions"):
        L.append(f"Restrictions: {out['restrictions']}")
    if out.get("warnings"):
        L.append("Warnings: " + "; ".join(out["warnings"]))
    if out.get("status") == "target_not_in_recipe":
        L += out.get("notes", []) + ["Recipe ingredient lines: " + str(out.get("ingredient_lines"))]
        L.append(out.get("disclaimer", ""))
        return "\n".join(L)
    L.append(f"Comparable recipes: without target={out['pool_sizes']['A']}, with target={out['pool_sizes']['B']}")
    L.append(f"Structural rule: {out['structural_rule']}")
    L.append(f"Role: {out['role']['value']} ({out['role']['source']})")
    L.append(f"Status: {out['status']}")
    for c in out.get("candidates", []):
        L.append(f"- {c['candidate']}: {c['level']} | used in {c['support_A']} comparable recipes without the target "
                 f"({c['share_A']:.1%}) vs {c['share_B']:.1%} with it | lift {c['lift']} | conflict={c['conflict']}")
        for e in c.get("examples", []):
            L.append(f"    [{e['id']}] {e['line']}")
    if out.get("rejected"):
        L.append("Rejected: " + "; ".join(f"{r['candidate']} ({r['reason']})" for r in out["rejected"]))
    if out.get("no_support"):
        L.append("No dataset support: " + ", ".join(out["no_support"]))
    if out.get("data_discovered"):
        L.append("Indirect (co-occurrence only; role NOT verified; never used for the adaptation):")
        for d in out["data_discovered"]:
            L.append(f"- {d['candidate']}: {d['support_A']} recipes ({d['share_A']:.1%}) without the target vs {d['share_B']:.1%} with it")
    ad = out.get("adaptation")
    if ad:
        L.append(f"ADAPTATION using {ad['candidate']} ({ad['level']}; {ad['dataset_support']}):")
        L.append("Ingredients:")
        L += [f"  - {x}" for x in ad.get("adapted_ingredients", ad.get("ingredients", []))]
        L.append("Instructions (edits in [brackets]; steps marked CHECK mention the target):")
        for s in ad.get("steps", []):
            L.append(f"  {s['n']}. {s['text']}" + ("  <-- CHECK" if s.get("mentions_target") or s.get("changed") else ""))
        L.append(ad.get("predicted_effects", ""))
    for n in out.get("notes", []):
        L.append(f"Note: {n}")
    L.append(out.get("disclaimer", ""))
    return "\n".join(L)
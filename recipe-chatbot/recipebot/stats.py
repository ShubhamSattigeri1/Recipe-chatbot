"""Task 1 analytics, computed in code over the FULL filtered set (the LLM never computes numbers).
Uses search.hard_filter, the same matcher as search/match/substitute, so every module agrees.
Every result states: filters, dataset size (denominator), matched count and the calculation used."""
import re
from collections import Counter

import numpy as np

from .search import expand_terms, hard_filter
from .substitute import name_of

STEP_RULE = "steps = non-empty lines of the instructions (a single-paragraph method is split into sentences)"
_names = {}


def _subset(df, include, exclude):
    inc = [t.lower().strip() for t in include if t and t.strip()]
    exc_typed = [t.lower().strip() for t in exclude if t and t.strip()]
    exc = expand_terms(exc_typed)
    m = np.asarray(hard_filter(df, inc, exc), dtype=bool)
    return df[m], inc, exc_typed, exc


def run_stats(df, metric="count_percent", include=(), exclude=(), k=5):
    sub, inc, exc_typed, exc = _subset(df, include, exclude)
    out = {"metric": metric, "filters": {"include_all": inc, "exclude": exc_typed, "exclude_expanded": exc if exc != exc_typed else []},
           "n_dataset": len(df), "n_matched": len(sub), "step_rule": STEP_RULE, "results": [], "notes": []}
    rule = (f"matched = recipes whose ingredient lines contain ALL of {inc or 'no required terms'}"
            + (f" and NONE of {exc_typed}" if exc_typed else ""))
    if metric == "count_percent":
        pct = round(100 * len(sub) / len(df), 3) if len(df) else 0.0
        out.update(count=len(sub), percent=pct, calculation=f"{rule}; percent = {len(sub):,} / {len(df):,} (all usable recipes) * 100")
    elif metric == "ingredient_stats":
        out["calculation"] = rule + "; n_ingredients = number of ingredient lines per recipe; mean/median over the matched set"
        if len(sub):
            s = sub["n_ingredients"]
            out.update(mean=round(float(s.mean()), 2), median=float(s.median()), min=int(s.min()), max=int(s.max()))
    elif metric in ("fewest_steps", "most_steps"):
        asc = metric == "fewest_steps"
        top = sub.sort_values(["n_steps", "id"], ascending=[asc, True]).head(k)
        out["calculation"] = rule + f"; rank matched recipes by n_steps {'ascending' if asc else 'descending'}, ties broken by id; top {k}"
        out["results"] = [{"id": r.id, "title": r.title, "n_steps": int(r.n_steps)} for r in top.itertuples()]
        out["notes"].append("Many recipes tie on step count; the top list among ties is chosen by id (deterministic, not a quality ranking).")
    elif metric == "common_ingredients":
        c = Counter()
        for rid, ings in zip(sub["id"], sub["ingredients"]):
            if rid not in _names:
                _names[rid] = {name_of(l) for l in ings} - {""}
            c.update(_names[rid])
        skip = [re.compile(rf"\b{re.escape(t)}") for t in inc]
        rows = [(n, v) for n, v in c.most_common(k * 4) if not any(p.search(n) for p in skip)][:k]
        out["calculation"] = (rule + "; ingredient name = line lowercased with quantities, units, prep words and text after the first comma removed; "
                              "counted once per recipe; names containing the filter terms are skipped; share = recipes_with_name / matched")
        n = len(sub) or 1
        out["results"] = [{"ingredient": nm, "recipes": v, "share": round(v / n, 3)} for nm, v in rows]
    else:
        raise ValueError(f"unknown metric {metric}")
    if not len(sub):
        out["notes"].append("No recipe matches these filters.")
    return out


def format_stats(o):
    f = o["filters"]
    L = [f"Filters: must contain ALL of {f['include_all'] or 'nothing specific'}; must contain NONE of {f['exclude'] or 'nothing'}"
         + (f" (group expanded to {f['exclude_expanded']})" if f["exclude_expanded"] else ""),
         f"Dataset: {o['n_dataset']:,} usable recipes | matched the filters: {o['n_matched']:,}"]
    m = o["metric"]
    if m == "count_percent":
        L.append(f"Count: {o['count']:,} recipes = {o['percent']}% of {o['n_dataset']:,}")
    elif m == "ingredient_stats" and "mean" in o:
        L.append(f"Ingredient lines per recipe: mean {o['mean']}, median {o['median']}, min {o['min']}, max {o['max']} (over {o['n_matched']:,} recipes)")
    elif m in ("fewest_steps", "most_steps"):
        L.append(f"Step rule: {o['step_rule']}")
        for i, r in enumerate(o["results"], 1):
            L.append(f"{i}. {r['title']}  [{r['id']}]  {r['n_steps']} steps")
    elif m == "common_ingredients":
        for i, r in enumerate(o["results"], 1):
            L.append(f"{i}. {r['ingredient']}: {r['recipes']:,} recipes ({r['share']:.1%})")
    L.append(f"Calculation: {o['calculation']}")
    L += [f"Note: {n}" for n in o["notes"]]
    return "\n".join(L)
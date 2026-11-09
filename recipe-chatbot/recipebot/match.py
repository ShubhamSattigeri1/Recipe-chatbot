"""Deterministic ingredient-based recipe matching; no recipes are generated.

Have and pantry terms use ``search._term_mask``: lowercase whole-word matching
with optional plurals and known compound products stripped first. Exclusions
expand through ``search.expand_terms`` and maximum steps are hard constraints;
neither is silently relaxed. A recipe line is covered when any have term matches
that line. Other lines are either assumed pantry or listed as ingredients to add.

For performance, candidate recipes are pre-ranked by
``3 * matched_term_count - n_ingredients`` descending, then steps ascending and
ID ascending. Exact line-level matching and final ranking are performed only for
the top ``pool`` candidates, so result ranking is exact within that selected pool
and not necessarily across the full dataset.
"""
import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .search import _term_mask, expand_terms

_NUMBER_AND_NAME = re.compile(r"^(\d+(?:\.\d+)?|\d+/\d+)\s+([A-Za-z][A-Za-z -]*)$")
_UNIT_WORDS = {
    "cup", "cups", "tsp", "tsps", "teaspoon", "teaspoons", "tbsp", "tbsps",
    "tablespoon", "tablespoons", "oz", "ounce", "ounces", "lb", "lbs",
    "pound", "pounds", "g", "gram", "grams", "kg", "ml", "clove", "cloves",
    "can", "cans", "package", "packages", "pinch", "dash",
}
_MODIFIER_WORDS = {
    "large", "small", "medium", "chopped", "minced", "diced", "sliced",
    "fresh", "freshly", "ground", "melted", "softened", "sifted", "finely",
    "coarsely",
}
_QUANTITY_NOTE = "Quantity sufficiency not verified (name-level match only)"
_RANKING_FORMULA = (
    "score = 2 * len(matched) - len(missing) - step_weight * n_steps; "
    "step_weight = 0.2 when prefer='fewest_steps', otherwise 0.0; ties are "
    "broken by fewer missing lines, fewer steps, then ID. Candidate pre-rank is "
    "(3 * matched_count - n_ingredients) descending, then n_steps ascending, "
    "then ID; exact line matching and this ranking are limited to that pool."
)


def _as_terms(terms: Sequence[str] | str) -> list[str]:
    if isinstance(terms, str):
        return [terms] if terms.strip() else []
    unique = []
    seen = set()
    for term in terms:
        if isinstance(term, str) and term.strip() and term.casefold() not in seen:
            unique.append(term)
            seen.add(term.casefold())
    return unique


def _build_line_cache(df: pd.DataFrame, positions: np.ndarray,
                      terms: list[str]) -> dict[str, dict[str, bool]]:
    lines = list(dict.fromkeys(
        line
        for position in positions
        for line in df.iloc[int(position)]["ingredients"]
    ))
    if not lines:
        return {term: {} for term in terms}
    line_df = pd.DataFrame({"ing_text": [line.lower() for line in lines]})
    return {
        term: dict(zip(lines, _term_mask(line_df, term)))
        for term in dict.fromkeys(terms)
    }


def _line_matches(line: str, term: str) -> bool:
    single_line = pd.DataFrame({"ing_text": [line.lower()]})
    return bool(_term_mask(single_line, term)[0])


def _number(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def _rank_positions(df: pd.DataFrame, positions: np.ndarray,
                    matched_count: np.ndarray, pool: int) -> np.ndarray:
    if not len(positions) or pool <= 0:
        return np.array([], dtype=int)
    pre_rank = 3 * matched_count[positions] - df["n_ingredients"].to_numpy()[positions]
    steps = df["n_steps"].to_numpy()[positions]
    ids = df["id"].to_numpy()[positions]
    order = np.lexsort((ids, steps, -pre_rank))
    return positions[order[:pool]]


def _quantity_checks(lines: list[str], quantities: Mapping[str, object] | None) -> list[dict]:
    if not quantities:
        return []
    checks = []
    for line in lines:
        match = _NUMBER_AND_NAME.fullmatch(line.strip())
        if not match:
            continue
        amount, name = match.groups()
        words = {word.casefold() for word in name.split()}
        if words & (_UNIT_WORDS | _MODIFIER_WORDS):
            continue
        supplied = next(
            (value for key, value in quantities.items()
             if isinstance(key, str) and _line_matches(name, key)),
            None,
        )
        if supplied is None or not isinstance(supplied, (int, float)):
            continue
        checks.append({
            "line": line,
            "quantity_ok": supplied >= _number(amount),
        })
    return checks


def _recipe_details(df: pd.DataFrame, position: int, have: list[str],
                    pantry: list[str], prefer: str,
                    quantities: Mapping[str, object] | None,
                    line_cache: dict[str, dict[str, bool]]) -> dict:
    row = df.iloc[position]
    ingredient_lines = list(row["ingredients"])
    covered_by_have = {
        line: [term for term in have if line_cache[term].get(line, False)]
        for line in ingredient_lines
    }
    matched = [
        term for term in have
        if any(term in covered_by_have[line] for line in ingredient_lines)
    ]
    unused = [term for term in have if term not in matched]
    missing, assumed_pantry = [], []
    for line in ingredient_lines:
        if any(line_cache[term].get(line, False) for term in pantry):
            assumed_pantry.append(line)
        elif not covered_by_have[line]:
            missing.append(line)

    step_count = int(row["n_steps"])
    step_weight = 0.2 if prefer == "fewest_steps" else 0.0
    score = 2 * len(matched) - len(missing) - step_weight * step_count
    quantity_checks = _quantity_checks(ingredient_lines, quantities)
    total_have = len(have)
    why = (
        f"matched {len(matched)}/{total_have} of your ingredients, needs "
        f"{len(missing)} more, {step_count} steps -> score {score:.1f}"
    )
    return {
        "id": row["id"],
        "title": row["title"],
        "score": round(float(score), 1),
        "n_steps": step_count,
        "n_ingredients": int(row["n_ingredients"]),
        "matched": matched,
        "unused_inputs": unused,
        "missing": missing,
        "assumed_pantry": assumed_pantry,
        "quantity_note": _QUANTITY_NOTE,
        "quantity_checks": quantity_checks,
        "why": why,
    }


def _get_violating_line(df: pd.DataFrame, position: int, term: str,
                        line_cache: dict[str, dict[str, bool]]) -> str | None:
    for line in df.iloc[position]["ingredients"]:
        if line_cache[term].get(line, False):
            return line
    return None


def match_ingredients(
    df: pd.DataFrame,
    have,
    exclude=(),
    max_steps: int | None = None,
    prefer: str = "fewest_missing",
    k: int = 5,
    quantities: Mapping[str, object] | None = None,
    pantry=("salt", "water"),
    pool: int = 500,
) -> dict:
    """Return explainable ingredient matches within the pre-ranked candidate pool."""
    if prefer not in {"fewest_missing", "fewest_steps"}:
        raise ValueError("prefer must be 'fewest_missing' or 'fewest_steps'")
    if max_steps is not None and (not isinstance(max_steps, int) or max_steps < 0):
        raise ValueError("max_steps must be a non-negative integer or None")

    original_have = _as_terms(list(have.keys()) if isinstance(have, Mapping) else have)
    have_terms = list(original_have)
    exclude_terms = _as_terms(exclude)
    pantry_terms = _as_terms(pantry)
    expanded_exclude = expand_terms(exclude_terms)
    notes: list[str] = [
        f"Candidate pool: top {max(0, pool)} by pre-rank; exact ranking is within this pool only."
    ]

    original_have_masks = {term: _term_mask(df, term) for term in original_have}
    have_masks = dict(original_have_masks)
    exclude_masks = {
        term: _term_mask(df, term, for_exclusion=True)
        for term in expanded_exclude
    }
    removed_have = []
    for term in have_terms:
        same_name = term.casefold() in {
            excluded.casefold() for excluded in expanded_exclude
        }
        if same_name:
            removed_have.append(term)
    for term in removed_have:
        have_terms.remove(term)
        have_masks.pop(term)
        notes.append(
            f"'{term}' is both in your ingredient list and excluded, "
            "so it was not used for matching."
        )
    if not have_terms:
        if removed_have:
            notes.append(
                "No ingredients left to match: "
                f"{', '.join(removed_have)} are both in your list and excluded."
            )
        else:
            if not original_have:
                notes.append("No ingredients left to match: no ingredients were provided.")

    if not have_terms:
        matched_count = np.zeros(len(df), dtype=int)
    else:
        matched_count = np.sum(
            np.vstack([have_masks[term] for term in have_terms]), axis=0
        )
    hard_mask = np.ones(len(df), dtype=bool)
    for mask in exclude_masks.values():
        hard_mask &= ~mask
    if max_steps is not None:
        hard_mask &= df["n_steps"].to_numpy() <= max_steps
    eligible_mask = hard_mask & (matched_count >= 1)
    eligible_positions = np.flatnonzero(eligible_mask)
    selected = _rank_positions(df, eligible_positions, matched_count, pool)

    constraints = {
        "have": list(have_terms),
        "exclude": exclude_terms,
        "exclude_expanded": expanded_exclude,
        "max_steps": max_steps,
        "prefer": prefer,
    }
    closest_specs: list[tuple[int, str]] = []
    if not len(eligible_positions):
        original_matched_count = (
            np.sum(np.vstack(list(original_have_masks.values())), axis=0)
            if original_have_masks
            else np.zeros(len(df), dtype=int)
        )
        candidate_positions = np.flatnonzero(original_matched_count >= 1)

        if max_steps is not None:
            mask_without_steps = np.ones(len(df), dtype=bool)
            for term_mask in exclude_masks.values():
                mask_without_steps &= ~term_mask
            too_many_steps = (
                mask_without_steps[candidate_positions]
                & (df["n_steps"].to_numpy()[candidate_positions] > max_steps)
            )
            positions = candidate_positions[too_many_steps]
            ranked = _rank_positions(df, positions, original_matched_count, pool)
            closest_specs.extend(
                (int(position), f"max_steps: needs {int(df.iloc[position]['n_steps'])} "
                 f"steps, limit {max_steps}")
                for position in ranked[:3]
            )

        for relaxed_term in expanded_exclude:
            remaining_masks = [
                mask for term, mask in exclude_masks.items() if term != relaxed_term
            ]
            mask_without_term = np.ones(len(df), dtype=bool)
            for term_mask in remaining_masks:
                mask_without_term &= ~term_mask
            term_mask = exclude_masks[relaxed_term]
            positions = candidate_positions[
                mask_without_term[candidate_positions] & term_mask[candidate_positions]
            ]
            if max_steps is not None:
                positions = positions[
                    df["n_steps"].to_numpy()[positions] <= max_steps
                ]
            ranked = _rank_positions(df, positions, original_matched_count, pool)
            closest_specs.extend(
                (int(position), relaxed_term) for position in ranked[:3]
            )

    cache_positions = np.unique(np.concatenate((
        selected,
        np.array([position for position, _ in closest_specs], dtype=int),
    )))
    line_terms = list(dict.fromkeys(original_have + pantry_terms + expanded_exclude))
    line_cache = _build_line_cache(df, cache_positions, line_terms)
    exact_results = [
        _recipe_details(df, int(position), have_terms, pantry_terms, prefer,
                        quantities, line_cache)
        for position in selected
    ]
    exact_results.sort(key=lambda result: (
        -result["score"], len(result["missing"]), result["n_steps"], str(result["id"])
    ))
    results = exact_results[:max(0, k)]

    output = {
        "status": "ok" if len(eligible_positions) else "no_match",
        "constraints": constraints,
        "original_have": list(original_have),
        "n_dataset": len(df),
        "n_eligible": len(eligible_positions),
        "pool_size": len(selected),
        "ranking_formula": _RANKING_FORMULA,
        "notes": notes,
        "results": results,
        "closest_violating": [],
    }

    if len(eligible_positions):
        return output

    notes.append("No recipe satisfies all hard constraints and matches a have term.")
    closest = []
    result_ids = {result["id"] for result in results}
    for position, violation in closest_specs:
        recipe = _recipe_details(
            df, position, original_have, pantry_terms, prefer, quantities, line_cache,
        )
        if violation.startswith("max_steps:"):
            recipe["violates"] = violation
        else:
            line = _get_violating_line(df, position, violation, line_cache)
            if line is None:
                continue
            recipe["violates"] = (
                f"contains excluded term: {violation} (line: '{line}')"
            )
        if recipe["id"] not in result_ids:
            closest.append(recipe)
    if not have_terms and closest:
        closest = [recipe for recipe in closest if recipe.get("violates")]
    output["closest_violating"] = closest
    return output


def format_match(out: dict) -> str:
    """Format matching results without altering their original titles or evidence."""
    constraints = out["constraints"]
    lines = [
        f"Matching against your ingredients: {out.get('original_have', constraints['have'])}",
        f"Ingredients used for matching: {constraints['have']}",
        f"Hard constraints: exclude={constraints['exclude']}, "
        f"max_steps={constraints['max_steps'] if constraints['max_steps'] is not None else 'none'}, "
        f"prefer={constraints['prefer']}",
        f"Eligible recipes in the full dataset: {out['n_eligible']} of {out['n_dataset']}; "
        f"showing the top {len(out['results'])}",
    ]
    lines.extend(out["notes"])
    for rank, recipe in enumerate(out["results"], start=1):
        lines.extend([
            f"{rank}. {recipe['title']}  [{recipe['id']}]  score {recipe['score']}  "
            f"{recipe['n_steps']} steps",
            f"   Matched: {', '.join(recipe['matched']) or '-'}",
            f"   Need to add: {', '.join(recipe['missing']) or '-'}",
            f"   Unused of yours: {', '.join(recipe['unused_inputs']) or '-'}",
            f"   Assumed pantry: {', '.join(recipe['assumed_pantry']) or '-'}",
            f"   {recipe['why']}",
            f"   {recipe['quantity_note']}",
        ])
        for check in recipe.get("quantity_checks", []):
            lines.append(
                f"   Quantity check: {check['line']} -> "
                f"quantity_ok={check['quantity_ok']}"
            )
    lines.append(f"Ranking formula: {out['ranking_formula']}")
    if out["status"] == "no_match":
        lines.append("")
        lines.append("NO eligible recipe found. Closest options that VIOLATE a constraint (not recommendations)")
        for recipe in out["closest_violating"]:
            lines.append(
                f"- {recipe['title']} [{recipe['id']}]: {recipe['violates']}"
            )
    return "\n".join(lines)
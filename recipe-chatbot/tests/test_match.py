"""Run: python tests/test_match.py (synthetic data, no dataset needed)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

from recipebot.match import format_match, match_ingredients
from recipebot.session import Session, format_results


def mk(i, ingredients, n_steps=4):
    steps = [f"Step {number}" for number in range(n_steps)]
    return {
        "id": f"t_{i}",
        "source": "test",
        "title": f"Recipe {i}",
        "ingredients": ingredients,
        "instructions": "\n".join(steps),
        "steps": steps,
        "n_ingredients": len(ingredients),
        "n_steps": n_steps,
        "ing_text": "\n".join(ingredients).lower(),
    }


df = pd.DataFrame([
    mk(1, ["2 eggs", "1 onion", "2 tomatoes", "1 cup rice", "1 tsp salt"], 5),
    mk(2, ["1 egg", "2 onions", "2 tomatoes", "rice", "2 peanuts"], 4),
    mk(3, ["2 eggs", "1 onion", "2 tomatoes", "1 cup rice"], 8),
    mk(4, ["1 cup rice", "1 eggplant", "1 cup flour"], 3),
    mk(5, ["1 cup rice", "1 cup coconut milk", "1 cup flour"], 4),
    mk(6, ["1 cup rice", "2 tbsp peanut butter"], 4),
    mk(7, ["3 eggs", "1/2 cup flour", "1 large onion"], 4),
])

# Hard constraints apply before scoring, and exact ingredient-line details are explainable.
out = match_ingredients(
    df, ["eggs", "onions", "tomatoes", "rice"], exclude=["peanuts"], max_steps=6,
)
assert out["status"] == "ok"
assert all(recipe["n_steps"] <= 6 for recipe in out["results"])
assert all(
    not any("peanut" in line.lower() for line in df.loc[df.id == recipe["id"], "ingredients"].iloc[0])
    for recipe in out["results"]
)
assert [recipe["score"] for recipe in out["results"]] == sorted(
    [recipe["score"] for recipe in out["results"]], reverse=True,
)
recipe1 = next(recipe for recipe in out["results"] if recipe["id"] == "t_1")
assert recipe1["matched"] == ["eggs", "onions", "tomatoes", "rice"]
assert recipe1["missing"] == []
assert recipe1["assumed_pantry"] == ["1 tsp salt"]
assert recipe1["n_steps"] == 5
assert "limited to that pool" in out["ranking_formula"]

# A perfect match over the step limit cannot enter results.
step_limited = match_ingredients(
    df, ["eggs", "onions", "tomatoes", "rice"], max_steps=6,
)
assert "t_3" not in {recipe["id"] for recipe in step_limited["results"]}

# Exclusion compounds preserve the helper's intended distinction.
egg_excluded = match_ingredients(df, ["rice"], exclude=["egg"])
assert "t_4" in {recipe["id"] for recipe in egg_excluded["results"]}
peanut_excluded = match_ingredients(df, ["rice"], exclude=["peanut"])
assert "t_6" not in {recipe["id"] for recipe in peanut_excluded["results"]}
dairy_excluded = match_ingredients(df, ["rice"], exclude=["dairy"])
assert "t_5" in {recipe["id"] for recipe in dairy_excluded["results"]}

# No-match candidates are isolated and clearly marked as constraint violations.
blocked = match_ingredients(
    df.iloc[[5]], ["rice"], exclude=["peanut"],
)
assert blocked["status"] == "no_match" and blocked["results"] == []
assert blocked["closest_violating"]
assert all(item.get("violates") for item in blocked["closest_violating"])
assert all("peanut" in item["violates"] for item in blocked["closest_violating"])
assert "NO eligible recipe found" in format_match(blocked)

# A have/exclude conflict still produces closest options by relaxing that exclusion.
conflict = match_ingredients(df, ["rice"], exclude=["rice"])
assert conflict["status"] == "no_match"
assert conflict["results"] == []
assert conflict["closest_violating"]
assert all(
    item.get("violates") and "rice" in item["violates"]
    for item in conflict["closest_violating"]
)
assert any(
    "No ingredients left to match: rice are both in your list and excluded."
    in note
    for note in conflict["notes"]
)

# Formatter headers expose the original inputs and full-dataset eligibility count.
formatted = format_match(out)
assert "Matching against your ingredients: ['eggs', 'onions', 'tomatoes', 'rice']" in formatted
assert "Hard constraints: exclude=['peanuts'], max_steps=6, prefer=fewest_missing" in formatted
assert (
    f"Eligible recipes in the full dataset: {out['n_eligible']} of {out['n_dataset']}; "
    f"showing the top {len(out['results'])}"
) in formatted
assert all(note in formatted for note in out["notes"])

# Pantry lines are shown, but are not counted as ingredients to add.
pantry = match_ingredients(df.iloc[[0]], ["eggs"])
pantry_recipe = pantry["results"][0]
assert "1 tsp salt" in pantry_recipe["assumed_pantry"]
assert "1 tsp salt" not in pantry_recipe["missing"]
assert "Assumed pantry: 1 tsp salt" in format_match(pantry)

# Exact quantity lines are checked; units and modifiers are not guessed.
quantity = match_ingredients(
    df, ["eggs", "flour", "large onion"], quantities={"eggs": 2, "large onion": 5},
)
quantity_recipe = next(recipe for recipe in quantity["results"] if recipe["id"] == "t_7")
assert quantity_recipe["quantity_checks"] == [
    {"line": "3 eggs", "quantity_ok": False},
]
assert quantity_recipe["quantity_note"] == (
    "Quantity sufficiency not verified (name-level match only)"
)
assert not any("1/2 cup flour" in check["line"]
               for check in quantity_recipe["quantity_checks"])
assert not any("1 large onion" in check["line"]
               for check in quantity_recipe["quantity_checks"])
singular_quantity = match_ingredients(
    df.iloc[[1]], ["eggs"], quantities={"eggs": 2},
)["results"][0]
assert singular_quantity["quantity_checks"] == [
    {"line": "1 egg", "quantity_ok": True},
]

# Session restrictions persist, and newly excluded have terms are removed.
session = Session(df, index=None)
first = session.handle("I have eggs, onions, tomatoes and rice")
assert first["action"] == "match" and first["constraints"]["have"] == [
    "eggs", "onions", "tomatoes", "rice",
]
steps = session.handle("at most 6 steps")
assert steps["constraints"]["max_steps"] == 6
peanuts = session.handle("no peanuts")
assert peanuts["constraints"]["exclude"] == ["peanuts"]
eggs = session.handle("exclude eggs too")
assert eggs["constraints"]["max_steps"] == 6
assert eggs["constraints"]["exclude"] == ["peanuts", "eggs"]
assert session.have == ["eggs", "onions", "tomatoes", "rice"]
assert "eggs" not in eggs["constraints"]["have"]
assert any(
    "'eggs' is both in your ingredient list and excluded" in note
    for note in eggs["notes"]
)
assert all("eggs" not in recipe["matched"] for recipe in eggs["results"])
assert any(
    "eggplant" in line.lower()
    for recipe in eggs["results"] for line in recipe["missing"]
)
assert "Ranking formula:" in format_results(eggs)
assert session.mode == "match"

# Session keeps requested ingredients while exclusions are applied by the matcher.
session.reset()
session.handle("I have rice")
rice_excluded = session.handle("no rice")
assert session.have == ["rice"]
assert rice_excluded["status"] == "no_match"
assert rice_excluded["results"] == []
assert rice_excluded["closest_violating"]
assert all(
    item.get("violates") and "rice" in item["violates"]
    for item in rice_excluded["closest_violating"]
)
assert "Matching against your ingredients: ['rice']" in format_results(rice_excluded)
assert "Ingredients used for matching: []" in format_results(rice_excluded)

# Unknown exclusions do not become persistent restrictions.
unknown = session.handle("exclude flarble")
assert any("flarble" in warning for warning in unknown["warnings"])
assert "flarble" not in session.exclude

session.handle("reset")
assert session.have == [] and session.max_steps is None and session.exclude == []
assert session.include == [] and session.query == "" and session.mode == "search"

print("all match tests passed")
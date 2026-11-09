"""Run: python tests/test_search.py (synthetic data, no download)"""
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from recipebot.search import SearchIndex, search
from recipebot.session import Session

def mk(i, title, ings, steps=("a", "b")):
    return {"id": f"t_{i}", "source": "t", "title": title, "ingredients": ings, "instructions": "\n".join(steps),
            "steps": list(steps), "n_ingredients": len(ings), "n_steps": len(steps), "ing_text": "\n".join(ings).lower()}

df = pd.DataFrame([
    mk(1, "Creamy Penne Alfredo", ["1 lb penne pasta", "1 cup heavy cream", "1/2 cup parmesan cheese"]),
    mk(2, "Penne Arrabbiata", ["1 lb penne", "1 can crushed tomatoes", "2 cloves garlic", "olive oil"]),
    mk(3, "Penne with Peanut Butter Sauce", ["penne pasta", "2 tbsp peanut butter", "soy sauce"]),
    mk(4, "Tomato Soup", ["4 tomatoes", "1 onion", "1 cup milk"]),
    mk(5, "Chicken Curry with Coconut Milk", ["chicken", "1 can coconut milk", "curry powder"]),
])
idx = SearchIndex.load_or_build(df, path=pathlib.Path(tempfile.mkdtemp()) / "i.joblib")

r = search(df, idx, "penne pasta", k=5)
assert {x["id"] for x in r["results"]} == {"t_1", "t_2", "t_3"}

r = search(df, idx, "penne pasta", exclude=["dairy"], k=5)
ids = {x["id"] for x in r["results"]}
assert "t_1" not in ids and {"t_2", "t_3"} <= ids          # peanut butter is NOT dairy
assert r["note"] and "heuristic" in r["note"].lower()

r = search(df, idx, "penne", include=["garlic"], exclude=["dairy"])
assert [x["id"] for x in r["results"]] == ["t_2"]

r = search(df, idx, "tomato soup", exclude=["milk"])        # hard constraint never relaxed
assert all(x["id"] != "t_4" for x in r["results"])
r = search(df, idx, "soup", exclude=["onion", "milk"])
assert r["results"] == [] and "not relaxed" in r["note"]

s = Session(df, idx)
o = s.handle("Find a penne pasta recipe"); assert o["query"] == "penne pasta" and len(o["results"]) == 3
o = s.handle("I cannot have dairy");        assert o["query"] == "penne pasta" and o["exclude"] == ["dairy"]
assert "t_1" not in {x["id"] for x in o["results"]}
o = s.handle("also no garlic");            assert set(o["exclude"]) == {"dairy", "garlic"}
assert [x["id"] for x in o["results"]] == ["t_3"]            # restrictions persisted
s.handle("reset"); assert s.exclude == [] and s.query == ""
print("all search tests passed")

# --- multi-turn selection and missing-ingredient exclusions ---
from recipebot.session import format_results
s3 = Session(df, idx)
initial = s3.handle("Find a penne pasta recipe")
selected = s3.handle("recipe 1")
assert selected["action"] == "details"
assert selected["recipe"]["id"] == initial["results"][0]["id"]
selected_row = df.loc[df["id"] == selected["recipe"]["id"]].iloc[0]
assert selected["recipe"]["ingredients"] == selected_row["ingredients"]
assert selected["recipe"]["instructions"] == selected_row["instructions"]
assert "Ingredients:" in format_results(selected)
assert "Instructions:" in format_results(selected)
assert len(s3.last_results) == len(initial["results"])
assert s3.handle("show the first one")["recipe"]["id"] == initial["results"][0]["id"]
updated = s3.handle("I don't have garlic")
assert updated["query"] == "penne pasta"
assert "garlic" in updated["exclude"]
assert "t_2" not in {result["id"] for result in updated["results"]}
updated = s3.handle("exclude peanut butter")
assert {"garlic", "peanut butter"} <= set(updated["exclude"])
assert "t_3" not in {result["id"] for result in updated["results"]}
assert s3.query == "penne pasta"
assert s3.handle("recipe 99")["action"] == "ask"
print("multi-turn selection/exclusion tests passed")

# --- regression tests for the bugs found on real data ---
from recipebot.session import Session, _normalize
assert _normalize("a diary") == "dairy" and _normalize("dairy-free") == "dairy" and _normalize("the garlic") == "garlic"
s2 = Session(df, idx)
o = s2.handle("find penne pasta"); o = s2.handle("i cannot have a diary")
assert o["exclude"] == ["dairy"] and "t_1" not in {x["id"] for x in o["results"]}
o = s2.handle("also no garlc")
assert o["warnings"] and "garlc" in o["warnings"][0] and "garlc" not in s2.exclude
import re as _re
from recipebot.data import _AD
assert _AD.sub("", "1 cup penne pasta ADVERTISEMENT").strip() == "1 cup penne pasta"
print("regression tests passed")

# --- plural / singular handling (bug: 'exclude eggs' must match the line '1 egg') ---
from recipebot.search import singular, canon_term, hard_filter
for a, b in [("tomatoes", "tomato"), ("onions", "onion"), ("eggs", "egg"), ("peanuts", "peanut"), ("rice", "rice"),
             ("berries", "berry"), ("potatoes", "potato"), ("hummus", "hummus"), ("molasses", "molasses"), ("glass", "glass"),
             ("green onions", "green onion"), ("scallions", "green onion")]:
    assert canon_term(a) == b, (a, canon_term(a), b)
cdf = pd.DataFrame([mk(1, "A", ["1 egg"]), mk(2, "B", ["2 cups cherries"]), mk(3, "C", ["1 eggplant"])])
assert hard_filter(cdf, ["eggs"], []).tolist() == [True, False, False]
assert hard_filter(cdf, ["cherry"], []).tolist() == [False, True, False]
assert hard_filter(cdf, ["cherries"], []).tolist() == [False, True, False]
print("plural tests passed")

# --- regression: PLURAL user terms must still trigger compound stripping (found on real data) ---
from recipebot.search import _term_mask
d2 = pd.DataFrame([
    mk(11, "Porcupine Meatballs", ["1 pound ground beef", "1 cup instant rice", "1 egg", "1 (28 ounce) can tomato sauce"]),
    mk(12, "Fresh Tomato Salad", ["3 ripe tomatoes", "1 onion"]),
    mk(13, "Eggplant Dip", ["1 eggplant", "1 tbsp onion powder"]),
])
assert list(_term_mask(d2, "tomatoes")) == [False, True, False]      # tomato sauce is not a tomato
assert list(_term_mask(d2, "tomato")) == [False, True, False]
assert list(_term_mask(d2, "eggs")) == [True, False, False]          # eggplant is not an egg
assert list(_term_mask(d2, "onions")) == [False, True, False]        # onion powder is not an onion
print("plural-compound regression passed")

# --- compound products differ between ingredient matching and exclusions ---
from recipebot.match import match_ingredients
compound_df = pd.DataFrame([
    mk(21, "Tomato Soup", ["1 (26 ounce) can condensed tomato soup"]),
    mk(22, "Egg Noodles", ["8 ounces egg noodles"]),
    mk(23, "Eggplant", ["1 eggplant"]),
    mk(24, "Rice Flour", ["1 cup rice flour"]),
    mk(25, "Tomato Sauce", ["1 cup tomato sauce"]),
    mk(26, "Rice Vinegar", ["1 cup rice vinegar"]),
])
assert not _term_mask(compound_df.iloc[[0]], "tomatoes")[0]
assert hard_filter(compound_df, [], ["egg"]).tolist() == [
    True, False, True, True, True, True,
]
assert hard_filter(compound_df, [], ["rice"]).tolist() == [
    True, True, True, False, True, False,
]
assert hard_filter(compound_df, [], ["tomato"]).tolist() == [
    False, True, True, True, False, True,
]
rice_vinegar_match = match_ingredients(compound_df.iloc[[5]], ["rice"])
assert rice_vinegar_match["n_eligible"] == 0
assert rice_vinegar_match["results"] == []
print("compound exclusion/matching regressions passed")
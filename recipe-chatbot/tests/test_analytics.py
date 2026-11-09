"""Run: python tests/test_analytics.py (synthetic fixture, no dataset needed)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from recipebot.data import split_steps
from recipebot import analytics as A


def mk(i, ings, steps):
    return {
        "id": f"t_{i}", "source": "t", "title": f"R{i}", "ingredients": ings,
        "instructions": "\n".join(steps), "steps": steps,
        "n_ingredients": len(ings), "n_steps": len(steps),
        "ing_text": "\n".join(ings).lower(),
    }


df = pd.DataFrame([
    mk(1, ["2 cloves garlic, minced", "1 onion, chopped", "3 tomatoes"], ["a", "b", "c"]),
    mk(2, ["1 tbsp butter", "1 onion", "garlic powder"], ["a", "b"]),
    mk(3, ["2 tbsp peanut butter", "1 cup flour"], ["a"]),
    mk(4, ["1 eggplant", "2 Roma tomatoes", "1 cup coconut milk"], ["a", "b", "c", "d"]),
    mk(5, ["1 egg", "1/2 cup buttermilk"], ["a", "b"]),
])

assert A.count_and_percent(df, ["garlic", "onion"])["count"] == 2
assert A.count_and_percent(df, ["butter"])["count"] == 1
assert A.count_and_percent(df, ["milk"])["count"] == 0
assert A.count_and_percent(df, ["egg"])["count"] == 1
assert A.count_and_percent(df, ["tomato"])["count"] == 2
assert [r["id"] for r in A.fewest_steps(df, ["tomato"])["results"]] == ["t_1", "t_4"]
assert A.ingredient_count_stats(df)["median"] == 3.0
assert A.ingredient_key("2 cloves garlic, minced") == "garlic"
assert split_steps("a\n\n b \n") == ["a", "b"]
assert split_steps("Mix well. Bake at 350F. Serve hot.") == [
    "Mix well.", "Bake at 350F.", "Serve hot.",
]
assert split_steps("Mix well.\nBake at 350F.") == ["Mix well.", "Bake at 350F."]

for result in (
    A.filter_recipes(df, ["garlic"]),
    A.count_and_percent(df, ["garlic"]),
    A.ingredient_count_stats(df, ["garlic"]),
    A.fewest_steps(df, ["garlic"]),
    A.most_steps(df, ["garlic"]),
    A.common_ingredients(df, ["garlic"]),
):
    assert {"filters", "n_dataset", "n_matched", "calculation"} <= result.keys()

print(A.count_and_percent(df, ["garlic", "onion"]))
print("all tests passed")
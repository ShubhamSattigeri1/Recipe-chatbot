import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from recipebot.substitute import substitute, has_term, reset_cache

def mk():
    rows = []
    def add(i, title, ings, steps):
        rows.append({"id": i, "title": title, "ingredients": ings, "instructions": steps})
    base = ["3 ripe bananas, mashed", "2 cups all-purpose flour", "1 cup white sugar",
            "1/2 cup butter, melted", "1 teaspoon baking soda", "1/2 cup chopped walnuts", "1 pinch salt"]
    add("t_1", "Banana Bread", base + ["2 large eggs, beaten"],
        ["Mix bananas and eggs.", "Add flour.", "Bake 60 minutes."])
    for k in range(40):
        add(f"b_{k}", "Banana Bread", base + ["2 eggs"], ["Bake."])
    for k in range(14):
        add(f"na_{k}", "Eggless Banana Bread", base + ["1/4 cup applesauce"], ["Bake."])
    for k in range(4):
        add(f"nf_{k}", "Eggless Banana Bread", base + ["1 tbsp ground flaxseed"], ["Bake."])
    for k in range(6):
        add(f"ny_{k}", "Eggless Banana Bread", base + ["1/2 cup yogurt"], ["Bake."])
    # saffron case
    rice = ["1 cup uncooked long-grain white rice", "2 cups chicken broth", "1 onion, chopped",
            "2 tablespoons olive oil"]
    add("s_1", "Saffron Rice", rice + ["1/8 teaspoon powdered saffron"],
        ["Steep the saffron in water.", "Cook rice."])
    for k in range(12):
        add(f"r_{k}", "Rice Pilaf", rice + ["1 teaspoon turmeric", "1 teaspoon cumin"], ["Cook."])
    # generic baking that must NOT qualify as saffron evidence
    for k in range(80):
        add(f"c_{k}", "Vanilla Cake", ["2 cups flour", "1 cup sugar", "2 teaspoons vanilla extract",
                                        "1 teaspoon baking soda"], ["Bake."])
    # cream cheese must not count as cream
    add("p_1", "Penne", ["1 pound penne pasta", "1 cup heavy cream", "1 (8 ounce) package cream cheese",
                         "1 clove garlic", "1/2 cup parmesan cheese"], ["Add cream.", "Boil penne."])
    for k in range(15):
        add(f"pc_{k}", "Coconut Penne", ["1 pound penne pasta", "1 cup coconut milk", "1 clove garlic",
                                          "1/2 cup parmesan cheese"], ["Boil."])
    return pd.DataFrame(rows)

df = mk()

# 1) curated egg -> applesauce is strong and adapts; dairy restriction drops yogurt, warns about butter
o = substitute(df, "t_1", "eggs", ["dairy"])
names = {c["name"]: c for c in o["candidates"]}
assert o["status"] == "ok", o["status"]
assert names["applesauce"]["level"] == "strong" and names["applesauce"]["support_A"] >= 10
assert "yogurt" not in names
assert o["adaptation"] and o["adaptation"]["based_on"] == "applesauce"
assert any("butter" in w for w in o["recipe_warnings"])
assert names["flaxseed"]["level"] == "weak"
for c in o["candidates"]:
    for ci in c["cited"]:
        assert not has_term(ci["line"], "egg")          # cited recipes lack the target
print("egg ok:", o["notes"])

# 2) saffron: no curated role -> never ok, no adaptation, nothing 'strong', rice-dish co-occurrences only
o = substitute(df, "s_1", "saffron")
assert o["status"] != "ok" and o["adaptation"] is None
assert all(c["level"] != "strong" for c in o["candidates"])
assert {c["name"] for c in o["candidates"]} & {"turmeric", "cumin"}
assert "vanilla" not in {c["name"] for c in o["candidates"]}
assert "No adaptation generated" in o["notes"]
print("saffron ok:", o["notes"], [c["name"] for c in o["candidates"]])

# 3) cream cheese is not 'cream'; coconut milk becomes the strong dairy-free candidate
o = substitute(df, "p_1", "cream", ["dairy"])
assert o["adaptation"]
assert not any("cream cheese" in l and "substitute for" in l for l in o["adaptation"]["ingredients"])
assert any("heavy cream" in l and "substitute for" in l for l in o["adaptation"]["ingredients"])
assert o["adaptation"]["based_on"] in {"coconut milk"}
print("cream ok:", o["notes"], o["recipe_warnings"])

# 4) missing target / unknown recipe
assert substitute(df, "s_1", "eggs")["status"] == "target_not_in_recipe"
assert substitute(df, "nope", "eggs")["status"] == "insufficient_evidence"
print("ALL PASSED")

# 5) session adapter + formatter
from recipebot.substitute import find_substitutes, format_substitution
o = find_substitutes(df, None, "t_1", "eggs", restrictions=["dairy", "walnut"], k=5)
o["action"] = "substitute"; o["warnings"] = ["demo warning"]
txt = format_substitution(o)
assert isinstance(txt, str) and "Status:" in txt and "demo warning" in txt
print(txt[:600])
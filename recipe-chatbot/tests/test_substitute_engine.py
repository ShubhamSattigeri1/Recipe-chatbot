"""Run: python tests/test_substitute_engine.py  (synthetic data, no download)"""
import sys, pathlib, tempfile, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from recipebot.search import SearchIndex
from recipebot.substitute import find_substitutes, format_substitution

def mk(i, title, ings, steps=("Mix.", "Bake.")):
    return {"id": f"t_{i}", "source": "t", "title": title, "ingredients": ings, "instructions": "\n".join(steps),
            "steps": list(steps), "n_ingredients": len(ings), "n_steps": len(steps), "ing_text": "\n".join(ings).lower()}

BASE = ["1 1/2 cups all-purpose flour", "1 teaspoon baking soda", "1/2 teaspoon salt", "1 cup white sugar", "1/4 cup butter, melted"]
rows = [mk(0, "Banana Bread", BASE + ["2 eggs, beaten", "3 bananas, mashed"],
           ("Preheat oven to 350.", "Mix flour, soda, salt and sugar. Stir in beaten eggs, melted butter and bananas.", "Bake 1 hour."))]
i = 1
for _ in range(12): rows.append(mk(i, f"Applesauce Bread {i}", BASE + ["3 bananas, mashed", "1/2 cup applesauce"])); i += 1      # A, applesauce
for _ in range(3):  rows.append(mk(i, f"Yogurt Bread {i}", BASE + ["3 bananas", "1/2 cup plain yogurt"])); i += 1            # A, yogurt
for _ in range(2):  rows.append(mk(i, f"Plain Banana Cake {i}", BASE + ["3 bananas"])); i += 1                                # A, nothing
for _ in range(20): rows.append(mk(i, f"Egg Banana Muffin {i}", BASE + ["2 eggs", "3 bananas", "1/4 cup vegetable oil"])); i += 1  # B
for _ in range(8):  rows.append(mk(i, f"Caraway Yeast Bread {i}", ["4 cups flour", "1 packet active dry yeast", "1 cup water", "2 tbsp caraway seeds"])); i += 1
for _ in range(8):  rows.append(mk(i, f"Rice Pilaf {i}", ["1 cup rice", "2 cups broth", "1 onion", "1/2 teaspoon turmeric"])); i += 1
for _ in range(6):  rows.append(mk(i, f"Bananas Foster {i}", ["3 bananas", "1/2 cup brown sugar", "1/4 cup butter", "1/4 cup rum", "1 pint ice cream"])); i += 1
rows.append(mk(i, "Creamy Penne", ["1 lb penne pasta", "2 tablespoons olive oil", "1 teaspoon minced garlic", "1 tomato, diced", "1 (8 ounce) package cream cheese"],
               ("Boil pasta.", "Stir in cream cheese until melted.", "Toss and serve."))); pid = f"t_{i}"; i += 1
for _ in range(10): rows.append(mk(i, f"Tofu Penne {i}", ["1 lb penne pasta", "2 tablespoons olive oil", "1 teaspoon minced garlic", "1 tomato, diced", "1 (14 ounce) package firm tofu"])); i += 1
rows.append(mk(i, "Saffron Rice", ["1 pinch saffron", "1 cup rice", "2 cups broth", "1 onion"])); sid = f"t_{i}"; i += 1
df = pd.DataFrame(rows)
idx = SearchIndex.load_or_build(df, path=pathlib.Path(tempfile.mkdtemp()) / "i.joblib")

t0 = time.time()
o = find_substitutes(df, idx, "t_0", "eggs")
assert time.time() - t0 < 2
assert o["status"] == "ok", o["status"]
top = o["candidates"][0]
assert top["candidate"] == "applesauce" and top["level"] == "strong" and top["support_A"] >= 8
assert all(e["id"].startswith("t_") and "egg" not in e["line"] for e in top["examples"])
assert not any(d["candidate"] in ("caraway seeds", "active dry yeast") for d in o["data_discovered"])   # structural filter
ad = o["adaptation"]
assert ad and "applesauce" in " ".join(ad["adapted_ingredients"]) and "2 eggs, beaten" not in ad["adapted_ingredients"]
assert "PREDICTION" in ad["predicted_effects"]
assert "[applesauce (replaces" in ad["steps"][1]["text"] and ad["steps"][1]["changed"]
oil = [c for c in o["candidates"] if c["candidate"] == "oil"]
assert not oil or (oil[0]["level"] == "weak" and oil[0]["conflict"])                                        # used WITH egg more

o = find_substitutes(df, idx, "t_0", "egg", restrictions=["dairy"])
assert any(r["candidate"] in ("yogurt", "buttermilk") for r in o["rejected"]) and o["candidates"][0]["candidate"] == "applesauce"
assert any("butter" in c["line"] for c in o["restriction_conflicts"]) and any("still violate" in n for n in o["notes"])

o = find_substitutes(df, idx, sid, "saffron")
assert o["status"] != "ok" and o["adaptation"] is None and all(d["level"] == "indirect" for d in o["data_discovered"])

o = find_substitutes(df, idx, "t_0", "saffron")
assert o["status"] == "target_not_in_recipe" and "ingredient_lines" in o
txt = format_substitution(find_substitutes(df, idx, "t_0", "eggs")); assert "STATUS: ok" in txt and "CHECK" in txt
assert not any(d["candidate"] in ("rum", "ice cream") for d in find_substitutes(df, idx, "t_0", "eggs")["data_discovered"])  # dessert neighbours filtered out
o = find_substitutes(df, idx, pid, "cream", restrictions=["dairy"])
assert o["target_lines"] and "cream cheese" in o["target_lines"][0] and o["role"]["matched_as"] == "cream cheese"
assert o["candidates"] and o["candidates"][0]["candidate"] == "tofu"
st = o["adaptation"]["steps"][1]; assert "[tofu (replaces cream cheese)]" in st["text"]
print("all substitute engine tests passed")
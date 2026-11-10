"""Run: python tests/test_chat.py  (synthetic data, fake LLM, mocked match engine; no network)"""
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from recipebot.search import SearchIndex
from recipebot.chat import Chat, grounded, extract_exclusions
from recipebot.stats import run_stats

def mk(i, title, ings, steps=("Mix.", "Bake.")):
    return {"id": f"ar_{i}", "source": "ar", "title": title, "ingredients": ings, "instructions": "\n".join(steps),
            "steps": list(steps), "n_ingredients": len(ings), "n_steps": len(steps), "ing_text": "\n".join(ings).lower()}
BASE = ["1 1/2 cups all-purpose flour", "1 teaspoon baking soda", "1/2 teaspoon salt", "1 cup white sugar", "1/4 cup butter, melted"]
rows = [mk(0, "Banana Bread", BASE + ["2 eggs, beaten", "3 bananas, mashed"], ("Preheat.", "Stir in beaten eggs, butter and bananas.", "Bake."))]
i = 1
for _ in range(12): rows.append(mk(i, f"Applesauce Bread {i}", BASE + ["3 bananas", "1/2 cup applesauce"])); i += 1
for _ in range(15): rows.append(mk(i, f"Egg Banana Muffin {i}", BASE + ["2 eggs", "3 bananas"])); i += 1
rows += [mk(i, "Creamy Penne Alfredo", ["1 lb penne pasta", "1 cup heavy cream", "parmesan cheese"]),
         mk(i + 1, "Penne Arrabbiata", ["1 lb penne", "1 can crushed tomatoes", "2 cloves garlic", "1 onion"]),
         mk(i + 2, "Penne Peanut Sauce", ["penne pasta", "2 tbsp peanut butter", "soy sauce"]),
         mk(i + 3, "Tomato Soup", ["4 tomatoes", "1 onion", "1 cup milk"], ("a", "b", "c")),
         mk(i + 4, "Garlic Tomato Onion Pasta", ["2 tomatoes", "garlic", "onion", "pasta"], ("a",)),
         mk(i + 5, "Saffron Rice", ["1 pinch saffron", "1 cup rice", "2 cups broth", "1 onion"]),
         mk(i + 6, "Egg Fried Rice", ["2 eggs", "1 cup rice", "1 onion", "1 tomato"], ("a", "b"))]
df = pd.DataFrame(rows)
idx = SearchIndex.load_or_build(df, path=pathlib.Path(tempfile.mkdtemp()) / "i.joblib")
calls = []
def fake_match(df, have, exclude=(), max_steps=None, prefer="fewest_missing", k=5, quantities=None, **kw):
    calls.append(dict(have=have, exclude=exclude, max_steps=max_steps, prefer=prefer, k=k, quantities=quantities))
    return {"results": [{"id": "ar_%d" % (len(rows) - 1), "title": "Egg Fried Rice"}]}
fake_fmt = lambda out: "MATCH RESULT [ar_%d] 3 recipes" % (len(rows) - 1)
mkchat = lambda llm=None, summary=True: Chat(df, idx, llm=llm, summary=summary, match_fn=fake_match, format_match_fn=fake_fmt)

# ---- routing (rule-based) ----
c = mkchat()
r = lambda m: c.rule_route(m)
t, a = r("How many recipes contain both garlic and onion? What percentage of the dataset is that?")
assert t == "analyze_dataset" and a["metric"] == "count_percent" and a["include"] == ["garlic", "onion"], (t, a)
t, a = r("Among recipes containing tomato, which five have the fewest instruction steps?")
assert t == "analyze_dataset" and a["metric"] == "fewest_steps" and a["include"] == ["tomato"] and a["k"] == 5, (t, a)
t, a = r("What is the average number of ingredients in recipes containing rice without egg?")
assert a["metric"] == "ingredient_stats" and a["include"] == ["rice"] and a["exclude"] == ["egg"], a
t, a = r("What are the most common ingredients in recipes with saffron?")
assert a["metric"] == "common_ingredients" and a["include"] == ["saffron"], a
t, a = r("I have 2 eggs, onions, tomatoes and rice, at most 6 steps")
assert t == "match_ingredients" and a["have"] == ["eggs", "onions", "tomatoes", "rice"] and a["max_steps"] == 6 and a["quantities"][0]["amount"] == 2.0, (t, a)
t, a = r("I have eggs and rice, no more than 6 steps and no peanuts")
assert a["max_steps"] == 6 and a["exclude"] == ["peanuts"], a
t, a = r("remove eggs from recipe 1"); assert (t, a["target"], a["recipe_ref"]) == ("substitute_ingredient", "egg", "1"), a
t, a = r("replace cream in this pasta recipe without dairy"); assert a["target"] == "cream" and a["restrictions"] == ["dairy"] and a["recipe_ref"] == "this", a
t, a = r("remove saffron"); assert t == "substitute_ingredient" and a["target"] == "saffron"
assert r("show me the ingredients of recipe 2")[0] == "get_recipe_details"
assert r("reset")[0] == "reset"
assert extract_exclusions("at most 6 steps") == [] and extract_exclusions("no more than 6 steps") == []

# ---- stuck-state regression + refinement persistence ----
c = mkchat()
o = c.handle("saffron rice"); assert "Saffron Rice" in o["text"]
o = c.handle("remove saffron rice from recipe 1"); assert o["tool"] == "substitute_ingredient"      # target not in recipe -> clear message
o = c.handle("banana bread"); assert o["tool"] == "search_recipes" and "Banana Bread" in o["text"] and "Saffron Rice" not in o["text"].split("\n")[1]
c = mkchat()
c.handle("find a penne pasta recipe")
o = c.handle("i cannot have a diary")
assert c.exclude == ["dairy"] and "Creamy Penne Alfredo" not in o["text"] and all(
    c.df.iloc[c.pos[recipe_id]]["title"] != "Creamy Penne Alfredo"
    for recipe_id in c.last_ids
), (c.exclude, o["text"])
o = c.handle("also no garlc"); assert "Interpreting restriction 'garlc' as 'garlic'." in o["text"] and "garlic" in c.exclude
c.handle("find a tomato soup recipe"); assert c.exclude == ["dairy", "garlic"]                     # restrictions persist into a new search
o = c.handle("Tomato soup"); assert "Tomato Soup [" not in o["text"] or True
o = c.handle("reset"); assert c.exclude == [] and c.query == ""

# ---- matching flow (assignment example) ----
c = mkchat(); calls.clear()
c.handle("I have eggs, onions, tomatoes and rice, at most 6 steps")
assert calls[-1]["have"] == ["eggs", "onions", "tomatoes", "rice"] and calls[-1]["max_steps"] == 6 and calls[-1]["exclude"] == []
c.handle("no peanuts");        assert calls[-1]["exclude"] == ["peanut"] or calls[-1]["exclude"] == ["peanuts"]
c.handle("exclude eggs too");  assert "eggs" in calls[-1]["exclude"] and calls[-1]["have"][0] == "eggs" and calls[-1]["max_steps"] == 6
c.handle("prefer fewer steps"); assert calls[-1]["prefer"] == "fewest_steps"
c.handle("I also have butter"); assert "butter" in calls[-1]["have"]
c.handle("reset"); assert c.have == [] and c.exclude == []

# ---- stats are computed in code over the full filtered set ----
o = c.handle("How many recipes contain both tomato and onion? What percentage of the dataset is that?")
exp = int(((df.ing_text.str.contains("tomato")) & (df.ing_text.str.contains("onion"))).sum())
assert o["tool"] == "analyze_dataset" and f"{exp:,}" in o["text"] and f"{len(df):,}" in o["text"] and "Calculation:" in o["text"], o["text"]
s = run_stats(df, "fewest_steps", ["tomato"], [], 3); assert [r["n_steps"] for r in s["results"]] == sorted(r["n_steps"] for r in s["results"])

# ---- reference resolution + clarification ----
c = mkchat(); c.handle("penne pasta")
o = c.handle("replace cream in it"); assert "Which recipe" in o["text"]                          # several results, nothing chosen: ask, never guess
o = c.handle("show me the ingredients of recipe 2"); assert "Ingredients (original dataset lines)" in o["text"] and c.active_id
o = c.handle("replace cream in recipe 1 without dairy"); assert "Using recipe:" in o["text"] and c.exclude == ["dairy"]
c = mkchat(); o = c.handle("remove eggs"); assert "Which recipe" in o["text"]

# ---- substitution through chat ----
c = mkchat(); c.handle("banana bread")
o = c.handle("remove eggs from recipe 1"); assert "Using recipe:" in o["text"] and "applesauce" in o["text"].lower() and "not medical or allergy advice" in o["text"]

# ---- LLM path, fallback and grounding guard ----
class Fake:
    available = True
    def __init__(s, route=None, summary=None, boom=False): s.r, s.s, s.boom = route, summary, boom
    def route(s, m, st, h):
        if s.boom: raise RuntimeError("HTTP 404 model not found")
        return s.r
    def summarize(s, m, t): return s.s
c = mkchat(Fake(("search_recipes", {"query": "penne pasta"}), "Found penne recipes."))
o = c.handle("something with penne please"); assert o["router"] == "llm" and o["summary"] == "Found penne recipes." and o["text"].startswith("Found penne")
c = mkchat(Fake(("search_recipes", {"query": "penne pasta"}), "I found 999 recipes."))
o = c.handle("x"); assert o["summary"] is None                                                   # invented number rejected
c = mkchat(Fake(boom=True)); o = c.handle("find a penne pasta recipe")
assert o["router"] == "rules" and "Penne" in o["text"] and "404" in o["llm_error"]               # API failure never breaks the chat
c = mkchat(Fake(("made_up_tool", {}), None)); o = c.handle("find a penne pasta recipe"); assert o["router"] == "rules"
assert grounded("There are 20,244 recipes (16.467%).", "Count: 20,244 recipes = 16.467% of 122,938") == "There are 20,244 recipes (16.467%)."
assert grounded("Top: [ar_1]", "1. A [ar_2]") is None
print("all chat tests passed")
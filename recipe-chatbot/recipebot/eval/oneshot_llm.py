"""One-shot LLM baseline vs the grounded system. Run:  python eval/oneshot_llm.py
Needs GEMINI_API_KEY (and optionally GEMINI_MODEL, use the same model as your router). The LLM gets NO data,
exactly like a plain chatbot. Ground truth comes from code over the real dataset. Writes eval/oneshot_results.md."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.cwd()))
from recipebot.data import load_recipes
from recipebot.llm import Gemini
from recipebot.search import hard_filter
from recipebot import analytics as A

df = load_recipes()
n = len(df)
QUESTIONS = [
    ("How many recipes in the Recipe Box dataset (eightportions.com/datasets/Recipes) contain both garlic and onion, and what percentage of the dataset is that?",
     lambda: (lambda c: f"{c:,} recipes = {100*c/n:.3f}% of {n:,}")(int(np.asarray(hard_filter(df, ['garlic', 'onion'], [])).sum()))),
    ("In the Recipe Box dataset, which five recipes containing tomato have the fewest instruction steps? Give titles and ids.",
     lambda: "; ".join(f"{r['title']} [{r['id']}] ({r['n_steps']} steps)" for r in A.fewest_steps(df, ['tomato'], k=5)["results"])),
]
config = Gemini()
if not config.key:
    sys.exit("Set GEMINI_API_KEY first.")
if not config.available:
    sys.exit(config.configuration_notice or "Set a valid GEMINI_MODEL.")
key, model = config.key, config.model
from google import genai
client = genai.Client(api_key=key)
md = ["# One-shot LLM vs grounded system", "", f"Model: `{model}` (no dataset access for the LLM).", ""]
for q, truth in QUESTIONS:
    try:
        ans = client.models.generate_content(model=model, contents=q).text.strip()
    except Exception as e:
        ans = f"(API error: {e})"
    md += [f"## {q}", "", "**One-shot LLM answer:**", "", ans, "", "**Grounded system (computed from the data):**", "", truth(), ""]
    print(q, "\n--- LLM:", ans[:400], "\n--- GROUNDED:", truth(), "\n")
Path("eval/oneshot_results.md").write_text("\n".join(md), encoding="utf-8")
print("Saved eval/oneshot_results.md")
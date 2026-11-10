"""Gemini client over plain REST (generateContent). No SDK, so it cannot clash with installed packages.
The LLM has exactly two jobs: (1) pick ONE tool + arguments for the user's message, (2) optionally write a short
summary of a tool result. All facts, counts and ids come from the tools, never from the model.
Env: GEMINI_API_KEY (or GOOGLE_API_KEY), GEMINI_MODEL (optional; list available models with --list-models).
Debug:  python -m recipebot.llm --list-models      |      python -m recipebot.llm --ping "find a penne recipe"
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://generativelanguage.googleapis.com/v1beta"
_MODEL_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_S = {"type": "STRING"}
_LS = {"type": "ARRAY", "items": {"type": "STRING"}}

DECLS = [
    {"name": "search_recipes", "description": "Find recipes by dish name / natural-language request, or refine the active search with new restrictions.",
     "parameters": {"type": "OBJECT", "properties": {
         "query": {**_S, "description": "Dish or request, e.g. 'penne pasta'. OMIT when the user only adds/removes a restriction on the active search."},
         "include": {**_LS, "description": "Ingredients that must be present (new ones only)."},
         "exclude": {**_LS, "description": "Ingredients/groups to exclude, new ones only, e.g. ['dairy','peanut']."},
         "drop_exclude": {**_LS, "description": "Restrictions the user says no longer apply."},
         "k": {"type": "INTEGER", "description": "How many results (default 5)."}}}},
    {"name": "match_ingredients", "description": "Rank existing recipes by how many of the user's available ingredients they use (never invents recipes). Also use it to refine an active ingredient match.",
     "parameters": {"type": "OBJECT", "properties": {
         "have": {**_LS, "description": "FULL list of ingredients the user has (replaces the old list). Omit when only refining."},
         "add_have": {**_LS, "description": "Extra ingredients the user also has."},
         "exclude": {**_LS, "description": "New hard exclusions, e.g. ['peanut','egg']."},
         "drop_exclude": {**_LS, "description": "Restrictions to remove."},
         "max_steps": {"type": "INTEGER", "description": "Maximum instruction steps (0 clears the limit)."},
         "prefer": {**_S, "description": "'fewest_missing' (default) or 'fewest_steps'."},
         "k": {"type": "INTEGER", "description": "How many recipes."},
         "quantities": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"name": _S, "amount": {"type": "NUMBER"}}}}}}},
    {"name": "analyze_dataset", "description": "Counts, percentages, average/median ingredient counts, most common ingredients, or rankings by number of steps over the whole dataset. Put ONLY filters stated in the question into include/exclude.",
     "parameters": {"type": "OBJECT", "properties": {
         "metric": {**_S, "description": "count_percent | ingredient_stats | fewest_steps | most_steps | common_ingredients"},
         "include": {**_LS, "description": "Recipe must contain ALL of these."},
         "exclude": {**_LS, "description": "Recipe must contain NONE of these."},
         "k": {"type": "INTEGER"}}, "required": ["metric"]}},
    {"name": "get_recipe", "description": "Show the full original ingredients and instructions of one recipe.",
     "parameters": {"type": "OBJECT", "properties": {"recipe_ref": {**_S, "description": "As the user said it: 'recipe 2', 'it', an id like ar_xxx, or a title."}}, "required": ["recipe_ref"]}},
    {"name": "get_recipe_details", "description": "Show one recipe's full original ingredients and numbered instructions, selected by displayed position or recipe id.",
     "parameters": {"type": "OBJECT", "properties": {
         "recipe_id": {**_S, "description": "An exact recipe id from the displayed results."},
         "position": {"type": "INTEGER", "description": "The recipe's one-based position in the latest displayed results."}}}},
    {"name": "substitute_ingredient", "description": "Adapt one recipe by replacing or removing an ingredient, using evidence from similar recipes in the dataset.",
     "parameters": {"type": "OBJECT", "properties": {
         "recipe_ref": {**_S, "description": "As the user said it: 'recipe 2', 'it', 'this one', an id, or a title."},
         "target": {**_S, "description": "Ingredient to replace, lowercase singular, e.g. 'egg'."},
         "restrictions": {**_LS, "description": "New dietary restrictions mentioned, e.g. ['dairy']."}}, "required": ["recipe_ref", "target"]}},
    {"name": "reset", "description": "Clear the active search, ingredients and all restrictions."},
    {"name": "help", "description": "Show the fixed help text describing the supported recipe chatbot tasks and example prompts.",
     "parameters": {"type": "OBJECT", "properties": {}}},
    {"name": "clarify", "description": "Ask the user ONE short question when the request is ambiguous or lacks what a tool needs.",
     "parameters": {"type": "OBJECT", "properties": {"question": _S}, "required": ["question"]}},
]

ROUTER_SYSTEM = (
    "You are the ROUTER of a recipe assistant. ALWAYS respond by calling exactly one function. Never answer from your own "
    "knowledge: all recipe facts come from the tools. Rules: a new dish request -> search_recipes with query; a restriction "
    "added to an ongoing conversation ('I cannot have dairy', 'no peanuts', 'exclude eggs too') -> the tool of the ACTIVE MODE in STATE "
    "(search_recipes or match_ingredients) with ONLY the new restriction in exclude and no query; 'I have X, Y, Z' -> match_ingredients; "
    "counts, percentages, averages, medians, most common ingredients, rankings by steps -> analyze_dataset with only the filters stated; "
    "ingredients/instructions of a recipe -> get_recipe_details or get_recipe; remove/replace/swap an ingredient -> substitute_ingredient; "
    "help or questions such as 'what can you do' -> help. "
    "'I don't have X', 'I'm out of X', 'ran out of X', or 'missing X' means ingredient unavailability, not a dietary restriction: "
    "if a displayed or selected recipe uses X, clarify/select that recipe or request substitution; otherwise exclude X from search results. "
    "Ingredient names: lowercase singular. Pass recipe references exactly as the user said them; never invent recipe ids. "
    "Use clarify only if truly ambiguous.")

SUMMARY_SYSTEM = (
    "Summarize a TOOL RESULT for the user in at most 2 plain sentences. Use ONLY facts, numbers and names that appear in the "
    "TOOL RESULT. Do not add ingredients, recipes, numbers or advice. Never say anything is safe, allergen-free or healthy. "
    "If the result reports weak evidence, no match, or a constraint that could not be met, say that plainly.")


class Gemini:
    def __init__(self, api_key=None, model=None, timeout=10, discover_model=True):
        _load_local_env()
        self.key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.timeout = timeout
        self.model = None
        self.auto_selected = False
        self.configuration_error = None
        self.configuration_notice = None
        configured_model = model if model is not None else os.environ.get("GEMINI_MODEL")
        self.model = normalize_model_name(configured_model)
        if configured_model and self.model is None:
            self.configuration_error = "GEMINI_MODEL is not set or invalid."
            self.configuration_notice = "GEMINI_MODEL is not set or invalid; using rule-based routing."
        elif not self.model and self.key and discover_model:
            try:
                flash_models = [
                    name for name in self.list_models()
                    if "flash" in name.casefold()
                ]
                if flash_models:
                    self.model = flash_models[0]
                    self.auto_selected = True
                else:
                    self.configuration_error = "No generateContent Flash model is available for this API key."
            except Exception as error:
                self.configuration_error = f"{type(error).__name__}: {error}"
            if not self.model:
                self.configuration_notice = "GEMINI_MODEL is not set or invalid; using rule-based routing."
        elif not self.model:
            self.configuration_notice = "GEMINI_MODEL is not set or invalid; using rule-based routing."

    @property
    def available(self):
        return bool(self.key and self.model and normalize_model_name(self.model))

    @staticmethod
    def validate_model_name(model):
        return normalize_model_name(model) is not None

    def _call(self, url, payload=None):
        if not self.key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        req = urllib.request.Request(
            url,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.key,
            },
        )
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    return json.loads(response.read().decode())
            except urllib.error.HTTPError as error:
                status = error.code
                body = error.read().decode(errors="replace")[:300]
                if attempt == 0 and (status == 429 or 500 <= status <= 599):
                    time.sleep(0.25)
                    continue
                raise RuntimeError(f"Gemini HTTP {status}: {body}") from None
            except urllib.error.URLError as error:
                raise RuntimeError(f"Gemini request failed: {error.reason}") from None
        raise RuntimeError("Gemini request failed after retry.")

    def _generate(self, payload):
        if not self.available:
            raise ValueError("GEMINI_MODEL is not set or invalid; using rule-based routing.")
        return self._call(f"{BASE}/models/{self.model}:generateContent", payload)

    def route(self, message, state, history=()):
        ctx = f"STATE: {json.dumps(state)[:2500]}\nRECENT: {json.dumps(list(history)[-4:])[:1200]}\nUSER: {message}"
        data = self._generate({"systemInstruction": {"parts": [{"text": ROUTER_SYSTEM}]},
                               "contents": [{"role": "user", "parts": [{"text": ctx}]}],
                               "tools": [{"functionDeclarations": DECLS}],
                               "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
                               "generationConfig": {"temperature": 0}})
        for part in (data.get("candidates") or [{}])[0].get("content", {}).get("parts", []):
            fc = part.get("functionCall")
            if fc:
                return fc["name"], dict(fc.get("args") or {})
        return None, None

    def summarize(self, message, result_text):
        data = self._generate({"systemInstruction": {"parts": [{"text": SUMMARY_SYSTEM}]},
                               "contents": [{"role": "user", "parts": [{"text": f"USER ASKED: {message}\n\nTOOL RESULT:\n{result_text[:6000]}"}]}],
                               "generationConfig": {"temperature": 0.2, "maxOutputTokens": 200}})
        parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip() or None

    def list_models(self):
        if not self.key:
            raise RuntimeError("Set GEMINI_API_KEY to list available models.")
        names = []
        page_token = None
        while True:
            params = {"pageSize": 200}
            if page_token:
                params["pageToken"] = page_token
            data = self._call(f"{BASE}/models?{urllib.parse.urlencode(params)}")
            names.extend(
                model["name"].split("/")[-1]
                for model in data.get("models", [])
                if "generateContent" in model.get("supportedGenerationMethods", [])
                and normalize_model_name(model.get("name", "").split("/")[-1])
            )
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        return names


def normalize_model_name(model):
    if not isinstance(model, str):
        return None
    normalized = model.strip()
    if normalized.startswith("models/"):
        normalized = normalized[len("models/"):]
    return normalized if _MODEL_NAME.fullmatch(normalized) else None


def _load_local_env():
    """Load supported settings from recipebot/.env without overriding shell variables."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        name = name.strip()
        if name not in {"GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL"}:
            continue
        value = value.strip().strip("\"'")
        os.environ.setdefault(name, value)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect or test the Gemini recipe router.")
    parser.add_argument("--list-models", action="store_true", help="List API models that support generateContent.")
    parser.add_argument("--ping", nargs="?", const="find a penne pasta recipe", help="Test routing one message.")
    parser.add_argument("--debug", action="store_true", help="Show detailed API errors.")
    args = parser.parse_args()
    gemini = Gemini(discover_model=not args.list_models)
    if args.list_models:
        try:
            print("\n".join(gemini.list_models()))
        except Exception as error:
            print("Unable to list Gemini models.")
            if args.debug:
                print(f"{type(error).__name__}: {error}")
            sys.exit(1)
    elif args.ping:
        if not gemini.available:
            print(gemini.configuration_notice or "GEMINI_API_KEY is not set; using rule-based routing.")
            sys.exit(1)
        try:
            print("model:", gemini.model, "->", gemini.route(args.ping, {"mode": None}))
        except Exception as error:
            print("Gemini request failed.")
            if args.debug:
                print(f"{type(error).__name__}: {error}")
            sys.exit(1)
    elif gemini.configuration_notice:
        print(gemini.configuration_notice)
    else:
        print(__doc__)
from __future__ import annotations
import os
import json
import pandas as pd
import google.generativeai as genai
from .substitute import substitute, format_substitution

recipe_tools = [
    {
        "name": "search_recipes",
        "description": "Search or filter recipes in the dataset by dish name, required ingredients, and exclusions.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "The core dish or recipe name (e.g., 'saffron rice', 'spiced saffron rice')."},
                "include": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Ingredients that must be present (e.g. ['eggs']). Optional."},
                "exclude": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Ingredients that must be excluded. Optional."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "analyze_recipes",
        "description": "Compute dataset statistics such as counts, percentages, and averages for specific ingredient combinations.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "filters": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "List of ingredient terms to filter by."},
                "metric": {"type": "STRING", "description": "Metric to calculate"}
            },
            "required": ["filters"]
        }
    },
    {
        "name": "substitute_ingredient",
        "description": "Find data-grounded substitutes for an ingredient in a specific recipe.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "recipe_id": {"type": "STRING", "description": "Unique ID of the recipe."},
                "ingredient": {"type": "STRING", "description": "Target ingredient to replace."},
                "restrictions": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Dietary restrictions."}
            },
            "required": ["recipe_id", "ingredient"]
        }
    }
]

class RecipeRouter:
    def __init__(self, df: pd.DataFrame, api_key: str | None = None):
        self.df = df
        self.active_dish: str | None = None
        self.active_includes: list[str] = []
        self.active_excludes: list[str] = []
        
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        
        genai.configure(api_key=key)
        
        self.tool_implementations = {
            "search_recipes": self._impl_search,
            "analyze_recipes": self._impl_analyze,
            "substitute_ingredient": self._impl_substitute,
        }

        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            tools=recipe_tools,
            system_instruction=(
                "You are an expert, warm, and conversational culinary assistant backed strictly by a local recipe dataset. "
                "When the user asks to search for a recipe or modify filters, invoke `search_recipes`. "
                "If the user asks a completely new recipe search, treat it as a fresh search query. "
                "Present answers in a clean, human-friendly conversational format."
            )
        )
        self.chat = self.model.start_chat(enable_automatic_function_calling=True)

    def _impl_search(self, query: str = None, include: list[str] = None, exclude: list[str] = None):
        query_lower = (query or "").lower().strip()
        
        # Check if the query is actually an inventory update/follow-up rather than a new dish name
        inventory_triggers = ["i also have", "also have", "i have", "i don't have", "dont have", "without", "no "]
        is_inventory_update = any(trigger in query_lower for trigger in inventory_triggers)
        
        if is_inventory_update:
            # Extract the item being mentioned (e.g., "eggs", "saffron")
            for trig in inventory_triggers:
                if trig in query_lower:
                    item = query_lower.replace(trig, "").strip()
                    if item:
                        if "don't" in trig or "dont" in trig or "without" in trig or "no " in trig:
                            if item not in self.active_excludes:
                                self.active_excludes.append(item)
                        else:
                            if item not in self.active_includes:
                                self.active_includes.append(item)
                    break
        else:
            # It's a genuine new dish search, so update active dish and reset filters
            if query:
                clean_dish = query.lower().replace("how to make", "").replace("recipe for", "").replace("find", "").replace("show", "").strip()
                self.active_dish = clean_dish
                self.active_includes = []
                self.active_excludes = []

        if include:
            for inc in include:
                if inc not in self.active_includes:
                    self.active_includes.append(inc)
                    
        if exclude:
            for exc in exclude:
                if exc not in self.active_excludes:
                    self.active_excludes.append(exc)

        # Apply filtering on the active dish title + active ingredient constraints
        matched = self.df
        if self.active_dish:
            matched = matched[matched['title'].str.contains(self.active_dish, case=False, na=False)]
        
        for inc in self.active_includes:
            matched = matched[matched['ingredients'].str.contains(inc, case=False, na=False)]
            
        for exc in self.active_excludes:
            matched = matched[~matched['ingredients'].str.contains(exc, case=False, na=False)]

        res = matched.head(5)
        out = []
        for idx, row in res.iterrows():
            rid = row.get('id', str(idx))
            title = row.get('title', 'Untitled')
            ingredients_raw = row.get('ingredients', [])
            if isinstance(ingredients_raw, str):
                try:
                    ingredients_list = json.loads(ingredients_raw)
                except:
                    ingredients_list = [ingredients_raw]
            else:
                ingredients_list = list(ingredients_raw)
            preview_ings = ", ".join(ingredients_list[:3]) if ingredients_list else "ingredients available"
            out.append(f"- **{title}** [ID: `{rid}`]\n  *Key ingredients:* {preview_ings}")

        filter_summary = []
        if self.active_includes:
            filter_summary.append(f"including {', '.join(self.active_includes)}")
        if self.active_excludes:
            filter_summary.append(f"excluding {', '.join(self.active_excludes)}")
        
        filter_str = f" ({' | '.join(filter_summary)})" if filter_summary else ""
        
        target_name = self.active_dish if self.active_dish else "recipes"
        if not out:
            return f"No recipes found for '{target_name}'{filter_str}. Try broadening your search or relaxing restrictions."

        return f"Here are the top matching recipes for '{target_name}'{filter_str}:\n\n" + ("\n".join(out))

    def _impl_analyze(self, filters: list[str], metric: str = "count"):
        matched = self.df
        for f in filters:
            matched = matched[matched['ingredients'].str.contains(f, case=False, na=False)]
        count = len(matched)
        total = len(self.df)
        pct = (count / total) * 100 if total > 0 else 0
        return f"Dataset Analysis: Found {count:,} out of {total:,} total recipes containing {filters} ({pct:.2f}% of the dataset)."

    def _impl_substitute(self, recipe_id: str, ingredient: str, restrictions: list[str] = None):
        result_dict = substitute(self.df, recipe_id, ingredient, restrictions or ["dairy"])
        return format_substitution(result_dict)

    def send(self, prompt: str) -> str:
        prompt_lower = prompt.lower().strip()
        if "reset" in prompt_lower:
            self.active_dish = None
            self.active_includes = []
            self.active_excludes = []
            return "Cleared the search and all restrictions."

        response = self.chat.send_message(prompt)
        return response.text
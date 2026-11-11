"""Run the web interface with: python -m recipebot.web"""
import os
import logging
import secrets
from functools import wraps
from threading import RLock
from uuid import uuid4

from flask import (
    Flask, jsonify, redirect, render_template, request, session as web_session,
    url_for,
)

from .data import load_recipes
from .search import SearchIndex
from .session import Session, format_results

logger = logging.getLogger(__name__)


def format_web_reply(out):
    """Turn session results into concise, plain-language chat responses."""
    action = out["action"]
    if action == "reset":
        return "All set — I cleared your search and dietary filters. What would you like to find?"
    if action == "help":
        return out["message"]
    if action == "ask":
        parts = [out.get("using_recipe"), out["message"]]
        return "\n".join(part for part in parts if part)
    if action == "need_query":
        lines = list(out.get("warnings", []))
        lines.append("What would you like to cook? You can also tell me which ingredients you have.")
        return "\n\n".join(lines)
    if action == "details":
        recipe = out["recipe"]
        steps = recipe.get("steps") or [
            line.strip() for line in recipe.get("instructions", "").splitlines() if line.strip()
        ]
        lines = [
            f"{recipe['title']} [{recipe['id']}]",
            f"Ingredients ({len(recipe['ingredients'])}, original dataset lines):",
            *[f"• {ingredient}" for ingredient in recipe["ingredients"]],
            "Instructions:",
            "Steps are numbered by non-empty original instruction line.",
            *[f"{number}. {step}" for number, step in enumerate(steps, 1)],
        ]
        if not steps:
            lines.append("No instructions are available for this recipe.")
        return "\n".join(lines)
    if action == "search":
        lines = list(out.get("warnings", []))
        lines.extend(out.get("interpreting", []))
        query = out["query"]
        results = out["results"]
        if not results:
            lines.extend([
                f'I couldn\'t find a recipe for "{query}" that matches all your preferences.',
                "Try a broader search or remove a restriction and search again.",
            ])
            return "\n\n".join(lines)

        lines.append(
            f'I found {len(results)} recipe{"s" if len(results) != 1 else ""} '
            f'for "{query}". Here are a few ideas:'
        )
        for number, recipe in enumerate(results, 1):
            step_label = "step" if recipe["n_steps"] == 1 else "steps"
            lines.append(f"{number}. {recipe['title']} — {recipe['n_steps']} {step_label}")
            if recipe.get("evidence"):
                lines.append(f"   Ingredients mentioned: {', '.join(recipe['evidence'])}")
        if out.get("include"):
            lines.append(f"Showing recipes that include: {', '.join(out['include'])}.")
        if out.get("exclude"):
            lines.append(f"Filtered out recipes mentioning: {', '.join(out['exclude'])}.")
            lines.append("Ingredient filters are based on recipe text and are not an allergy-safety guarantee.")
        lines.append(
            "Would you like to see one? Just tell me its number and I'll share "
            "the ingredients and instructions."
        )
        return "\n\n".join(lines)
    if action == "match":
        lines = list(out.get("warnings", []))
        recipes = out["results"]
        if recipes:
            lines.append(
                f"Nice, with {', '.join(out['original_have'])}, these recipes look like "
                "good places to start:"
            )
            for number, recipe in enumerate(recipes, 1):
                step_label = "step" if recipe["n_steps"] == 1 else "steps"
                lines.append(f"{number}. {recipe['title']} — {recipe['n_steps']} {step_label}")
                if recipe["matched"]:
                    lines.append(f"   Uses: {', '.join(recipe['matched'])}")
                if recipe["missing"]:
                    lines.append(f"   You may need: {', '.join(recipe['missing'])}")
                else:
                    lines.append("   You appear to have all listed ingredients.")
                if recipe["assumed_pantry"]:
                    lines.append(
                        f"   Assumed pantry staples (like salt or water): "
                        f"{', '.join(recipe['assumed_pantry'])}"
                    )
            lines.append(
                "Ingredient matching checks names, not amounts, so check the quantities in the recipe."
            )
        else:
            lines.append(
                "I couldn't find a recipe that matches all those ingredients and "
                "restrictions. We could try a few more ingredients or loosen one "
                "of the restrictions."
            )
            if out.get("closest_violating"):
                lines.append("These are closest options, but each one misses a requirement:")
                for recipe in out["closest_violating"]:
                    lines.append(f"• {recipe['title']}: {recipe['violates']}")
            lines.append("Try adding more ingredients or relaxing a restriction.")
        return "\n\n".join(lines)
    if action == "substitute":
        recipe = out.get("recipe") or {}
        lines = [f"Let's see what could work for {recipe.get('title', 'your recipe')}."]
        if out.get("status") == "target_not_in_recipe":
            lines.append(
                f'I couldn\'t find "{out["target"]}" in this recipe. Here are its ingredients:'
            )
            lines.extend(f"• {ingredient}" for ingredient in out.get("ingredient_lines", []))
            return "\n".join(lines)
        if out.get("target_lines"):
            lines.append(f"Ingredient to replace: {', '.join(out['target_lines'])}")
        candidates = out.get("candidates", [])
        if candidates:
            lines.append("Possible alternatives based on similar recipes:")
            for candidate in candidates:
                confidence = {
                    "strong": "good dataset support",
                    "weak": "limited dataset support",
                }.get(candidate["level"], "uncertain")
                lines.append(f"• {candidate['candidate']} ({confidence})")
            adaptation = out.get("adaptation")
            if adaptation:
                lines.append(
                    f"\nTo try: replace the ingredient with {adaptation['candidate']}."
                )
                lines.append("Updated ingredients:")
                lines.extend(f"• {ingredient}" for ingredient in adaptation["adapted_ingredients"])
                lines.append("Instructions:")
                lines.extend(
                    f"{step['n']}. {step['text']}" for step in adaptation["steps"]
                )
                lines.append(
                    "The replacement amount has not been verified; adjust it to taste and check the result."
                )
        else:
            lines.append(
                f'I don\'t have enough evidence to recommend a reliable replacement for "{out["target"]}".'
            )
        lines.append(out.get("disclaimer", ""))
        return "\n\n".join(line for line in lines if line)

    return format_results(out)


def create_app(df=None, index=None, gemini_client=None):
    if df is None:
        df = load_recipes()
    if index is None:
        index = SearchIndex.load_or_build(df, semantic=True, build_semantic=False)

    app = Flask(__name__)
    if index.semantic_notice:
        app.logger.warning(index.semantic_notice)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("FLASK_COOKIE_SECURE") == "1",
    )
    login_id = os.environ.get("RECIPEBOT_LOGIN_ID", "TEST1234")
    login_password = os.environ.get("RECIPEBOT_LOGIN_PASSWORD", "TEST1234")
    gemini_model = os.environ.get("GEMINI_MODEL")
    gemini_rewrite_enabled = os.environ.get("GEMINI_REWRITE") == "1"
    if gemini_client is None and gemini_rewrite_enabled:
        from .llm import Gemini

        model_config = Gemini()
        if model_config.available:
            from google import genai

            gemini_client = genai.Client(api_key=model_config.key)
            gemini_model = model_config.model
        else:
            logger.warning(model_config.configuration_notice or "Gemini is unavailable; using local replies.")
    sessions = {}
    sessions_lock = RLock()

    def login_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not web_session.get("authenticated"):
                return redirect(url_for("login"))
            return view(*args, **kwargs)

        return wrapped

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            submitted_id = request.form.get("login_id", "")
            submitted_password = request.form.get("password", "")
            valid_id = secrets.compare_digest(submitted_id, login_id)
            valid_password = secrets.compare_digest(submitted_password, login_password)
            if valid_id and valid_password:
                web_session.clear()
                web_session["authenticated"] = True
                return redirect(url_for("home"))
            error = "That login ID or password doesn't look right. Please try again."
        return render_template("login.html", error=error)

    @app.post("/logout")
    @login_required
    def logout():
        web_session.clear()
        return redirect(url_for("login"))

    @app.get("/")
    @login_required
    def home():
        return render_template(
            "index.html",
            gemini_enabled=gemini_client is not None and gemini_rewrite_enabled,
        )

    @app.post("/api/chat")
    @login_required
    def chat():
        payload = request.get_json(silent=True)
        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, str):
            return jsonify(error="Enter a message to get started."), 400
        message = message.strip()
        if not message:
            return jsonify(error="Enter a message to get started."), 400
        if len(message) > 1000:
            return jsonify(error="Messages must be 1,000 characters or fewer."), 400

        conversation_id = web_session.setdefault("conversation_id", uuid4().hex)
        with sessions_lock:
            recipe_session = sessions.get(conversation_id)
            if recipe_session is None:
                recipe_session = sessions[conversation_id] = Session(df, index)
            reply = format_web_reply(recipe_session.handle(message))
        if gemini_client is not None and gemini_rewrite_enabled:
            prompt = (
                "Rewrite the recipe assistant reply below so it sounds like a warm, "
                "natural conversation between two people. Keep it clear and concise. "
                "Preserve every recipe title, ingredient, amount, instruction, caveat, "
                "and recommendation exactly in meaning. Do not invent facts, add recipes, "
                "or mention internal scores or IDs. Return only the rewritten reply.\n\n"
                f"Assistant reply:\n{reply}"
            )
            try:
                response = gemini_client.models.generate_content(
                    model=gemini_model,
                    contents=prompt,
                )
                if not response.text or not response.text.strip():
                    raise RuntimeError("Gemini returned an empty reply.")
                reply = response.text.strip()
            except Exception:
                logger.exception("Gemini could not rewrite the recipe reply.")
                return jsonify(
                    error=(
                        "I couldn't reach the conversation service just now. "
                        "Please try again in a moment."
                    )
                ), 502
        return jsonify(reply=reply)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000)

"""Focused conversation-flow regressions; no network or external API calls."""
import pathlib
import sys
import tempfile
from unittest.mock import MagicMock, patch
import io
import urllib.error

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from recipebot.chat import Chat
from recipebot.llm import Gemini, normalize_model_name
from recipebot.search import SearchIndex, normalize_query
from recipebot.session import Session


def recipe(recipe_id, title, ingredients, steps=("Prepare ingredients.", "Cook and serve.")):
    return {
        "id": recipe_id,
        "source": "test",
        "title": title,
        "ingredients": ingredients,
        "instructions": "\n".join(steps),
        "steps": list(steps),
        "n_ingredients": len(ingredients),
        "n_steps": len(steps),
        "ing_text": "\n".join(ingredients).lower(),
    }


df = pd.DataFrame([
    recipe("ar_saffron", "Saffron Rice", ["1 pinch saffron", "1 cup rice", "2 cups broth", "1 onion"]),
    recipe("ar_saffron_two", "Indian Saffron Rice", ["saffron", "rice", "water", "salt"]),
    recipe("ar_vanilla", "Vanilla Ice Cream", ["vanilla", "cream", "sugar", "milk"]),
    recipe("ar_penne", "Penne Pasta", ["penne pasta", "tomato", "garlic", "olive oil"]),
    recipe("ar_cream_penne", "Creamy Penne", ["penne pasta", "heavy cream", "parmesan cheese"]),
])
index = SearchIndex.load_or_build(
    df,
    path=pathlib.Path(tempfile.mkdtemp()) / "session-flow.joblib",
)


def new_chat():
    return Chat(df, index, llm=None, summary=False)


def test_typos_compounds_and_visible_interpretation():
    result = new_chat().handle("Vanila Icecream")
    assert result["action"] == "search_recipes"
    assert result["text"].splitlines()[0] == "Interpreting 'vanila icecream' as 'vanilla ice cream'."
    first_result_line = next(line for line in result["text"].splitlines() if line.startswith("1. "))
    assert "ice cream" in first_result_line.lower()
    assert normalize_query("vanila", {"van", "ila", "vanilla"}) == "vanilla"
    assert normalize_query("icecream", {"icecream", "ice", "cream"}) == "ice cream"
    assert normalize_query("cheesecake pancakes", {"cheese", "cake", "pan", "cakes"}) == "cheesecake pancakes"


def test_help_is_fixed_text_and_does_not_search():
    chat = new_chat()
    result = chat.handle("what all can you make")
    assert result["action"] == "help"
    assert "ingredient-based" in result["text"] or "ingredients you have" in result["text"]
    assert chat.query == "" and chat.mode is None and chat.last_ids == []


def test_recipe_reference_shows_original_lines_and_numbered_steps():
    chat = new_chat()
    chat.handle("saffron rice")
    expected_id = chat.last_ids[0]
    selected = chat.handle("recipe 1")
    recipe_row = df.loc[df["id"] == expected_id].iloc[0]
    assert selected["action"] == "details"
    assert chat.selected_recipe_id == expected_id
    for line in recipe_row["ingredients"]:
        assert line in selected["text"]
    for number, step in enumerate(recipe_row["steps"], 1):
        assert f"{number}. {step}" in selected["text"]
    assert "non-empty original instruction line" in selected["text"]


def test_missing_saffron_uses_refusal_not_generated_adaptation():
    chat = new_chat()
    chat.handle("saffron rice")
    chat.handle("recipe 1")
    result = chat.handle("i dont have saffron")
    assert result["action"] == "substitute"
    assert result["status"] in {"weak_evidence", "insufficient_evidence"}
    assert result["adaptation"] is None
    assert "No adaptation generated" in result["text"]


def test_reference_without_results_does_not_search():
    chat = new_chat()
    result = chat.handle("recipe 1")
    assert result["action"] == "details"
    assert "I haven't shown any recipes yet" in result["text"]
    assert chat.query == "" and chat.mode is None and chat.last_ids == []


def test_dairy_restriction_refines_search():
    chat = new_chat()
    chat.handle("penne pasta")
    result = chat.handle("I cannot have dairy")
    assert result["action"] == "search_recipes"
    assert chat.query == "penne pasta"
    assert "dairy" in chat.exclude
    assert "Creamy Penne" not in result["text"]


def test_model_name_validation_and_no_request_for_invalid_models():
    assert normalize_model_name("<a name from the list>") is None
    assert normalize_model_name("") is None
    assert normalize_model_name("models/foo bar") is None
    assert normalize_model_name("models/xyz-1.5") == "xyz-1.5"
    client = Gemini(api_key="not-a-real-key", model="models/foo bar", discover_model=False)
    with patch.object(client, "_call") as call:
        try:
            client._generate({})
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid model name should be rejected.")
        call.assert_not_called()


def test_model_discovery_uses_header_and_retries_transient_errors():
    client = Gemini(api_key="unit-test-key", model="models/xyz-1.5", discover_model=False)
    response = MagicMock()
    response.read.return_value = b'{"models":[{"name":"models/gemini-flash",'
    response.read.return_value += b'"supportedGenerationMethods":["generateContent"]},'
    response.read.return_value += b'{"name":"models/gemini-embed",'
    response.read.return_value += b'"supportedGenerationMethods":["embedContent"]}]}'
    response.__enter__.return_value = response
    with patch("recipebot.llm.urllib.request.urlopen", return_value=response) as urlopen:
        assert client.list_models() == ["gemini-flash"]
    request = urlopen.call_args.args[0]
    assert request.full_url.startswith("https://generativelanguage.googleapis.com/v1beta/models?")
    assert "unit-test-key" not in request.full_url
    assert request.get_header("X-goog-api-key") == "unit-test-key"
    assert urlopen.call_args.kwargs["timeout"] == client.timeout

    retry_response = MagicMock()
    retry_response.read.return_value = b'{"ok":true}'
    retry_response.__enter__.return_value = retry_response
    too_many_requests = urllib.error.HTTPError(
        "https://generativelanguage.googleapis.com/v1beta/models",
        429,
        "rate limited",
        {},
        io.BytesIO(b"retry"),
    )
    with patch("recipebot.llm.time.sleep"), patch(
        "recipebot.llm.urllib.request.urlopen",
        side_effect=[too_many_requests, retry_response],
    ) as urlopen:
        assert client._call("https://generativelanguage.googleapis.com/v1beta/models") == {"ok": True}
        assert urlopen.call_count == 2


def test_missing_model_selects_available_flash_without_hardcoding():
    with patch.dict("os.environ", {"GEMINI_MODEL": ""}), patch.object(
        Gemini, "list_models", return_value=["gemini-pro", "gemini-next-flash"]
    ):
        client = Gemini(api_key="unit-test-key")
    assert client.available
    assert client.model == "gemini-next-flash"
    assert client.auto_selected


def test_missing_ingredient_without_results_is_an_explicit_exclusion():
    chat = new_chat()
    result = chat.handle("i dont have saffron")
    assert "Treating 'saffron' as an ingredient to exclude from results." in result["text"]
    assert "saffron" in chat.exclude
    assert chat.query == "" and chat.mode is None


def test_missing_ingredient_prompts_for_a_displayed_recipe_and_remembers_target():
    chat = new_chat()
    chat.handle("saffron rice")
    question = chat.handle("i dont have saffron")
    assert question["action"] == "clarify"
    assert "Which recipe do you mean?" in question["text"]
    assert chat.pending_target == "saffron"
    result = chat.handle("recipe 1")
    assert result["action"] == "substitute"
    assert result["args"]["target"] == "saffron"
    assert chat.pending_target is None


def test_legacy_web_session_handles_help_and_missing_ingredient_followup():
    session = Session(df, index)
    help_result = session.handle("what all can you make")
    assert help_result["action"] == "help"
    assert session.query == "" and session.last_results == []

    session.handle("saffron rice")
    question = session.handle("i dont have saffron")
    assert question["action"] == "ask" and session.pending_target == "saffron"
    substituted = session.handle("recipe 1")
    assert substituted["action"] == "substitute"
    assert substituted["status"] in {"weak_evidence", "insufficient_evidence"}
    assert substituted["adaptation"] is None

    fresh = Session(df, index)
    result = fresh.handle("i dont have saffron")
    assert result["action"] == "need_query"
    assert "Treating 'saffron' as an ingredient to exclude from results." in result["warnings"]
    assert fresh.query == ""


if __name__ == "__main__":
    for test in (
        test_typos_compounds_and_visible_interpretation,
        test_help_is_fixed_text_and_does_not_search,
        test_recipe_reference_shows_original_lines_and_numbered_steps,
        test_missing_saffron_uses_refusal_not_generated_adaptation,
        test_reference_without_results_does_not_search,
        test_dairy_restriction_refines_search,
        test_model_name_validation_and_no_request_for_invalid_models,
        test_model_discovery_uses_header_and_retries_transient_errors,
        test_missing_model_selects_available_flash_without_hardcoding,
        test_missing_ingredient_without_results_is_an_explicit_exclusion,
        test_missing_ingredient_prompts_for_a_displayed_recipe_and_remembers_target,
        test_legacy_web_session_handles_help_and_missing_ingredient_followup,
    ):
        test()
    print("all session flow tests passed")

import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from recipebot.search import SearchIndex
from recipebot.web import create_app


class WebInterfaceTests(unittest.TestCase):
    def setUp(self):
        recipes = [
            {
                "id": "test_chicken_plain",
                "source": "test",
                "title": "Simple Chicken Rice",
                "ingredients": ["chicken", "rice", "broth"],
                "instructions": "Cook the chicken and rice.",
                "steps": ["Cook the chicken and rice."],
                "n_ingredients": 3,
                "n_steps": 1,
                "ing_text": "chicken\nrice\nbroth",
            },
            {
                "id": "test_chicken_cream",
                "source": "test",
                "title": "Creamy Chicken Rice",
                "ingredients": ["chicken", "rice", "cream"],
                "instructions": "Cook the chicken and rice with cream.",
                "steps": ["Cook the chicken and rice with cream."],
                "n_ingredients": 3,
                "n_steps": 1,
                "ing_text": "chicken\nrice\ncream",
            },
        ]
        self.df = pd.DataFrame(recipes)
        self.temp_dir = tempfile.TemporaryDirectory()
        index = SearchIndex.load_or_build(
            self.df, path=pathlib.Path(self.temp_dir.name) / "index.joblib"
        )
        with patch.dict("os.environ", {"GEMINI_API_KEY": "", "GEMINI_REWRITE": ""}):
            app = create_app(self.df, index)
        self.client = app.test_client()
        self.client.post(
            "/login",
            data={"login_id": "TEST1234", "password": "TEST1234"},
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_homepage_renders_chat_interface(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"tailwindcss.com", response.data)
        self.assertIn(b"What sounds good today?", response.data)
        self.assertIn(b"Sign out", response.data)

    def test_restrictions_persist_for_conversation(self):
        first = self.client.post("/api/chat", json={"message": "Find chicken rice recipes"})
        second = self.client.post("/api/chat", json={"message": "I cannot have dairy"})

        self.assertEqual(first.status_code, 200)
        self.assertIn("Creamy Chicken Rice", first.json["reply"])
        self.assertIn("I found", first.json["reply"])
        self.assertIn("— 1 step", first.json["reply"])
        self.assertNotIn("score", first.json["reply"])
        self.assertNotIn("test_chicken", first.json["reply"])
        self.assertNotIn("[", first.json["reply"])
        self.assertEqual(second.status_code, 200)
        self.assertIn("Simple Chicken Rice", second.json["reply"])
        self.assertNotIn("Creamy Chicken Rice", second.json["reply"])

    def test_selecting_recipe_shows_ingredients_and_instructions(self):
        self.client.post("/api/chat", json={"message": "Find chicken rice recipes"})
        response = self.client.post("/api/chat", json={"message": "recipe 1"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("Ingredients (3, original dataset lines)", response.json["reply"])
        self.assertIn("Instructions:", response.json["reply"])
        self.assertIn("1. Cook the chicken and rice.", response.json["reply"])
        self.assertIn("Cook the chicken and rice.", response.json["reply"])

    def test_help_is_fixed_and_does_not_run_a_recipe_search(self):
        response = self.client.post("/api/chat", json={"message": "what all can you make"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("I can help you find recipes", response.json["reply"])
        self.assertNotIn("Simple Chicken Rice", response.json["reply"])

    def test_ingredient_match_is_explained_in_plain_language(self):
        response = self.client.post("/api/chat", json={"message": "I have chicken and rice"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("good places to start", response.json["reply"])
        self.assertIn("Uses:", response.json["reply"])
        self.assertIn("You may need:", response.json["reply"])
        self.assertNotIn("Ranking formula", response.json["reply"])

    def test_empty_message_returns_validation_error(self):
        response = self.client.post("/api/chat", json={"message": "  "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)

    def test_non_string_message_returns_validation_error(self):
        response = self.client.post("/api/chat", json={"message": 5})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)

    def test_oversized_message_returns_validation_error(self):
        response = self.client.post("/api/chat", json={"message": "a" * 1001})
        self.assertEqual(response.status_code, 400)
        self.assertIn("1,000 characters", response.json["error"])

    def test_login_is_required_and_default_credentials_work(self):
        client = self.client.application.test_client()
        response = client.get("/")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, "/login")

        page = client.get("/login")
        self.assertEqual(page.status_code, 200)
        self.assertNotIn(b"TEST1234", page.data)
        failed_login = client.post(
            "/login",
            data={"login_id": "wrong", "password": "wrong"},
        )
        self.assertEqual(failed_login.status_code, 200)
        self.assertIn(b"doesn&#39;t look right", failed_login.data)
        successful_login = client.post(
            "/login",
            data={"login_id": "TEST1234", "password": "TEST1234"},
        )
        self.assertEqual(successful_login.status_code, 302)
        self.assertEqual(successful_login.location, "/")

    def test_logout_requires_login_again(self):
        response = self.client.post("/logout")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, "/login")
        self.assertEqual(self.client.get("/").location, "/login")

    def test_gemini_makes_reply_conversational_without_exposing_key(self):
        class FakeModels:
            def __init__(self):
                self.prompt = None

            def generate_content(self, *, model, contents):
                self.prompt = contents
                return type("Response", (), {"text": "That sounds lovely! Here are a few ideas."})()

        class FakeGemini:
            def __init__(self):
                self.models = FakeModels()

        gemini = FakeGemini()
        index = SearchIndex.load_or_build(
            self.df, path=pathlib.Path(self.temp_dir.name) / "gemini-index.joblib"
        )
        with patch.dict("os.environ", {"GEMINI_REWRITE": "1"}):
            client = create_app(self.df, index, gemini_client=gemini).test_client()
        client.post("/login", data={"login_id": "TEST1234", "password": "TEST1234"})
        response = client.post("/api/chat", json={"message": "Find chicken rice"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["reply"], "That sounds lovely! Here are a few ideas.")
        self.assertIn("Do not invent facts", gemini.models.prompt)

    def test_api_key_alone_does_not_add_gemini_latency(self):
        from google import genai

        with (
            patch.dict(
                "os.environ",
                {"GEMINI_API_KEY": "test-key", "GEMINI_REWRITE": ""},
            ),
            patch.object(genai, "Client") as gemini_constructor,
        ):
            app = create_app(self.df, SearchIndex.load_or_build(
                self.df, path=pathlib.Path(self.temp_dir.name) / "fast-index.joblib"
            ))

        gemini_constructor.assert_not_called()
        self.assertEqual(app.test_client().get("/").status_code, 302)


if __name__ == "__main__":
    unittest.main()

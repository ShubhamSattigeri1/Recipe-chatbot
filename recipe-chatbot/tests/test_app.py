"""Run: python tests/test_app.py  (Flask test client, synthetic data, no network)"""
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_chat as T                                  # reuses the synthetic dataset
from recipebot.app import create_app
import recipebot.chat as chatmod
chatmod.match_ingredients, chatmod.format_match = T.fake_match, T.fake_fmt
app = create_app(df=T.df, index=T.idx, llm=None)
c = app.test_client()
page = c.get("/").get_data(as_text=True)
assert "SpeechRecognition" in page and "speechSynthesis" in page and "/api/chat" in page
assert c.get("/health").get_json()["recipes"] == len(T.df)
r = c.post("/api/chat", json={"message": "find a penne pasta recipe", "sid": "a"}).get_json()
assert r["tool"] == "search_recipes" and "Penne" in r["text"] and r["speech"] and r["router"] == "rules"
r = c.post("/api/chat", json={"message": "I cannot have dairy", "sid": "a"}).get_json()
assert "Creamy Penne Alfredo" not in r["text"]
r = c.post("/api/chat", json={"message": "penne pasta", "sid": "b"}).get_json()            # another browser: separate state
assert "Creamy Penne Alfredo" in r["text"]
assert c.post("/api/chat", json={"message": ""}).status_code == 400
print("all app tests passed")
# Nosh Recipe Finder

> A lightweight, ingredient-aware recipe-finder chat application running locally with Flask and Tailwind CSS.

---

## 🌟 Overview

**Nosh** combines a curated recipe collection with intelligent ingredient matching. Tell Nosh what you want to eat, what ingredients you have, or what you need to avoid, and it will instantly search its database to suggest matching recipes.

---

## ⚙️ Key Features

* **Smart Search:** Search recipes by dish name or any keywords in titles and ingredients.
* **Hybrid Ranking:** Combines TF-IDF keyword ranking with local sentence-embedding ranking for superior results.
* **Dietary Filters:** Ask for recipes that include or exclude specific ingredients or ingredient groups.
* **Pantry Matching:** Find possible recipes from the ingredients you have on hand and see what's still missing.
* **Interactive Navigation:** Open any recipe by replying with its number to view full ingredients and instructions.
* **Ingredient Substitutions:** Get smart substitution suggestions based on comparable recipes and curated cooking roles.
* **Conversation Memory:** Maintain restrictions and search state throughout your chat, or type `reset` / `clear` to start fresh.
* **Optional Gemini Integration:** Use Google Gemini to rewrite replies into a more conversational tone (disabled by default to ensure instant responses).

---

## 🔑 Demo Login

Nosh comes with hard-coded demo credentials for local use:

| Field | Default Value |
| --- | --- |
| **Login ID** | `TEST1234` |
| **Password** | `TEST1234` |

> 📌 **Sign-in URL:** `[http://127.0.0.1:5000/login](http://127.0.0.1:5000/login)`
> *Use the "Sign out" button inside the app to end your session. Override these using `RECIPEBOT_LOGIN_ID` and `RECIPEBOT_LOGIN_PASSWORD` environment variables in production.*

---

## 🚀 Getting Started on Windows

1. Open **PowerShell** and navigate to your project directory:
```powershell
cd "C:\Users\Admin\OneDrive\Desktop\Shubham\Nosh_Robotics_Task\recipe-chatbot"

```


2. Install dependencies:
```powershell
..\venv\Scripts\python.exe -m pip install -r requirements.txt

```


3. Start the Flask application:
```powershell
..\venv\Scripts\python.exe -m recipebot.web

```


4. Open your browser and go to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**, then sign in with the demo credentials.
> ⚠️ *Keep the PowerShell window open while using Nosh. Press **Ctrl+C** to stop the server.*



---

## 💻 Command-Line Interface (CLI)

You can also run Nosh directly from your terminal:

```powershell
..\venv\Scripts\python.exe -m recipebot.cli

```

### Building the Hybrid Index

To unlock semantic search alongside keyword matching, build the hybrid index once:

```powershell
..\venv\Scripts\python.exe -m recipebot.cli --build-hybrid-index

```

*This downloads the local embedding model and saves vectors to `data/tfidf.semantic.joblib`. Subsequent searches will run locally using hybrid ranking.*

---

## 🎛️ Optional Server Settings

| Environment Variable | Description | Default |
| --- | --- | --- |
| `RECIPEBOT_LOGIN_ID` | Override demo login ID | `TEST1234` |
| `RECIPEBOT_LOGIN_PASSWORD` | Override demo password | `TEST1234` |
| `FLASK_SECRET_KEY` | Flask session-signing key | Random value per startup |
| `FLASK_COOKIE_SECURE` | Set to `1` for HTTPS secure cookies | Disabled |
| `GEMINI_REWRITE` | Set to `1` to enable Gemini reply rewriting | Disabled |
| `GEMINI_API_KEY` | Google Gemini API key (or `GOOGLE_API_KEY`) | Not set |
| `GEMINI_MODEL` | Valid Gemini model name | Discovered automatically |

---

## 📂 Project Structure

```text
recipe-chatbot/
├── README.md
├── requirements.txt
├── data/
│   ├── recipes_raw_nosource_ar.json
│   ├── recipes_raw_nosource_epi.json
│   ├── recipes_raw_nosource_fn.json
│   ├── recipes.pkl
│   └── tfidf.joblib
├── recipebot/
│   ├── data.py          # Load and prepare recipe data
│   ├── search.py        # Hybrid search and ingredient filters
│   ├── match.py         # Match recipes to ingredients on hand
│   ├── substitute.py    # Evidence-based substitution suggestions
│   ├── session.py       # Conversation state, parsing, and recipe actions
│   ├── web.py           # Flask routes, authentication, and chat responses
│   └── templates/
│       ├── index.html   # Chat interface
│       └── login.html   # Sign-in page
└── tests/               # Search, matching, substitution, analytics, and web tests

```

---

## 🧪 Running Tests

Run the test suite using PowerShell:

```powershell
..\venv\Scripts\python.exe tests\test_session_flow.py
..\venv\Scripts\python.exe tests\test_search.py
..\venv\Scripts\python.exe tests\test_match.py
..\venv\Scripts\python.exe tests\test_chat.py
..\venv\Scripts\python.exe tests\test_substitute_engine.py
..\venv\Scripts\python.exe tests\test_analytics.py
..\venv\Scripts\python.exe tests\test_web.py
..\venv\Scripts\python.exe tests\test_app.py

```

---

## 🔮 Future Scope: Voice Communication

Nosh is built to expand! Future iterations could include:

1. **Speech-to-Text:** Browser microphone integration to convert spoken requests into chat messages.
2. **Unified Logic:** Routing voice queries through the existing `/api/chat` pipeline.
3. **Text-to-Speech:** Reading replies aloud with integrated playback controls.
4. **Enhanced Features:** Persistent user accounts, pantry tracking, and robust cloud deployment monitoring.

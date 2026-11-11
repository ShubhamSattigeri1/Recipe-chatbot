# Nosh Recipe Finder

Nosh is a small, locally run recipe-finder chat application. It combines a recipe collection with ingredient-aware search and serves a browser-based chat interface using Flask.

## In plain language

Tell Nosh what you would like to eat, what ingredients you have, or what ingredients you need to avoid. It searches the recipes in its collection and suggests matches. You can ask for a recipe's ingredients and instructions, or ask about replacing an ingredient.

Nosh uses recipe text and keyword matching. It is not a nutrition service, and its dietary filters are not a guarantee that a recipe is safe for an allergy.

## Features

- Search recipes by dish name or other words in the recipe titles and ingredients.
<<<<<<< HEAD
<<<<<<< HEAD
- Combine TF-IDF keyword ranking with local sentence-embedding ranking once the hybrid index is built.
=======
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd
=======
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd
- Ask for recipes that include an ingredient or exclude an ingredient or ingredient group.
- Find possible recipes from ingredients you have, and see which ingredients may still be needed.
- Open a recipe by replying with its number to see its ingredients and instructions.
- Ask for an ingredient substitution for a selected recipe. Suggestions are based on comparable recipes and curated cooking roles; availability of evidence varies.
- Keep restrictions and search state during the current conversation, or say `reset` or `clear` to start over.
- Optionally use Google Gemini to rewrite a local reply in a more conversational style. This is off by default so a Google API request does not delay every chat response.

## Login

The demo login defaults are:

| Field | Default |
|---|---|
| Login ID | `TEST1234` |
| Password | `TEST1234` |

Enter these on the sign-in page at `http://127.0.0.1:5000/login`. Use **Sign out** in the app to end the session.

These are shared, hard-coded demo defaults—not private credentials. Do not expose this development app to the public internet with them. You can override them with the `RECIPEBOT_LOGIN_ID` and `RECIPEBOT_LOGIN_PASSWORD` environment variables. A production deployment should use unique credentials and a production-grade authentication setup.

## Run on Windows

Open PowerShell and run:

```powershell
cd "C:\Users\Admin\OneDrive\Desktop\Shubham\Nosh_Robotics_Task\recipe-chatbot"
..\venv\Scripts\python.exe -m pip install -r requirements.txt
..\venv\Scripts\python.exe -m recipebot.web
```

Open **http://127.0.0.1:5000** and sign in with the demo credentials above. Leave the PowerShell window open while using Nosh; press **Ctrl+C** in that window to stop the server.

<<<<<<< HEAD
<<<<<<< HEAD
The server binds to `127.0.0.1`, which makes it available on the same computer. On its first run, Nosh may take longer to build its TF-IDF index, which is cached in `data/tfidf.joblib`. The server does not build the much larger semantic index during startup; use the one-time build command below to enable hybrid ranking.

### Run the command-line chatbot

From the `recipe-chatbot` directory, run:

```powershell
..\venv\Scripts\python.exe -m recipebot.cli
```

The CLI uses local rule-based routing when Gemini is disabled or unavailable. Use `--no-llm` to disable Gemini routing and `--debug` to print detailed API errors. Normal CLI and web startup use hybrid retrieval when its cache exists; otherwise they clearly report and use TF-IDF without blocking to build the full embedding corpus.

Build the hybrid index once from the `recipe-chatbot` directory:

```powershell
..\venv\Scripts\python.exe -m recipebot.cli --build-hybrid-index
```

This downloads the local embedding model if needed and embeds the recipe corpus. The one-time CPU build can take a long time; it shows progress and saves vectors to `data/tfidf.semantic.joblib`. After it finishes, normal CLI and web searches use hybrid ranking locally. To list available Gemini models instead, run `python -m recipebot.cli --list-models`.

### Optional Gemini routing and replies

The CLI uses Gemini only to choose a tool; recipe facts and substitutions are retrieved and computed locally. It uses `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) and the optional `GEMINI_MODEL`. If the model is omitted, the CLI discovers models that support `generateContent` and selects an available Flash-class model. Invalid model names are rejected before a model request is built. Requests have a timeout and retry once for HTTP 429 or 5xx responses; failures fall back to rules.

Copy `recipe-chatbot/.env.example` to `recipe-chatbot/recipebot/.env` and set your key locally, or use environment variables. `.env` is not a place to store a key in tracked source; do not commit it, paste it into chat, or include it in screenshots. To list available model IDs, run `python -m recipebot.cli --list-models`. With a key configured, you can explicitly select a model by setting `GEMINI_MODEL`; a leading `models/` is optional.

The browser app's Gemini reply rewriting remains opt-in via `GEMINI_REWRITE=1`; it uses the same configured/discovered model and adds network latency. Without it, web replies remain local. The API key stays server-side.
=======
=======
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd
The server binds to `127.0.0.1`, which makes it available on the same computer. On its first run, Nosh may take longer to load or build its recipe search index. The index is cached in `data/tfidf.joblib` for subsequent starts.

### Optional Google Gemini replies

Gemini rewriting is disabled unless `GEMINI_REWRITE` is set to `1`. To enable it for a PowerShell session:

```powershell
$env:GEMINI_API_KEY = "your-Google-AI-Studio-key"
$env:GEMINI_REWRITE = "1"
..\venv\Scripts\python.exe -m recipebot.web
```

The default model is `gemini-3.8-flash`. You can override it with `GEMINI_MODEL`. When enabled, each reply is sent to Google for rewriting, so replies depend on network availability and may take longer. If the Gemini request fails, Nosh reports an error instead of returning the unrevised reply.

The API key is read by the server from the `GEMINI_API_KEY` environment variable; it is not sent to the browser. Do not put an API key in source code, commit it, or share it in screenshots or messages. Setting the key alone does not enable Gemini.

The app does not automatically load `.env` files. If your key is stored there, load it into the PowerShell environment before starting the server.
<<<<<<< HEAD
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd
=======
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd

Other optional server settings:

| Environment variable | Purpose | Default |
|---|---|---|
| `RECIPEBOT_LOGIN_ID` | Override the demo login ID | `TEST1234` |
| `RECIPEBOT_LOGIN_PASSWORD` | Override the demo password | `TEST1234` |
| `FLASK_SECRET_KEY` | Set a stable Flask session-signing key | Random value generated at startup |
| `FLASK_COOKIE_SECURE` | Set to `1` to mark session cookies secure when using HTTPS | Disabled |
| `GEMINI_REWRITE` | Set to `1` to enable Gemini reply rewriting | Disabled |
<<<<<<< HEAD
<<<<<<< HEAD
| `GEMINI_API_KEY` | Google Gemini API key; also accepts `GOOGLE_API_KEY` | Not set |
| `GOOGLE_API_KEY` | Alternate Google Gemini API key variable | Not set |
| `GEMINI_MODEL` | Optional valid Gemini model name; discovered when omitted | Not set |
=======
| `GEMINI_API_KEY` | Google Gemini API key; required for Gemini rewriting | Not set |
| `GEMINI_MODEL` | Gemini model name | `gemini-3.8-flash` |
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd
=======
| `GEMINI_API_KEY` | Google Gemini API key; required for Gemini rewriting | Not set |
| `GEMINI_MODEL` | Gemini model name | `gemini-3.8-flash` |
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd

If `FLASK_SECRET_KEY` is not set, a new key is generated at each startup. Existing browser sessions will need to sign in again after restarting.

## Technical overview

### Request and response flow

1. Flask serves the Tailwind-styled chat page and requires a signed-in session for the application and chat API.
2. The browser sends the user's message to `POST /api/chat`.
3. The server keeps a `recipebot.session.Session` for that browser conversation and passes the message to the existing deterministic recipe logic.
4. `format_web_reply` turns the structured result into a human-readable response. Recipe IDs and ranking scores are not included in normal recipe result summaries.
5. If Gemini rewriting is explicitly enabled, the server sends that response to Gemini and returns the rewritten text.

The API accepts JSON in the form `{"message": "..."}`. Empty messages and messages longer than 1,000 characters are rejected.

### Recipe data and search

`recipebot.data.load_recipes()` loads the three recipe JSON collections in `data/` into a pandas DataFrame. It removes advertisement-only ingredient artifacts and drops entries without a title, ingredients, or instructions. The prepared DataFrame is cached as `data/recipes.pkl`.

<<<<<<< HEAD
<<<<<<< HEAD
`recipebot.search.SearchIndex` builds or loads a TF-IDF index from recipe titles and ingredient text plus a local dense sentence-embedding index (`sentence-transformers/all-MiniLM-L6-v2`). Retrieval combines the two rankings with weighted reciprocal-rank fusion (55% TF-IDF, 45% semantic), so exact ingredient/title overlap remains important while semantically related wording can also contribute. Lexical weight is applied only to recipes with actual TF-IDF matches, so a zero-match query is ranked by semantic similarity instead of arbitrary dataset order. Both indexes are local and cached (`data/tfidf.joblib` and `data/tfidf.semantic.joblib`). Build the dense corpus cache explicitly with `python -m recipebot.cli --build-hybrid-index` to avoid delaying ordinary app startup; later starts load the cached vectors and embed queries locally. Set `RECIPEBOT_EMBEDDING_MODEL` to another locally supported Sentence Transformers model if needed. If the optional embedding package or model is unavailable, the application reports the fallback and continues with TF-IDF.

Ingredient inclusions and exclusions are still enforced as hard filters after ranking; semantic similarity cannot bypass a requested restriction. The vector model ranks recipes but does not generate recipe facts or alter original ingredient strings.
=======
`recipebot.search.SearchIndex` builds or loads a TF-IDF index from recipe titles and ingredient text. Search results are ranked by text similarity, while requested ingredient inclusions and exclusions are applied as hard filters. The index is cached in `data/tfidf.joblib`.
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd
=======
`recipebot.search.SearchIndex` builds or loads a TF-IDF index from recipe titles and ingredient text. Search results are ranked by text similarity, while requested ingredient inclusions and exclusions are applied as hard filters. The index is cached in `data/tfidf.joblib`.
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd

Ingredient-based matching reports matched ingredients and ingredients to add. The matching code treats salt and water as assumed pantry staples by default. Ingredient quantities are not generally verified.

Substitution suggestions use similar recipes, ingredient-role heuristics, and evidence levels. Suggestions are not guaranteed to work in every recipe; replacement amounts and results may need testing.

<<<<<<< HEAD
<<<<<<< HEAD
### Worked technical trace: CLI search for "saffron rice"

This example traces the actual request through the current implementation. The user searches for **saffron rice**.

1. **The CLI reads the message.** `python -m recipebot.cli` loads the recipe DataFrame and hybrid search indexes once, then reads `saffron rice` from the terminal.
2. **The router identifies a search.** `Chat.handle()` sends the message to Gemini for a tool choice when a valid Gemini model is configured; otherwise, the rule-based router handles it. The LLM only routes the request and never supplies recipe facts.
4. **The query is normalized.** `search()` lowercases tokens, splits recognizable compounds such as `icecream`, and fuzzy-corrects unknown words against a cached vocabulary built from titles and original ingredient lines. Any changes are shown to the user. Known vocabulary words are left intact.
5. **Two query representations are computed locally.** TF-IDF generates a sparse lexical vector; the cached MiniLM model generates a dense query embedding. The two recipe rankings are combined with weighted reciprocal-rank fusion, avoiding direct comparison of their differently scaled scores.
6. **Recipes are ranked and filtered.** The fused order is scanned and ingredient inclusions/exclusions are applied as hard filters; the embedding model cannot bypass those constraints. With no constraints, a query with no matching lexical terms can still retrieve semantically similar recipes.
7. **The CLI formats a grounded answer.** Each result has a title, step count, and up to three original ingredient lines matching query tokens. A follow-up such as `recipe 1` resolves by displayed position and shows all original ingredient lines and numbered instruction lines.

**Observed example output:** using the local recipe data, the query returns saffron-specific matches such as `Indian Saffron Rice`, `Saffron Rice`, and `Saffron Rice with Currants`. The CLI uses the hybrid index when the local embedding dependency/model is available; otherwise its startup message identifies the TF-IDF fallback. Combined ranking is for retrieval, not a probability or measure of cooking quality.
=======
=======
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd
### Worked technical trace: searching for "saffron rice"

This example traces the actual request through the current implementation. The user searches for **saffron rice**.

1. **The browser submits the message.** The chat page sends `{"message": "Find saffron rice recipes"}` as JSON to `POST /api/chat`. Flask first checks that the browser is logged in and that the message is a non-empty string no longer than 1,000 characters.
2. **The server retrieves the conversation state.** Flask uses the conversation ID in the signed session cookie to retrieve (or create) a `recipebot.session.Session`. That object already references the loaded recipe DataFrame and search index.
3. **The session parser identifies a search.** `Session.handle()` lowercases and trims the text. It checks for reset commands, recipe-number selection, substitution requests, ingredient exclusions/inclusions, and "I have ..." inventory messages. This sentence matches none of those special cases, but starts with the recognized search verb `Find`, so it is handled as a normal recipe search.
4. **The query is cleaned.** `_clean_query()` removes the leading `Find` and trailing `recipes`, leaving the exact search string `saffron rice`. This step does **not** spell-correct search queries. The parser's fuzzy matching applies only to recognized dietary restriction group names, not dish searches.
5. **The search string becomes a TF-IDF query vector.** `search()` passes the lowercase text to the fitted scikit-learn `TfidfVectorizer`. In this example, both `saffron` and `rice` are present in the index vocabulary, so the query contributes meaningful features for both words.
6. **Recipes are ranked and filtered.** The vectorizer transforms the query into a sparse vector. The search code multiplies the precomputed sparse recipe matrix by that query vector, sorts recipe scores from highest to lowest, and examines candidates in batches of 2,000 (up to 60,000 positive-score candidates). There are no include/exclude filters in this example, so the candidates are not removed by dietary constraints. The first five matches are returned.
7. **The server formats a friendly answer.** Each result includes a title, step count, and up to three ingredient lines whose text matches query tokens. The browser receives a JSON `reply`; it displays that text without exposing internal recipe IDs or ranking scores.
8. **A follow-up can open a result.** If the user replies `recipe 1`, the session maps that position in its saved `last_results` back to the recipe row, then returns the recipe's ingredient list and instructions.

**Observed example output:** using the current local recipe data and TF-IDF cache, the correctly spelled query returned saffron-specific matches such as `Indian Saffron Rice`, `Saffron Rice`, and `Saffron Rice with Currants`. The top scores reflect how strongly each recipe text matches the query terms; they are not probabilities that the recipe is a good match. This example illustrates the intended behavior of the recipe search: exact keyword terms such as `saffron` and `rice` drive retrieval, while filtering remains separate from ranking.
<<<<<<< HEAD
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd
=======
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd

### Main files

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
<<<<<<< HEAD
<<<<<<< HEAD
│   ├── search.py        # Hybrid search and ingredient filters
=======
│   ├── search.py        # TF-IDF search and ingredient filters
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd
=======
│   ├── search.py        # TF-IDF search and ingredient filters
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd
│   ├── match.py         # Match recipes to ingredients on hand
│   ├── substitute.py    # Evidence-based substitution suggestions
│   ├── session.py       # Conversation state, parsing, and recipe actions
│   ├── web.py           # Flask routes, authentication, and chat responses
│   └── templates/
│       ├── index.html   # Chat interface
│       └── login.html   # Sign-in page
└── tests/               # Search, matching, substitution, analytics, and web tests
```

## Dependencies and interface styling

<<<<<<< HEAD
<<<<<<< HEAD
Dependencies are listed in `requirements.txt`. Local hybrid retrieval uses Sentence Transformers and its selected PyTorch backend; the initial model download requires network access. Once cached, embedding and retrieval run locally. The Google GenAI SDK is used only when optional web rewriting is enabled.
=======
Dependencies are listed in `requirements.txt`: pandas, scikit-learn, joblib, NumPy, Flask, and the Google GenAI SDK. The Google SDK is only used when Gemini rewriting is enabled.
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd
=======
Dependencies are listed in `requirements.txt`: pandas, scikit-learn, joblib, NumPy, Flask, and the Google GenAI SDK. The Google SDK is only used when Gemini rewriting is enabled.
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd

Tailwind CSS and the DM Sans font are loaded from external CDNs by the templates. Internet access is needed for those hosted styles and fonts; the recipe search itself runs locally.

## Tests

<<<<<<< HEAD
<<<<<<< HEAD
From the `recipe-chatbot` directory, run all available test scripts with:

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

The historical `tests/test_substitute.py` is not present in this checkout; `tests/test_substitute_engine.py` covers the current substitution engine.

The current `test_analytics.py` run has a pre-existing failing assertion: it expects a sentence with three periods to count as three steps, while the repository's `split_steps()` rule counts non-empty original instruction lines. Recipe detail output follows the latter rule; this unrelated behavior was not changed.
=======
=======
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd
From the `recipe-chatbot` directory, run the focused web tests with:

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -p test_web.py
```

The other test scripts are `test_search.py`, `test_match.py`, `test_substitute_engine.py`, `test_substitute.py`, and `test_analytics.py`. Some existing standalone tests may not pass with the current code: `test_analytics.py` has an assertion failure, and `test_substitute.py` imports a `substitute` function that is not present in the current substitution module. These are separate from the web tests.
<<<<<<< HEAD
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd
=======
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd

## Limitations and security notes

- Search, restrictions, and ingredient matching are based on recipe text and keyword heuristics. They can miss ingredient variants and must not be relied on for allergy safety.
- Recipe quantities are not comprehensively checked against the ingredients a user has.
- Substitution suggestions are evidence-informed estimates, not tested or guaranteed cooking instructions.
- Gemini is an optional language rewriter, not the source of recipe search results. Enabling it sends the generated reply text to Google's API and adds network latency.
- Conversation state is held in server memory and is not saved as a durable chat history.
- The default credentials are for local demonstration only. The built-in sign-in is intentionally simple and is not sufficient protection for a public or production service.
- The development server is intended for local use, not as a production web server.

## Future scope

### Voice communication

Nosh could support spoken conversation in a future version:

1. Add a microphone control and browser speech recognition to convert speech to text.
2. Send the recognized text through the existing `/api/chat` endpoint so voice and typed messages use the same recipe logic.
3. Read replies aloud with browser text-to-speech, with controls to pause, replay, or mute.
4. Add clear microphone permission, transcription review, and privacy notices. A later option could use a speech-to-text or text-to-speech provider for browsers without the required built-in support.

Other possible extensions include persistent user accounts and saved recipes, improved nutrition and dietary metadata, a pantry inventory, and deployment with production authentication and monitoring.

# Nosh Recipe Finder

Nosh is a small, locally run recipe-finder chat application. It combines a recipe collection with ingredient-aware search and serves a browser-based chat interface using Flask.

## In plain language

Tell Nosh what you would like to eat, what ingredients you have, or what ingredients you need to avoid. It searches the recipes in its collection and suggests matches. You can ask for a recipe's ingredients and instructions, or ask about replacing an ingredient.

Nosh uses recipe text and keyword matching. It is not a nutrition service, and its dietary filters are not a guarantee that a recipe is safe for an allergy.

## Features

- Search recipes by dish name or other words in the recipe titles and ingredients.
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

Other optional server settings:

| Environment variable | Purpose | Default |
|---|---|---|
| `RECIPEBOT_LOGIN_ID` | Override the demo login ID | `TEST1234` |
| `RECIPEBOT_LOGIN_PASSWORD` | Override the demo password | `TEST1234` |
| `FLASK_SECRET_KEY` | Set a stable Flask session-signing key | Random value generated at startup |
| `FLASK_COOKIE_SECURE` | Set to `1` to mark session cookies secure when using HTTPS | Disabled |
| `GEMINI_REWRITE` | Set to `1` to enable Gemini reply rewriting | Disabled |
| `GEMINI_API_KEY` | Google Gemini API key; required for Gemini rewriting | Not set |
| `GEMINI_MODEL` | Gemini model name | `gemini-3.8-flash` |

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

`recipebot.search.SearchIndex` builds or loads a TF-IDF index from recipe titles and ingredient text. Search results are ranked by text similarity, while requested ingredient inclusions and exclusions are applied as hard filters. The index is cached in `data/tfidf.joblib`.

Ingredient-based matching reports matched ingredients and ingredients to add. The matching code treats salt and water as assumed pantry staples by default. Ingredient quantities are not generally verified.

Substitution suggestions use similar recipes, ingredient-role heuristics, and evidence levels. Suggestions are not guaranteed to work in every recipe; replacement amounts and results may need testing.

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
│   ├── search.py        # TF-IDF search and ingredient filters
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

Dependencies are listed in `requirements.txt`: pandas, scikit-learn, joblib, NumPy, Flask, and the Google GenAI SDK. The Google SDK is only used when Gemini rewriting is enabled.

Tailwind CSS and the DM Sans font are loaded from external CDNs by the templates. Internet access is needed for those hosted styles and fonts; the recipe search itself runs locally.

## Tests

From the `recipe-chatbot` directory, run the focused web tests with:

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -p test_web.py
```

The other test scripts are `test_search.py`, `test_match.py`, `test_substitute_engine.py`, `test_substitute.py`, and `test_analytics.py`. Some existing standalone tests may not pass with the current code: `test_analytics.py` has an assertion failure, and `test_substitute.py` imports a `substitute` function that is not present in the current substitution module. These are separate from the web tests.

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

"""Step 2b: session state. Restrictions PERSIST across turns until the user says 'reset'/'clear'.
Rule-based message parser (no LLM yet); in Step 3 Gemini will produce the same structured fields."""
import re

import difflib

from .search import GROUPS, _term_mask, search, singular

_EXCL = re.compile(r"(?:\bi\s+)?(?:can\s?not|can't|cant|cannot)\s+(?:have|eat)\s+(?P<a>[^.?!]+)"
                   r"|(?:\bi\s+)?(?:do\s+not|don['’]t|dont)\s+have\s+(?P<c>[^.?!]+)"
                   r"|(?:\bno\b|\bwithout\b|\bexclude\b|\bavoid\b|allergic to|do not want|don't want)\s+(?P<b>[^.?!]+)")
_INCL = re.compile(r"(?:must (?:have|contain)|include|that (?:has|have|contains?))\s+(?P<a>[^.?!]+)")
_FILLER = {"i", "also", "and", "but", "please", "sorry", "ok", "okay", "now", "so", "then", "pls"}
_HAVE = re.compile(r"^(?:i have|i've got|using|with only)\s+(.+)$")
_MAX_STEPS = re.compile(
    r"\b(?:at most|no more than|max)\s+(\d+)\s+steps?\b|\bunder\s+(\d+)\s+steps?\b"
)
_RECIPE_COUNT = re.compile(r"\b(?:suggest\s+(\d+)\s+recipes?|top\s+(\d+))\b")
_SUB_ACTION = re.compile(
    r"\b(?:remove|replace|swap(?:\s+out)?|substitute|take\s+out|omit|leave\s+out)\b",
    re.IGNORECASE,
)
_RECIPE_ID = re.compile(r"\b(?:ar|epi|fn)_[A-Za-z0-9./_-]+", re.IGNORECASE)
_SUB_TARGET = re.compile(
    r"\b(?:remove|replace|swap\s+out|swap|substitute|take\s+out|omit|leave\s+out)"
    r"\s+(?:the\s+)?(?P<t>[a-z][a-z ]*?)\s+(?:from|in|out of)\b",
    re.IGNORECASE,
)
_WITHOUT_TARGET = re.compile(
    r"\bwithout\s+(?P<t>[a-z][a-z ]*?)\s+in\b", re.IGNORECASE,
)
_MAKE_FREE = re.compile(
    r"\bmake\s+(?:it|recipe\s+\d+)\s+(?P<t>[a-z][a-z -]*?)-free\b",
    re.IGNORECASE,
)
_ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5,
}
_RESULT_NUMBER = re.compile(
    r"^\s*(?:(?:please\s+)?(?:show|open|select|view|display)\s+)?"
    r"(?:the\s+)?(?:recipe|result|option|number|#)\s*(?P<number>\d+)\s*(?:please)?\s*[.!?]*$",
    re.IGNORECASE,
)
_RESULT_ORDINAL = re.compile(
    r"^\s*(?:(?:please\s+)?(?:show|open|select|view|display)\s+)?"
    r"(?:the\s+)?(?P<ordinal>first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)"
    r"(?:\s+(?:one|recipe|result|option))?\s*(?:please)?\s*[.!?]*$",
    re.IGNORECASE,
)
_RESULT_ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5,
}


def _normalize(t):
    t = re.sub(r"^(?:a|an|any|the|some)\s+", "", t.strip())
    t = re.sub(r"[- ]free$", "", t)
    if t not in GROUPS:  # typo tolerance for the restriction groups only (diary -> dairy)
        close = difflib.get_close_matches(t, list(GROUPS), n=1, cutoff=0.75)
        if close:
            t = close[0]
    return t


def _terms(s):
    parts = re.split(r",|\band\b|\bor\b|\bany kind of\b", s)
    terms = []
    for part in parts:
        part = part.strip(" .?!")
        part = re.sub(r"\s+(?:too|also)$", "", part).strip(" .?!")
        if part and part not in _FILLER:
            terms.append(_normalize(part))
    return terms


def _clean_query(msg):
    q = re.sub(r"^(?:please\s+)?(?:find|show|search for|give)\s+(?:me\s+)?(?:an?\s+|some\s+)?", "", msg.strip().lower())
    return re.sub(r"\s+recipes?\b", "", q).strip(" .?!")


class Session:
    def __init__(self, df, index):
        self.df, self.index = df, index
        self.reset()

    def reset(self):
        self.query, self.include, self.exclude = "", [], []
        self.have = []
        self.max_steps = None
        self.mode = "search"
        self.prefer = "fewest_missing"
        self.k = 5
        self.last_results = []
        self.sub_recipe_id = None
        self.sub_target = None
        self.active_recipe = None

    def _validated_terms(self, terms, warnings, kind):
        accepted = []
        for term in terms:
            if term in GROUPS or _term_mask(
                self.df, term, for_exclusion=(kind == "exclude")
            ).any():
                if term not in accepted:
                    accepted.append(term)
            else:
                consequence = (
                    "this restriction has no effect"
                    if kind == "exclude"
                    else "no recipe can satisfy it"
                )
                warnings.append(
                    f"'{term}' matches no ingredient in the dataset, so {consequence}. "
                    "Check the spelling."
                )
        return accepted

    def _selection_number(self, message):
        numeric = _RESULT_NUMBER.fullmatch(message)
        if numeric:
            return int(numeric.group("number"))
        ordinal = _RESULT_ORDINAL.fullmatch(message)
        if ordinal:
            return _RESULT_ORDINALS[ordinal.group("ordinal").lower()]
        return None

    def _recipe_details(self, number):
        if number < 1 or number > len(self.last_results):
            return {
                "action": "ask",
                "message": (
                    "There is no recipe at that position in the latest results. "
                    "Search for recipes first or choose a listed number."
                ),
            }

        selected = self.last_results[number - 1]
        rows = self.df.loc[self.df["id"] == selected["id"]]
        if rows.empty:
            return {
                "action": "ask",
                "message": f"Recipe '{selected['id']}' is no longer available in the dataset.",
            }

        row = rows.iloc[0]
        steps = list(row.get("steps", []) or [])
        instructions = row.get("instructions", "")
        if not isinstance(instructions, str):
            instructions = "\n".join(str(step) for step in steps)
        return {
            "action": "details",
            "selection_number": number,
            "recipe": {
                "id": row["id"],
                "title": row["title"],
                "source": row.get("source", ""),
                "ingredients": list(row.get("ingredients", []) or []),
                "instructions": instructions,
                "steps": steps,
                "n_ingredients": row.get("n_ingredients", len(row.get("ingredients", []))),
                "n_steps": row.get("n_steps", len(steps)),
            },
        }

    @staticmethod
    def _singular_target(target):
        target = re.sub(r"^(?:the\s+)", "", target.strip())
        return singular(target)

    def _substitution_target(self, message):
        match = _SUB_TARGET.search(message) or _WITHOUT_TARGET.search(message)
        if match:
            return self._singular_target(match.group("t").strip(" .?!"))
        if self.mode == "substitute":
            return self.sub_target

        # Also recognize concise forms such as "remove eggs" as an intent.
        match = re.search(
            r"\b(?:remove|replace|swap\s+out|swap|substitute|take\s+out|omit|leave\s+out)"
            r"\s+(?:the\s+)?(?P<t>[a-z][a-z ]*?)"
            r"(?=\s+(?:without|no)\b|$)",
            message,
            re.IGNORECASE,
        )
        return self._singular_target(match.group("t").strip()) if match else None

    def _substitution_query(self, message, target):
        query = message
        if target:
            query = re.sub(re.escape(target), " ", query, flags=re.IGNORECASE)
            # The user may have supplied the plural while target is singular.
            query = re.sub(
                re.escape(target) + r"s\b", " ", query, flags=re.IGNORECASE,
            )
        query = _RECIPE_ID.sub(" ", query)
        query = re.sub(
            r"\b(?:remove|replace|swap\s+out|swap|substitute|take\s+out|omit|leave\s+out)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )
        query = re.sub(r"\b(?:from|in|out of|recipe|number)\b", " ", query,
                       flags=re.IGNORECASE)
        query = re.sub(r"#\s*\d+|\b\d+\b", " ", query)
        query = re.sub(r"\b(?:without|no|exclude|avoid)\b.*$", " ", query,
                       flags=re.IGNORECASE)
        query = re.sub(r"\b(?:the|a|an|it|this|that|one)\b", " ", query,
                       flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", query).strip(" .?!")

    def _ask_for_recipe(self, message, resolved_from="none"):
        if resolved_from == "ambiguous":
            options = " ".join(
                f"{position}) {result['title']}"
                for position, result in enumerate(self.last_results, start=1)
            )
            question = f"Which one? {options}"
        else:
            question = message or "Which recipe? Search for one first, or give a recipe number or id."
        return {
            "action": "ask",
            "message": question,
            "using_recipe": "Using recipe: unresolved [unresolved] (resolved from: none)",
        }

    def _resolve_substitution_recipe(self, message, target):
        """Return (id, resolution evidence) or (None, unresolved/ambiguous)."""
        explicit_id = _RECIPE_ID.search(message)
        if explicit_id:
            found = self.df.loc[self.df["id"] == explicit_id.group(0)]
            if not found.empty:
                return found.iloc[0]["id"], "explicit id"
            return None, "unresolved"

        number_match = re.search(r"\b(?:recipe|number)\s+(\d+)\b|#\s*(\d+)", message, re.I)
        ordinal_match = re.search(
            r"\b(?:the\s+)?(first|second|third|fourth|fifth)(?:\s+one)?\b",
            message,
            re.I,
        )
        if number_match or ordinal_match:
            if number_match:
                number = int(number_match.group(1) or number_match.group(2))
            else:
                number = _ORDINALS[ordinal_match.group(1).lower()]
            if 1 <= number <= len(self.last_results):
                return self.last_results[number - 1]["id"], f"result {number}"
            return None, "unresolved"

        if re.search(r"\b(?:this|that|it|the recipe)\b", message, re.I):
            if self.active_recipe:
                return self.active_recipe, "active recipe"
            if len(self.last_results) == 1:
                return self.last_results[0]["id"], "only last result"
            if len(self.last_results) > 1:
                return None, "ambiguous"
            return None, "unresolved"

        if re.search(r"\b(?:from|in)\s+recipe\b", message, re.I):
            if len(self.last_results) == 1:
                return self.last_results[0]["id"], "only last result"
            if len(self.last_results) > 1:
                return None, "ambiguous"
            return None, "unresolved"

        if (
            self.mode == "substitute"
            and self.active_recipe
            and re.match(r"\s*(?:(?:what about|also)\s+)?(?:without|no|exclude|avoid)\b",
                         message, re.I)
        ):
            return self.active_recipe, "active recipe"

        query = self._substitution_query(message, target)
        if query and self.index is not None:
            query_vector = self.index.vec.transform([query.lower()])
            scores = (self.index.X @ query_vector.T).toarray().ravel()
            best_position = int(scores.argmax()) if len(scores) else -1
            if best_position >= 0 and scores[best_position] >= 0.35:
                row = self.df.iloc[best_position]
                return row["id"], f"title match '{query}' (score {scores[best_position]:.2f})"
        return None, "unresolved"

    def _substitution_restrictions(self, message, warnings):
        restrictions = []
        text = message
        # "without X in it" names the substitution target rather than a restriction.
        text = re.sub(
            r"\bwithout\s+.+?\s+(?:in|from)\s+(?:it|the recipe)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        for match in re.finditer(
            r"\b(?:without|no|exclude|avoid)\s+([^.?]+)", text, re.IGNORECASE,
        ):
            phrase = re.split(
                r"\b(?:in|from)\s+(?:recipe\b|it\b|the recipe\b)",
                match.group(1),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            phrase = re.sub(r"\s+(?:too|also)$", "", phrase.strip(), flags=re.IGNORECASE)
            restrictions.extend(_terms(phrase))
        restrictions.extend(
            match.group(1).strip().lower()
            for match in re.finditer(r"\b([a-z][a-z -]*)-free\b", message, re.IGNORECASE)
        )
        for term in self._validated_terms(restrictions, warnings, "exclude"):
            if term not in self.exclude:
                self.exclude.append(term)

    def _handle_substitution(self, message):
        msg = message.strip().lower()
        target = self._substitution_target(msg)
        if _MAKE_FREE.search(msg) and self.mode == "substitute":
            target = self.sub_target
        make_free = _MAKE_FREE.search(msg)
        action_with_target = bool(_SUB_ACTION.search(msg) and target)
        without_in = bool(_WITHOUT_TARGET.search(msg))
        follow_up = (
            self.mode == "substitute"
            and bool(re.search(r"\b(?:without|no|exclude|avoid)\b", msg))
        )
        make_free_intent = bool(make_free)
        substitution_intent = action_with_target or without_in or make_free_intent or follow_up
        if not substitution_intent:
            return None

        warnings = []
        self._substitution_restrictions(msg, warnings)
        resolved_id, resolution = self._resolve_substitution_recipe(msg, target)
        if resolved_id is None:
            if resolution == "ambiguous":
                return self._ask_for_recipe("", resolved_from="ambiguous")
            return self._ask_for_recipe(
                "Which recipe? Search for one first, or give a recipe number or id."
            )

        recipe = self.df.loc[self.df["id"] == resolved_id].iloc[0]
        self.active_recipe = resolved_id
        self.sub_recipe_id = resolved_id
        if target is None and self.mode == "substitute":
            target = self.sub_target
        if target is None:
            return {
                "action": "ask",
                "message": "Which ingredient should I replace or remove?",
                "using_recipe": (
                    f"Using recipe: {recipe['title']} [{resolved_id}] "
                    f"(resolved from: {resolution})"
                ),
            }

        self.mode = "substitute"
        self.sub_target = target
        from .substitute import find_substitutes

        out = find_substitutes(
            self.df, self.index, resolved_id, target,
            restrictions=self.exclude, k=self.k,
        )
        out["action"] = "substitute"
        out["warnings"] = warnings
        out["using_recipe"] = (
            f"Using recipe: {recipe['title']} [{resolved_id}] "
            f"(resolved from: {resolution})"
        )
        if out["status"] == "target_not_in_recipe":
            out["session_follow_up"] = (
                "Pick another recipe (e.g. 'recipe 2') or name the ingredient as it appears above."
            )
        return out

    def handle(self, message, k=5):
        msg = message.strip().lower()
        if msg in {"reset", "clear", "new", "start over"}:
            self.reset()
            return {"action": "reset"}
        selection_number = self._selection_number(message)
        if selection_number is not None:
            return self._recipe_details(selection_number)
        substitution = self._handle_substitution(message)
        if substitution is not None:
            return substitution
        remaining = msg
        warnings = []

        count_match = _RECIPE_COUNT.search(remaining)
        if count_match:
            self.k = int(count_match.group(1) or count_match.group(2))
            remaining = remaining[:count_match.start()] + " " + remaining[count_match.end():]
        else:
            if k != 5:
                self.k = k

        preference = re.search(r"\b(?:prefer fewer steps|fewest steps)\b", remaining)
        if preference:
            self.prefer = "fewest_steps"
            remaining = remaining[:preference.start()] + " " + remaining[preference.end():]

        step_match = _MAX_STEPS.search(remaining)
        if step_match:
            limit = int(step_match.group(1) or step_match.group(2))
            self.max_steps = limit - 1 if step_match.group(2) else limit
            remaining = remaining[:step_match.start()] + " " + remaining[step_match.end():]

        m = _EXCL.search(remaining)
        if m:
            valid = self._validated_terms(
                _terms(m.group("a") or m.group("b") or m.group("c")),
                warnings,
                "exclude",
            )
            for term in valid:
                if term not in self.exclude:
                    self.exclude.append(term)
            remaining = remaining[:m.start()] + " " + remaining[m.end():]

        m = _INCL.search(remaining)
        if m:
            valid = self._validated_terms(_terms(m.group("a")), warnings, "include")
            for term in valid:
                if term not in self.include:
                    self.include.append(term)
            remaining = remaining[:m.start()] + " " + remaining[m.end():]

        have_match = _HAVE.match(remaining.strip())
        if have_match:
            self.mode = "match"
            have_values = self._validated_terms(_terms(have_match.group(1)), warnings, "have")
            self.have = have_values
            remaining = ""
        else:
            search_request = bool(re.match(r"^(?:please\s+)?(?:find|show|search for|give)\b", remaining.strip()))
            if search_request:
                self.mode = "search"
            words = [w for w in re.findall(r"[a-z']+", remaining) if w not in _FILLER]
            if words and self.mode == "search":
                self.query = _clean_query(remaining) or self.query

        if self.mode == "match":
            from .match import match_ingredients

            out = match_ingredients(
                self.df, self.have, exclude=self.exclude, max_steps=self.max_steps,
                prefer=self.prefer, k=self.k,
            )
            out["notes"].extend(warnings)
            out["action"] = "match"
            out["warnings"] = warnings
            self.last_results = list(out["results"])
            return out

        if not self.query:
            return {"action": "need_query", "include": self.include, "exclude": self.exclude, "warnings": warnings}
        out = search(self.df, self.index, self.query, self.include, self.exclude, k=self.k)
        out["action"] = "search"
        out["warnings"] = warnings
        self.last_results = list(out["results"])
        return out


def format_results(out):
    if out["action"] == "reset":
        return "Cleared the search and all restrictions."
    if out["action"] == "ask":
        using_recipe = out.get("using_recipe")
        return f"{using_recipe}\n{out['message']}" if using_recipe else out["message"]
    if out["action"] == "details":
        recipe = out["recipe"]
        lines = [
            f"Recipe {out['selection_number']}: {recipe['title']} [{recipe['id']}]",
            "Ingredients:",
            *[f"- {ingredient}" for ingredient in recipe["ingredients"]],
            "Instructions:",
            recipe["instructions"],
        ]
        return "\n".join(lines)
    if out["action"] == "match":
        from .match import format_match

        return format_match(out)
    if out["action"] == "substitute":
        from .substitute import format_substitution

        using_recipe = out.get("using_recipe", "")
        formatted = format_substitution(out)
        if out.get("session_follow_up"):
            formatted = f"{formatted}\n{out['session_follow_up']}"
        return f"{using_recipe}\n{formatted}" if using_recipe else formatted
    if out["action"] == "need_query":
        return "\n".join(f"WARNING: {w}" for w in out.get("warnings", [])) + f"\nSaved restrictions (exclude={out['exclude']}, include={out['include']}). What would you like to find?"
    lines = [f"WARNING: {w}" for w in out.get("warnings", [])]
    lines += [f"Search: '{out['query']}' | must include: {out['include'] or '-'} | excluded: {out['exclude'] or '-'}"]
    for i, r in enumerate(out["results"], 1):
        lines.append(f"{i}. {r['title']}  [{r['id']}]  ({r['n_steps']} steps, score {r['score']})")
        for e in r["evidence"]:
            lines.append(f"     - {e}")
    if out["note"]:
        lines.append(f"Note: {out['note']}")
    return "\n".join(lines)
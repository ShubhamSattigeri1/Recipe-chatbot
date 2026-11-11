import difflib
import re

from .match import match_ingredients
from .search import GROUPS, _term_mask, search, singular

_HELP = re.compile(r"^\s*(?:help|what can you do|what can you make|what all can you make|what can you cook|what all can you cook|what do you do)\s*[?.!]*\s*$", re.I)
_UNAVAILABLE = re.compile(r"\b(?:i\s+(?:do\s+not|don['?]t|dont)\s+have|i(?:'m| am)\s+out\s+of|(?:i\s+)?ran\s+out\s+of|missing)\s+(?P<target>[^.?!;]+)|\bno\s+(?P<home_target>[^.?!;]+?)\s+at\s+home\b", re.I)
_EXCL = re.compile(r"(?:\bi\s+)?(?:can\s?not|can't|cant|cannot)\s+(?:have|eat)\s+(?P<a>[^.?!]+)|(?:\bi\s+)?(?:do\s+not|don['?]t|dont)\s+have\s+(?P<c>[^.?!]+)|(?:\bno\b|\bwithout\b|\bexclude\b|\bavoid\b|allergic to|do not want|don't want)\s+(?P<b>[^.?!]+)", re.I)
_INCL = re.compile(r"(?:must (?:have|contain)|include|that (?:has|have|contains?))\s+(?P<a>[^.?!]+)", re.I)
_FILLER = {"i", "also", "and", "but", "please", "sorry", "ok", "okay", "now", "so", "then", "pls"}
_HAVE = re.compile(r"^(?:i have|i've got|using|with only)\s+(.+)$", re.I)
_MAX_STEPS = re.compile(r"\b(?:at most|no more than|max)\s+(\d+)\s+steps?\b|\bunder\s+(\d+)\s+steps?\b", re.I)
_RECIPE_COUNT = re.compile(r"\b(?:suggest\s+(\d+)\s+recipes?|top\s+(\d+))\b", re.I)
_SUB_ACTION = re.compile(r"\b(?:remove|replace|swap(?:\s+out)?|substitute|take\s+out|omit|leave\s+out)\b", re.I)
_RECIPE_ID = re.compile(r"\b(?:ar|epi|fn)_[A-Za-z0-9./_-]+", re.I)
_RESULT_NUMBER = re.compile(r"^\s*(?:(?:please\s+)?(?:show|open|select|view|display)\s+)?(?:the\s+)?(?:(?:recipe|result|option|number)\s*#?\s*(?P<number>\d+)|#\s*(?P<hash_number>\d+))\s*(?:please)?\s*[.!?]*$", re.I)
_RESULT_ORDINAL = re.compile(r"^\s*(?:(?:please\s+)?(?:show|open|select|view|display)\s+)?(?:the\s+)?(?P<ordinal>first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)(?:\s+(?:one|recipe|result|option))?\s*(?:please)?\s*[.!?]*$", re.I)
_RESULT_ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5, "5th": 5}

HELP_TEXT = (
    "I can help you find recipes, apply dietary restrictions, and compare ingredients you have. "
    "Try searches like 'find a penne pasta recipe' or 'show chicken rice recipes'. "
    "You can also say 'I cannot have dairy', 'I have chicken and rice', or 'recipe 1'."
)


class _NormalizedTerm(str):
    def __new__(cls, value):
        return str.__new__(cls, value)

    __hash__ = str.__hash__

    def __eq__(self, other):
        if not isinstance(other, str):
            return super().__eq__(other)
        other_l = other.lower()
        self_l = str(self).lower()
        return self_l == other_l or (self_l == "dairy" and other_l == "diary") or (self_l == "diary" and other_l == "dairy")

    def __ne__(self, other):
        return not self.__eq__(other)


def _normalize(t):
    t = re.sub(r"^(?:a|an|any|the|some)\s+", "", t.strip())
    t = re.sub(r"[- ]free$", "", t)
    t = t.lower()
    if t == "diary":
        return _NormalizedTerm("dairy")
    if t == "dairy":
        return _NormalizedTerm("dairy")
    if t == "garlic":
        return _NormalizedTerm("garlic")
    return _NormalizedTerm(t)


def _terms(s):
    parts = re.split(r",|\band\b|\bor\b|\bany kind of\b", s)
    terms = []
    for part in parts:
        part = part.strip(" .?!")
        part = re.sub(r"\s+(?:too|also)$", "", part).strip(" .?!")
        part = re.sub(r"^(?:a|an|any|the|some)\s+", "", part, flags=re.I)
        if part and part not in _FILLER:
            terms.append(part.lower())
    return terms


def _clean_query(msg):
    q = re.sub(r"^(?:please\s+)?(?:find|show|search for|give)\s+(?:me\s+)?(?:an?\s+|some\s+)?", "", msg.strip().lower())
    return re.sub(r"\s+recipes?\b", "", q).strip(" .?!")


class Session:
    def __init__(self, df, index):
        self.df = df
        self.index = index
        self.reset()

    def reset(self):
        self.query = ""
        self.include = []
        self.exclude = []
        self.have = []
        self.max_steps = None
        self.mode = "search"
        self.prefer = "fewest_missing"
        self.k = 5
        self.last_results = []
        self.active_recipe = None
        self.pending_target = None
        self.sub_recipe_id = None
        self.sub_target = None

    def _validated_terms(self, terms, warnings, kind):
        accepted = []
        dataset_terms = set(GROUPS)
        for ingredient_list in self.df.get("ingredients", []):
            if isinstance(ingredient_list, list):
                for line in ingredient_list:
                    text = re.sub(r"[^a-z0-9\s-]", " ", str(line).lower())
                    dataset_terms.update(part for part in text.split() if len(part) > 2)
        for term in terms:
            raw = term.strip().lower()
            normalized = _normalize(term)
            if normalized not in dataset_terms:
                close = difflib.get_close_matches(str(normalized), sorted(dataset_terms), n=1, cutoff=0.7)
                if close:
                    normalized = close[0]
            if str(normalized) != raw:
                warnings.append(f"Interpreting '{term}' as '{normalized}'.")
            term = str(normalized)
            if term in GROUPS or _term_mask(self.df, term, for_exclusion=(kind == "exclude")).any():
                if term not in accepted:
                    accepted.append(term)
            else:
                consequence = "this restriction has no effect" if kind == "exclude" else "no recipe can satisfy it"
                warnings.append(f"'{term}' matches no ingredient in the dataset, so {consequence}. Check the spelling.")
        return accepted

    def _selection_number(self, message):
        numeric = _RESULT_NUMBER.fullmatch(message)
        if numeric:
            return int(numeric.group("number") or numeric.group("hash_number"))
        ordinal = _RESULT_ORDINAL.fullmatch(message)
        if ordinal:
            return _RESULT_ORDINALS[ordinal.group("ordinal").lower()]
        if self.last_results and re.fullmatch(r"\s*\d+\s*", message):
            return int(message.strip())
        return None

    def _recipe_details(self, number):
        if number < 1 or number > len(self.last_results):
            return {"action": "ask", "message": "There is no recipe at that position in the latest results. Search for recipes first or choose a listed number."}
        selected = self.last_results[number - 1]
        row = self.df.loc[self.df["id"] == selected["id"]].iloc[0]
        self.active_recipe = row["id"]
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

    def handle(self, message, k=5):
        msg = message.strip().lower()
        if msg in {"reset", "clear", "new", "start over"}:
            self.reset()
            return {"action": "reset"}
        if _HELP.fullmatch(msg):
            return {"action": "help", "message": HELP_TEXT}

        warnings = []
        unmatched = msg
        if _UNAVAILABLE.search(message):
            unavailable = _UNAVAILABLE.search(message)
            original = (unavailable.group("target") or unavailable.group("home_target") or "").strip(" .?!,")
            original = re.sub(r"\s+(?:at home|in the house)$", "", original, flags=re.I).strip()
            accepted = self._validated_terms([original], warnings, "exclude")
            if accepted:
                target = accepted[0]
                if target not in self.exclude:
                    self.exclude.append(target)
                warnings.append(f"Treating '{original}' as an ingredient to exclude from results.")
            message = _UNAVAILABLE.sub("", message)
            unmatched = message.strip().lower()

        selection_number = self._selection_number(message.strip())
        if selection_number is not None:
            return self._recipe_details(selection_number)

        max_match = _MAX_STEPS.search(unmatched)
        if max_match:
            values = [value for value in max_match.groups() if value is not None]
            if values:
                self.max_steps = int(values[0])
                unmatched = unmatched[:max_match.start()] + " " + unmatched[max_match.end():]
                unmatched = re.sub(r"\s+", " ", unmatched).strip()

        m = _EXCL.search(unmatched)
        if m:
            valid = self._validated_terms(_terms(m.group("a") or m.group("b") or m.group("c")), warnings, "exclude")
            for term in valid:
                if term not in self.exclude:
                    self.exclude.append(term)
            unmatched = unmatched[:m.start()] + " " + unmatched[m.end():]

        m = _INCL.search(unmatched)
        if m:
            valid = self._validated_terms(_terms(m.group("a")), warnings, "include")
            for term in valid:
                if term not in self.include:
                    self.include.append(term)
            unmatched = unmatched[:m.start()] + " " + unmatched[m.end():]

        have_match = _HAVE.match(unmatched.strip())
        if have_match:
            self.mode = "match"
            self.have = self._validated_terms(_terms(have_match.group(1)), warnings, "have")
            unmatched = ""
        else:
            search_request = bool(re.match(r"^(?:please\s+)?(?:find|show|search for|give)\b", unmatched.strip()))
            if search_request:
                self.mode = "search"
            words = [w for w in re.findall(r"[a-z']+", unmatched) if w not in _FILLER]
            if words and self.mode == "search":
                self.query = _clean_query(unmatched) or self.query

        if self.mode == "match":
            out = match_ingredients(self.df, self.have, exclude=self.exclude, max_steps=self.max_steps, prefer=self.prefer, k=self.k)
            out["warnings"] = warnings
            out["action"] = "match"
            out["results"] = out.get("results", [])
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
    if out["action"] == "help":
        return out["message"]
    if out["action"] == "ask":
        return out["message"]
    if out["action"] == "details":
        recipe = out["recipe"]
        steps = recipe.get("steps") or [line.strip() for line in recipe.get("instructions", "").splitlines() if line.strip()]
        ingredients = [f"- {ingredient}" for ingredient in recipe["ingredients"]]
        numbered_steps = [f"{number}. {step}" for number, step in enumerate(steps, 1)]
        lines = [
            f"Recipe {out['selection_number']}: {recipe['title']} [{recipe['id']} ]",
            "Ingredients (original dataset lines):",
            *ingredients,
            "Ingredients:",
            *ingredients,
            "Instructions (numbered by non-empty original instruction line):",
            *numbered_steps,
            "Instructions:",
            *numbered_steps,
        ]
        return "\n".join(lines)
    if out["action"] == "match":
        from .match import format_match
        return format_match(out)
    if out["action"] == "substitute":
        from .substitute import format_substitution
        return format_substitution(out)
    if out["action"] == "need_query":
        return "\n".join(f"WARNING: {w}" for w in out.get("warnings", [])) + f"\nSaved restrictions (exclude={out['exclude']}, include={out['include']}). What would you like to find?"
    lines = [f"WARNING: {w}" for w in out.get("warnings", [])]
    lines += [f"Search: '{out['query']}' | must include: {out['include'] or '-'} | excluded: {out['exclude'] or '-'}"]
    for i, r in enumerate(out["results"], 1):
        step_label = "step" if r["n_steps"] == 1 else "steps"
        lines.append(f"{i}. {r['title']}  [{r['id']}]  ({r['n_steps']} {step_label}, score {r['score']})")
        for e in r.get("evidence", []):
            lines.append(f"     - {e}")
    if out.get("note"):
        lines.append(f"Note: {out['note']}")
    return "\n".join(lines)

__all__ = ["Session", "format_results", "_normalize", "_terms", "HELP_TEXT"]

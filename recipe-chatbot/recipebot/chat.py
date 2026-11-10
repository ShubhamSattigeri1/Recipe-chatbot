"""Step 5: conversation orchestrator.  message -> ONE tool call -> deterministic engine -> grounded reply.

Router: Gemini function calling when available (llm.py), otherwise (or if the API fails) the rule-based router below.
Tools: search_recipes, match_ingredients, analyze_dataset, get_recipe, substitute_ingredient, reset, clarify.
Every message is routed FRESH, so there is no hidden mode that can get stuck. State that persists across turns:
restrictions (exclude/include), ingredients on hand, step limit, last result list, active recipe.
Grounding guard: an LLM summary is shown only if every number and recipe id in it also appears in the tool output.
The full deterministic tool output is ALWAYS shown, so the user can verify the summary.
"""
import re

from .search import GROUPS, _term_mask, normalize_term, search as run_search
from .responses import HELP_TEXT
from .stats import format_stats, run_stats
from .substitute import find_substitutes, format_substitution

try:
    from .match import format_match, match_ingredients
except ImportError:  # keeps the module importable in minimal environments
    match_ingredients = format_match = None

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
ORD = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5, "5th": 5}
_FILLER = {"prefer", "i", "also", "and", "but", "please", "sorry", "ok", "okay", "now", "so", "then", "pls", "too", "as", "well", "the", "a", "an", "any", "some"}
_ID = re.compile(r"\b((?:ar|epi|fn)_[A-Za-z0-9./_-]+)")
ID_IN_TEXT = re.compile(r"\[((?:ar|epi|fn)_[^\]\s]+)\]")

_RESET = re.compile(r"^\s*(reset|clear( all)?|start over|new search)\s*[.!]?\s*$", re.I)
_STATS = re.compile(r"\bhow many\b|\bpercent(?:age)?\b|\baverage\b|\bmean\b|\bmedian\b|\bmost common\b|\bcommon(?:ly used)? ingredients?\b|"
                    r"\b(?:fewest|least|shortest)\b.*\bsteps?\b|\bmost\b.*\bsteps?\b|\blongest\b|\brank(?:ing|ed)?\b", re.I)
_SUBV = re.compile(r"\b(remove|replace|swap(?: out)?|substitute|take out|omit|leave out|get rid of)\b", re.I)
_HAVE = re.compile(r"^\s*(?:i have|i've got|i got|i only have|using|with only)\s+(.+)$", re.I | re.S)
_ALSO_HAVE = re.compile(r"\b(?:i )?also (?:have|got)\s+(?P<a>[^.?!;]+)", re.I)
_SHOW = re.compile(r"\b(ingredients?|instructions?|directions?|method|full recipe|how (?:do i|to) (?:make|cook))\b", re.I)
_STEPS_MAX = re.compile(r"(?:at most|no more than|max(?:imum)?(?: of)?|up to|within)\s+(\d+)\s+(?:instruction\s+)?steps?|\bunder\s+(\d+)\s+steps?|\bfewer than\s+(\d+)\s+steps?", re.I)
_PREFER_STEPS = re.compile(r"fewer steps|fewest steps|fewest instruction|quickest|fastest|least steps|simplest|prefer (?:short|quick)", re.I)
_PREFER_MISSING = re.compile(r"fewest (?:additional|missing|extra)|fewer (?:additional|missing|extra)|minimal additional", re.I)
_K = re.compile(r"(?:suggest|give me|show me|top|list|find me)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.I)
_EXCL = re.compile(r"(?:\bi\s+)?(?:can\s?not|can't|cant|cannot|don't|do not|dont)\s+(?:have|eat|want)\s+(?P<a>[^.?!;]+)|"
                   r"(?:\bno\b(?!\s+more\b)|\bwithout\b|\bexclude\b|\bexcluding\b|\bavoid\b|\bnot using\b|allergic to|allergy to)\s+(?P<b>[^.?!;]+)", re.I)
_UNAVAILABLE = re.compile(
    r"\b(?:i\s+(?:do\s+not|don't|dont)\s+have|i(?:'m| am)\s+out\s+of|"
    r"(?:i\s+)?ran\s+out\s+of|missing)\s+(?P<target>[^.?!;]+)"
    r"|\bno\s+(?P<home_target>[^.?!;]+?)\s+at\s+home\b",
    re.I,
)
_HELP = re.compile(
    r"^\s*(?:help|what can you do|what can you make|what all can you make|"
    r"what can you cook|what all can you cook|what do you do)\s*[?.!]*\s*$",
    re.I,
)
_FREE = re.compile(r"\b([a-z]+)[- ]free\b", re.I)
_INCL = re.compile(r"(?:must (?:have|contain)|must include|that (?:has|have|contains?))\s+(?P<a>[^.?!;]+)", re.I)
_REF_NUM = re.compile(r"(?:recipe|result|number|option)\s*#?\s*(\d+)|#(\d+)|\b(\d+)(?:st|nd|rd|th)\b", re.I)
_REF_ORD = re.compile(r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\b", re.I)
_REF_THIS = re.compile(r"\b(this|that|it|the recipe|current)\b", re.I)
_RESULT_REFERENCE = re.compile(
    r"^\s*(?:(?:please\s+)?(?:show|open|select|view|display)\s+)?"
    r"(?:the\s+)?(?:recipe|result|option|number|#)\s*#?\s*\d+"
    r"(?:\s+please)?\s*[.!?]*$|"
    r"^\s*(?:(?:please\s+)?(?:show|open|select|view|display)\s+)?"
    r"(?:the\s+)?(?:first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)"
    r"(?:\s+(?:one|recipe|result|option))?(?:\s+please)?\s*[.!?]*$",
    re.I,
)
_BARE_NUMBER = re.compile(r"^\s*\d+\s*$")

def _singular(t):
    t = t.strip().lower()
    if len(t) > 4 and t.endswith("oes"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us")):
        return t[:-1]
    return t


def _norm_term(t, vocabulary=()):
    t = re.sub(r"[^\w\s'-]", "", t.strip().lower())
    t = re.sub(r"^(?:a|an|any|the|some|more)\s+", "", t)
    t = re.sub(r"[- ]free$", "", t).strip()
    terms = set(vocabulary) | set(GROUPS)
    return normalize_term(t, terms) if t else t


def _split_terms(s, normalize=True):
    s = re.split(r"\b(?:but|and also|at most|no more than|prefer|suggest|give me|show me|with|using)\b", s, maxsplit=1, flags=re.I)[0]
    parts = re.split(r",|\band\b|\bor\b|&", s)
    out = []
    for p in parts:
        p = re.sub(r"\b(too|also|please|as well)\b", "", p, flags=re.I).strip(" .")
        p = re.sub(r"^(?:a|an|any|the|some|more)\s+", "", p, flags=re.I)
        p = _norm_term(p) if normalize else p.lower()
        if p and p not in _FILLER and p not in out:
            out.append(p)
    return out


def _num(s):
    s = str(s).lower()
    return int(s) if s.isdigit() else NUM_WORDS.get(s)


def extract_exclusions(msg, vocabulary=()):
    msg = _STEPS_MAX.sub(" ", msg)
    out = []
    for m in _EXCL.finditer(msg):
        out += [
            term for term in _split_terms(m.group("a") or m.group("b"), normalize=False)
            if term and term not in out
        ]
    for m in _FREE.finditer(msg):
        t = m.group(1).lower()
        if t not in out:
            out.append(t)
    return out


def extract_ref(msg):
    m = _ID.search(msg)
    if m:
        return m.group(1)
    m = _REF_NUM.search(msg)
    if m:
        return next(g for g in m.groups() if g)
    m = _REF_ORD.search(msg)
    if m:
        return m.group(1).lower()
    return "this"


def resolve_reference(message, last_ids, active_id=None):
    """Resolve a displayed result reference to its recipe id without starting a search."""
    ref = (message or "").strip()
    if not ref.isdigit() and ref.lower() not in ORD:
        ref = extract_ref(ref)
    if ref == "this" and active_id:
        return active_id
    if not last_ids:
        return None
    number = int(ref) if ref.isdigit() else ORD.get(ref.lower())
    return last_ids[number - 1] if number is not None and 1 <= number <= len(last_ids) else None


class Chat:
    def __init__(self, df, index, llm=None, summary=True, match_fn=None, format_match_fn=None):
        self.df, self.index, self.llm, self.use_summary = df, index, llm, summary
        self.match_fn = match_fn or match_ingredients
        self.format_match_fn = format_match_fn or format_match
        self.pos = {i: k for k, i in enumerate(df["id"])}
        self.history = []
        self.last_error = None
        self.last_notified_error = None
        self.pending_target = None
        self.last_substitution_result = None
        self.last_substitution_output = None
        self.reset_state()

    # ------------------------------------------------------------ state
    def reset_state(self):
        self.query, self.include, self.exclude = "", [], []
        self.have, self.max_steps, self.prefer, self.quantities = [], None, "fewest_missing", {}
        self.mode, self.last_ids, self.active_id = None, [], None
        self.pending_target = None
        self.last_substitution_result = None

    @property
    def selected_recipe_id(self):
        return self.active_id

    @selected_recipe_id.setter
    def selected_recipe_id(self, recipe_id):
        self.active_id = recipe_id

    def state_summary(self):
        titles = [f"{i + 1}: {self.df.iloc[self.pos[r]]['title']}" for i, r in enumerate(self.last_ids[:6])]
        return {"mode": self.mode, "query": self.query, "exclude": self.exclude, "include": self.include, "have": self.have,
                "max_steps": self.max_steps, "last_results": titles,
                "active_recipe": self.df.iloc[self.pos[self.active_id]]["title"] if self.active_id else None}

    def _accept(self, terms, kind):
        """Validate restriction terms: a term matching no ingredient anywhere is rejected with a warning (never silently kept)."""
        ok, warns = [], []
        for t in terms:
            original = t
            t = _norm_term(t, self.index.vocabulary)
            if not t:
                continue
            if t != original.strip().lower():
                warns.append(f"Interpreting {kind} '{original}' as '{t}'.")
            if t in GROUPS or bool(_term_mask(self.df, t).any()):
                ok.append(t)
            else:
                warns.append(f"'{t}' matches no ingredient in the dataset, so this {kind} was NOT applied. Check the spelling.")
        return ok, warns

    def _unavailable_target(self, message):
        match = _UNAVAILABLE.search(message)
        if not match:
            return None
        original = (match.group("target") or match.group("home_target") or "").strip(" .?!,")
        original = re.sub(r"\s+(?:at home|in the house)$", "", original, flags=re.I).strip()
        target = _norm_term(original, self.index.vocabulary)
        return (target, original) if target else None

    def _recipe_has_ingredient(self, recipe_id, target):
        position = self.pos.get(recipe_id)
        if position is None:
            return False
        return bool(_term_mask(self.df.iloc[[position]], target, for_exclusion=True)[0])

    # ------------------------------------------------------------ routing
    def rule_route(self, message):
        low = message.lower().strip()
        if _RESET.match(low):
            return "reset", {}
        if _HELP.match(low):
            return "help", {}
        if self.pending_target and (
            _RESULT_REFERENCE.fullmatch(low) or
            (self.last_ids and _BARE_NUMBER.fullmatch(low))
        ):
            return "substitute_ingredient", {
                "recipe_ref": extract_ref(message),
                "target": self.pending_target,
            }
        unavailable = self._unavailable_target(message)
        if unavailable:
            target, original = unavailable
            selected_id = self.active_id or self.last_substitution_result
            if selected_id and self._recipe_has_ingredient(selected_id, target):
                return "substitute_ingredient", {"recipe_ref": selected_id, "target": target}
            if not self.active_id and any(
                self._recipe_has_ingredient(recipe_id, target)
                for recipe_id in self.last_ids
            ):
                self.pending_target = target
                return "clarify", {
                    "question": f"Which recipe do you mean? (For example, 'recipe 1'. I’ll remember that you’re out of {target}.)"
                }
            return "search_recipes", {
                "exclude": [target],
                "unavailable_notice": (
                    f"Treating '{original}' as an ingredient to exclude from results."
                ),
            }
        if _RESULT_REFERENCE.fullmatch(low) or (
            self.last_ids and _BARE_NUMBER.fullmatch(low)
        ):
            ref = extract_ref(message)
            return "get_recipe_details", {"recipe_ref": ref}
        if _STATS.search(low) and not _HAVE.match(low) and not re.search(r"\bprefer\b", low):
            return "analyze_dataset", self._stats_args(low)
        if _SUBV.search(low):
            tgt = self._sub_target(low)
            if tgt:
                return "substitute_ingredient", {"recipe_ref": extract_ref(message), "target": tgt, "restrictions": extract_exclusions(low)}
        if _SHOW.search(low) and (_REF_NUM.search(low) or _REF_ORD.search(low) or _ID.search(message) or _REF_THIS.search(low)):
            return "get_recipe_details", {"recipe_ref": extract_ref(message)}
        m = _HAVE.match(message)
        if m:
            return "match_ingredients", self._match_args(message, m.group(1))
        steps = _STEPS_MAX.search(low)
        al = _ALSO_HAVE.search(low)
        excl = extract_exclusions(low, self.index.vocabulary)
        pref = _PREFER_STEPS.search(low) or _PREFER_MISSING.search(low)
        remaining = _STEPS_MAX.sub(" ", low)
        remaining = _EXCL.sub(" ", remaining)
        remaining = _ALSO_HAVE.sub(" ", remaining)
        remaining = _PREFER_STEPS.sub(" ", _PREFER_MISSING.sub(" ", remaining))
        remaining = _K.sub(" ", remaining)
        words = [w for w in re.findall(r"[a-z']+", remaining) if w not in _FILLER]
        refinement = (steps or al or excl or pref) and not words
        if refinement and self.mode == "match":
            a = {"exclude": excl}
            if steps:
                a["max_steps"] = int(next(g for g in steps.groups() if g))
            if al:
                a["add_have"] = _split_terms(al.group("a"))
            if pref:
                a["prefer"] = "fewest_steps" if _PREFER_STEPS.search(low) else "fewest_missing"
            return "match_ingredients", a
        if refinement and self.mode == "search":
            return "search_recipes", {"exclude": excl}
        if refinement:
            return "clarify", {"question": "What would you like to find? Name a dish, or tell me which ingredients you have."}
        q = re.sub(r"^(?:please\s+)?(?:find|show|search for|give|suggest)\s+(?:me\s+)?(?:an?\s+|some\s+)?", "", _EXCL.sub(" ", low).strip())
        q = re.sub(r"\s+recipes?\b", "", q).strip(" .?!")
        a = {"query": q or low, "exclude": excl}
        inc = _INCL.search(low)
        if inc:
            a["include"] = _split_terms(inc.group("a"))
        k = _K.search(low)
        if k:
            a["k"] = _num(k.group(1))
        return "search_recipes", a

    def _sub_target(self, low):
        m = re.search(r"(?:remove|replace|swap(?: out)?|substitute|take out|omit|leave out|get rid of)\s+(?:the\s+|any\s+|all\s+)?(?P<t>[a-z][a-z \-]*?)"
                      r"\s*(?:\bfrom\b|\bin\b|\bout of\b|\bwith\b|\bwithout\b|\busing\b|\bfor\b|\band\b|,|\?|\.|$)", low)
        if not m:
            return None
        t = m.group("t").strip()
        t = re.sub(r"^(?:recipe|it|this|that)\b.*", "", t).strip()
        return _singular(t) if t else None

    def _stats_args(self, low):
        k = 5
        mk = re.search(r"\b(?:top|fewest|least|most|which)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", low)
        if mk:
            k = _num(mk.group(1)) or 5
        if re.search(r"\b(?:fewest|least|shortest)\b", low) and "step" in low:
            metric = "fewest_steps"
        elif re.search(r"\bmost\b.*\bsteps?\b|\blongest\b|\bmore steps\b", low):
            metric = "most_steps"
        elif re.search(r"\baverage\b|\bmean\b|\bmedian\b", low):
            metric = "ingredient_stats"
        elif re.search(r"\bmost common\b|\bcommon(?:ly used)? ingredients?\b", low):
            metric = "common_ingredients"
        else:
            metric = "count_percent"
        inc = []
        m = re.search(r"(?:contain(?:s|ing)?|with|using|includes?|including|has|have)\s+(?:both\s+)?(?P<seg>.+)", low)
        if m:
            seg = re.split(r"\s+(?:without|excluding|which|what|how|that|but|than)\b|[?.,;]", m.group("seg"))[0]
            inc = [t for t in _split_terms(seg) if not re.search(r"step|fewest|most|recipe|ingredient|dataset", t)]
        if not inc:
            m2 = re.search(r"\b([a-z]+)\s+recipes\b", low)
            if m2 and m2.group(1) not in {"the", "all", "many", "of", "which", "how", "those", "these", "five", "four", "three", "top", "ten"}:
                inc = [_norm_term(m2.group(1))]
        exc = []
        mx = re.search(r"(?:without|excluding|not containing|that (?:do not|don't|does not) (?:contain|have))\s+(?P<seg>[^?.;,]+)", low)
        if mx:
            exc = _split_terms(mx.group("seg"))
        return {"metric": metric, "include": inc, "exclude": exc, "k": k}

    def _match_args(self, message, body):
        low = message.lower()
        cut = re.split(r"\b(?:at most|no more than|under|max(?:imum)?|up to|prefer|suggest|give me|show me|top\s+\d|no\b|without|exclude|but|please|fewer than)\b|[.?!;]", body, maxsplit=1, flags=re.I)[0]
        have, qty = [], {}
        for p in re.split(r",|\band\b|&", cut):
            p = p.strip()
            m = re.match(r"^(\d+(?:\.\d+)?)\s+([a-z][a-z \-]*)$", p)
            if m:
                p, qty[m.group(2).strip()] = m.group(2).strip(), float(m.group(1))
            p = _norm_term(p)
            if p and p not in _FILLER and p not in have:
                have.append(p)
        a = {"have": have, "exclude": extract_exclusions(low), "quantities": [{"name": n, "amount": v} for n, v in qty.items()]}
        s = _STEPS_MAX.search(low)
        if s:
            g = s.groups()
            a["max_steps"] = int(g[0] or g[1] or g[2]) - (1 if (g[1] or g[2]) else 0)
        if _PREFER_STEPS.search(low):
            a["prefer"] = "fewest_steps"
        k = _K.search(low)
        if k:
            a["k"] = _num(k.group(1))
        return a

    def _route(self, message):
        low = message.lower().strip()
        if _HELP.fullmatch(low) or self._unavailable_target(message) or (
            _RESULT_REFERENCE.fullmatch(low) or
            (self.last_ids and _BARE_NUMBER.fullmatch(low))
        ):
            return (*self.rule_route(message), "rules")
        if self.llm is not None and getattr(self.llm, "available", True):
            try:
                tool, args = self.llm.route(message, self.state_summary(), self.history)
                if tool in TOOLS:
                    return tool, args, "llm"
            except Exception as e:  # network, quota, bad model name ... fall back, never crash the chat
                self.last_error = f"{type(e).__name__}: {e}"
        return (*self.rule_route(message), "rules")

    # ------------------------------------------------------------ reference resolution
    def resolve(self, ref):
        ref = (ref or "this").strip()
        m = _ID.search(ref)
        if m and m.group(1) in self.pos:
            return m.group(1), "explicit id", None
        low = ref.lower()
        if low.isdigit() or low in ORD:
            if not self.last_ids:
                return None, None, "I haven't shown any recipes yet. Try 'find a saffron rice recipe'."
            n = int(low) if low.isdigit() else ORD[low]
            resolved = resolve_reference(ref, self.last_ids, self.active_id)
            if resolved:
                return resolved, f"result {n}", None
            return None, None, f"There is no result {n} in the last list." + self._list_hint()
        if re.match(r"(this|that|it|the recipe|current)\b", low):
            if self.active_id:
                return self.active_id, "the recipe we were working on", None
            if len(self.last_ids) == 1:
                return self.last_ids[0], "the only result", None
            if self.last_ids:
                return None, None, "Which recipe do you mean?" + self._list_hint()
            return None, None, "Which recipe? Search for one first, or give a recipe number or id."
        out = run_search(self.df, self.index, ref, k=1)
        if out["results"] and out["results"][0]["score"] >= 0.35:
            return out["results"][0]["id"], f"best title match for '{ref}'", None
        return None, None, f"I could not tell which recipe '{ref}' is." + self._list_hint()

    def _list_hint(self):
        if not self.last_ids:
            return ""
        return "\n" + "\n".join(f"  {i + 1}) {self.df.iloc[self.pos[r]]['title']}" for i, r in enumerate(self.last_ids[:5])) + "\nSay e.g. 'recipe 2'."

    # ------------------------------------------------------------ tools
    def t_reset(self, a):
        self.reset_state()
        return "Cleared the search, the ingredient list and all restrictions.", "Everything is cleared.", []

    def t_clarify(self, a):
        q = a.get("question") or "Could you tell me a bit more?"
        return q, q, []

    def _apply_constraints(self, a):
        warns = []
        ex, w = self._accept(a.get("exclude") or [], "restriction")
        warns += w
        for t in ex:
            if t not in self.exclude:
                self.exclude.append(t)
        inc, w = self._accept(a.get("include") or [], "requirement")
        warns += w
        for t in inc:
            if t not in self.include:
                self.include.append(t)
        for t in a.get("drop_exclude") or []:
            t = _norm_term(t)
            if t in self.exclude:
                self.exclude.remove(t)
        return warns

    def t_search(self, a):
        warns = self._apply_constraints(a)
        q = (a.get("query") or "").strip()
        if q:
            self.query = q
            self.active_id = None
        if not self.query:
            lines = [a["unavailable_notice"]] if a.get("unavailable_notice") else []
            lines += [f"WARNING: {w}" for w in warns]
            lines.append("What would you like to find?")
            return "\n".join(lines), "What would you like to find?", []
        k = int(a.get("k") or 5)
        out = run_search(self.df, self.index, self.query, self.include, self.exclude, k=k)
        self.query = out["query"]
        self.mode, self.last_ids = "search", [r["id"] for r in out["results"]]
        self.active_id = None
        L = [f"WARNING: {w}" for w in warns]
        if a.get("unavailable_notice"):
            L.append(a["unavailable_notice"])
        L.extend(out.get("interpreting", []))
        L.append(f"Search: '{out['query']}' | must include: {out['include'] or '-'} | excluded: {self.exclude or '-'}")
        for i, r in enumerate(out["results"], 1):
            step_label = "step" if r["n_steps"] == 1 else "steps"
            L.append(f"{i}. {r['title']}  ({r['n_steps']} {step_label})")
            L += [f"     - {e}" for e in r["evidence"]]
        if out["note"]:
            L.append(f"Note: {out['note']}")
        if out["results"]:
            L.append("Showing the top matches only. For counts or statistics over all matching recipes, ask a statistics question.")
        return "\n".join(L), self._speech_list(out["results"]), out["results"]

    def t_match(self, a):
        if self.match_fn is None:
            return "The ingredient-matching module is not available.", "Matching is unavailable.", []
        warns = self._apply_constraints({"exclude": a.get("exclude"), "drop_exclude": a.get("drop_exclude")})
        if a.get("have"):
            self.have = [t for t in (_norm_term(x) for x in a["have"]) if t]
        for t in a.get("add_have") or []:
            t = _norm_term(t)
            if t and t not in self.have:
                self.have.append(t)
        if a.get("max_steps") is not None:
            self.max_steps = int(a["max_steps"]) or None
        if a.get("prefer") in ("fewest_missing", "fewest_steps"):
            self.prefer = a["prefer"]
        for q in a.get("quantities") or []:
            try:
                self.quantities[_norm_term(q["name"])] = float(q["amount"])
            except (KeyError, TypeError, ValueError):
                pass
        if not self.have:
            return "Which ingredients do you have?", "Which ingredients do you have?", []
        out = self.match_fn(self.df, have=list(self.have), exclude=list(self.exclude), max_steps=self.max_steps, prefer=self.prefer,
                            k=int(a.get("k") or 5), quantities=dict(self.quantities) or None)
        self.mode, self.last_ids, self.active_id = "match", [r["id"] for r in out.get("results", [])], None
        text = "\n".join([f"WARNING: {w}" for w in warns] + [self.format_match_fn(out)])
        return text, self._speech_list(out.get("results", [])), out.get("results", [])

    def t_analyze(self, a):
        inc, w1 = self._accept(a.get("include") or [], "filter")
        exc, w2 = self._accept(a.get("exclude") or [], "exclusion")
        out = run_stats(self.df, a.get("metric") or "count_percent", inc, exc, int(a.get("k") or 5))
        text = "\n".join([f"WARNING: {w}" for w in w1 + w2] + [format_stats(out)])
        lines = text.splitlines()
        speech = next((l for l in lines if l.startswith(("Count:", "Ingredient lines", "1."))), lines[0])
        return text, speech + (" The calculation is on screen." if True else ""), []

    def t_recipe(self, a):
        ref = a.get("recipe_ref")
        if a.get("recipe_id"):
            ref = a["recipe_id"]
        elif a.get("position") is not None:
            ref = str(a["position"])
        if not ref and not self.last_ids:
            err = "I haven't shown any recipes yet. Try 'find a saffron rice recipe'."
            return err, err, []
        rid, how, err = self.resolve(ref)
        if err:
            return err, err.splitlines()[0], []
        r = self.df.iloc[self.pos[rid]]
        self.selected_recipe_id = rid
        steps = list(r.get("steps", []) or [])
        if not steps:
            steps = [line.strip() for line in str(r["instructions"]).splitlines() if line.strip()]
        text = (
            f"Recipe: {r['title']} [{rid}] (resolved from: {how})\n\n"
            "Ingredients (original dataset lines):\n"
            + "\n".join(f"  - {line}" for line in r["ingredients"])
            + "\n\nInstructions (numbered by non-empty original instruction line):\n"
            + "\n".join(f"  {number}. {step}" for number, step in enumerate(steps, 1))
        )
        return text, f"Here is {r['title']}. The ingredients and numbered steps are on screen.", []

    def t_help(self, _args):
        return HELP_TEXT, HELP_TEXT, []

    def t_substitute(self, a):
        rid, how, err = self.resolve(a.get("recipe_ref"))
        if err:
            return err, err.splitlines()[0], []
        target = _singular(a.get("target") or "")
        if not target:
            return "Which ingredient should be replaced?", "Which ingredient should be replaced?", []
        warns = self._apply_constraints({"exclude": a.get("restrictions")})
        self.active_id = rid
        out = find_substitutes(self.df, self.index, rid, target, list(self.exclude))
        self.last_substitution_result = rid
        self.last_substitution_output = out
        self.pending_target = None
        r = self.df.iloc[self.pos[rid]]
        text = "\n".join([f"WARNING: {w}" for w in warns] + [f"Using recipe: {r['title']} [{rid}] (resolved from: {how})", format_substitution(out)])
        if (out.get("status") in {"weak_evidence", "insufficient_evidence"}
                and out.get("adaptation") is None
                and "no adaptation generated" not in text.lower()):
            text += "\nNo adaptation generated."
        cand = out["adaptation"]["candidate"] if out.get("adaptation") else None
        speech = (f"For {r['title']}, the evidence status is {out['status'].replace('_', ' ')}."
                  + (f" The best supported replacement is {cand}, but the amount is not verified." if cand else " I could not support a replacement from the dataset.")
                  + " Details are on screen.")
        return text, speech, []

    # ------------------------------------------------------------ main entry
    def handle(self, message):
        self.last_error = None
        self.last_substitution_output = None
        tool, args, router = self._route(message)
        fn = {"search_recipes": self.t_search, "match_ingredients": self.t_match, "analyze_dataset": self.t_analyze,
              "get_recipe": self.t_recipe, "get_recipe_details": self.t_recipe, "help": self.t_help,
              "substitute_ingredient": self.t_substitute, "reset": self.t_reset, "clarify": self.t_clarify}.get(tool)
        if fn is None:
            tool, args, router = self.rule_route(message)
            fn = {"search_recipes": self.t_search, "match_ingredients": self.t_match, "analyze_dataset": self.t_analyze,
                  "get_recipe": self.t_recipe, "get_recipe_details": self.t_recipe, "help": self.t_help,
                  "substitute_ingredient": self.t_substitute, "reset": self.t_reset, "clarify": self.t_clarify}[tool]
        text, speech, _ = fn(args)
        summary = None
        if self.use_summary and self.llm is not None and tool not in ("reset", "clarify", "get_recipe", "get_recipe_details", "help"):
            try:
                summary = grounded(self.llm.summarize(message, text), text)
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
        self.history.append({"user": message[:200], "tool": tool})
        self.history = self.history[-6:]
        notice = None
        if self.last_error and self.last_error != self.last_notified_error:
            notice = "Gemini unavailable; used rule-based routing."
            self.last_notified_error = self.last_error
        output = {"text": (summary + "\n\n" if summary else "") + text, "summary": summary, "speech": summary or speech, "tool": tool,
                "action": {"get_recipe": "details", "get_recipe_details": "details",
                           "substitute_ingredient": "substitute"}.get(tool, tool),
                "args": args, "router": router,
                "llm_error": self.last_error, "llm_notice": notice,
                "llm_notice_key": self.last_error if notice else None}
        if self.last_substitution_output is not None:
            output["status"] = self.last_substitution_output["status"]
            output["adaptation"] = self.last_substitution_output["adaptation"]
        return output

    @staticmethod
    def _speech_list(results):
        titles = [r["title"] for r in results[:3]]
        return ("Top results: " + "; ".join(titles) + ". The details are on screen.") if titles else "I found no recipe that meets all the constraints."


TOOLS = {"search_recipes", "match_ingredients", "analyze_dataset", "get_recipe", "get_recipe_details",
         "substitute_ingredient", "reset", "help", "clarify"}


def grounded(summary, tool_text):
    """Return the summary only if every number and recipe id in it also appears in the tool output; else None."""
    if not summary:
        return None
    nums = lambda s: set(re.findall(r"\d+(?:\.\d+)?", s.replace(",", "")))
    if not nums(summary) <= nums(tool_text):
        return None
    if not set(ID_IN_TEXT.findall(summary)) <= set(ID_IN_TEXT.findall(tool_text)):
        return None
    return summary.strip()
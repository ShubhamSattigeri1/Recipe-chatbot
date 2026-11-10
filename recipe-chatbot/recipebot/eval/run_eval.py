"""Reproducible evaluation. Run from the repo root:
    python eval/run_eval.py                    # runs eval/cases.json
    python eval/run_eval.py --only T1          # only cases whose id starts with T1
    python eval/run_eval.py --cases eval/other.json
Deterministic: it drives Chat (rule-based router, llm=None) and the engines directly, no LLM and no network.
Every check re-verifies the reply against the DATASET (ids are looked up in the DataFrame and re-filtered with
search.hard_filter), so it tests grounding, not just wording. Writes eval/results.md."""
import argparse, json, re, sys, time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from recipebot.data import load_recipes
from recipebot.search import SearchIndex, expand_terms, hard_filter
from recipebot.chat import Chat

ID_RE = re.compile(r"\[((?:ar|epi|fn)_[^\]\s]+)\]")


def mask(df, include=(), exclude=()):
    return np.asarray(hard_filter(df, [t.lower() for t in include], expand_terms(exclude)), dtype=bool)


class Ctx:
    def __init__(self, df, index):
        self.df, self.index = df, index
        self.pos = {i: k for k, i in enumerate(df["id"])}

    def rows(self, ids):
        return self.df.iloc[[self.pos[i] for i in ids if i in self.pos]]


def check_text(ctx, text, c, elapsed):
    """Returns (ok, detail) for one check dict against a reply text."""
    t = c["type"]
    scope = text.split(c["before"])[0] if c.get("before") and c["before"] in text else text
    if t == "contains":
        return c["text"].lower() in text.lower(), f"contains {c['text']!r}"
    if t == "not_contains":
        return c["text"].lower() not in text.lower(), f"not_contains {c['text']!r}"
    if t == "regex":
        return bool(re.search(c["pattern"], text, re.I | re.S)), f"regex {c['pattern']!r}"
    if t == "not_regex":
        return not re.search(c["pattern"], text, re.I | re.S), f"not_regex {c['pattern']!r}"
    if t == "max_seconds":
        return elapsed <= c["value"], f"{elapsed:.1f}s <= {c['value']}s"
    if t == "ids_count":
        n = len(set(ID_RE.findall(scope)))
        return n == c["equals"], f"ids before marker = {n}, expected {c['equals']}"
    if t == "ids_satisfy":
        ids = list(dict.fromkeys(ID_RE.findall(scope)))
        if not ids:
            return False, "no recipe ids found in reply"
        sub = ctx.rows(ids)
        bad = []
        if c.get("include") is not None or c.get("exclude") is not None:
            ok = mask(sub, c.get("include", []), c.get("exclude", []))
            bad += [f"{i} violates include/exclude" for i, o in zip(sub["id"], ok) if not o]
        if c.get("max_steps") is not None:
            bad += [f"{i} has {n} steps" for i, n in zip(sub["id"], sub["n_steps"]) if n > c["max_steps"]]
        return not bad, f"{len(ids)} ids re-verified against dataset; " + (f"FAILED: {bad[:3]}" if bad else "all satisfy")
    raise ValueError(f"unknown check type {t}")


def get_path(d, path):
    for k in path.split("."):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def check_tool(ctx, out, c):
    t = c["type"]
    if t == "field_equals":
        v = c["value"]
        v = len(ctx.df) if v == "@n_dataset" else v
        got = get_path(out, c["field"])
        return got == v, f"{c['field']} = {got}, expected {v}"
    if t == "independent_count":
        exp = int(mask(ctx.df, c.get("include", []), c.get("exclude", [])).sum())
        got = get_path(out, c["field"])
        return got == exp, f"{c['field']} = {got}; independent recount with search.hard_filter = {exp}"
    if t == "sorted_by":
        vals = [r[c["key"]] for r in get_path(out, c["list"])]
        ok = vals == sorted(vals, reverse=(c.get("order") == "desc"))
        return ok and len(vals) > 0, f"{c['key']} sequence {vals[:8]}"
    if t == "results_satisfy":
        ids = [r["id"] for r in get_path(out, c["list"])]
        sub = ctx.rows(ids)
        ok = mask(sub, c.get("include", []), c.get("exclude", []))
        return bool(len(ids)) and bool(ok.all()), f"{len(ids)} result ids re-verified (include={c.get('include')}, exclude={c.get('exclude')})"
    if t == "status_in":
        return out["status"] in c["values"], f"status = {out['status']}, allowed {c['values']}"
    if t == "field_is_none":
        return get_path(out, c["field"]) is None, f"{c['field']} is {get_path(out, c['field'])!r}"
    raise ValueError(f"unknown tool check {t}")


def call_tool(ctx, name, args):
    mod, fn = name.split(".")
    if mod == "analytics":
        from recipebot import analytics as A
        return getattr(A, fn)(ctx.df, **args)
    if mod == "match":
        from recipebot import match as M
        return getattr(M, fn)(ctx.df, **args)
    if mod == "substitute":
        from recipebot import substitute as S
        return getattr(S, fn)(ctx.df, ctx.index, **args)
    raise ValueError(name)


def run_case(ctx, case):
    results = []
    if "tool" in case:
        t0 = time.time()
        try:
            out = call_tool(ctx, case["tool"], case.get("args", {}))
        except Exception as e:
            return [("tool call", False, f"EXCEPTION {type(e).__name__}: {e}")]
        el = time.time() - t0
        for c in case["checks"]:
            ok, d = check_tool(ctx, out, c)
            results.append((f"{case['tool']}", ok, d))
        return results
    s = Chat(ctx.df, ctx.index, llm=None)  # rule-based router: deterministic, no network
    for turn in case["turns"]:
        t0 = time.time()
        try:
            text = s.handle(turn["say"])["text"]
        except Exception as e:
            results.append((turn["say"], False, f"EXCEPTION {type(e).__name__}: {e}"))
            continue
        el = time.time() - t0
        for c in turn.get("checks", []):
            ok, d = check_text(ctx, text, c, el)
            results.append((turn["say"], ok, d))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="eval/cases.json")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    cases = [c for c in json.load(open(a.cases, encoding="utf-8")) if c["id"].startswith(a.only)]
    print("Loading data + index ...")
    df = load_recipes()
    ctx = Ctx(df, SearchIndex.load_or_build(df))
    md = ["# Evaluation results", "", f"Dataset: {len(df):,} recipes. Cases: {len(cases)}. Every check re-verifies against the dataset.", "",
          "| Case | Task | What it tests | Result |", "|---|---|---|---|"]
    detail, total, passed = [], 0, 0
    for case in cases:
        res = run_case(ctx, case)
        ok_all = all(ok for _, ok, _ in res)
        total += 1; passed += ok_all
        print(f"[{'PASS' if ok_all else 'FAIL'}] {case['id']}: {case['name']}")
        md.append(f"| {case['id']} | {case['task']} | {case['name']} | {'PASS' if ok_all else '**FAIL**'} |")
        detail.append(f"\n### {case['id']} {case['name']}\n_{case.get('why', '')}_\n")
        for say, ok, d in res:
            line = f"- {'PASS' if ok else 'FAIL'} | `{say[:60]}` | {d}"
            detail.append(line)
            if not ok:
                print("    ", line)
    md += ["", f"**{passed}/{total} cases passed.**", "", "## Details"] + detail
    Path("eval/results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n{passed}/{total} cases passed. Details: eval/results.md")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
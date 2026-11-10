"""Run from the repo root:  python tools/snapshot.py
Prints a compact summary of the project (files, function signatures, key settings, test results)
and also saves it to snapshot.txt so it can be pasted into the chat. Never prints secret values."""
import ast, hashlib, inspect, os, subprocess, sys, io, contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
out = io.StringIO()
def p(*a): print(*a); print(*a, file=out)

p("== ENV ==")
p("python", sys.version.split()[0], "| GEMINI_API_KEY set:", bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
  "| GEMINI_MODEL:", os.environ.get("GEMINI_MODEL", "-"))
p("requirements.txt:", (ROOT / "requirements.txt").read_text().split() if (ROOT / "requirements.txt").exists() else "MISSING")

p("\n== FILES (lines, hash) ==")
files = sorted([*ROOT.glob("recipebot/*.py"), *ROOT.glob("tests/*.py"), *ROOT.glob("tools/*.py"), *ROOT.glob("*.py"), *ROOT.glob("*.md"), *ROOT.glob("eval/*")])
for f in files:
    if f.is_file():
        b = f.read_bytes(); p(f"{f.relative_to(ROOT)}  {len(b.splitlines())} lines  {hashlib.md5(b).hexdigest()[:8]}")

p("\n== SIGNATURES (recipebot/*.py) ==")
for f in sorted(ROOT.glob("recipebot/*.py")):
    try:
        tree = ast.parse(f.read_text(encoding="utf-8"))
    except SyntaxError as e:
        p(f"[{f.name}] SYNTAX ERROR: {e}"); continue
    doc = (ast.get_docstring(tree) or "").splitlines()
    p(f"\n[{f.name}] {doc[0] if doc else '(no docstring)'}")
    imps = sorted({("." * n.level) + (n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level})
    if imps: p("  imports from package:", imps)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            p(f"  def {n.name}({ast.unparse(n.args)})")
        elif isinstance(n, ast.ClassDef):
            p(f"  class {n.name}: " + ", ".join(m.name for m in n.body if isinstance(m, ast.FunctionDef)))
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets):
            name = [t.id for t in n.targets][0]
            try: v = ast.literal_eval(n.value); keys = list(v) if isinstance(v, (dict, set, list, tuple)) else v
            except Exception: keys = "..."
            s = str(keys); p(f"  {name} = {s[:160]}{'...' if len(s) > 160 else ''}")

p("\n== KEY FACTS ==")
try:
    from recipebot import search
    sig = inspect.signature(search._term_mask)
    p("search._term_mask:", sig, "| DERIVATIVES present:", hasattr(search, "DERIVATIVES"), "| DIFFERENT_ITEMS present:", hasattr(search, "DIFFERENT_ITEMS"))
    p("search.GROUPS:", list(search.GROUPS))
except Exception as e:
    p("search import FAILED:", repr(e))
for mod, fn in [("match", "match_ingredients"), ("substitute", "find_substitutes"), ("session", "Session"), ("chat", "Chat"), ("stats", "run_stats"), ("llm", "Gemini")]:
    try:
        m = __import__(f"recipebot.{mod}", fromlist=[fn]); p(f"recipebot.{mod}.{fn}:", inspect.signature(getattr(m, fn)))
    except Exception as e:
        p(f"recipebot.{mod}.{fn}: not available ({type(e).__name__}: {str(e)[:80]})")
pk = ROOT / "data" / "recipes.pkl"
if pk.exists():
    try:
        import pandas as pd; df = pd.read_pickle(pk); p("recipes.pkl:", len(df), "rows | columns:", list(df.columns))
    except Exception as e: p("recipes.pkl unreadable:", repr(e))
p("data/ files:", sorted(x.name for x in (ROOT / "data").glob("*")) if (ROOT / "data").exists() else "no data dir")

p("\n== TESTS ==")
for t in sorted(ROOT.glob("tests/test_*.py")):
    try:
        r = subprocess.run([sys.executable, str(t)], capture_output=True, text=True, timeout=600, cwd=ROOT)
        tail = (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0 else (r.stderr.strip().splitlines() or ["?"])[-1]
        p(f"{t.name}: {'PASS' if r.returncode == 0 else 'FAIL'} | {tail[:200]}")
    except subprocess.TimeoutExpired:
        p(f"{t.name}: TIMEOUT")
(ROOT / "snapshot.txt").write_text(out.getvalue(), encoding="utf-8")
print("\nSaved to snapshot.txt")
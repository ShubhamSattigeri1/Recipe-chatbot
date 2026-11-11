"""Step 1a: load the 3 Recipe Box JSON files into one DataFrame with stable IDs."""
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ZIP_URL = "https://storage.googleapis.com/recipe-box/recipes_raw.zip"  # linked from eightportions.com/datasets/Recipes
SOURCES = {  # prefix -> file. ID = "<prefix>_<original key>", e.g. ar_wXu3QbyFGoH
    "ar": "recipes_raw_nosource_ar.json",
    "epi": "recipes_raw_nosource_epi.json",
    "fn": "recipes_raw_nosource_fn.json",
}
CACHE = DATA_DIR / "recipes.pkl"
_AD = re.compile(r"\s*ADVERTISEMENT\s*$", re.IGNORECASE)  # often glued to the END of a real line
JUNK_INGREDIENTS = {"advertisement"}  # scraper artifact, not an ingredient


def download_if_missing():
    DATA_DIR.mkdir(exist_ok=True)
    if all((DATA_DIR / f).exists() for f in SOURCES.values()):
        return
    zpath = DATA_DIR / "recipes_raw.zip"
    print(f"Downloading {ZIP_URL} ...")
    urllib.request.urlretrieve(ZIP_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():  # flatten any folder structure
            if name.endswith(".json"):
                (DATA_DIR / Path(name).name).write_bytes(z.read(name))


def split_steps(instructions: str) -> list:
    """Split a recipe instruction string into sentence-like steps."""
    if not instructions:
        return []
    chunks = re.split(r"\n+|(?<=[.!?])\s+", str(instructions).strip())
    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]


def load_recipes(use_cache=True, data_dir=None) -> pd.DataFrame:
    """Returns one row per usable recipe. Titles/ingredients/instructions are the ORIGINAL strings;
    only the 'ADVERTISEMENT' junk ingredient lines are removed. Recipes missing a title,
    ingredients or instructions are dropped (counted in df.attrs['n_dropped'])."""
    global DATA_DIR
    if data_dir is not None:
        DATA_DIR = Path(data_dir)
    cache = DATA_DIR / "recipes.pkl"
    if use_cache and cache.exists():
        df = pd.read_pickle(cache)
        return df
    download_if_missing()

    rows, dropped = [], 0
    for src, fname in SOURCES.items():
        with open(DATA_DIR / fname, encoding="utf-8") as f:
            raw = json.load(f)
        for key, r in raw.items():
            if not isinstance(r, dict):
                dropped += 1
                continue
            title = (r.get("title") or "").strip()
            ings = [_AD.sub("", i).strip() for i in (r.get("ingredients") or []) if isinstance(i, str)]
            ings = [i for i in ings if i and i.lower() not in JUNK_INGREDIENTS]
            instr = r.get("instructions") or ""
            steps = split_steps(instr)
            if not (title and ings and steps):
                dropped += 1
                continue
            rows.append({
                "id": f"{src}_{key}", "source": src, "title": title,
                "ingredients": ings, "instructions": instr, "steps": steps,
                "n_ingredients": len(ings), "n_steps": len(steps),
                "ing_text": "\n".join(ings).lower(),  # precomputed for fast matching
            })
    df = pd.DataFrame(rows)
    df.attrs["n_dropped"] = dropped
    df.to_pickle(cache)
    return df


if __name__ == "__main__":
    d = load_recipes(use_cache=False)
    print(f"{len(d):,} usable recipes | dropped {d.attrs.get('n_dropped')} incomplete")
    print(d.groupby("source").size())
    print(d.iloc[0][["id", "title", "n_ingredients", "n_steps"]])
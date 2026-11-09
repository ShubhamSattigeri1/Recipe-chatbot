"""Run: python -m recipebot.cli   (type 'reset' to clear restrictions, 'quit' to exit)"""
from .data import load_recipes
from .search import SearchIndex
from .session import Session, format_results

def main():
    df = load_recipes()
    print(f"Loaded {len(df):,} recipes.")
    s = Session(df, SearchIndex.load_or_build(df))
    print("Ask for a recipe, e.g. 'Find a penne pasta recipe', then 'I cannot have dairy'.")
    while True:
        try:
            msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in {"quit", "exit"}:
            break
        if msg:
            print(format_results(s.handle(msg)))

if __name__ == "__main__":
    main()
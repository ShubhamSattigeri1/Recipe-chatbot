<<<<<<< HEAD
"""Run: python -m recipebot.cli [--debug] [--no-llm]."""
import argparse

from .chat import Chat
from .data import load_recipes
from .llm import Gemini
from .search import SearchIndex


def main(argv=None):
    parser = argparse.ArgumentParser(description="Chat with the local recipe collection.")
    parser.add_argument("--debug", action="store_true", help="Show detailed Gemini errors.")
    parser.add_argument("--no-llm", action="store_true", help="Disable Gemini and use rule-based routing.")
    parser.add_argument("--list-models", action="store_true", help="List Gemini models supporting generateContent.")
    parser.add_argument(
        "--build-hybrid-index",
        action="store_true",
        help="Build and cache the local semantic index (one-time, may take hours).",
    )
    args = parser.parse_args(argv)

    if args.list_models:
        gemini = Gemini(discover_model=False)
        try:
            print("\n".join(gemini.list_models()))
        except Exception as error:
            print("Unable to list Gemini models.")
            if args.debug:
                print(f"{type(error).__name__}: {error}")
            return 1
        return 0

    df = load_recipes()
    if args.build_hybrid_index:
        index = SearchIndex.load_or_build(df, semantic=True)
        if index.semantic_X is None:
            print(index.semantic_notice or "Unable to build the hybrid search index.")
            return 1
        print(f"Hybrid search index is ready for {len(df):,} recipes.")
        return 0

    gemini = None if args.no_llm else Gemini()
    if gemini and gemini.configuration_notice:
        print(gemini.configuration_notice)
        if args.debug and gemini.configuration_error:
            print(gemini.configuration_error)
    elif gemini and gemini.auto_selected:
        print(f"GEMINI_MODEL not set; selected available Flash model {gemini.model}.")
    index = SearchIndex.load_or_build(df, semantic=True, build_semantic=False)
    if index.semantic_notice:
        print(index.semantic_notice)
    chat = Chat(df, index, llm=gemini if gemini and gemini.available else None)
    print(f"Loaded {len(df):,} recipes | router: {'Gemini ' + gemini.model if gemini and gemini.available else 'rules'}")
    print("Try: 'find a penne pasta recipe', then 'I cannot have dairy'.")
    last_notice_key = None
=======
"""Run: python -m recipebot.cli   (type 'reset' to clear restrictions, 'quit' to exit)"""
from .data import load_recipes
from .search import SearchIndex
from .session import Session, format_results

def main():
    df = load_recipes()
    print(f"Loaded {len(df):,} recipes.")
    s = Session(df, SearchIndex.load_or_build(df))
    print("Ask for a recipe, e.g. 'Find a penne pasta recipe', then 'I cannot have dairy'.")
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd
    while True:
        try:
            msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in {"quit", "exit"}:
            break
        if msg:
<<<<<<< HEAD
            out = chat.handle(msg)
            print(out["text"])
            notice = out.get("llm_notice")
            notice_key = out.get("llm_notice_key")
            if notice and notice_key != last_notice_key:
                print(notice)
                last_notice_key = notice_key
            if out.get("llm_error"):
                if args.debug:
                    print(f"[Gemini error: {out['llm_error']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
=======
            print(format_results(s.handle(msg)))

if __name__ == "__main__":
    main()
>>>>>>> 13df6dde3f08ec0c782e7972ef15bc78752194cd

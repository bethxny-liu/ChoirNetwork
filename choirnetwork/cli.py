"""Build, search, evaluate, and serve the hymn index."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from choirnetwork.engine import HymnSimilarityEngine, load_engine
from choirnetwork.scraper import load_hymns, save_hymns, save_missing_lyrics, scrape_hymnal


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChoirNetwork: semantic hymn search")
    commands = parser.add_subparsers(dest="command", required=True)

    scrape = commands.add_parser("scrape", help="Fetch website lyrics into a separate staging file")
    scrape.add_argument("--output", type=Path, default=Path("data/raw/scraped_hymns.json"))
    scrape.add_argument("--missing-output", type=Path, default=Path("data/raw/missing_lyrics.json"))

    build = commands.add_parser("build", help="Embed the reviewed local corpus")
    build.add_argument("--input", type=Path, default=Path("data/raw/hymns.json"))
    build.add_argument("--output", type=Path, default=Path("data/index"))

    search = commands.add_parser("search", help="Recommend hymns for a sermon title")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, choices=range(1, 51), default=10, metavar="1-50")

    serve = commands.add_parser("serve", help="Run the web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    evaluate = commands.add_parser("eval", help="Compare BM25 and dense retrieval ablations")
    evaluate.add_argument("--eval-file", type=Path, default=Path("eval/datasets/service_hymns.csv"))
    evaluate.add_argument("--top-k", type=int, choices=range(1, 51), default=5, metavar="1-50")
    evaluate.add_argument("--split", choices=("development", "test"), default="development")
    evaluate.add_argument("--results-dir", type=Path, default=Path("eval/results/current"))
    evaluate.add_argument("--confirm-held-out", action="store_true")
    for command in (search, serve, evaluate):
        command.add_argument("--index", type=Path, default=Path("data/index"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "scrape":
        destinations = {args.output.resolve(), args.missing_output.resolve()}
        protected_corpus = Path("data/raw/hymns.json").resolve()
        if protected_corpus in destinations or len(destinations) != 2:
            raise SystemExit("Choose separate output files and keep the reviewed corpus path unchanged.")
        hymns, missing = scrape_hymnal()
        save_hymns(hymns, args.output)
        save_missing_lyrics(missing, args.missing_output)
        print(f"Saved {len(hymns)} hymns to {args.output}; {len(missing)} missing lyrics")
    elif args.command == "build":
        engine = HymnSimilarityEngine.from_raw_hymns(load_hymns(args.input))
        engine.index.save(args.output)
        print(f"Built {len(engine.index.slugs)} hymns at {args.output}")
    elif args.command == "search":
        for rank, hymn in enumerate(load_engine(args.index).search(args.query, top_k=args.top_k), 1):
            print(f"{rank}. Hymn {hymn.label} — {hymn.title}")
    elif args.command == "serve":
        import uvicorn

        os.environ["CHOIRNETWORK_INDEX_PATH"] = str(args.index)
        uvicorn.run("choirnetwork.api:app", host=args.host, port=args.port)
    elif args.command == "eval":
        from choirnetwork.eval import compare_retrieval_configs, write_evaluation_report

        if args.split == "test" and not args.confirm_held_out:
            raise SystemExit("Freeze the configuration before using --confirm-held-out. Do not tune on test results.")
        report = compare_retrieval_configs(args.index, args.eval_file, k=args.top_k, split=args.split)
        path = write_evaluation_report(report, args.results_dir)
        for result in report["results"]:
            print(f"{result['name']}: nDCG@{args.top_k}={result['metrics']['ndcg']:.4f}")
        print(f"Wrote {path}")

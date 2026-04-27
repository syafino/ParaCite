"""ParaCite command-line interface.

Subcommands:

    paracite serve                              # run the FastAPI app
    paracite ingest <path>                      # synchronous ingest
    paracite ask "<text>" [--top-k N]           # synchronous ask
    paracite ask --file <path> [--top-k N]      # ask using file contents

Both ``ingest`` and ``ask`` go through the exact same core services as the
HTTP API so behavior stays identical between transports.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import API_HOST, API_PORT


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "src.app.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from src.app import get_ingest_service

    path = Path(args.path)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    def on_progress(stage: str, info: dict) -> None:
        if args.verbose:
            extras = " ".join(f"{k}={v}" for k, v in info.items())
            print(f"  [{stage}] {extras}", file=sys.stderr)

    try:
        result = get_ingest_service().ingest(path, on_progress=on_progress)
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from src.app import get_ask_service

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 1
        text = path.read_text(encoding="utf-8", errors="replace")
    elif args.text:
        text = args.text
    else:
        print("error: provide either positional <text> or --file PATH", file=sys.stderr)
        return 1

    result = get_ask_service().ask(text, top_k=args.top_k, style=args.style)
    print(json.dumps(result, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paracite",
        description="ParaCite -- citable-claim extraction + retrieval",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose logging",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="run the HTTP API (uvicorn)")
    p_serve.add_argument("--host", default=API_HOST)
    p_serve.add_argument("--port", type=int, default=API_PORT)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    p_ingest = sub.add_parser("ingest", help="ingest a .txt/.md/.pdf file (sync)")
    p_ingest.add_argument("path", type=str, help="path to file")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", help="run claim extraction + retrieval (sync)")
    p_ask.add_argument("text", nargs="?", default=None, help="text to analyze")
    p_ask.add_argument("--file", default=None, help="read text from a file instead")
    p_ask.add_argument("--top-k", type=int, default=3)
    p_ask.add_argument("--style", default="apa")
    p_ask.set_defaults(func=_cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

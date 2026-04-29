"""ParaCite entrypoint -- delegates to the CLI in ``src.app.cli``."""

from src.app.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

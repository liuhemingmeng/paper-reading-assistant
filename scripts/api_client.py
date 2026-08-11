"""Call a JSON API and save the response without third-party dependencies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_env_file(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs without overwriting existing variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        os.environ.setdefault(key.strip(), value.strip())


def fetch_json(url: str, api_key: str | None = None, timeout: float = 10.0) -> object:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"API returned HTTP {error.code}") from error
    except URLError as error:
        raise RuntimeError(f"API request failed: {error.reason}") from error
    except TimeoutError as error:
        raise RuntimeError("API request timed out") from error
    except json.JSONDecodeError as error:
        raise RuntimeError("API response was not valid JSON") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch JSON from an API and save it.")
    parser.add_argument("url", nargs="?", help="JSON API URL; defaults to API_BASE_URL from .env.")
    parser.add_argument("--output", type=Path, default=Path("data/api_response.json"))
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    load_env_file()
    url = args.url or os.getenv("API_BASE_URL")
    if not url or url == "https://example.com/api":
        raise SystemExit("Provide a real URL or set API_BASE_URL in .env.")

    try:
        result = fetch_json(url, os.getenv("API_KEY"), args.timeout)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved API response to {args.output}")


if __name__ == "__main__":
    main()

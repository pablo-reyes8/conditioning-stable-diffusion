"""Legacy compatibility wrapper.

Use `python3 scripts/data.py build-manifest` and
`python3 scripts/data.py filter-archive` instead of editing this file manually.
"""

from scripts.data import main


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail if the same module-level function name is defined in more than one
file under scripts/, tools/, or bot/.

Catches copy-pasted helpers before they rot. Imports do NOT count: a function
defined in a shared module and imported by consumers has a single definition,
so this passes. Exemptions are curated per-name (see EXEMPT) for deliberate
reimplementations and known pre-existing duplication.

Usage:
    python .github/scripts/check_duplicate_defs.py [DIR ...]
"""

import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIRS = ["scripts", "tools", "bot"]

# Curated exemptions. Each name is a deliberate reimplementation — NOT new
# drift:
#   main / run       — entry-point convention in standalone scripts
#   init_mt5         — mt5_connect.init_mt5 (RPyC bridge init) vs
#                      tune_scaleout.init_mt5 (login wrapper)
#   mt5_order_send   — frame-sensitive MT5 order wrappers: execution.py
#                      (retry/timeout) + main.py (thin alias; see comment at
#                      main.py top)
EXEMPT = {
    "main", "run",
    "init_mt5", "mt5_order_send",
}


def module_level_funcs(path: Path):
    """Yield module-level function names defined in `path`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name


def check(dirs):
    owners = defaultdict(list)
    for d in dirs:
        base = Path(d) if Path(d).is_absolute() else ROOT / d
        for py in sorted(base.glob("*.py")):
            try:
                names = list(module_level_funcs(py))
            except (SyntaxError, OSError) as e:
                print(f"ERROR: could not parse {py}: {e}")
                return 1
            for name in names:
                if name not in EXEMPT:
                    owners[name].append(str(py))
    dupes = {n: fs for n, fs in owners.items() if len(fs) > 1}
    if not dupes:
        print("OK: no duplicate module-level function names across "
              + ", ".join(str(d) for d in dirs))
        return 0
    for name in sorted(dupes):
        print(f"DUPLICATE '{name}' defined in:")
        for f in sorted(dupes[name]):
            print(f"  {f}")
    print("\nRefactor these into a shared _common.py "
          "(see scripts/_common.py, tools/_common.py).")
    return 1


def main():
    dirs = sys.argv[1:] or DEFAULT_DIRS
    sys.exit(check(dirs))


if __name__ == "__main__":
    main()

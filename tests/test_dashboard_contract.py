"""Dashboard writer/template contract (architecture plan C6).

Every key `bot/dashboard.py` emits must be consumed by the template — dead
fields are found by this test, not by audit. The writer previously shipped
`ticket` (and before that margin/exec_quality/correlation/timestamp) that no
consumer read; the template grep below makes that class of drift a build failure.
"""

import re
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent.parent / "bot"
TEMPLATE = Path(__file__).resolve().parent.parent / "dashboard" / "templates" / "index.html"


def _emitted_keys():
    """Quoted dict keys at line-start in dashboard.py (the state dict literal,
    including the nested positions_detail/health dicts)."""
    src = (BOT_DIR / "dashboard.py").read_text(encoding="utf-8")
    return set(re.findall(r'^\s*"(\w+)":', src, re.M))


def test_every_emitted_key_is_consumed_by_template():
    template = TEMPLATE.read_text(encoding="utf-8")
    unused = {k for k in _emitted_keys() if k not in template}
    assert unused == set(), (
        f"dashboard.py emits keys the template never reads: {sorted(unused)}. "
        "Remove them from the writer or render them in index.html."
    )


def test_template_reference_points_are_written():
    """The other direction: every state.* the template reads must be emitted."""
    template = TEMPLATE.read_text(encoding="utf-8")
    src = (BOT_DIR / "dashboard.py").read_text(encoding="utf-8")
    for m in re.finditer(r"state\.(\w+)", template):
        key = m.group(1)
        if key not in src:
            raise AssertionError(f"template reads state.{key} but dashboard.py never emits it")
